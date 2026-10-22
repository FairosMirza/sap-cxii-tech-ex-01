"""Relevance cutoff for semantic search, from the development labels (rule fixed in docs/decisions.md D12).

The cutoff applies to the cosine between the query's semantic text and a product under the served model, before
bonuses. It is the highest value that keeps every judged development pair graded 2 or 3, rounded down to 4
decimals. Reported: what it removes from the served model's development top-10 (and any lower-ranked item that
moves up), the queries left empty, and a leave-one-query-out estimate (cutoff from 9 queries, applied to the 10th).
Run from the repository root:  python -m scripts.relevance_cutoff
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from app.artifacts import load
from app.query_parser import parse_query
from app.semantic import SemanticSearcher
from app.settings import load_config, load_serving
from scripts.evaluate import check_fresh

K = 10
USEFUL = 2


def cutoff(similarities: list[float]) -> float | None:
    return math.floor(min(similarities) * 10_000) / 10_000 if similarities else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--out", default="reports/relevance_cutoff.json")
    args = parser.parse_args()

    qrels = json.loads(Path("eval/qrels_dev.json").read_text())
    check_fresh(qrels, Path("eval/labels.jsonl"), Path("eval/pools"), "dev")
    judged = qrels["judgements"]["semantic"]
    store, _ = load(Path(args.artifacts))
    model = load_serving()["semantic_model"]
    modality, ids, index = store.modalities[model], store.catalog.ids, store.catalog.index
    queries = json.loads(Path("eval/semantic_queries.json").read_text())["dev"]

    vectors = {q: modality.query_encoder(parse_query(q).semantic_text) for q in queries}
    pairs = {q: [(pid, grade, float(modality.vectors[index[pid]] @ vectors[q])) for pid, grade in judged[q].items()] for q in queries}
    useful = {q: [s for _, g, s in pairs[q] if g >= USEFUL] for q in queries}
    value = cutoff([s for q in queries for s in useful[q]])

    plain = SemanticSearcher(store, model, load_config())
    gated = SemanticSearcher(store, model, load_config(), min_similarity=value)
    removed, added, per_query, loqo = Counter(), Counter(), {}, Counter()
    for q in queries:
        grade_of = judged[q]
        before = [ids[r] for r in plain.search(q, K).rows]
        after = [ids[r] for r in gated.search(q, K).rows]
        sims = {pid: s for pid, _, s in pairs[q]}
        for pid in set(before) - set(after):
            removed[grade_of[pid]] += 1
        for pid in set(after) - set(before):
            added[grade_of.get(pid, "unjudged")] += 1
        per_query[q] = {"results_before": len(before), "results_after": len(after),
                        "useful_judged": len(useful[q]), "min_useful_similarity": round(min(useful[q]), 4) if useful[q] else None}
        held_out = cutoff([s for other in queries if other != q for s in useful[other]])
        for pid in before:
            if sims[pid] < held_out:
                loqo[{0: "grade 0", 1: "grade 1"}.get(grade_of[pid], "grade 2-3")] += 1

    shown = Counter(judged[q][ids[r]] for q in queries for r in plain.search(q, K).rows)
    served = {q: [(judged[q][ids[r]], float(modality.vectors[r] @ vectors[q])) for r in plain.search(q, K).rows] for q in queries}
    no_useful = [q for q in queries if not useful[q]]
    emptying = max((s for q in no_useful for _, s in served[q]), default=None)
    report = {
        "rule": "docs/decisions.md D12", "model": model, "min_similarity": value,
        "judged_pairs": sum(len(p) for p in pairs.values()), "useful_pairs": sum(len(u) for u in useful.values()),
        "served_top10_before": {f"grade {g}": shown[g] for g in sorted(shown)},
        "removed_from_served_top10": {f"grade {g}": removed[g] for g in sorted(removed)},
        "moved_up_into_top10": {str(g): n for g, n in added.items()},
        "queries_left_empty": [q for q, row in per_query.items() if row["results_after"] == 0],
        "leave_one_query_out_removed": dict(loqo),
        "alternative_emptying_queries_without_useful_results": None if emptying is None else {
            "queries": no_useful, "cutoff_above": round(emptying, 4),
            "useful_judged_pairs_below_it": sum(s <= emptying for q in queries for s in useful[q]),
            "useful_served_results_below_it": sum(g >= USEFUL and s <= emptying for q in queries for g, s in served[q])},
        "per_query": per_query,
    }
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_query"}, indent=1, ensure_ascii=False))
    for q, row in per_query.items():
        print(f"  {row['results_before']:>2} -> {row['results_after']:>2}  useful {row['useful_judged']:>2}  min useful sim {row['min_useful_similarity']}  {q}")


if __name__ == "__main__":
    main()
