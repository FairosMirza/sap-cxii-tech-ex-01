"""Reporting checks on the development and holdout evaluations: verification only, nothing is tuned.

For each split and task, from the merged judgements and the pooled rankers' top-10:
- P@5 over all queries, a query without any relevant judged product counting as 0, beside the reported P@5,
  which (like nDCG@10) leaves such queries out;
- judgement coverage: every result each ranker returns and how many are unjudged (never read as grade 0);
- each query without a relevant judged product, with the grades of what every ranker returned for it;
- paired differences for every pair of compared rankers: nDCG@10 and P@5 over the scored queries, P@5 over all.
Intervals are the evaluation's own (scripts.evaluate): 95% percentile bootstrap over queries, resampled with
replacement 10,000 times with a fixed seed. A paired difference resamples per-query differences, so each query's
two scores stay together. Each interval covers one pair and is not adjusted for multiple comparisons.
The holdout rankings are recomputed only after the freeze check (scripts.freeze), so they are the evaluated ones.
Run from the repository root:  python -m scripts.evaluation_checks
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.build_label_pools import SERVED_RANKERS, semantic_rankings, similar_rankings, store_with_embeddings
from scripts.check_labels import labels_file, ranker_specs
from scripts.evaluate import BOOTSTRAP, RELEVANT, SEED, check_fresh, interval, paired, query_metrics

COMPARED = {"similar": ("C_bge", "C_minilm", "B_tfidf", "A_structured"), "semantic": ("bge-small", "minilm", "tfidf")}
POOLS = Path("eval/pools")


def precision_at_5(ranking: list[str], judged: dict[str, int]) -> float | None:
    grades = [judged.get(pid) for pid in ranking[:5]]
    return None if None in grades else sum(g >= RELEVANT for g in grades) / 5


def _status(grade: int | None) -> str:
    return "unjudged" if grade is None else f"grade_{grade}"


def task_checks(task: str, rankings: dict[str, dict[str, list[str]]], judgements: dict[str, dict[str, int]]) -> dict:
    specs = sorted({spec for ranked in rankings.values() for spec in ranked})
    queries = list(rankings)
    scored = [q for q in queries if query_metrics([], judgements.get(q, {})) is not None]
    values: dict[str, dict[str, list]] = {}
    report: dict = {"queries": len(queries), "scored_queries": len(scored), "rankers": {}}
    for spec in specs:
        rows = [query_metrics(rankings[q][spec], judgements[q]) for q in scored]
        values[spec] = {"ndcg": [r["ndcg"] for r in rows], "p5": [r["p5"] for r in rows],
                        "p5_all": [precision_at_5(rankings[q][spec], judgements.get(q, {})) for q in queries]}
        entry = {"results_shown": sum(len(rankings[q][spec]) for q in queries),
                 "unjudged_results": sum(pid not in judgements.get(q, {}) for q in queries for pid in rankings[q][spec])}
        for name, key in (("p5_scored_queries", "p5"), ("p5_all_queries", "p5_all")):
            if None not in values[spec][key]:
                entry[name] = round(float(np.mean(values[spec][key])), 4)
                entry[f"{name}_ci95"] = interval(np.array(values[spec][key]))
        report["rankers"][spec] = entry
    report["queries_without_relevant_product"] = {
        q: {"judged": len(judgements.get(q, {})), "grades": dict(Counter(_status(g) for g in judgements.get(q, {}).values())),
            "returned": {spec: dict(Counter(_status(judgements.get(q, {}).get(pid)) for pid in rankings[q][spec])) for spec in specs}}
        for q in queries if q not in scored}
    report["pairs"] = {}
    for a, b in itertools.combinations([s for s in COMPARED[task] if s in specs], 2):
        report["pairs"][f"{a} - {b}"] = {
            label: paired(np.array(values[a][key]), np.array(values[b][key]))
            for label, key in (("ndcg_scored", "ndcg"), ("p5_scored", "p5"), ("p5_all", "p5_all"))
            if None not in values[a][key] and None not in values[b][key]}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/evaluation_checks.json")
    args = parser.parse_args()

    store = store_with_embeddings()
    report = {"intervals": {"method": "percentile bootstrap over queries; paired differences resample per-query differences",
                            "resamples": BOOTSTRAP, "seed": SEED, "level": 0.95, "multiple_comparisons": "not adjusted"},
              "splits": {}}
    for split in ("dev", "holdout"):
        qrels = json.loads(Path(f"eval/qrels_{split}.json").read_text())
        check_fresh(qrels, labels_file(split), POOLS, split)
        if split == "holdout":
            from scripts.freeze import check

            check()
        report["splits"][split] = {}
        for task, rank in (("similar", similar_rankings), ("semantic", semantic_rankings)):
            specs = tuple(s for s in ranker_specs(POOLS, task, split) if s in COMPARED[task] or s in SERVED_RANKERS)
            report["splits"][split][task] = task_checks(task, rank(store, split, specs), qrels["judgements"][task])
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")

    for split, tasks in report["splits"].items():
        for task, r in tasks.items():
            print(f"\n{split} / {task}: {r['scored_queries']} of {r['queries']} queries scored")
            for spec, e in r["rankers"].items():
                print(f"  {spec:14s} P@5 scored {e.get('p5_scored_queries')} | all queries {e.get('p5_all_queries')} "
                      f"{e.get('p5_all_queries_ci95')} | shown {e['results_shown']}, unjudged {e['unjudged_results']}")
            for q, info in r["queries_without_relevant_product"].items():
                print(f"  no relevant product: {q[:40]!r} judged {info['judged']} {info['grades']}; returned {info['returned']}")
            for pair, d in r["pairs"].items():
                print(f"  {pair}: " + "; ".join(f"{k} {v['mean']:+.4f} {v['ci95']}" for k, v in d.items()))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
