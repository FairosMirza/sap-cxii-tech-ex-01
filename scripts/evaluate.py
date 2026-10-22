"""Development evaluation from the merged human judgements (eval/qrels_<split>.json, see scripts.check_labels).

Definitions and the rule for choosing an approach are fixed in docs/decisions.md D07: nDCG@10 with gain
2^grade - 1 (linear gains as a check), Precision@5 with grade >= 2 relevant, 95% bootstrap intervals over
queries, paired differences against the lead, and the one-standard-error rule over a fixed complexity order.
Rankers with an unjudged top-10 result are reported but never compared. Variants (SYSTEM+VARIANT, read from the
pool membership files) are compared with their base system.
Run from the repository root:  python -m scripts.evaluate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from scripts.build_label_pools import K, SEMANTIC_SYSTEMS, SIMILAR_SYSTEMS
from scripts.check_labels import pool_files, ranker_specs

ORDER = {"similar": SIMILAR_SYSTEMS, "semantic": SEMANTIC_SYSTEMS}  # least complex first (D07)
RELEVANT = 2
BOOTSTRAP, SEED = 10_000, 20261024


def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def query_metrics(ranking: list[str], judged: dict[str, int]) -> dict | None:
    """Metrics of one ranked list; None when no judged product has grade >= 1 (there is no ideal ranking)."""
    ideal = sorted(judged.values(), reverse=True)[:K]
    if not ideal or ideal[0] == 0:
        return None
    grades = [judged.get(pid) for pid in ranking[:K]]
    metrics = {"returned": len(grades), "judged": sum(g is not None for g in grades),
               "p5": sum((g or 0) >= RELEVANT for g in grades[:5]) / 5}
    for name, gain in (("ndcg", lambda g: 2 ** g - 1), ("ndcg_linear", lambda g: g)):
        metrics[name] = dcg([gain(g or 0) for g in grades]) / dcg([gain(g) for g in ideal])
    return metrics


def interval(values: np.ndarray) -> list[float]:
    means = values[np.random.default_rng(SEED).integers(0, len(values), (BOOTSTRAP, len(values)))].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    return {"mean": round(float(d.mean()), 4), "se": round(float(d.std(ddof=1) / math.sqrt(len(d))), 4), "ci95": interval(d)}


def choose(order: tuple[str, ...], scores: dict[str, np.ndarray]) -> dict:
    """The lead has the highest mean; the choice is the least complex system within one paired SE of the lead."""
    candidates = [s for s in order if s in scores]
    lead = max(candidates, key=lambda s: scores[s].mean())
    versus = {s: paired(scores[s], scores[lead]) for s in candidates if s != lead}
    choice = next(s for s in candidates if s == lead or versus[s]["mean"] >= -versus[s]["se"])
    return {"lead": lead, "choice": choice, "versus_lead": versus}


def title_repeats(ids: list[str], title_of: dict[str, int]) -> int:
    """Results whose normalised title already appeared higher in the same list (a redundancy proxy)."""
    seen: set[int] = set()
    repeats = 0
    for pid in ids[:K]:
        code = title_of[pid]
        repeats += code >= 0 and code in seen
        seen.add(code)
    return repeats


def evaluate_task(task: str, rankings: dict[str, dict[str, list[str]]], judgements: dict[str, dict[str, int]],
                  title_of: dict[str, int]) -> dict:
    specs = sorted({spec for ranked in rankings.values() for spec in ranked})
    excluded = [q for q in rankings if query_metrics([], judgements.get(q, {})) is None]
    queries = [q for q in rankings if q not in excluded]
    per_query = {spec: [query_metrics(rankings[q][spec], judgements[q]) for q in queries] for spec in specs}
    arrays = {spec: {m: np.array([row[m] for row in rows]) for m in ("ndcg", "ndcg_linear", "p5")} for spec, rows in per_query.items()}
    summary = {}
    for spec, rows in per_query.items():
        complete = all(row["judged"] == row["returned"] for row in rows)
        summary[spec] = {"complete": complete, "unjudged_results": sum(row["returned"] - row["judged"] for row in rows),
                         "mean_title_repeats": round(float(np.mean([title_repeats(rankings[q][spec], title_of) for q in queries])), 2)}
        if complete:
            for m in ("ndcg", "ndcg_linear", "p5"):
                summary[spec][m] = round(float(arrays[spec][m].mean()), 4)
                summary[spec][f"{m}_ci95"] = interval(arrays[spec][m])
    comparable = {spec: arrays[spec] for spec in specs if summary[spec]["complete"]}
    base = {spec: a["ndcg"] for spec, a in comparable.items() if spec in ORDER[task]}
    result = {"queries": len(queries), "excluded_no_relevant_product": excluded, "rankers": summary}
    if base:
        result["choice"] = choose(ORDER[task], base)
        result["choice_with_linear_gains"] = choose(ORDER[task], {s: comparable[s]["ndcg_linear"] for s in base})["choice"]
    result["variants"] = {spec: {"base": spec.partition("+")[0],
                                 "ndcg_difference": paired(comparable[spec]["ndcg"], comparable[spec.partition("+")[0]]["ndcg"]),
                                 "p5_difference": paired(comparable[spec]["p5"], comparable[spec.partition("+")[0]]["p5"])}
                          for spec in comparable if "+" in spec and spec.partition("+")[0] in comparable}
    result["per_query"] = {spec: {q: {k: round(v, 4) for k, v in row.items()} for q, row in zip(queries, rows)}
                           for spec, rows in per_query.items()}
    return result


def check_fresh(qrels: dict, labels: Path, pools: Path, split: str) -> None:
    """Refuse judgements merged from older labels or pools: re-run scripts.check_labels first."""
    current = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for task in ORDER for p in pool_files(pools, task, split)}
    if hashlib.sha256(labels.read_bytes()).hexdigest() != qrels["sources"]["labels"]["sha256"] or current != qrels["sources"]["pools"]:
        raise SystemExit("the merged judgements are stale: run `python -m scripts.check_labels --rankings` first")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "holdout"], default="dev")
    parser.add_argument("--labels", default="eval/labels.jsonl")
    parser.add_argument("--pools", default="eval/pools")
    parser.add_argument("--out", help="default: reports/<split>_evaluation.json")
    args = parser.parse_args()

    from scripts.build_label_pools import semantic_rankings, similar_rankings, store_with_embeddings

    qrels = json.loads(Path(f"eval/qrels_{args.split}.json").read_text())
    check_fresh(qrels, Path(args.labels), Path(args.pools), args.split)
    store = store_with_embeddings()
    title_of = dict(zip(store.catalog.ids, store.catalog.title_codes.tolist()))
    report = {"split": args.split, "definitions": "docs/decisions.md D07", "k": K, "bootstrap_resamples": BOOTSTRAP,
              "judgements": {"file": f"eval/qrels_{args.split}.json", "created": qrels["created"]}, "tasks": {}}
    for task, rank in (("similar", similar_rankings), ("semantic", semantic_rankings)):
        specs = tuple(ranker_specs(Path(args.pools), task, args.split))
        report["tasks"][task] = evaluate_task(task, rank(store, args.split, specs), qrels["judgements"][task], title_of)

    out = Path(args.out or f"reports/{args.split}_evaluation.json")
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    for task, result in report["tasks"].items():
        print(f"\n{task}: {result['queries']} queries scored; no relevant product judged: {len(result['excluded_no_relevant_product'])}")
        for spec, s in result["rankers"].items():
            scores = (f"nDCG@10 {s['ndcg']:.3f} {s['ndcg_ci95']}  linear {s['ndcg_linear']:.3f}  P@5 {s['p5']:.3f} {s['p5_ci95']}"
                      if s["complete"] else f"not compared: {s['unjudged_results']} unjudged results")
            print(f"  {spec:24s} {scores}  title repeats {s['mean_title_repeats']}")
        if "choice" in result:
            print(f"  lead {result['choice']['lead']}, choice {result['choice']['choice']} "
                  f"(linear gains: {result['choice_with_linear_gains']}); vs lead: {json.dumps(result['choice']['versus_lead'])}")
        for spec, v in result["variants"].items():
            print(f"  {spec} vs {v['base']}: nDCG {json.dumps(v['ndcg_difference'])}, P@5 {json.dumps(v['p5_difference'])}")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
