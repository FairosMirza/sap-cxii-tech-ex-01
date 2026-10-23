"""Development evaluation from the merged human judgements (eval/qrels_<split>.json, see scripts.check_labels).

Definitions and the rule for choosing an approach are fixed in docs/decisions.md D07: nDCG@10 with gain
2^grade - 1 (linear gains as a check), Precision@5 with grade >= 2 relevant, 95% bootstrap intervals over
queries, paired differences against the lead, and the one-standard-error rule over a fixed complexity order.
Rankers with an unjudged top-10 result are reported but never compared, and a list with an unjudged result gets no
metrics: unjudged is never read as grade 0. Variants (SYSTEM+VARIANT, read from the pool membership files) are
compared with their base system. The frozen configuration's rankers are never chosen: `served` (the deployed path)
is compared with `served_exact` (the same configuration searched exhaustively), which is compared with the ranker it
configures; the reference gives coverage and paired differences, the human judgements give relevance. Results shown
and grade-0 results shown count every query, including those left out of the means.
The holdout is evaluated once, against the freeze record (scripts.freeze).
Run from the repository root:  python -m scripts.evaluate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.build_label_pools import (K, SEMANTIC_SYSTEMS, SERVED, SERVED_EXACT, SERVED_RANKERS, SIMILAR_SYSTEMS,
                                       ranker_config, served_base)
from scripts.check_labels import labels_file, pool_files, ranker_specs

ORDER = {"similar": SIMILAR_SYSTEMS, "semantic": SEMANTIC_SYSTEMS}  # least complex first (D07)
RELEVANT = 2
BOOTSTRAP, SEED = 10_000, 20261024
STATUS = ("relevant", "grade_1", "grade_0", "unjudged")


def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def query_metrics(ranking: list[str], judged: dict[str, int]) -> dict | None:
    """Metrics of one ranked list; None when no judged product has grade >= 1 (there is no ideal ranking).
    A list with an unjudged result gets None metrics."""
    ideal = sorted(judged.values(), reverse=True)[:K]
    if not ideal or ideal[0] == 0:
        return None
    grades = [judged.get(pid) for pid in ranking[:K]]
    metrics = {"returned": len(grades), "judged": sum(g is not None for g in grades)}
    if metrics["judged"] < metrics["returned"]:
        return {**metrics, "p5": None, "ndcg": None, "ndcg_linear": None}
    metrics["p5"] = sum(g >= RELEVANT for g in grades[:5]) / 5
    for name, gain in (("ndcg", lambda g: 2 ** g - 1), ("ndcg_linear", lambda g: g)):
        metrics[name] = dcg([gain(g) for g in grades]) / dcg([gain(g) for g in ideal])
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


def choose_text_weight(base_weight: float, sweep: dict[float, dict]) -> float:
    """D05: keep the base weight unless another beats it by more than one paired SE; then the closest to the base."""
    better = [w for w, d in sweep.items() if d["mean"] > d["se"]]
    return min(better, key=lambda w: (abs(w - base_weight), -sweep[w]["mean"])) if better else base_weight


def _status(pid: str, judged: dict[str, int]) -> str:
    grade = judged.get(pid)
    return "unjudged" if grade is None else "relevant" if grade >= RELEVANT else f"grade_{grade}"


def compare_paths(rankings: dict[str, dict[str, list[str]]], judgements: dict[str, dict[str, int]],
                  path: str, reference: str, comparable: dict[str, dict[str, np.ndarray]]) -> dict:
    """`path` against `reference` over every query: coverage of the reference's top-10, the results `path` drops and
    adds by judgement (relevant = grade >= 2, unjudged counted apart), and paired nDCG@10 and P@5 differences over
    the scored queries when both rankers are fully judged."""
    coverage, dropped, added, below = [], Counter(), Counter(), []
    for q, ranked in rankings.items():
        got, ref, judged = ranked[path], ranked[reference], judgements.get(q, {})
        lost, new = [p for p in ref if p not in got], [p for p in got if p not in ref]
        share = 1 - len(lost) / len(ref) if ref else 1.0
        coverage.append(share)
        dropped.update(_status(p, judged) for p in lost)
        added.update(_status(p, judged) for p in new)
        if share < 1:
            below.append({"query": q, "coverage": round(share, 2), "dropped": dict(Counter(_status(p, judged) for p in lost)),
                          "added": dict(Counter(_status(p, judged) for p in new))})
    result = {"path": path, "reference": reference, "queries": len(coverage),
              "coverage": {"mean": round(float(np.mean(coverage)), 4), "min": round(float(np.min(coverage)), 2),
                           "fully_covered": sum(c == 1 for c in coverage)},
              "dropped": {s: dropped[s] for s in STATUS}, "added": {s: added[s] for s in STATUS},
              "queries_below_full_coverage": below}
    if path in comparable and reference in comparable:
        result["ndcg_difference"] = paired(comparable[path]["ndcg"], comparable[reference]["ndcg"])
        result["p5_difference"] = paired(comparable[path]["p5"], comparable[reference]["p5"])
    else:
        result["paired_differences"] = "not computed: a ranker has unjudged results"
    return result


def evaluate_task(task: str, rankings: dict[str, dict[str, list[str]]], judgements: dict[str, dict[str, int]],
                  title_of: dict[str, int], configured: str | None = None) -> dict:
    """`configured`: the ranker the served configuration applies its search settings to (C_bge, bge-small)."""
    specs = sorted({spec for ranked in rankings.values() for spec in ranked})
    excluded = [q for q in rankings if query_metrics([], judgements.get(q, {})) is None]
    queries = [q for q in rankings if q not in excluded]
    per_query = {spec: [query_metrics(rankings[q][spec], judgements[q]) for q in queries] for spec in specs}
    arrays = {spec: {m: np.array([row[m] for row in rows]) for m in ("ndcg", "ndcg_linear", "p5")} for spec, rows in per_query.items()}
    summary = {}
    for spec, rows in per_query.items():
        complete = all(row["judged"] == row["returned"] for row in rows)
        summary[spec] = {"complete": complete, "unjudged_results": sum(row["returned"] - row["judged"] for row in rows),
                         "mean_results_shown": round(float(np.mean([len(ranked[spec]) for ranked in rankings.values()])), 2),
                         "grade_0_results_shown": sum(judgements.get(q, {}).get(pid) == 0 for q, ranked in rankings.items()
                                                      for pid in ranked[spec]),
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
    for base in {spec.partition("+")[0] for spec in result["variants"] if "+text_" in spec}:
        sweep = {float(spec.partition("+text_")[2]): v["ndcg_difference"] for spec, v in result["variants"].items()
                 if spec.startswith(f"{base}+text_")}
        base_weight = ranker_config(base).block_weights["text"]
        result.setdefault("text_weight_choice", {})[base] = {"base_weight": base_weight, "chosen": choose_text_weight(base_weight, sweep),
                                                           "rule": "docs/decisions.md D05"}
    if SERVED in specs and SERVED_EXACT in specs:
        result["deployed_vs_exact"] = compare_paths(rankings, judgements, SERVED, SERVED_EXACT, comparable)
    if configured in specs and SERVED_EXACT in specs:
        result["exact_vs_base"] = compare_paths(rankings, judgements, SERVED_EXACT, configured, comparable)
    result["per_query"] = {spec: {q: {k: v if v is None else round(v, 4) for k, v in row.items()} for q, row in zip(queries, rows)}
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
    parser.add_argument("--labels", help="default: eval/labels.jsonl (dev) or eval/labels_holdout.jsonl")
    parser.add_argument("--pools", default="eval/pools")
    parser.add_argument("--out", help="default: reports/<split>_evaluation.json")
    args = parser.parse_args()

    from scripts.build_label_pools import semantic_rankings, similar_rankings, store_with_embeddings

    qrels = json.loads(Path(f"eval/qrels_{args.split}.json").read_text())
    check_fresh(qrels, Path(args.labels) if args.labels else labels_file(args.split), Path(args.pools), args.split)
    out = Path(args.out or f"reports/{args.split}_evaluation.json")
    if args.split == "holdout":
        from scripts.freeze import check

        check()
        if out.exists():
            raise SystemExit(f"{out} exists: the holdout is evaluated once (D07)")
    store = store_with_embeddings()
    title_of = dict(zip(store.catalog.ids, store.catalog.title_codes.tolist()))
    report = {"split": args.split, "definitions": "docs/decisions.md D07", "k": K, "bootstrap_resamples": BOOTSTRAP,
              "judgements": {"file": f"eval/qrels_{args.split}.json", "created": qrels["created"]}, "tasks": {}}
    for task, rank in (("similar", similar_rankings), ("semantic", semantic_rankings)):
        specs = tuple(ranker_specs(Path(args.pools), task, args.split))
        configured = served_base(task) if set(SERVED_RANKERS) & set(specs) else None
        report["tasks"][task] = evaluate_task(task, rank(store, args.split, specs), qrels["judgements"][task], title_of, configured)

    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    for task, result in report["tasks"].items():
        print(f"\n{task}: {result['queries']} queries scored; no relevant product judged: {len(result['excluded_no_relevant_product'])}")
        for spec, s in result["rankers"].items():
            scores = (f"nDCG@10 {s['ndcg']:.3f} {s['ndcg_ci95']}  linear {s['ndcg_linear']:.3f}  P@5 {s['p5']:.3f} {s['p5_ci95']}"
                      if s["complete"] else f"not compared: {s['unjudged_results']} unjudged results")
            print(f"  {spec:24s} {scores}  title repeats {s['mean_title_repeats']}  shown {s['mean_results_shown']}"
                  f"  grade 0 {s['grade_0_results_shown']}")
        if "choice" in result:
            print(f"  lead {result['choice']['lead']}, choice {result['choice']['choice']} "
                  f"(linear gains: {result['choice_with_linear_gains']}); vs lead: {json.dumps(result['choice']['versus_lead'])}")
        for spec, v in result["variants"].items():
            print(f"  {spec} vs {v['base']}: nDCG {json.dumps(v['ndcg_difference'])}, P@5 {json.dumps(v['p5_difference'])}")
        for base, choice in result.get("text_weight_choice", {}).items():
            print(f"  text weight for {base}: {choice['chosen']} (base {choice['base_weight']}, rule D05)")
        for name in ("deployed_vs_exact", "exact_vs_base"):
            if name in result:
                c = result[name]
                paired_text = (f"nDCG {json.dumps(c['ndcg_difference'])}, P@5 {json.dumps(c['p5_difference'])}"
                               if "ndcg_difference" in c else c["paired_differences"])
                print(f"  {c['path']} vs {c['reference']}: coverage {json.dumps(c['coverage'])}; dropped {c['dropped']}; "
                      f"added {c['added']}; {paired_text}")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
