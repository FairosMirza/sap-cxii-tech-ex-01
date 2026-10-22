"""Run approaches A (structured attributes) and B (TF-IDF + attributes) on the selected queries.

Saves every query's top-10 for later pooled labelling. Diagnostics are computed on development
queries only; holdout outputs are stored but not summarised until the final comparison.
Run from the repository root:  python -m scripts.run_baselines
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from app.pipeline import build_store
from app.retrieval import SimilarityEngine
from app.scoring import FeatureStore
from app.settings import DEFAULT_CONFIG, load_config

K = 10
METHODS = ("A_structured", "B_tfidf")
WARMUP_RUNS, TIMED_RUNS = 1, 5
LATENCY_SCOPE = (
    "In-process scoring and ranking of one query against the full catalogue (score_all + rank). "
    "Excludes catalogue/feature building, HTTP handling and JSON serialisation."
)


def _diagnostics(store: FeatureStore, query: int, rows: list[int]) -> dict:
    """Sanity checks only; category agreement is not a relevance label."""
    c = store.catalog
    query_categories = set(c.category_matrix[query].indices)
    out: dict = {"returned": len(rows)}
    if query_categories and rows:
        out["category_agreement"] = float(np.mean([bool(query_categories & set(c.category_matrix[r].indices)) for r in rows]))
    if c.brand_codes[query] >= 0 and rows:
        out["brand_agreement"] = float(np.mean([c.brand_codes[r] == c.brand_codes[query] for r in rows]))
    ratios = [abs(np.log(c.price[r] / c.price[query])) for r in rows if np.isfinite(c.price[r]) and np.isfinite(c.price[query])]
    if ratios:
        out["median_abs_log_price_ratio"] = float(np.median(ratios))
    return out


def _mean(items: list[dict], key: str) -> float | None:
    values = [d[key] for d in items if key in d]
    return round(float(np.mean(values)), 4) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", default="eval/queries.json")
    parser.add_argument("--out", default="reports/day1_baselines.json")
    args = parser.parse_args()

    queries = json.loads(Path(args.queries).read_text())["queries"]
    started = time.perf_counter()
    store = build_store()
    build_seconds = time.perf_counter() - started
    c = store.catalog

    report: dict = {
        "k": K, "build_seconds": round(build_seconds, 1), "title_cap": "off (primary comparison)",
        "text_anchors": store.modalities["text"].anchors.__dict__,
        "latency_method": {
            "scope": LATENCY_SCOPE,
            "statistic": f"median and p95 over {TIMED_RUNS} timed runs per development query after {WARMUP_RUNS} warm-up run",
            "hardware": f"{platform.machine()} {platform.platform()}, Python {platform.python_version()}, default threading",
        },
        "methods": {},
    }
    top_sets: dict[str, dict[str, set]] = {}
    for method in METHODS:
        engine = SimilarityEngine(store, load_config(DEFAULT_CONFIG, method))
        per_query, timings = [], []
        for q in queries:
            row = engine.row_of(q["id"])
            result = engine.search(row, K)
            entry = {"query": q["id"], "split": q["split"], "stratum": q["stratum"],
                     "title_cap_shortfall": result.title_cap_shortfall,
                     "results": [{"id": c.ids[r], "score": round(s, 4)} for r, s in result.hits]}
            if q["split"] == "dev":
                entry["diagnostics"] = _diagnostics(store, row, [r for r, _ in result.hits])
                for run in range(WARMUP_RUNS + TIMED_RUNS):
                    t0 = time.perf_counter()
                    engine.search(row, K)
                    if run >= WARMUP_RUNS:
                        timings.append(1000 * (time.perf_counter() - t0))
            per_query.append(entry)
        top_sets[method] = {e["query"]: {r["id"] for r in e["results"]} for e in per_query}
        dev = [e["diagnostics"] for e in per_query if "diagnostics" in e]
        report["methods"][method] = {
            "config": load_config(DEFAULT_CONFIG, method).component_weights(),
            "dev_summary": {
                "queries": len(dev),
                "mean_results_returned": _mean(dev, "returned"),
                "category_agreement_at_10": _mean(dev, "category_agreement"),
                "brand_agreement_at_10": _mean(dev, "brand_agreement"),
                "median_abs_log_price_ratio": _mean(dev, "median_abs_log_price_ratio"),
            },
            "latency_ms": {"median": round(float(np.median(timings)), 2), "p95": round(float(np.percentile(timings, 95)), 2),
                           "timed_runs": len(timings)},
            "queries": per_query,
        }
    dev_ids = [q["id"] for q in queries if q["split"] == "dev"]
    a, b = top_sets[METHODS[0]], top_sets[METHODS[1]]
    report["dev_top10_jaccard_A_vs_B"] = round(float(np.mean([len(a[i] & b[i]) / max(len(a[i] | b[i]), 1) for i in dev_ids])), 4)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"store built in {report['build_seconds']}s; text anchors {report['text_anchors']}")
    print("latency:", report["latency_method"]["statistic"], "|", LATENCY_SCOPE)
    for method in METHODS:
        m = report["methods"][method]
        print(method, m["dev_summary"], "latency_ms", m["latency_ms"])
    print("dev top-10 Jaccard A vs B:", report["dev_top10_jaccard_A_vs_B"])
    for q in [q for q in queries if q["split"] == "dev"][:2]:
        print(f"\nQUERY: {q['name'][:90]}")
        for method in METHODS:
            entry = next(e for e in report["methods"][method]["queries"] if e["query"] == q["id"])
            for hit in entry["results"][:4]:
                print(f"  {method[:1]} {hit['score']:.3f} {c.names[c.index[hit['id']]][:85]}")


if __name__ == "__main__":
    main()
