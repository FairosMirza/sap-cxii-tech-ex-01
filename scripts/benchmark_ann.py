"""Approximate (FAISS HNSW) versus exact search: the evidence for the default search mode (D08).

Approximate mode takes HNSW candidates from the text vectors and re-ranks them with the exact hybrid score, so
two effects are measured apart: how well HNSW recovers the true nearest text vectors (vector recall), and how
much of the exact hybrid top-10 lies among the true top-N text neighbours at all (the ceiling of text-only
candidates). Coverage is the share of the exact top-10 that approximate mode returns. Filtered search and
in-process latency are measured too. Queries: the development queries (the set D08 names) and a seeded sample
of catalogue products; holdout queries are excluded.
Run from the repository root:  python -m scripts.benchmark_ann
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from app.artifacts import load
from app.retrieval import SimilarFilters, SimilarityEngine
from app.scoring import eligible_mask
from app.semantic import SemanticSearcher
from app.settings import load_config, load_serving

K = 10
CANDIDATES = (200, 400, 800, 1600, 3200)
SAMPLE, SEARCH_SAMPLE, SEED = 1000, 300, 20261025
WARMUP = 20
COVERAGE_BAR = 0.95  # D08


def summary(values: list[float]) -> dict:
    return {"mean": round(float(np.mean(values)), 4), "min": round(float(np.min(values)), 4)}


def timing(milliseconds: list[float]) -> dict:
    return {"median_ms": round(float(np.median(milliseconds)), 2), "p95_ms": round(float(np.percentile(milliseconds, 95)), 2)}


def timed(fn) -> tuple[object, float]:
    started = time.perf_counter()
    result = fn()
    return result, 1000 * (time.perf_counter() - started)


def coverage(approx: list[int], exact: list[int]) -> float:
    return len(set(approx) & set(exact)) / len(exact) if exact else 1.0


def similar_benchmark(engine: SimilarityEngine, rows: list[int], with_vectors: bool) -> dict:
    store, config = engine.store, engine.config
    index = store.indexes[config.text_model]
    vectors = store.modalities[config.text_model].vectors
    for row in rows[:WARMUP]:
        engine.search(row, K, mode="exact"), engine.search(row, K, mode="ann")
    exact, exact_ms = {}, []
    for row in rows:
        result, ms = timed(lambda: engine.search(row, K, mode="exact"))
        exact[row] = [r for r, _ in result.hits]
        exact_ms.append(ms)
    report = {"queries": len(rows), "exact": timing(exact_ms), "approximate": {}}
    for n in CANDIDATES:
        cfg = replace(config, ann_candidates=n)
        cover, ceiling, recall, ms_list = [], [], [], []
        for row in rows:
            result, ms = timed(lambda: engine.search(row, K, config=cfg, mode="ann"))
            cover.append(coverage([r for r, _ in result.hits], exact[row]))
            ms_list.append(ms)
            if with_vectors:
                allowed = eligible_mask(store, row, config)
                sims = vectors @ vectors[row]
                sims[~allowed] = -np.inf
                true_top = np.argpartition(-sims, n)[:n]
                ceiling.append(coverage(list(true_top), exact[row]))
                recall.append(len(set(index.search(vectors[row], n, allowed=allowed)) & set(true_top)) / n)
        entry = {"coverage": summary(cover), "share_of_queries_fully_covered": round(float(np.mean(np.array(cover) == 1.0)), 3),
                 **timing(ms_list)}
        if with_vectors:
            entry["text_candidate_ceiling"] = summary(ceiling)
            entry["hnsw_vector_recall"] = summary(recall)
        report["approximate"][n] = entry
    return report


def filtered_benchmark(engine: SimilarityEngine, rows: list[int], candidates: int) -> dict:
    """Coverage and latency of filtered search (D08 / v2 Part 4) at the given candidate count."""
    c = engine.store.catalog
    config = replace(engine.config, ann_candidates=candidates)
    report = {"candidates": candidates}
    cases = {"max_price = query price": lambda row: SimilarFilters(max_price=float(c.price[row])) if np.isfinite(c.price[row]) else None,
             "same_brand": lambda row: SimilarFilters(same_brand=True) if c.brand_codes[row] >= 0 else None}
    for name, make in cases.items():
        cover, exact_ms, ann_ms, allowed = [], [], [], []
        for row in rows:
            filters = make(row)
            if filters is None:
                continue
            exact, e_ms = timed(lambda: engine.search(row, K, filters=filters, mode="exact"))
            approx, a_ms = timed(lambda: engine.search(row, K, config=config, filters=filters, mode="ann"))
            cover.append(coverage([r for r, _ in approx.hits], [r for r, _ in exact.hits]))
            exact_ms.append(e_ms)
            ann_ms.append(a_ms)
            allowed.append(int((eligible_mask(engine.store, row, engine.config) & filters.mask(engine.store, row)).sum()))
        report[name] = {"queries": len(cover), "median_products_allowed": int(np.median(allowed)),
                        "coverage": summary(cover), "exact": timing(exact_ms), "approximate": timing(ann_ms)}
    return report


def search_benchmark(searcher: SemanticSearcher, queries: list[str]) -> dict:
    base = searcher.config
    for query in queries[:WARMUP]:
        searcher.search(query, K, mode="exact"), searcher.search(query, K, mode="ann")
    exact, exact_ms = {}, []
    for query in queries:
        result, ms = timed(lambda: searcher.search(query, K, mode="exact"))
        exact[query] = list(result.rows)
        exact_ms.append(ms)
    report = {"queries": len(queries), "exact": timing(exact_ms), "approximate": {}}
    for n in CANDIDATES:
        searcher.config = replace(base, ann_candidates=n)
        cover, ms_list = [], []
        for query in queries:
            result, ms = timed(lambda: searcher.search(query, K, mode="ann"))
            cover.append(coverage(list(result.rows), exact[query]))
            ms_list.append(ms)
        report["approximate"][n] = {"coverage": summary(cover),
                                    "share_of_queries_fully_covered": round(float(np.mean(np.array(cover) == 1.0)), 3), **timing(ms_list)}
    searcher.config = base
    return report


def rule(dev: dict, sample: dict) -> dict:
    """D08: the smallest candidate count whose mean coverage reaches the bar on both query sets, and its speed."""
    passing = [n for n in CANDIDATES if dev["approximate"][n]["coverage"]["mean"] >= COVERAGE_BAR
               and sample["approximate"][n]["coverage"]["mean"] >= COVERAGE_BAR]
    if not passing:
        return {"meets_coverage_bar": False}
    n = passing[0]
    return {"meets_coverage_bar": True, "candidates": n,
            "approximate_median_ms": sample["approximate"][n]["median_ms"], "exact_median_ms": sample["exact"]["median_ms"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--out", default="reports/ann_benchmark.json")
    args = parser.parse_args()

    store, manifest = load(Path(args.artifacts))
    serving = load_serving()
    engine = SimilarityEngine(store, load_config(method=serving["similar_method"]))
    searcher = SemanticSearcher(store, serving["semantic_model"], load_config())
    c = store.catalog
    queries = json.loads(Path("eval/queries.json").read_text())["queries"]
    dev_rows = [c.index[q["id"]] for q in queries if q["split"] == "dev"]
    holdout = {c.index[q["id"]] for q in queries if q["split"] == "holdout"}
    available = store.modalities[engine.config.text_model].available
    rng = np.random.default_rng(SEED)
    pool = [r for r in np.flatnonzero(available) if r not in holdout and r not in dev_rows]
    sample_rows = [int(r) for r in rng.choice(pool, SAMPLE, replace=False)]
    title_rows = [int(r) for r in rng.choice(pool, 3 * SEARCH_SAMPLE, replace=False) if len(c.names[r].split()) >= 3]
    title_queries = list(dict.fromkeys(" ".join(c.names[r].split()[:5]) for r in title_rows))[:SEARCH_SAMPLE]
    dev_searches = json.loads(Path("eval/semantic_queries.json").read_text())["dev"]

    rep = manifest["representations"][engine.config.text_model]
    report = {
        "index": {"type": rep["index"]["type"], "params": rep["index"]["params"], "products": len(c),
                  "build_seconds": rep["index"]["build_seconds"], "bytes": rep["index"]["bytes"]},
        "method": (f"in-process latency (no HTTP), {WARMUP} warm-up queries; coverage = share of the exact top-{K} "
                   f"returned; sample: {SAMPLE} products and {SEARCH_SAMPLE} title-derived searches, seed {SEED}, "
                   "holdout queries excluded"),
        "similar_products": {"method": engine.config.name,
                             "dev": similar_benchmark(engine, dev_rows, with_vectors=True),
                             "sample": similar_benchmark(engine, sample_rows, with_vectors=True)},
        "semantic_search": {"model": serving["semantic_model"],
                            "dev": search_benchmark(searcher, dev_searches),
                            "sample": search_benchmark(searcher, title_queries)},
    }
    for task in ("similar_products", "semantic_search"):
        report[task]["rule_d08"] = rule(report[task]["dev"], report[task]["sample"])
    similar_rule = report["similar_products"]["rule_d08"]
    report["similar_products"]["filtered_sample"] = filtered_benchmark(
        engine, sample_rows[:SEARCH_SAMPLE], similar_rule.get("candidates", engine.config.ann_candidates))
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")

    for task in ("similar_products", "semantic_search"):
        for split in ("dev", "sample"):
            part = report[task][split]
            print(f"\n{task} / {split} ({part['queries']} queries): exact {part['exact']}")
            for n, entry in part["approximate"].items():
                extra = (f" | ceiling {entry['text_candidate_ceiling']['mean']} | HNSW recall {entry['hnsw_vector_recall']['mean']}"
                         if "hnsw_vector_recall" in entry else "")
                print(f"  candidates {n:>4}: coverage {entry['coverage']['mean']} (min {entry['coverage']['min']}, "
                      f"full {entry['share_of_queries_fully_covered']}){extra} | {entry['median_ms']} / {entry['p95_ms']} ms")
        print(f"  D08 rule: {report[task]['rule_d08']}")
    print("\nfiltered:", json.dumps(report["similar_products"]["filtered_sample"], indent=1))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
