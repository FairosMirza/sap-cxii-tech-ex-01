"""Approximate (FAISS HNSW) versus exhaustive search for the served configuration (D08).

Coverage is the share of the exhaustive top-10 (every eligible product scored) that a pipeline returns. At each
candidate count N, three pipelines use the served scorer and settings, so the sources of loss are measured apart:
  exact_text  the exact text top-N, reranked by the full scorer: loss from choosing candidates by text alone;
  faiss       the FAISS (HNSW) text top-N, reranked the same way: candidate choice plus approximation;
  served      the deployed path (FAISS candidates, widened when fewer than 10 results survive), also timed.
Semantic search applies the relevance cutoff (D12) in every pipeline and in the exhaustive reference; the served
path without the cutoff is reported too. Per-query distributions and the worst cases accompany the means. Coverage
measures fidelity to the scorer, mistakes included: it says nothing about whether the results are relevant.
Query sets (holdout queries excluded): for similar products, the 20 development products, a seeded sample of 1,000
catalogue products and 300 of those with filters; for search, the 10 development queries (natural language), 300
pseudo-queries made of the first five words of product titles (title-like text, not real user searches) and the
same 300 with a price ceiling appended ("<words> under <that product's price>").
The results hold for the scorer measured (recorded in the report): rerun after changing its weights, text model
or blocks. Run from the repository root (ARTIFACTS_DIR and CONFIG_PATH as for the service):
    python -m scripts.benchmark_ann
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from app.ann import candidate_rows
from app.artifacts import load_service
from app.features import SOURCE_FALLBACK, SOURCE_NONE, SOURCE_PRIMARY
from app.retrieval import SimilarFilters, SimilarityEngine
from app.scoring import eligible_mask, order_rows, score_all, select_distinct
from app.semantic import SemanticSearcher

K = 10
CANDIDATES = (200, 400, 800, 1600, 3200)
SAMPLE, SEARCH_SAMPLE, SEED = 1000, 300, 20261025
WARMUP, WORST = 20, 8
COVERAGE_BAR = 0.95  # D08
PIPELINES = ("exact_text", "faiss", "served")
SOURCES = {SOURCE_NONE: "none", SOURCE_PRIMARY: "sales rank", SOURCE_FALLBACK: "browse node"}


def coverage(got: list[int], reference: list[int]) -> float:
    return len(set(got) & set(reference)) / len(reference) if reference else 1.0


def distribution(values: list[float]) -> dict:
    v = np.array(values)
    return {"mean": round(float(v.mean()), 4), "min": round(float(v.min()), 2), "queries": len(v), "full": int((v == 1).sum()),
            "0.9_to_1": int(((v >= 0.9) & (v < 1)).sum()), "0.8_to_0.9": int(((v >= 0.8) & (v < 0.9)).sum()),
            "0.5_to_0.8": int(((v >= 0.5) & (v < 0.8)).sum()), "below_0.5": int((v < 0.5).sum())}


def timing(milliseconds: list[float]) -> dict:
    return {"median_ms": round(float(np.median(milliseconds)), 2), "p95_ms": round(float(np.percentile(milliseconds, 95)), 2)}


def timed(fn) -> tuple[object, float]:
    started = time.perf_counter()
    result = fn()
    return result, 1000 * (time.perf_counter() - started)


def served_candidates(index, vector: np.ndarray, allowed: np.ndarray, n: int) -> np.ndarray:
    """Candidates as the served path chooses them: every allowed product when there are at most n."""
    return np.flatnonzero(allowed) if n >= int(allowed.sum()) else candidate_rows(index, vector, allowed, n)


def rerank_similar(engine: SimilarityEngine, row: int, rows: np.ndarray) -> list[int]:
    """The served similar-products ranking, restricted to `rows`."""
    scores = np.full(len(engine.store.catalog), -np.inf)
    scores[rows] = score_all(engine.store, row, engine.config, rows=rows)
    return select_distinct(engine.store, order_rows(engine.store, rows, scores), K, engine.config).rows


def rerank_search(searcher: SemanticSearcher, parsed, vector: np.ndarray, rows: np.ndarray) -> list[int]:
    """The served search ranking, restricted to `rows`, relevance cutoff included."""
    similarity = searcher.store.modalities[searcher.text_key].vectors[rows] @ vector
    scores = np.full(len(searcher.store.catalog), -np.inf)
    scores[rows] = similarity + searcher.bonuses(parsed, rows)
    if searcher.min_similarity is not None:
        rows = rows[similarity >= searcher.min_similarity]
    return select_distinct(searcher.store, order_rows(searcher.store, rows, scores), K, searcher.config).rows


def with_settings(searcher: SemanticSearcher, candidates: int, min_similarity: float | None) -> SemanticSearcher:
    variant = copy.copy(searcher)
    variant.config, variant.min_similarity = searcher.config.with_search("ann", candidates), min_similarity
    return variant


def summarise(per_query: dict, deployed: int, describe) -> dict:
    """Coverage of each pipeline per candidate count; at the deployed count, where the loss arises and the worst cases."""
    report = {"by_candidates": {}}
    for n in CANDIDATES:
        entry = {name: distribution([q[n][name] for q in per_query.values()]) for name in PIPELINES}
        entry["served"].update(timing([q[n]["served_ms"] for q in per_query.values()]))
        if all("served_without_cutoff" in q[n] for q in per_query.values()):
            entry["served_without_cutoff"] = distribution([q[n]["served_without_cutoff"] for q in per_query.values()])
        recall = [q[n]["hnsw_recall"] for q in per_query.values()]
        entry["hnsw_vector_recall"] = {"mean": round(float(np.mean(recall)), 4), "min": round(float(np.min(recall)), 4)}
        report["by_candidates"][n] = entry
    at = [q[deployed] for q in per_query.values()]
    report["loss_at_deployed"] = {
        "queries": len(at),
        "below_full_coverage": sum(q["served"] < 1 for q in at),
        "text_only_candidates_lose": sum(q["exact_text"] < 1 for q in at),
        "approximation_loses_more": sum(q["faiss"] < q["exact_text"] for q in at),
        "approximation_recovers_some": sum(q["faiss"] > q["exact_text"] for q in at),
        "widening_recovers_some": sum(q["served"] > q["faiss"] for q in at),
    }
    order = sorted(per_query, key=lambda key: (per_query[key][deployed]["served"], per_query[key][deployed]["exact_text"]))
    report["worst_at_deployed"] = [{**describe(key), **{name: per_query[key][deployed][name] for name in PIPELINES}}
                                   for key in order[:WORST] if per_query[key][deployed]["served"] < 1]
    return report


def similar_benchmark(engine: SimilarityEngine, rows: list[int]) -> dict:
    store, config, deployed = engine.store, engine.config, engine.config.ann_candidates
    c, index = store.catalog, store.indexes[config.text_model]
    vectors = store.modalities[config.text_model].vectors
    settings = {n: config.with_search("ann", n) for n in CANDIDATES}
    for row in rows[:WARMUP]:
        engine.search(row, K, mode="exact"), engine.search(row, K)
    per_query, exhaustive_ms, best = {}, [], {}
    for row in rows:
        result, ms = timed(lambda: engine.search(row, K, mode="exact"))
        exhaustive = [r for r, _ in result.hits]
        exhaustive_ms.append(ms)
        allowed = eligible_mask(store, row, config)
        similarity = np.where(allowed, vectors @ vectors[row], -np.inf)
        order = np.argsort(-similarity, kind="stable")[:int(allowed.sum())]
        best[row] = float(similarity[order[0]])
        per_query[row] = {}
        for n in CANDIDATES:
            text_rows, faiss_rows = order[:n], served_candidates(index, vectors[row], allowed, n)
            served, served_ms = timed(lambda: engine.search(row, K, config=settings[n]))
            per_query[row][n] = {"exact_text": coverage(rerank_similar(engine, row, text_rows), exhaustive),
                                 "faiss": coverage(rerank_similar(engine, row, faiss_rows), exhaustive),
                                 "served": coverage([r for r, _ in served.hits], exhaustive), "served_ms": served_ms,
                                 "hnsw_recall": len(np.intersect1d(faiss_rows, text_rows)) / len(text_rows)}

    def describe(row: int) -> dict:
        return {"id": c.ids[row], "name": c.names[row][:80],
                "category": [c.category_names[j] for j in c.category_matrix[row].indices],
                "category_source": SOURCES[int(c.category_source[row])], "best_text_cosine": round(best[row], 3)}

    served = {row: per_query[row][deployed]["served"] for row in rows}
    by_source = {}
    for code, name in SOURCES.items():
        members = [served[row] for row in rows if c.category_source[row] == code]
        if members:
            by_source[name] = {"queries": len(members), "mean": round(float(np.mean(members)), 4)}
    strengths = np.array([best[row] for row in rows])
    weak = strengths <= np.percentile(strengths, 5)
    coverages = np.array([served[row] for row in rows])
    return {"queries": len(rows), "exhaustive": timing(exhaustive_ms),
            "best_text_neighbour_cosine": {"p1": round(float(np.percentile(strengths, 1)), 3),
                                           "p5": round(float(np.percentile(strengths, 5)), 3),
                                           "median": round(float(np.median(strengths)), 3)},
            "served_at_deployed_by_category_source": by_source,
            "served_at_deployed_by_neighbour_strength": {
                "weakest_5_percent": {"queries": int(weak.sum()), "mean": round(float(coverages[weak].mean()), 4)},
                "rest": {"queries": int((~weak).sum()), "mean": round(float(coverages[~weak].mean()), 4)}},
            **summarise(per_query, deployed, describe)}


def filtered_benchmark(engine: SimilarityEngine, rows: list[int]) -> dict:
    """Filtered similar products (v2 Part 4) with the served settings. A filter that leaves at most
    max(candidates, exact_below) products is scored exhaustively, so FAISS is only used above that."""
    store, c = engine.store, engine.store.catalog
    params = store.indexes[engine.config.text_model].params
    threshold = max(engine.config.ann_candidates, params.exact_below)
    report = {"candidates": engine.config.ann_candidates, "scored_exhaustively_at_or_below": threshold}
    cases = {"max_price = query price": lambda row: SimilarFilters(max_price=float(c.price[row])) if np.isfinite(c.price[row]) else None,
             "same_brand": lambda row: SimilarFilters(same_brand=True) if c.brand_codes[row] >= 0 else None}
    for name, make in cases.items():
        cover, faiss_cover, exhaustive_ms, served_ms, allowed = [], [], [], [], []
        for row in rows:
            filters = make(row)
            if filters is None:
                continue
            exact, e_ms = timed(lambda: engine.search(row, K, filters=filters, mode="exact"))
            served, s_ms = timed(lambda: engine.search(row, K, filters=filters))
            value = coverage([r for r, _ in served.hits], [r for r, _ in exact.hits])
            count = int((eligible_mask(store, row, engine.config) & filters.mask(store, row)).sum())
            cover.append(value)
            if count > threshold:
                faiss_cover.append(value)
            exhaustive_ms.append(e_ms)
            served_ms.append(s_ms)
            allowed.append(count)
        report[name] = {"queries": len(cover), "median_products_allowed": int(np.median(allowed)),
                        "queries_using_faiss": len(faiss_cover), "coverage": distribution(cover),
                        "coverage_when_faiss_used": distribution(faiss_cover) if faiss_cover else None,
                        "exhaustive": timing(exhaustive_ms), "served": timing(served_ms)}
    return report


def search_benchmark(searcher: SemanticSearcher, queries: list[str]) -> dict:
    store, deployed, cutoff = searcher.store, searcher.config.ann_candidates, searcher.min_similarity
    index = store.indexes[searcher.text_key]
    vectors = store.modalities[searcher.text_key].vectors
    threshold = max(deployed, index.params.exact_below)
    gated = {n: with_settings(searcher, n, cutoff) for n in CANDIDATES}
    ungated = {n: with_settings(searcher, n, None) for n in CANDIDATES}
    for query in queries[:WARMUP]:
        searcher.search(query, K, mode="exact"), searcher.search(query, K)
    per_query, exhaustive_ms, facts = {}, [], {}
    for query in queries:
        result, ms = timed(lambda: searcher.search(query, K, mode="exact"))
        exhaustive = list(result.rows)
        exhaustive_ms.append(ms)
        exhaustive_ungated = list(ungated[deployed].search(query, K, mode="exact").rows)
        parsed = result.parsed
        mask, _ = searcher.filter_mask(parsed)
        vector = searcher.encode_query(parsed.semantic_text)
        similarity = np.where(mask, vectors @ vector, -np.inf)
        order = np.argsort(-similarity, kind="stable")[:int(mask.sum())]
        filters = [name for name, active in (("price", parsed.price is not None), ("audience", bool(parsed.audience)),
                                             ("colour", bool(parsed.colors))) if active]
        facts[query] = {"filters": filters, "allowed": int(mask.sum()), "exhaustive_results": len(exhaustive)}
        per_query[query] = {}
        for n in CANDIDATES:
            text_rows, faiss_rows = order[:n], served_candidates(index, vector, mask, n)
            served, served_ms = timed(lambda: gated[n].search(query, K))
            per_query[query][n] = {"exact_text": coverage(rerank_search(searcher, parsed, vector, text_rows), exhaustive),
                                   "faiss": coverage(rerank_search(searcher, parsed, vector, faiss_rows), exhaustive),
                                   "served": coverage(list(served.rows), exhaustive), "served_ms": served_ms,
                                   "served_without_cutoff": coverage(list(ungated[n].search(query, K).rows), exhaustive_ungated),
                                   "hnsw_recall": len(np.intersect1d(faiss_rows, text_rows)) / len(text_rows) if len(text_rows) else 1.0}

    def split(keep) -> dict | None:
        values = [per_query[q][deployed]["served"] for q in queries if keep(facts[q])]
        return distribution(values) if values else None

    results = np.array([facts[q]["exhaustive_results"] for q in queries])
    return {"queries": len(queries), "exhaustive_with_cutoff": timing(exhaustive_ms),
            "queries_with_a_filter": sum(bool(facts[q]["filters"]) for q in queries),
            "median_products_allowed": int(np.median([facts[q]["allowed"] for q in queries])),
            "scored_exhaustively_at_or_below": threshold,
            "queries_using_faiss": sum(facts[q]["allowed"] > threshold for q in queries),
            "cutoff_leaves_fewer_than_k": int((results < K).sum()), "cutoff_leaves_none": int((results == 0).sum()),
            "served_at_deployed_by_filter": {"with_filter": split(lambda f: bool(f["filters"])),
                                             "without_filter": split(lambda f: not f["filters"]),
                                             "using_faiss": split(lambda f: f["allowed"] > threshold)},
            **summarise(per_query, deployed, lambda q: {"query": q, **facts[q]})}


def rule(dev: dict, sample: dict, exhaustive_key: str) -> dict:
    """D08 (unchanged): the smallest candidate count whose served coverage averages >= 0.95 on the development
    set and on the sample, and its speed."""
    passing = [n for n in CANDIDATES if all(s["by_candidates"][n]["served"]["mean"] >= COVERAGE_BAR for s in (dev, sample))]
    if not passing:
        return {"meets_coverage_bar": False}
    n = passing[0]
    return {"meets_coverage_bar": True, "candidates": n, "served_median_ms": sample["by_candidates"][n]["served"]["median_ms"],
            "exhaustive_median_ms": sample[exhaustive_key]["median_ms"]}


def query_sets(engine: SimilarityEngine) -> dict:
    """The benchmark's query sets (fixed seed, holdout excluded), shared with the semantic-ID experiment."""
    store, c = engine.store, engine.store.catalog
    queries = json.loads(Path("eval/queries.json").read_text())["queries"]
    dev_rows = [c.index[q["id"]] for q in queries if q["split"] == "dev"]
    holdout = {c.index[q["id"]] for q in queries if q["split"] == "holdout"}
    rng = np.random.default_rng(SEED)
    pool = [r for r in np.flatnonzero(store.modalities[engine.config.text_model].available) if r not in holdout and r not in dev_rows]
    sample_rows = [int(r) for r in rng.choice(pool, SAMPLE, replace=False)]
    title_rows = [int(r) for r in rng.choice(pool, 3 * SEARCH_SAMPLE, replace=False) if len(c.names[r].split()) >= 3]
    source: dict[str, int] = {}
    for row in title_rows:
        source.setdefault(" ".join(c.names[row].split()[:5]), row)
    title_queries = list(source)[:SEARCH_SAMPLE]
    priced_queries = [f"{q} under {int(c.price[source[q]])}" for q in title_queries if np.isfinite(c.price[source[q]])]
    return {"dev_rows": dev_rows, "sample_rows": sample_rows, "title_queries": title_queries,
            "priced_queries": priced_queries, "dev_searches": json.loads(Path("eval/semantic_queries.json").read_text())["dev"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/ann_benchmark.json")
    args = parser.parse_args()

    engine, searcher, info = load_service()
    c = engine.store.catalog
    sets = query_sets(engine)
    dev_rows, sample_rows, dev_searches = sets["dev_rows"], sets["sample_rows"], sets["dev_searches"]
    title_queries, priced_queries = sets["title_queries"], sets["priced_queries"]

    rep = info["representations"][engine.config.text_model]
    similar = {"dev": similar_benchmark(engine, dev_rows), "sample": similar_benchmark(engine, sample_rows),
               "filtered_sample": filtered_benchmark(engine, sample_rows[:SEARCH_SAMPLE])}
    similar["rule_d08"] = rule(similar["dev"], similar["sample"], "exhaustive")
    search = {"dev": search_benchmark(searcher, dev_searches), "title_pseudo_queries": search_benchmark(searcher, title_queries),
              "title_pseudo_queries_with_price_ceiling": search_benchmark(searcher, priced_queries)}
    search["rule_d08"] = rule(search["dev"], search["title_pseudo_queries"], "exhaustive_with_cutoff")
    report = {
        "scorer": {"code_commit": info.get("code_commit"), "similar_method": engine.config.name,
                   "text_model": engine.config.text_model, "block_weights": engine.config.block_weights,
                   "similar_candidates": engine.config.ann_candidates, "semantic_model": searcher.text_key,
                   "semantic_candidates": searcher.config.ann_candidates, "relevance_cutoff": searcher.min_similarity},
        "index": {"type": rep["index"]["type"], "params": rep["index"]["params"], "products": len(c),
                  "build_seconds": rep["index"]["build_seconds"], "bytes": rep["index"]["bytes"]},
        "method": f"in-process latency (no HTTP), {WARMUP} warm-up queries, seed {SEED}; see scripts/benchmark_ann.py",
        "similar_products": similar, "semantic_search": search,
    }
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")

    for task, sets in (("similar_products", ("dev", "sample")),
                       ("semantic_search", ("dev", "title_pseudo_queries", "title_pseudo_queries_with_price_ceiling"))):
        for name in sets:
            s = report[task][name]
            print(f"\n{task} / {name} ({s['queries']} queries); exhaustive {s.get('exhaustive') or s.get('exhaustive_with_cutoff')}")
            for n, p in s["by_candidates"].items():
                served = p["served"]
                print(f"  N={n:>4}: exact-text {p['exact_text']['mean']:.4f} | faiss {p['faiss']['mean']:.4f} | served {served['mean']:.4f} "
                      f"(min {served['min']}, full {served['full']}/{served['queries']}, <0.5 {served['below_0.5']}) "
                      f"{served['median_ms']}/{served['p95_ms']} ms | vector recall {p['hnsw_vector_recall']['mean']}")
            print(f"  loss at deployed: {s['loss_at_deployed']}")
        print(f"  D08: {report[task]['rule_d08']}")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
