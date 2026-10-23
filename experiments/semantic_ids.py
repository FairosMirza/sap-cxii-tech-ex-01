"""Semantic IDs as a candidate index, compared with HNSW under the D08 benchmark's conditions (D15, pre-registered).

Codes: residual k-means on the L2-normalised bge-small product vectors, 3 levels of 64 centroids (FAISS k-means,
25 iterations, fixed seed); a product's semantic ID is its 3 codes. Retrieval for a query vector q: lookup tables
of q's inner products with each level's centroids; two-code prefixes ranked by the sum of their two entries;
allowed products collected prefix by prefix until N are found; the collected products ordered by their three-code
score ("sid") or by exact cosine ("sid_exact_order"); the top N go to the same exact reranker as HNSW's candidates.
Every allowed product is scored when there are at most N, or at most 2,000, as for HNSW.
Same query sets, seed, scorer, coverage definition, process and thread settings as scripts.benchmark_ann.
Run from the repository root:  python -m experiments.semantic_ids
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np

from app.ann import HNSWIndex
from app.artifacts import load_service
from app.scoring import eligible_mask
from scripts.benchmark_ann import (CANDIDATES, K, WARMUP, WORST, coverage, distribution, query_sets, rerank_search,
                                   rerank_similar, served_candidates, timed, timing)

LEVELS, CODEWORDS, NITER, SEED = 3, 64, 25, 20261023
DEPLOYED = {"similar_products": 800, "semantic_search": 200}
PIPELINES = ("exact_text", "hnsw", "sid", "sid_exact_order")


class SemanticIdIndex:
    def __init__(self, centroids: list[np.ndarray], codes: np.ndarray, exact_below: int) -> None:
        self.centroids, self.codes, self.exact_below = centroids, codes, exact_below
        self.prefix = codes[:, 0].astype(np.int64) * CODEWORDS + codes[:, 1]
        self.order = np.argsort(self.prefix, kind="stable")
        self.offsets = np.searchsorted(self.prefix[self.order], np.arange(CODEWORDS * CODEWORDS + 1))

    @classmethod
    def build(cls, vectors: np.ndarray, available: np.ndarray, exact_below: int) -> "SemanticIdIndex":
        residual = vectors.astype(np.float32).copy()
        centroids, codes = [], np.zeros((len(vectors), LEVELS), dtype=np.uint8)
        for level in range(LEVELS):
            kmeans = faiss.Kmeans(vectors.shape[1], CODEWORDS, niter=NITER, seed=SEED + level, verbose=False)
            kmeans.train(np.ascontiguousarray(residual[available]))
            _, assigned = kmeans.index.search(np.ascontiguousarray(residual), 1)
            codes[:, level] = assigned[:, 0]
            centroids.append(kmeans.centroids.copy())
            residual -= kmeans.centroids[assigned[:, 0]]
        return cls(centroids, codes, exact_below)

    @property
    def nbytes(self) -> int:
        return (sum(c.nbytes for c in self.centroids) + self.codes.nbytes + self.prefix.nbytes + self.order.nbytes
                + self.offsets.nbytes)

    def candidates(self, vector: np.ndarray, allowed: np.ndarray, n: int, product_vectors: np.ndarray | None = None) -> np.ndarray:
        count = int(allowed.sum())
        if n >= count or count <= self.exact_below:
            return np.flatnonzero(allowed)
        tables = [c @ vector for c in self.centroids]
        prefix_order = np.argsort(-(tables[0][:, None] + tables[1][None, :]).ravel(), kind="stable")
        sizes = np.bincount(self.prefix[allowed], minlength=CODEWORDS * CODEWORDS)[prefix_order]
        needed = int(np.searchsorted(np.cumsum(sizes), n)) + 1
        rows = np.concatenate([self.order[self.offsets[p]:self.offsets[p + 1]] for p in prefix_order[:needed]])
        rows = rows[allowed[rows]]
        if product_vectors is None:
            c = self.codes[rows]
            score = tables[0][c[:, 0]] + tables[1][c[:, 1]] + tables[2][c[:, 2]]
        else:
            score = product_vectors[rows] @ vector
        keep = np.argpartition(-score, n - 1)[:n] if len(rows) > n else np.arange(len(rows))
        return rows[keep]


def summarise(per_query: dict, deployed: int, describe) -> dict:
    report = {"by_candidates": {}}
    for n in CANDIDATES:
        entry = {name: distribution([q[n][name] for q in per_query.values()]) for name in PIPELINES}
        entry["latency_ms"] = {name: {part: timing([q[n][f"{name}_{part}_ms"] for q in per_query.values()]) for part in ("candidates", "total")}
                               for name in ("hnsw", "sid", "sid_exact_order")}
        report["by_candidates"][n] = entry
    gap = sorted(per_query, key=lambda key: per_query[key][deployed]["sid"] - per_query[key][deployed]["hnsw"])
    report["worst_sid_minus_hnsw_at_deployed"] = [
        {**describe(key), **{name: per_query[key][deployed][name] for name in PIPELINES}}
        for key in gap[:WORST] if per_query[key][deployed]["sid"] < per_query[key][deployed]["hnsw"]]
    return report


def similar_set(engine, sid: SemanticIdIndex, rows: list[int]) -> dict:
    store, config = engine.store, engine.config
    index, vectors = store.indexes[config.text_model], store.modalities[config.text_model].vectors
    c = store.catalog
    per_query = {}
    warm = rows[:WARMUP]
    for repeat, row in enumerate(warm + rows):
        exhaustive = [r for r, _ in engine.search(row, K, mode="exact").hits]
        allowed = eligible_mask(store, row, config)
        text_order = np.argsort(-np.where(allowed, vectors @ vectors[row], -np.inf), kind="stable")[:int(allowed.sum())]
        result = {}
        for n in CANDIDATES:
            result[n] = {"exact_text": coverage(rerank_similar(engine, row, text_order[:n]), exhaustive)}
            for name, generate in (("hnsw", lambda: served_candidates(index, vectors[row], allowed, n)),
                                   ("sid", lambda: sid.candidates(vectors[row], allowed, n)),
                                   ("sid_exact_order", lambda: sid.candidates(vectors[row], allowed, n, vectors))):
                cands, cand_ms = timed(generate)
                ranked, rank_ms = timed(lambda: rerank_similar(engine, row, cands))
                result[n].update({name: coverage(ranked, exhaustive), f"{name}_candidates_ms": cand_ms, f"{name}_total_ms": cand_ms + rank_ms})
        if repeat >= len(warm):
            per_query[row] = result
    return {"queries": len(rows), **summarise(per_query, DEPLOYED["similar_products"], lambda r: {"id": c.ids[r], "name": c.names[r][:70]})}


def search_set(searcher, sid: SemanticIdIndex, queries: list[str]) -> dict:
    index = searcher.store.indexes[searcher.text_key]
    vectors = searcher.store.modalities[searcher.text_key].vectors
    per_query = {}
    warm = queries[:WARMUP]
    for repeat, query in enumerate(warm + queries):
        exhaustive = list(searcher.search(query, K, mode="exact").rows)
        (parsed, mask, vector), prep_ms = timed(lambda: _prepare(searcher, query))
        text_order = np.argsort(-np.where(mask, vectors @ vector, -np.inf), kind="stable")[:int(mask.sum())]
        result = {}
        for n in CANDIDATES:
            result[n] = {"exact_text": coverage(rerank_search(searcher, parsed, vector, text_order[:n]), exhaustive)}
            for name, generate in (("hnsw", lambda: served_candidates(index, vector, mask, n)),
                                   ("sid", lambda: sid.candidates(vector, mask, n)),
                                   ("sid_exact_order", lambda: sid.candidates(vector, mask, n, vectors))):
                cands, cand_ms = timed(generate)
                ranked, rank_ms = timed(lambda: rerank_search(searcher, parsed, vector, cands))
                result[n].update({name: coverage(ranked, exhaustive), f"{name}_candidates_ms": cand_ms,
                                  f"{name}_total_ms": prep_ms + cand_ms + rank_ms})
        if repeat >= len(warm):
            per_query[query] = result
    return {"queries": len(queries), **summarise(per_query, DEPLOYED["semantic_search"], lambda q: {"query": q})}


def _prepare(searcher, query: str):
    from app.query_parser import parse_query

    parsed = parse_query(query)
    mask, _ = searcher.filter_mask(parsed)
    return parsed, mask, searcher.encode_query(parsed.semantic_text)


def decide(report: dict) -> dict:
    """D15's rule: replace HNSW only with coverage no lower on every query set and lower median latency on every set."""
    verdict = {}
    for variant in ("sid", "sid_exact_order"):
        checks = []
        for task, n in DEPLOYED.items():
            for name, part in report[task].items():
                entry = part["by_candidates"][n]
                checks.append({"task": task, "set": name, "candidates": n,
                               "coverage_not_lower": entry[variant]["mean"] >= entry["hnsw"]["mean"],
                               "median_latency_lower": entry["latency_ms"][variant]["total"]["median_ms"] < entry["latency_ms"]["hnsw"]["total"]["median_ms"]})
        verdict[variant] = {"adopt": all(c["coverage_not_lower"] and c["median_latency_lower"] for c in checks), "checks": checks}
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/experiments/semantic_ids.json")
    args = parser.parse_args()

    engine, searcher, info = load_service()
    store = engine.store
    modality = store.modalities[engine.config.text_model]
    hnsw = store.indexes[engine.config.text_model]
    faiss.omp_set_num_threads(1)
    sid, sid_ms = timed(lambda: SemanticIdIndex.build(modality.vectors, modality.available, hnsw.params.exact_below))
    _, hnsw_ms = timed(lambda: HNSWIndex.build(modality.vectors, hnsw.params))
    prefixes = np.bincount(sid.prefix, minlength=CODEWORDS * CODEWORDS)
    sets = query_sets(engine)
    report = {
        "protocol": "docs/decisions.md D15, pre-registered before any code was trained",
        "codes": {"levels": LEVELS, "codewords": CODEWORDS, "niter": NITER, "seed": SEED,
                  "distinct_ids": len({tuple(c) for c in sid.codes.tolist()}), "products": len(sid.codes),
                  "non_empty_prefixes": int((prefixes > 0).sum()), "largest_prefix": int(prefixes.max()),
                  "median_non_empty_prefix": int(np.median(prefixes[prefixes > 0]))},
        "build_seconds_single_thread": {"semantic_ids": round(sid_ms / 1000, 2), "hnsw": round(hnsw_ms / 1000, 2)},
        "memory_bytes": {"semantic_ids_structures": sid.nbytes, "hnsw_index_with_its_vector_copy": hnsw.memory_bytes,
                         "hnsw_graph_without_vectors": hnsw.memory_bytes - modality.vectors.nbytes,
                         "product_vectors_both_rerankers_need": modality.vectors.nbytes},
        "method": "in-process, same query sets, scorer and coverage definition as scripts.benchmark_ann; FAISS on 1 thread",
        "similar_products": {"dev": similar_set(engine, sid, sets["dev_rows"]), "sample": similar_set(engine, sid, sets["sample_rows"])},
        "semantic_search": {"dev": search_set(searcher, sid, sets["dev_searches"]),
                            "title_pseudo_queries": search_set(searcher, sid, sets["title_queries"]),
                            "title_pseudo_queries_with_price_ceiling": search_set(searcher, sid, sets["priced_queries"])},
    }
    report["rule_d15"] = decide(report)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")

    print(json.dumps({k: report[k] for k in ("codes", "build_seconds_single_thread", "memory_bytes")}, indent=1))
    for task in ("similar_products", "semantic_search"):
        for name, part in report[task].items():
            print(f"\n{task} / {name} ({part['queries']} queries)")
            for n, e in part["by_candidates"].items():
                lat = e["latency_ms"]
                print(f"  N={n:>4}: exact-text {e['exact_text']['mean']:.4f} | hnsw {e['hnsw']['mean']:.4f} {lat['hnsw']['total']['median_ms']} ms"
                      f" | sid {e['sid']['mean']:.4f} {lat['sid']['total']['median_ms']} ms"
                      f" | sid-exact-order {e['sid_exact_order']['mean']:.4f} {lat['sid_exact_order']['total']['median_ms']} ms")
    print("\nrule:", json.dumps({v: r["adopt"] for v, r in report["rule_d15"].items()}))
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
