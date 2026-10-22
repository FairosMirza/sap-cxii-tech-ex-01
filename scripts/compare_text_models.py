"""Compare text representations (TF-IDF and pinned embedding models) without human labels.

Measures model size, product encoding time, single-query encoding latency, how often a product's
10 nearest text neighbours share its observed category, and fused-scorer diagnostics on the
development queries. These are diagnostics for choosing approach C, not relevance verdicts.
Run from the repository root:  python -m scripts.compare_text_models
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy import sparse

from app.embeddings import EMBEDDING_MODELS, ONNX_MODEL, TextEncoder, cached_product_vectors
from app.features import SOURCE_PRIMARY
from app.pipeline import build_store, embedding_modality
from app.retrieval import SimilarityEngine
from app.scoring import FeatureStore
from app.settings import load_config

SEED = 7
PURITY_SAMPLE = 2000
K = 10
QUERY_WARMUP, QUERY_RUNS = 5, 50
SAMPLE_QUERIES = [
    "comfortable running shoes under $100", "warm winter jackets size M", "affordable summer dresses for women",
    "black leather wallet for men", "durable workout gear",
]


def neighbour_category_purity(store: FeatureStore, key: str, rows: np.ndarray) -> float:
    """Share of each sampled product's top-10 text neighbours (other ASINs and families) sharing a category."""
    c, vectors = store.catalog, store.modalities[key].vectors
    shares = []
    for start in range(0, len(rows), 250):
        chunk = rows[start:start + 250]
        sims = vectors[chunk] @ vectors.T
        sims = sims.toarray() if sparse.issparse(sims) else np.asarray(sims)
        for i, q in enumerate(chunk):
            blocked = c.asins == c.asins[q]
            if c.family_codes[q] >= 0:
                blocked |= c.family_codes == c.family_codes[q]
            sims[i, blocked | ~store.modalities[key].available] = -np.inf
            top = np.argpartition(-sims[i], K)[:K]
            shared = np.asarray((c.category_matrix[top] @ c.category_matrix[q].T).todense()).ravel() > 0
            shares.append(shared.mean())
    return round(float(np.mean(shares)), 4)


def dev_diagnostics(store: FeatureStore, key: str, queries: list[dict], baseline_tops: dict) -> dict:
    config = replace(load_config(method="B_tfidf"), text_model=key, name=f"C_{key}")
    engine = SimilarityEngine(store, config)
    c = store.catalog
    category, brand, overlap = [], [], []
    for q in queries:
        row = engine.row_of(q["id"])
        hits = [r for r, _ in engine.search(row, K).hits]
        q_cats = set(c.category_matrix[row].indices)
        if q_cats:
            category.append(np.mean([bool(q_cats & set(c.category_matrix[r].indices)) for r in hits]))
        if c.brand_codes[row] >= 0:
            brand.append(np.mean([c.brand_codes[r] == c.brand_codes[row] for r in hits]))
        ids = {c.ids[r] for r in hits}
        overlap.append(len(ids & baseline_tops[q["id"]]) / len(ids | baseline_tops[q["id"]]))
    return {"category_agreement_at_10": round(float(np.mean(category)), 4),
            "brand_agreement_at_10": round(float(np.mean(brand)), 4),
            "top10_jaccard_vs_B": round(float(np.mean(overlap)), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", default="eval/queries.json")
    parser.add_argument("--models", nargs="+", default=list(EMBEDDING_MODELS))
    parser.add_argument("--out", default="reports/text_model_comparison.json")
    args = parser.parse_args()

    store = build_store()
    c = store.catalog
    dev = [q for q in json.loads(Path(args.queries).read_text())["queries"] if q["split"] == "dev"]
    rng = np.random.default_rng(SEED)
    observed = np.flatnonzero(c.category_source == SOURCE_PRIMARY)
    sample = rng.choice(observed, PURITY_SAMPLE, replace=False)

    baseline = SimilarityEngine(store, load_config(method="B_tfidf"))
    baseline_tops = {q["id"]: {c.ids[r] for r, _ in baseline.search(baseline.row_of(q["id"]), K).hits} for q in dev}

    report = {"environment": f"{platform.machine()} {platform.platform()}, Python {platform.python_version()}, CPU, ONNX Runtime, default threads",
              "query_latency_method": f"median and p95 of encoding one query, {QUERY_RUNS} runs after {QUERY_WARMUP} warm-up runs, CPU",
              "category_purity_method": f"{PURITY_SAMPLE} products with observed category; top-{K} text neighbours excluding same ASIN and variant family",
              "representations": {}}
    report["representations"]["tfidf"] = {
        "dimensions": int(store.modalities["tfidf"].vectors.shape[1]),
        "neighbour_category_purity_at_10": neighbour_category_purity(store, "tfidf", sample),
        "fused_dev_diagnostics": dev_diagnostics(store, "tfidf", dev, baseline_tops),
    }
    for key in args.models:
        spec = EMBEDDING_MODELS[key]
        encoder = TextEncoder(spec)
        started = time.perf_counter()
        vectors, from_cache = cached_product_vectors(encoder, c.texts)
        encode_seconds = time.perf_counter() - started
        store.modalities[key] = embedding_modality(c, vectors)
        timings = []
        for i in range(QUERY_WARMUP + QUERY_RUNS):
            t0 = time.perf_counter()
            encoder.encode_query(SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)])
            if i >= QUERY_WARMUP:
                timings.append(1000 * (time.perf_counter() - t0))
        report["representations"][key] = {
            "model_id": spec.model_id, "revision": encoder.revision, "dimensions": spec.dim,
            "onnx_model_mb": round((encoder.path / ONNX_MODEL).stat().st_size / 1e6, 1),
            "product_encoding_seconds": None if from_cache else round(encode_seconds, 1),
            "query_encoding_ms": {"median": round(float(np.median(timings)), 2), "p95": round(float(np.percentile(timings, 95)), 2)},
            "anchors": store.modalities[key].anchors.__dict__,
            "neighbour_category_purity_at_10": neighbour_category_purity(store, key, sample),
            "fused_dev_diagnostics": dev_diagnostics(store, key, dev, baseline_tops),
        }
        print(key, json.dumps(report["representations"][key]), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print("tfidf", json.dumps(report["representations"]["tfidf"]))


if __name__ == "__main__":
    main()
