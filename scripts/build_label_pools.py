"""Build blind candidate pools for human relevance labelling.

Similar products: top-10 of each approach (title cap off). Semantic search: top-10 of each text
representation, with no relevance cutoff. Candidates are de-duplicated and shuffled with a fixed seed.
Which system produced each candidate goes to a separate membership file the labelling page never reads.
Run from the repository root:  python -m scripts.build_label_pools --split dev
The holdout is pooled only for named finalists:  --split holdout --systems <finalists...>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from app.embeddings import EMBEDDING_MODELS, TextEncoder, cached_product_vectors
from app.pipeline import build_store, embedding_modality
from app.retrieval import SimilarityEngine
from app.scoring import FeatureStore
from app.semantic import SemanticSearcher
from app.settings import load_config

K = 10
SEED = 20261022
SIMILAR_SYSTEMS = ("A_structured", "B_tfidf", "C_minilm", "C_bge")
SEMANTIC_SYSTEMS = ("tfidf", "minilm", "bge-small")
POOL_DIR = Path("eval/pools")


def store_with_embeddings(keys: tuple[str, ...] = ("minilm", "bge-small")) -> FeatureStore:
    store = build_store()
    for key in keys:
        encoder = TextEncoder(EMBEDDING_MODELS[key])
        vectors, _ = cached_product_vectors(encoder, store.catalog.texts)
        store.modalities[key] = embedding_modality(store.catalog, vectors, encoder.encode_query)
    return store


def _write(task: str, split: str, queries: list[dict], membership: dict, systems: tuple[str, ...]) -> int:
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    pool = {"task": task, "split": split, "k": K, "seed": SEED, "queries": queries}
    (POOL_DIR / f"{task}_{split}.json").write_text(json.dumps(pool, indent=1, ensure_ascii=False) + "\n")
    (POOL_DIR / f"{task}_{split}_membership.json").write_text(
        json.dumps({"systems": list(systems), "queries": membership}, indent=1, ensure_ascii=False) + "\n")
    return sum(len(q["candidates"]) for q in queries)


def _blind(index: int, ids: set[str]) -> list[str]:
    ordered = sorted(ids)
    return [ordered[i] for i in np.random.default_rng([SEED, index]).permutation(len(ordered))]


def similar_pool(store: FeatureStore, split: str, systems: tuple[str, ...]) -> int:
    queries = [q for q in json.loads(Path("eval/queries.json").read_text())["queries"] if q["split"] == split]
    engines = {name: SimilarityEngine(store, load_config(method=name)) for name in systems}
    pooled, membership = [], {}
    for i, q in enumerate(queries):
        ranked = {name: [store.catalog.ids[r] for r, _ in e.search(e.row_of(q["id"]), K).hits] for name, e in engines.items()}
        membership[q["id"]] = ranked
        pooled.append({"query": q["id"], "candidates": _blind(i, {pid for ids in ranked.values() for pid in ids})})
    return _write("similar", split, pooled, membership, systems)


def semantic_pool(store: FeatureStore, split: str, systems: tuple[str, ...]) -> int:
    texts = json.loads(Path("eval/semantic_queries.json").read_text())[split]
    searchers = {key: SemanticSearcher(store, key, load_config()) for key in systems}
    pooled, membership = [], {}
    for i, text in enumerate(texts):
        ranked = {key: [store.catalog.ids[r] for r in s.search(text, K).rows] for key, s in searchers.items()}
        membership[text] = ranked
        pooled.append({"query": text, "candidates": _blind(1000 + i, {pid for ids in ranked.values() for pid in ids})})
    return _write("semantic", split, pooled, membership, systems)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "holdout"], default="dev")
    parser.add_argument("--systems", nargs="*", help="required for the holdout: the finalist systems only")
    args = parser.parse_args()
    if args.split == "holdout" and not args.systems:
        parser.error("the holdout is pooled only for explicitly named finalists (--systems ...)")

    store = store_with_embeddings()
    similar = tuple(s for s in (args.systems or SIMILAR_SYSTEMS) if s in SIMILAR_SYSTEMS)
    semantic = tuple(s for s in (args.systems or SEMANTIC_SYSTEMS) if s in SEMANTIC_SYSTEMS)
    if similar:
        print(f"similar-products pool ({args.split}): {similar_pool(store, args.split, similar)} judgements")
    if semantic:
        print(f"semantic-search pool ({args.split}): {semantic_pool(store, args.split, semantic)} judgements")


if __name__ == "__main__":
    main()
