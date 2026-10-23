"""Build blind candidate pools for human relevance labelling.

Similar products: top-10 of each approach (title cap off). Semantic search: top-10 of each text
representation, with no relevance cutoff. The ranker `served` is the deployed path of each task as configured in
configs/default.yaml (approximate search, and the relevance cutoff for search); `served_exact` is the same
configuration searched exhaustively (mode=exact). Their membership files record that configuration.
Candidates are de-duplicated and shuffled with a fixed seed.
Which system produced each candidate goes to a separate membership file the labelling page never reads.
Run from the repository root:  python -m scripts.build_label_pools --split dev
The holdout is pooled only for named finalists, after the freeze (scripts.freeze):  --split holdout --systems ...
With --supplement [NAME] the existing pools stay untouched: only pairs the rankers now retrieve that no pool of the
split holds (after a model change, or for a variant such as C_bge+title_cap) go to
{task}_{split}_supplement[_NAME].json, queries shuffled. A supplement is written even without new pairs, to record
its rankers. Pools are never overwritten.
"""

from __future__ import annotations

import argparse
import functools
import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np

from app.artifacts import load_service
from app.embeddings import EMBEDDING_MODELS, TextEncoder, cached_product_vectors
from app.pipeline import build_store, embedding_modality
from app.retrieval import SimilarityEngine
from app.scoring import FeatureStore
from app.semantic import SemanticSearcher
from app.settings import ScoringConfig, _validated, load_config
from scripts.check_labels import pool_files

K = 10
SEED = 20261022
SUPPLEMENT_SEED = 20261023
SIMILAR_SYSTEMS = ("A_structured", "B_tfidf", "C_minilm", "C_bge")
SEMANTIC_SYSTEMS = ("tfidf", "minilm", "bge-small")
# Similar-products rankers may carry one variant, written SYSTEM+VARIANT (e.g. C_bge+no_category, C_bge+text_0.3).
VARIANTS = {
    "title_cap": lambda config: replace(config, max_per_title=2),
    "no_category": lambda config: replace(config, attribute_weights={**config.attribute_weights, "category": 0.0}),
    **{f"text_{w}": lambda config, w=w: config.with_blocks(text=w, attributes=round(1 - w, 2)) for w in (0.3, 0.4, 0.6, 0.7)},
}
SEED_OFFSETS = {"similar": 0, "semantic": 1000}
POOL_DIR = Path("eval/pools")
SERVED, SERVED_EXACT = "served", "served_exact"
SERVED_RANKERS = (SERVED, SERVED_EXACT)


@functools.cache
def served_service() -> tuple[SimilarityEngine, SemanticSearcher, dict]:
    return load_service()


def served_base(task: str) -> str:
    """The ranker that the served configuration of `task` applies its search settings to."""
    serving = served_service()[2]["serving"]
    return serving["similar_method"] if task == "similar" else serving["semantic_model"]


def ranker_config(spec: str) -> ScoringConfig:
    method, _, variant = spec.partition("+")
    config = load_config(method=method)
    return _validated(VARIANTS[variant](config)) if variant else config


def is_similar_ranker(spec: str) -> bool:
    method, _, variant = spec.partition("+")
    return spec in SERVED_RANKERS or (method in SIMILAR_SYSTEMS and (not variant or variant in VARIANTS))


def is_semantic_ranker(spec: str) -> bool:
    return spec in SERVED_RANKERS or spec in SEMANTIC_SYSTEMS


def _mode(spec: str) -> str | None:
    return "exact" if spec == SERVED_EXACT else None


def store_with_embeddings(keys: tuple[str, ...] = ("minilm", "bge-small")) -> FeatureStore:
    store = build_store()
    for key in keys:
        encoder = TextEncoder(EMBEDDING_MODELS[key])
        vectors, _ = cached_product_vectors(encoder, store.catalog.texts)
        store.modalities[key] = embedding_modality(store.catalog, vectors, encoder.encode_query)
    return store


def _write(out: Path, name: str, task: str, split: str, seed: int, queries: list[dict], membership: dict,
           systems: tuple[str, ...]) -> int:
    out.mkdir(parents=True, exist_ok=True)
    pool = {"task": task, "split": split, "k": K, "seed": seed, "queries": queries}
    (out / f"{name}.json").write_text(json.dumps(pool, indent=1, ensure_ascii=False) + "\n")
    served = {}
    if set(SERVED_RANKERS) & set(systems):
        info = served_service()[2]
        served = {"served": {"serving": info["serving"], "code_commit": info["code_commit"], "artifacts_created": info["created"]}}
    (out / f"{name}_membership.json").write_text(
        json.dumps({"systems": list(systems), **served, "queries": membership}, indent=1, ensure_ascii=False) + "\n")
    return sum(len(q["candidates"]) for q in queries)


def _blind(index: int, ids: set[str]) -> list[str]:
    ordered = sorted(ids)
    return [ordered[i] for i in np.random.default_rng([SEED, index]).permutation(len(ordered))]


def similar_rankings(store: FeatureStore, split: str, systems: tuple[str, ...]) -> dict[str, dict[str, list[str]]]:
    """Top-k product IDs of each ranker for every query product of the split, in query order."""
    queries = [q["id"] for q in json.loads(Path("eval/queries.json").read_text())["queries"] if q["split"] == split]
    engines = {spec: served_service()[0] if spec in SERVED_RANKERS else SimilarityEngine(store, ranker_config(spec)) for spec in systems}
    return {q: {name: [e.store.catalog.ids[r] for r, _ in e.search(e.row_of(q), K, mode=_mode(name)).hits] for name, e in engines.items()}
            for q in queries}


def semantic_rankings(store: FeatureStore, split: str, systems: tuple[str, ...]) -> dict[str, dict[str, list[str]]]:
    texts = json.loads(Path("eval/semantic_queries.json").read_text())[split]
    searchers = {key: served_service()[1] if key in SERVED_RANKERS else SemanticSearcher(store, key, load_config()) for key in systems}
    return {text: {key: [s.store.catalog.ids[r] for r in s.search(text, K, mode=_mode(key)).rows] for key, s in searchers.items()}
            for text in texts}


def pool_queries(rankings: dict[str, dict[str, list[str]]], offset: int) -> list[dict]:
    return [{"query": q, "candidates": _blind(offset + i, {pid for ids in ranked.values() for pid in ids})}
            for i, (q, ranked) in enumerate(rankings.items())]


def supplement_queries(rankings: dict[str, dict[str, list[str]]], pooled: set[tuple[str, str]]) -> tuple[list[dict], dict]:
    """Retrieved pairs missing from `pooled`: queries in shuffled order, candidates blind, membership kept apart."""
    new = {q: {pid for ids in ranked.values() for pid in ids if (q, pid) not in pooled} for q, ranked in rankings.items()}
    queries = [q for q in rankings if new[q]]
    queries = [queries[i] for i in np.random.default_rng(SUPPLEMENT_SEED).permutation(len(queries))]
    pool = [{"query": q, "candidates": _blind(SUPPLEMENT_SEED + i, new[q])} for i, q in enumerate(queries)]
    membership = {q: {name: [pid for pid in ids if pid in new[q]] for name, ids in rankings[q].items()} for q in queries}
    return pool, membership


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "holdout"], default="dev")
    parser.add_argument("--systems", nargs="*", help="required for the holdout: the finalist systems only")
    parser.add_argument("--out-dir", default=str(POOL_DIR), help="use a scratch directory for checks; labels refer to eval/pools")
    parser.add_argument("--supplement", nargs="?", const="", default=None, metavar="NAME",
                        help="write only pairs no existing pool of the split holds, to a separate file")
    args = parser.parse_args()
    if args.split == "holdout" and not args.systems:
        parser.error("the holdout is pooled only for explicitly named finalists (--systems ...)")
    unknown = [s for s in args.systems or () if not is_similar_ranker(s) and not is_semantic_ranker(s)]
    if unknown:
        parser.error(f"unknown systems {unknown}; variants are written SYSTEM+VARIANT with VARIANT in {sorted(VARIANTS)}, "
                     f"and {SERVED!r} and {SERVED_EXACT!r} are the deployed configuration, searched as served or exactly")
    if args.supplement and not re.fullmatch(r"[a-z0-9_]+", args.supplement):
        parser.error("supplement names use lowercase letters, digits and underscores")
    if args.split == "holdout":
        from scripts.freeze import check

        check()

    out = Path(args.out_dir)
    suffix = "" if args.supplement is None else "_supplement" + (f"_{args.supplement}" if args.supplement else "")
    similar = tuple(s for s in (args.systems or SIMILAR_SYSTEMS) if is_similar_ranker(s))
    semantic = tuple(s for s in (args.systems or SEMANTIC_SYSTEMS) if is_semantic_ranker(s))
    jobs = [(task, systems, rank) for task, systems, rank in (("similar", similar, similar_rankings),
                                                            ("semantic", semantic, semantic_rankings)) if systems]
    for task, _, _ in jobs:
        name = f"{task}_{args.split}{suffix}"
        if (out / f"{name}.json").exists():
            parser.error(f"{out / name}.json exists; pools that may already be labelled are never overwritten")
        if suffix and not (out / f"{task}_{args.split}.json").exists():
            parser.error(f"no existing {task} pool for {args.split} to supplement")

    store = store_with_embeddings()
    for task, systems, rank in jobs:
        rankings = rank(store, args.split, systems)
        if not suffix:
            count = _write(out, f"{task}_{args.split}", task, args.split, SEED, pool_queries(rankings, SEED_OFFSETS[task]),
                           rankings, systems)
            print(f"{task} pool ({args.split}): {count} judgements")
            continue
        pooled = {(q["query"], c) for path in pool_files(out, task, args.split)
                  for q in json.loads(path.read_text())["queries"] for c in q["candidates"]}
        queries, membership = supplement_queries(rankings, pooled)
        count = _write(out, f"{task}_{args.split}{suffix}", task, args.split, SUPPLEMENT_SEED, queries, membership, systems)
        print(f"{task} supplemental pool ({args.split}{suffix}): {count} judgements over {len(queries)} queries"
              + ("; the existing pools already cover every retrieved pair (rankers recorded)" if not queries else ""))


if __name__ == "__main__":
    main()
