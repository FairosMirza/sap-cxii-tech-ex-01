from dataclasses import replace

import numpy as np
import pytest

from app.ann import HNSWIndex, HNSWParams, candidate_rows
from app.pipeline import build_store, embedding_modality
from app.retrieval import SimilarFilters, SimilarityEngine
from tests.conftest import config, make_record

N, DIM = 400, 16
CATEGORIES = ["WomensKurtasKurtis", "MensT_Shirts", "WomensSarees", "MensFormalShirts"]


@pytest.fixture(scope="module")
def engine():
    rng = np.random.default_rng(0)
    records = [make_record(f"p{i:03d}", f"Product {i}", brand=f"B{i % 7}", price=f"{200 + 5 * i}.00",
                           category=CATEGORIES[i % 4]) for i in range(N)]
    store = build_store(records, text=False)
    vectors = rng.normal(size=(N, DIM)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    store.modalities["fake"] = embedding_modality(store.catalog, vectors)
    store.indexes["fake"] = HNSWIndex.build(vectors, HNSWParams(exact_below=0))
    return SimilarityEngine(store, replace(config(text=0.5, attributes=0.5), text_model="fake", ann_candidates=60))


def rows(engine, query, k=10, **kwargs):
    return [r for r, _ in engine.search(query, k, **kwargs).hits]


def test_hnsw_search_respects_allowed_rows(engine):
    index = engine.store.indexes["fake"]
    allowed = np.zeros(N, dtype=bool)
    allowed[::3] = True
    found = index.search(engine.store.modalities["fake"].vectors[5], 20, allowed=allowed)
    assert len(found) == 20 and allowed[found].all()


def test_ann_results_match_exact_on_a_small_catalogue(engine):
    overlaps = [len(set(rows(engine, q, mode="ann")) & set(rows(engine, q, mode="exact"))) / 10 for q in range(0, N, 20)]
    assert np.mean(overlaps) >= 0.9


def test_widening_still_returns_k_results(engine):
    tiny = replace(engine.config, ann_candidates=1)
    assert len(rows(engine, 0, k=10, mode="ann", config=tiny)) == 10


def test_filters_apply_in_ann_mode(engine):
    c = engine.store.catalog
    hits = rows(engine, 0, k=10, mode="ann", filters=SimilarFilters(same_brand=True))
    assert len(hits) == 10 and all(c.brand_codes[h] == c.brand_codes[0] for h in hits)


def test_selective_filters_fall_back_to_exact_scoring():
    vectors = np.eye(4, dtype=np.float32)
    index = HNSWIndex.build(vectors, HNSWParams(exact_below=2))
    allowed = np.array([True, False, True, False])
    assert candidate_rows(index, vectors[0], allowed).tolist() == [0, 2]


def test_ann_without_index_is_an_error(engine):
    with pytest.raises(LookupError):
        engine.search(0, 5, config=replace(engine.config, text_model="missing"), mode="ann")
