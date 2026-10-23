import numpy as np

from experiments.semantic_ids import CODEWORDS, LEVELS, SemanticIdIndex


def _unit(rows: int, dim: int = 16, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=(rows, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_codes_reconstruct_better_with_each_level():
    vectors = _unit(3000)
    index = SemanticIdIndex.build(vectors, np.ones(len(vectors), dtype=bool), exact_below=0)
    assert index.codes.shape == (3000, LEVELS) and index.codes.max() < CODEWORDS
    errors, reconstruction = [], np.zeros_like(vectors)
    for level, centroids in enumerate(index.centroids):
        reconstruction += centroids[index.codes[:, level]]
        errors.append(float(np.linalg.norm(vectors - reconstruction, axis=1).mean()))
    assert errors[0] > errors[1] > errors[2]


def test_candidates_respect_budget_and_filter_and_fall_back_to_all_allowed():
    vectors = _unit(3000, seed=1)
    index = SemanticIdIndex.build(vectors, np.ones(len(vectors), dtype=bool), exact_below=0)
    allowed = np.ones(len(vectors), dtype=bool)
    allowed[::3] = False
    rows = index.candidates(vectors[5], allowed, 200)
    assert len(rows) == 200 == len(set(rows.tolist())) and allowed[rows].all()
    assert len(index.candidates(vectors[5], allowed, 200, vectors)) == 200
    few = np.zeros(len(vectors), dtype=bool)
    few[:50] = True
    assert sorted(index.candidates(vectors[5], few, 200).tolist()) == list(range(50))


def test_query_vector_finds_its_own_product_among_candidates():
    vectors = _unit(3000, seed=2)
    index = SemanticIdIndex.build(vectors, np.ones(len(vectors), dtype=bool), exact_below=0)
    hits = sum(q in index.candidates(vectors[q], np.ones(len(vectors), dtype=bool), 100).tolist() for q in range(0, 3000, 100))
    assert hits >= 27
