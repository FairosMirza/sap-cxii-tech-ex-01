"""Offline construction of product representations. Nothing here runs per request."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from app.data import DEFAULT_DATA_FILE, load_records
from app.features import Catalog, build_catalog
from app.scoring import Anchors, DenseModality, FeatureStore

ANCHOR_SEED = 13
ANCHOR_RANDOM_PAIRS = 20_000
ANCHOR_QUERIES = 2_000
_CHUNK = 250


def _rowwise_dot(vectors, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if sparse.issparse(vectors):
        return np.asarray(vectors[a].multiply(vectors[b]).sum(axis=1)).ravel()
    return np.einsum("ij,ij->i", vectors[a], vectors[b])


def fit_anchors(vectors, available: np.ndarray, asins: np.ndarray, seed: int = ANCHOR_SEED) -> Anchors:
    """Median similarity of random pairs, and median similarity of each sampled product's best match.

    Same-ASIN rows are excluded so duplicate listings do not inflate the 'best match' anchor.
    """
    rng = np.random.default_rng(seed)
    rows = np.flatnonzero(available)
    if len(rows) < 2:
        return Anchors(random_median=0.0, top1_median=1.0)
    a, b = rng.choice(rows, ANCHOR_RANDOM_PAIRS), rng.choice(rows, ANCHOR_RANDOM_PAIRS)
    keep = asins[a] != asins[b]
    random_median = float(np.median(_rowwise_dot(vectors, a[keep], b[keep])))

    queries = rng.choice(rows, min(ANCHOR_QUERIES, len(rows)), replace=False)
    best = []
    for start in range(0, len(queries), _CHUNK):
        chunk = queries[start:start + _CHUNK]
        sims = vectors[chunk] @ vectors.T
        sims = sims.toarray() if sparse.issparse(sims) else np.asarray(sims)
        sims[:, ~available] = -np.inf
        for i, q in enumerate(chunk):
            sims[i, asins == asins[q]] = -np.inf
        best.extend(sims.max(axis=1))
    return Anchors(random_median=random_median, top1_median=float(np.median(best)))


def tfidf_modality(catalog: Catalog) -> DenseModality:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", stop_words="english", ngram_range=(1, 2),
        min_df=2, max_df=0.5, sublinear_tf=True, dtype=np.float32,
    )
    try:
        vectors = vectorizer.fit_transform(catalog.texts).tocsr()
    except ValueError:  # no term survives the document-frequency limits (tiny catalogues)
        vectors = sparse.csr_matrix((len(catalog), 1), dtype=np.float32)
    available = np.asarray(vectors.getnnz(axis=1) > 0)
    return DenseModality(vectors, available, fit_anchors(vectors, available, catalog.asins))


def build_store(records: list[dict] | None = None, text: bool = True) -> FeatureStore:
    if records is None:
        records, _ = load_records(DEFAULT_DATA_FILE)
    store = FeatureStore(build_catalog(records))
    if text:
        store.modalities["text"] = tfidf_modality(store.catalog)
    return store
