"""Approximate nearest-neighbour candidate generation with FAISS HNSW.

HNSW: Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical
Navigable Small World graphs", IEEE TPAMI 2018 (arXiv:1603.09320). Library: FAISS (Douze et al., 2024,
arXiv:2401.08281). The index only proposes candidates; final ranking always uses the exact hybrid scorer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass(frozen=True)
class HNSWParams:
    m: int = 32
    ef_construction: int = 200
    ef_search: int = 128
    candidates: int = 200
    # When a filter leaves this many products or fewer, they are scored exactly instead of via the graph.
    exact_below: int = 2000


class HNSWIndex:
    """Inner-product HNSW over unit vectors (inner product equals cosine)."""

    def __init__(self, index: faiss.Index, params: HNSWParams) -> None:
        self.index = index
        self.params = params

    @classmethod
    def build(cls, vectors: np.ndarray, params: HNSWParams = HNSWParams()) -> "HNSWIndex":
        index = faiss.IndexHNSWFlat(vectors.shape[1], params.m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = params.ef_construction
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        return cls(index, params)

    @classmethod
    def load(cls, path: Path, params: HNSWParams) -> "HNSWIndex":
        return cls(faiss.read_index(str(path)), params)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @property
    def size(self) -> int:
        return self.index.ntotal

    @property
    def memory_bytes(self) -> int:
        return int(faiss.serialize_index(self.index).nbytes)

    def search(self, vector: np.ndarray, k: int, allowed: np.ndarray | None = None,
               ef_search: int | None = None) -> np.ndarray:
        """Rows with (approximately) the k highest inner products, restricted to `allowed` rows if given."""
        ef = max(ef_search or self.params.ef_search, k)
        query = np.ascontiguousarray(vector.reshape(1, -1), dtype=np.float32)
        if allowed is None:
            _, ids = self.index.search(query, k, params=faiss.SearchParametersHNSW(efSearch=ef))
            return ids[0][ids[0] >= 0]
        # The bitmap and selector must stay referenced until the search has finished.
        bits = np.packbits(allowed.astype(np.uint8), bitorder="little")
        selector = faiss.IDSelectorBitmap(len(bits), faiss.swig_ptr(bits))
        params = faiss.SearchParametersHNSW(efSearch=ef, sel=selector)
        _, ids = self.index.search(query, k, params=params)
        del params, selector, bits
        return ids[0][ids[0] >= 0]


def candidate_rows(index: HNSWIndex | None, vector: np.ndarray, allowed: np.ndarray, candidates: int | None = None) -> np.ndarray:
    """Rows to score exactly: all allowed rows when there is no index or the filter is selective, else HNSW candidates."""
    count = int(allowed.sum())
    if index is None or count <= index.params.exact_below:
        return np.flatnonzero(allowed)
    return index.search(vector, min(candidates or index.params.candidates, count), allowed=allowed)
