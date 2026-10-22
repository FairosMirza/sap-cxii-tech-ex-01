"""Similarity search entry points: the required function and the engine behind it."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np

from app.scoring import FeatureStore, rank, score_all
from app.settings import DEFAULT_CONFIG, ScoringConfig, load_config

logger = logging.getLogger(__name__)


class ProductNotFoundError(LookupError):
    """The requested product ID is not in the catalogue."""


@dataclass
class SearchResult:
    hits: list[tuple[int, float]]
    title_cap_skipped: int = 0
    title_cap_shortfall: int = 0


class SimilarityEngine:
    def __init__(self, store: FeatureStore, config: ScoringConfig) -> None:
        self.store = store
        self.config = config

    def validate_k(self, num_similar: object) -> int:
        valid = isinstance(num_similar, (int, np.integer)) and not isinstance(num_similar, bool)
        if not valid or not 1 <= int(num_similar) <= self.config.max_results:
            raise ValueError(f"num_similar must be an integer between 1 and {self.config.max_results}")
        return int(num_similar)

    def row_of(self, product_id: object) -> int:
        row = self.store.catalog.index.get(product_id) if isinstance(product_id, str) else None
        if row is None:
            raise ProductNotFoundError(product_id)
        return row

    def search(self, query: int, k: int, config: ScoringConfig | None = None) -> SearchResult:
        config = config or self.config
        scores = score_all(self.store, query, config)
        ranked = rank(self.store, query, scores, k, config)
        if ranked.title_cap_shortfall:
            logger.info("title cap left %d of %d result slots empty for product %s",
                        ranked.title_cap_shortfall, k, self.store.catalog.ids[query])
        hits = [(row, float(scores[row])) for row in ranked.rows]
        return SearchResult(hits, ranked.title_cap_skipped, ranked.title_cap_shortfall)

    def search_rows(self, query: int, k: int, config: ScoringConfig | None = None) -> list[tuple[int, float]]:
        return self.search(query, k, config).hits

    def find_similar(self, product_id: str, num_similar: int) -> list[str]:
        k = self.validate_k(num_similar)
        query = self.row_of(product_id)
        return [self.store.catalog.ids[row] for row, _ in self.search_rows(query, k)]


_engine: SimilarityEngine | None = None


def get_engine() -> SimilarityEngine:
    global _engine
    if _engine is None:
        from app.pipeline import build_store

        _engine = SimilarityEngine(build_store(), load_config(DEFAULT_CONFIG))
    return _engine


def find_similar_products(product_id: str, num_similar: int) -> List[str]:
    """IDs of the `num_similar` products most similar to `product_id`, most similar first.

    Raises ProductNotFoundError for an unknown ID and ValueError for an invalid count.
    Fewer IDs are returned only when fewer eligible products exist.
    """
    return get_engine().find_similar(product_id, num_similar)
