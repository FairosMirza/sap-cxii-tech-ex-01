"""Free-text product search: parse constraints, filter first, then rank by embedding similarity.

Filters exclude only confident conflicts: an explicit price limit removes products outside it (and
products without a price, which cannot be verified); a stated audience or colour removes products whose
known audience or colours contradict it. Unknown values stay eligible. Soft preferences (colour, material,
size, cheaper/pricier) add small bonuses that reorder close matches but cannot lift unrelated products.
An optional relevance cutoff on the query-product cosine (before bonuses) drops unrelated results, so fewer
than `k`, or none, may come back (D12).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from app.ann import candidate_rows
from app.features import COLOR_FAMILIES, SOURCE_NONE, normalize_key
from app.query_parser import ParsedQuery, parse_query
from app.scoring import FeatureStore, order_rows, select_distinct
from app.settings import ScoringConfig

ITEM_COLOUR_BONUS = 0.06
AVAILABLE_COLOUR_BONUS = 0.03
MATERIAL_BONUS = 0.04
SIZE_BONUS = 0.02
PRICE_PREFERENCE_BONUS = 0.05
_SIZE_TOKENS = {"S": {"s", "small"}, "M": {"m", "medium"}, "L": {"l", "large"}, "XL": {"xl"}, "XXL": {"xxl", "2xl"},
                "XS": {"xs"}, "FREE": {"free"}}
_MULTICOLOR = COLOR_FAMILIES.index("multicolor")


@dataclass
class SemanticResult:
    rows: list[int]
    scores: list[float]
    parsed: ParsedQuery
    eligible: int
    excluded: dict[str, int] = field(default_factory=dict)


class SemanticSearcher:
    def __init__(self, store: FeatureStore, text_key: str, config: ScoringConfig, encode_query=None,
                 min_similarity: float | None = None) -> None:
        self.store = store
        self.text_key = text_key
        self.encode_query = encode_query or store.modalities[text_key].query_encoder
        if self.encode_query is None:
            raise ValueError(f"representation {text_key!r} cannot encode free-text queries")
        self.config = config
        self.min_similarity = min_similarity
        c = store.catalog
        self.title_tokens = [set(normalize_key(text).split()) for text in c.texts]
        self.price_percentile = _price_percentile_within_category(store)

    def filter_mask(self, parsed: ParsedQuery) -> tuple[np.ndarray, dict[str, int]]:
        c = self.store.catalog
        mask = self.store.modalities[self.text_key].available.copy()
        excluded: dict[str, int] = {}

        def apply(name: str, keep: np.ndarray) -> None:
            excluded[name] = int((mask & ~keep).sum())
            mask[:] &= keep

        if parsed.price is not None:
            finite = np.isfinite(c.price)
            allowed = np.zeros(len(c), dtype=bool)
            allowed[finite] = [parsed.price.allows(p) for p in c.price[finite]]
            apply("price", allowed)
        if parsed.audience:
            apply("audience", (c.audience == 0) | ((c.audience & parsed.audience) > 0))
        if parsed.colors:
            wanted = np.zeros(len(COLOR_FAMILIES), dtype=np.float32)
            wanted[[COLOR_FAMILIES.index(name) for name in parsed.colors]] = 1.0
            compatible = (c.color_matrix @ wanted > 0) | (c.variant_color_matrix @ wanted > 0) | (c.color_matrix[:, _MULTICOLOR] > 0)
            apply("colour", (c.color_source == SOURCE_NONE) | compatible)
        return mask, excluded

    def bonuses(self, parsed: ParsedQuery, rows: np.ndarray) -> np.ndarray:
        c = self.store.catalog
        bonus = np.zeros(len(rows))
        if parsed.colors:
            wanted = np.zeros(len(COLOR_FAMILIES), dtype=np.float32)
            wanted[[COLOR_FAMILIES.index(name) for name in parsed.colors]] = 1.0
            is_colour = c.color_matrix[rows] @ wanted > 0
            available_in = c.variant_color_matrix[rows] @ wanted > 0
            bonus += np.where(is_colour, ITEM_COLOUR_BONUS, np.where(available_in, AVAILABLE_COLOUR_BONUS, 0.0))
        if parsed.materials:
            bonus += MATERIAL_BONUS * np.array([bool(parsed.materials & self.title_tokens[r]) for r in rows])
        if parsed.sizes:
            tokens = set().union(*(_SIZE_TOKENS.get(s, {s.lower()}) for s in parsed.sizes))
            bonus += SIZE_BONUS * np.array([bool(tokens & self.title_tokens[r]) for r in rows])
        if parsed.price_preference:
            percentile = np.nan_to_num(self.price_percentile[rows], nan=0.5)
            bonus += PRICE_PREFERENCE_BONUS * (1 - percentile if parsed.price_preference == "low" else percentile)
        return bonus

    def search(self, query: str, k: int, mode: str | None = None) -> SemanticResult:
        parsed = parse_query(query)
        mask, excluded = self.filter_mask(parsed)
        count = int(mask.sum())
        modality = self.store.modalities[self.text_key]
        query_vector = self.encode_query(parsed.semantic_text)
        index = self.store.indexes.get(self.text_key) if (mode or self.config.search_mode) == "ann" else None
        scores = np.full(len(self.store.catalog), -np.inf)
        wanted = self.config.ann_candidates
        while True:
            rows = np.flatnonzero(mask) if index is None or wanted >= count else candidate_rows(index, query_vector, mask, wanted)
            similarity = np.zeros(0)
            if len(rows):
                if sparse.issparse(modality.vectors):
                    similarity = np.asarray((modality.vectors[rows] @ query_vector.T).todense()).ravel()
                else:
                    similarity = modality.vectors[rows] @ query_vector
                scores[rows] = similarity + self.bonuses(parsed, rows)
            kept = rows if self.min_similarity is None else rows[similarity >= self.min_similarity]
            selected = select_distinct(self.store, order_rows(self.store, kept, scores), k, self.config)
            # Candidates are the most similar products: once one falls below the cutoff, wider searches add none.
            if len(selected.rows) == k or len(rows) >= count or len(kept) < len(rows):
                break
            wanted *= 4
        return SemanticResult(selected.rows, [float(scores[r]) for r in selected.rows], parsed, count, excluded)


def _price_percentile_within_category(store: FeatureStore) -> np.ndarray:
    """Each product's price percentile among products of its first category (NaN when unknown)."""
    c = store.catalog
    percentile = np.full(len(c), np.nan)
    first = np.full(len(c), -1)
    rows, cols = c.category_matrix.nonzero()
    for row, col in zip(rows[::-1], cols[::-1]):
        first[row] = col
    for category in np.unique(first[first >= 0]):
        members = np.flatnonzero((first == category) & np.isfinite(c.price))
        if len(members) > 1:
            order = c.price[members].argsort().argsort()
            percentile[members] = order / (len(members) - 1)
    return percentile
