"""Per-component similarities, their evidence-aware combination, and deterministic ranking.

Each component returns a similarity in [0, 1] plus a mask saying whether both products
have that evidence. Missing-to-missing is never treated as agreement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from app.features import Catalog
from app.settings import ScoringConfig

# Calibrated dense similarities: 0 = typical random pair, CALIBRATED_TOP = typical best match, 1 = identical.
CALIBRATED_TOP = 0.9
_RATING_RANGE = 4.0


@dataclass(frozen=True)
class Anchors:
    random_median: float
    top1_median: float

    def apply(self, raw: np.ndarray) -> np.ndarray:
        lo, hi = self.random_median, min(self.top1_median, 1 - 1e-6)
        raw = np.clip(raw, 0.0, 1.0)
        low_part = (raw - lo) / max(hi - lo, 1e-6) * CALIBRATED_TOP
        high_part = CALIBRATED_TOP + (raw - hi) / (1 - hi) * (1 - CALIBRATED_TOP)
        return np.where(raw <= lo, 0.0, np.where(raw <= hi, low_part, high_part))


@dataclass
class DenseModality:
    """L2-normalised vectors (rows without a usable vector are all-zero and masked out)."""

    vectors: np.ndarray | sparse.csr_matrix
    available: np.ndarray
    anchors: Anchors

    def raw_similarity(self, query: int) -> np.ndarray:
        if sparse.issparse(self.vectors):
            return np.asarray((self.vectors @ self.vectors[query].T).todense()).ravel()
        return self.vectors @ self.vectors[query]


@dataclass
class FeatureStore:
    catalog: Catalog
    modalities: dict[str, DenseModality] = field(default_factory=dict)

    def __post_init__(self) -> None:
        c = self.catalog
        self.has_category = np.asarray(c.category_matrix.getnnz(axis=1) > 0)
        self.color_counts = c.color_matrix.sum(axis=1)
        self.log_price = np.log(c.price)
        self.log_weight = np.log(c.weight_g)
        order = np.argsort(np.asarray(c.ids))
        self.id_rank = np.empty(len(c), dtype=np.int64)
        self.id_rank[order] = np.arange(len(c))


@dataclass
class ComponentScores:
    sims: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]


def _log_kernel(values: np.ndarray, query_value: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(values) & np.isfinite(query_value)
    sims = np.exp(-np.abs(np.where(mask, values, 0.0) - (query_value if np.isfinite(query_value) else 0.0)) / scale)
    return np.where(mask, sims, 0.0), mask


def component_scores(store: FeatureStore, query: int, config: ScoringConfig, needed: set[str]) -> ComponentScores:
    """Similarities between product `query` and every product, for the requested components."""
    c = store.catalog
    sims: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}

    for name, key in (("text", config.text_model), ("image", "image")):
        if name in needed:
            if key not in store.modalities:
                raise LookupError(f"{name} is weighted but representation {key!r} has not been built")
            modality = store.modalities[key]
            masks[name] = modality.available & modality.available[query]
            sims[name] = np.where(masks[name], modality.anchors.apply(modality.raw_similarity(query)), 0.0)
    if "brand" in needed:
        masks["brand"] = (c.brand_codes >= 0) & (c.brand_codes[query] >= 0)
        sims["brand"] = (masks["brand"] & (c.brand_codes == c.brand_codes[query])).astype(float)
    if "category" in needed:
        masks["category"] = store.has_category & store.has_category[query]
        shared = np.asarray((c.category_matrix @ c.category_matrix[query].T).todense()).ravel()
        sims["category"] = (masks["category"] & (shared > 0)).astype(float)
    if "color" in needed:
        masks["color"] = (store.color_counts > 0) & (store.color_counts[query] > 0)
        overlap = c.color_matrix @ c.color_matrix[query]
        union = store.color_counts + store.color_counts[query] - overlap
        sims["color"] = np.where(masks["color"], overlap / np.maximum(union, 1.0), 0.0)
    if "price" in needed:
        sims["price"], masks["price"] = _log_kernel(store.log_price, store.log_price[query], config.price_scale)
    if "weight" in needed:
        sims["weight"], masks["weight"] = _log_kernel(store.log_weight, store.log_weight[query], config.weight_scale)
    if "rating" in needed:
        masks["rating"] = np.isfinite(c.rating) & np.isfinite(c.rating[query])
        sims["rating"] = np.where(masks["rating"], 1 - np.abs(np.nan_to_num(c.rating) - np.nan_to_num(c.rating[query])) / _RATING_RANGE, 0.0)
    return ComponentScores(sims, masks)


def combine(scores: ComponentScores, weights: dict[str, float], min_evidence: float, size: int) -> np.ndarray:
    """Weighted average over evidence both products have, shrunk when that evidence is too thin.

    score = sum(w * s over shared components) / max(sum(w over shared components), min_evidence * sum(w))
    """
    active = {name: w for name, w in weights.items() if w > 0}
    total = sum(active.values())
    numerator = np.zeros(size)
    evidence = np.zeros(size)
    for name, weight in active.items():
        if name not in scores.masks:
            continue
        mask = scores.masks[name]
        numerator += weight * np.where(mask, scores.sims[name], 0.0)
        evidence += weight * mask
    return numerator / np.maximum(evidence, max(min_evidence * total, 1e-12))


def score_all(store: FeatureStore, query: int, config: ScoringConfig) -> np.ndarray:
    weights = config.component_weights()
    needed = {name for name, w in weights.items() if w > 0}
    scores = component_scores(store, query, config, needed)
    return combine(scores, weights, config.min_evidence, len(store.catalog))


def eligible_mask(store: FeatureStore, query: int, config: ScoringConfig) -> np.ndarray:
    c = store.catalog
    eligible = np.ones(len(c), dtype=bool)
    eligible[query] = False
    if config.exclude_same_asin:
        eligible &= c.asins != c.asins[query]
    if config.exclude_query_family and c.family_codes[query] >= 0:
        eligible &= c.family_codes != c.family_codes[query]
    return eligible


@dataclass
class RankResult:
    rows: list[int]
    title_cap_skipped: int = 0
    title_cap_shortfall: int = 0


def order_rows(store: FeatureStore, rows: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Rows sorted by score desc, then rating desc, review count desc, product ID asc."""
    c = store.catalog
    rounded = np.round(scores[rows], 9)
    rating = np.nan_to_num(c.rating[rows], nan=-1.0)
    reviews = np.nan_to_num(c.review_count[rows], nan=-1.0)
    return rows[np.lexsort((store.id_rank[rows], -reviews, -rating, -rounded))]


def select_distinct(store: FeatureStore, ordered: np.ndarray, k: int, config: ScoringConfig) -> RankResult:
    """Walk the ranked rows keeping one row per ASIN, `max_per_family` per variant family and, when
    configured, `max_per_title` per normalised title. The title cap is a result-diversity rule: it keeps
    the relevance order, fills from lower ranks, and records how many result slots it left empty.
    """
    c = store.catalog
    result = RankResult(rows=[])
    seen_asins: set[str] = set()
    family_counts: dict[int, int] = {}
    title_counts: dict[int, int] = {}
    for row in ordered:
        asin, family, title = c.asins[row], int(c.family_codes[row]), int(c.title_codes[row])
        if asin in seen_asins or (family >= 0 and family_counts.get(family, 0) >= config.max_per_family):
            continue
        if config.max_per_title is not None and title >= 0 and title_counts.get(title, 0) >= config.max_per_title:
            result.title_cap_skipped += 1
            continue
        seen_asins.add(asin)
        if family >= 0:
            family_counts[family] = family_counts.get(family, 0) + 1
        title_counts[title] = title_counts.get(title, 0) + 1
        result.rows.append(int(row))
        if len(result.rows) == k:
            break
    result.title_cap_shortfall = min(k - len(result.rows), result.title_cap_skipped)
    return result


def rank(store: FeatureStore, query: int, scores: np.ndarray, k: int, config: ScoringConfig,
         extra_mask: np.ndarray | None = None) -> RankResult:
    """Top-k rows similar to product `query`, after eligibility rules and any caller filter."""
    eligible = eligible_mask(store, query, config)
    if extra_mask is not None:
        eligible &= extra_mask
    return select_distinct(store, order_rows(store, np.flatnonzero(eligible), scores), k, config)
