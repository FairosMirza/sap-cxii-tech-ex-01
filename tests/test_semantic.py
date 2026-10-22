import numpy as np
import pytest

from app.pipeline import build_store, embedding_modality
from app.retrieval import FilterNotApplicableError, SimilarFilters
from app.semantic import SemanticSearcher
from tests.conftest import config, engine_for, make_record

VOCAB = ["running", "shoe", "shoes", "wallet", "leather", "dress", "jacket", "winter", "women", "men"]


def fake_encode(text: str) -> np.ndarray:
    """Bag-of-words stand-in for a sentence encoder, so tests need no model download."""
    words = text.lower().replace("'", " ").split()
    vector = np.array([float(sum(w.startswith(v) for w in words)) for v in VOCAB], dtype=np.float32)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def searcher_for(records: list[dict]) -> SemanticSearcher:
    store = build_store(records, text=False)
    vectors = np.stack([fake_encode(t) for t in store.catalog.texts])
    store.modalities["fake"] = embedding_modality(store.catalog, vectors)
    return SemanticSearcher(store, "fake", config(text=0.5, attributes=0.5), fake_encode)


def short_ids(searcher: SemanticSearcher, query: str, k: int = 10) -> list[str]:
    result = searcher.search(query, k)
    return [searcher.store.catalog.ids[r].rstrip("0") for r in result.rows]


def test_price_limit_applies_and_unknown_prices_are_excluded():
    searcher = searcher_for([
        make_record("cheap", "Running Shoes", price="450.00", category="Shoes"),
        make_record("pricey", "Running Shoes Pro", price="1999.00", category="Shoes"),
        make_record("noprice", "Running Shoes Lite", price=None, category="Shoes"),
        make_record("wallet", "Leather Wallet", price="300.00", category="Wallets"),
    ])
    result = searcher.search("comfortable running shoes under 500", 10)
    prices = [searcher.store.catalog.price[r] for r in result.rows]
    assert prices and all(p < 500 for p in prices)
    assert short_ids(searcher, "comfortable running shoes under 500")[0] == "cheap"
    assert result.excluded["price"] == 2


def test_semantic_similarity_orders_results():
    searcher = searcher_for([
        make_record("wallet", "Leather Wallet", category="Wallets"),
        make_record("shoe", "Running Shoes", category="Shoes"),
        make_record("jacket", "Winter Jacket", category="Jackets"),
    ])
    assert short_ids(searcher, "running shoes")[0] == "shoe"
    assert short_ids(searcher, "warm winter jacket")[0] == "jacket"


def test_audience_conflicts_are_excluded_but_unknown_audience_is_kept():
    searcher = searcher_for([
        make_record("womens", "Women's Leather Wallet", category=None),
        make_record("mens", "Men's Leather Wallet", category=None),
        make_record("unknown", "Leather Wallet", category=None),
    ])
    assert set(short_ids(searcher, "leather wallet for men")) == {"mens", "unknown"}


def test_colour_item_match_beats_available_in_and_conflicts_are_excluded():
    available = {**make_record("available", "Leather Wallet Brown", category=None), "colour": "Brown|Black"}
    searcher = searcher_for([
        make_record("black", "Black Leather Wallet", category=None),
        available,
        make_record("red", "Red Leather Wallet", category=None),
        make_record("unknown", "Leather Wallet", category=None),
    ])
    ranked = short_ids(searcher, "black leather wallet")
    assert ranked[0] == "black" and "red" not in ranked
    assert ranked.index("available") < ranked.index("unknown")


def test_affordable_prefers_cheaper_within_category():
    searcher = searcher_for([
        make_record("dear", "Women Dress", price="2999.00", category="WomensDresses"),
        make_record("cheap", "Women Dress", price="399.00", category="WomensDresses", brand="Other"),
    ])
    assert short_ids(searcher, "affordable dress for women")[0] == "cheap"


def test_filters_can_leave_no_results():
    searcher = searcher_for([make_record("shoe", "Running Shoes", price="900.00", category="Shoes")])
    result = searcher.search("running shoes under 100", 5)
    assert result.rows == [] and result.eligible == 0


def test_similar_products_same_brand_and_price_filters():
    records = [
        make_record("q", "Black Cotton Kurti", brand="Acme", price="500.00"),
        make_record("a1", "Red Cotton Kurti", brand="Acme", price="550.00"),
        make_record("a2", "Blue Cotton Kurti", brand="Acme", price="1500.00"),
        make_record("b1", "Black Rayon Kurti", brand="Other", price="480.00"),
    ]
    engine = engine_for(records)
    query = records[0]["uniq_id"]
    same_brand = engine.find_similar(query, 5, SimilarFilters(same_brand=True))
    assert {i.rstrip("0") for i in same_brand} == {"a1", "a2"}
    cheap = engine.find_similar(query, 5, SimilarFilters(max_price=600))
    assert {i.rstrip("0") for i in cheap} == {"a1", "b1"}


def test_same_brand_filter_needs_a_known_brand():
    records = [make_record("q", "Kurti", brand=None), make_record("x", "Kurti X")]
    with pytest.raises(FilterNotApplicableError):
        engine_for(records).find_similar(records[0]["uniq_id"], 1, SimilarFilters(same_brand=True))


@pytest.mark.parametrize("kwargs", [{"min_price": -1}, {"max_price": float("nan")}, {"min_price": 500, "max_price": 100}])
def test_invalid_price_filters_rejected(kwargs):
    with pytest.raises(ValueError):
        SimilarFilters(**kwargs)
