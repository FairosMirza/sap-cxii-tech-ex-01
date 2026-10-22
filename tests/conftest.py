import pytest

from app.pipeline import build_store
from app.retrieval import SimilarityEngine
from app.settings import ScoringConfig

ATTRIBUTE_WEIGHTS = {"category": 0.4, "price": 0.2, "brand": 0.15, "color": 0.15, "weight": 0.1, "rating": 0.0}


def make_record(uid: str, name: str, *, asin: str | None = None, brand: str | None = "Acme",
                price: str | None = "499.00", weight: str = "999999999", rating: str = "4.0",
                category: str | None = "WomensKurtasKurtis", reviews: int | None = 10, node: str | None = None) -> dict:
    record: dict[str, object] = {
        "uniq_id": uid.ljust(32, "0"),
        "asin": asin or f"ASIN{uid}",
        "product_name": name,
        "meta_keywords": name,
        "weight": weight,
        "rating": rating,
        "medium": f"https://images-na.ssl-images-amazon.com/images/I/{uid}.jpg",
    }
    if brand is not None:
        record["brand"] = brand
    if price is not None:
        record["sales_price"] = price
    if category is not None:
        record["sales_rank_in_child_category"] = {category: "#100"}
    if reviews is not None:
        record["product_details__k_v_pairs"] = {"Customer_Reviews": f"{rating} out of 5 stars {reviews} customer reviews"}
    if node is not None:
        record["browsenode"] = node
    return record


def config(text: float = 0.0, attributes: float = 1.0, **overrides) -> ScoringConfig:
    from app.settings import _validated

    return _validated(ScoringConfig(block_weights={"text": text, "image": 0.0, "attributes": attributes},
                                    attribute_weights=dict(ATTRIBUTE_WEIGHTS), **overrides))


def engine_for(records: list[dict], cfg: ScoringConfig | None = None, text: bool = False) -> SimilarityEngine:
    return SimilarityEngine(build_store(records, text=text), cfg or config())


@pytest.fixture
def uid():
    return lambda short: short.ljust(32, "0")
