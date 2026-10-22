"""Product type should normally outrank a wrong-type product that matches brand, colour or price.

A category weight of 0.40 does not guarantee this: the other attributes total 0.60 and renormalising
over missing features shifts their influence. Known failures of attribute-only scoring (approach A)
are strict xfails, so a future fix is noticed; the same cases are checked with text evidence (B/C).
"""

import numpy as np
import pytest

from app.scoring import Anchors, DenseModality
from tests.conftest import config, engine_for, make_record

QUERY = make_record("q", "Black Cotton Kurti", brand="Acme", price="499.00", category="WomensKurtasKurtis")
WALLET = dict(name="Black Leather Wallet", brand="Acme", price="499.00", category="MensWallets")
KURTI = dict(name="Red Rayon Kurti", brand="Other", price="499.00", category="WomensKurtasKurtis")
ATTRIBUTES_ONLY = config(text=0.0, attributes=1.0)
TEXT_AND_ATTRIBUTES = config(text=0.5, attributes=0.5)
# Cosine of each row's text vector to the query: same-type titles are close, wrong-type titles are not.
TEXT_COSINE = {"q": 1.0, "wallet": 0.2, "kurti": 0.8}


def winner(wallet: dict, kurti: dict, cfg=ATTRIBUTES_ONLY, with_text: bool = False) -> str:
    records = [QUERY, make_record("wallet", **wallet), make_record("kurti", **kurti)]
    engine = engine_for(records, cfg)
    if with_text:
        cos = np.array([TEXT_COSINE[short] for short in ("q", "wallet", "kurti")], dtype=np.float32)
        vectors = np.stack([cos, np.sqrt(1 - cos**2), np.zeros_like(cos)], axis=1)
        engine.store.modalities["text"] = DenseModality(vectors, np.ones(3, dtype=bool), Anchors(0.0, 0.8))
    return engine.find_similar(QUERY["uniq_id"], 1)[0].rstrip("0")


def test_type_beats_matching_brand_colour_and_price_at_equal_price():
    assert winner(WALLET, KURTI) == "kurti"


def test_correct_type_with_missing_brand_and_colour_still_wins():
    assert winner(WALLET, {**KURTI, "name": "Rayon Kurti", "brand": None}) == "kurti"


@pytest.mark.xfail(strict=True, reason="Attributes only: brand + colour + price (0.50) outweigh type (0.40) when the right type costs twice as much")
def test_attributes_only_type_at_double_price():
    assert winner(WALLET, {**KURTI, "price": "998.00"}) == "kurti"


@pytest.mark.xfail(strict=True, reason="Attributes only: an uncategorised wrong type is renormalised over brand, colour and price and scores 1.0")
def test_attributes_only_wrong_type_with_missing_category():
    assert winner({**WALLET, "category": None}, KURTI) == "kurti"


@pytest.mark.xfail(strict=True, reason="Attributes only: a mislabelled category leaves no type evidence for the right product")
def test_attributes_only_right_type_with_wrong_category():
    assert winner(WALLET, {**KURTI, "category": "MensWallets"}) == "kurti"


@pytest.mark.parametrize("wallet, kurti", [
    (WALLET, {**KURTI, "price": "998.00"}),
    ({**WALLET, "category": None}, KURTI),
    (WALLET, {**KURTI, "category": "MensWallets"}),
    (WALLET, {**KURTI, "name": "Rayon Kurti", "brand": None}),
], ids=["double-price", "wrong-type-uncategorised", "right-type-mislabelled", "right-type-sparse"])
def test_text_evidence_keeps_type_dominant(wallet, kurti):
    assert winner(wallet, kurti, TEXT_AND_ATTRIBUTES, with_text=True) == "kurti"
