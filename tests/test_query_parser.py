import pytest

from app.features import BOYS, GIRLS, MEN, WOMEN
from app.query_parser import parse_query, words_to_number


def price(query: str):
    return parse_query(query).price


@pytest.mark.parametrize("query, minimum, maximum, max_inclusive", [
    ("comfortable running shoes under $100", None, 100, False),
    ("shoes under a hundred bucks", None, 100, False),
    ("kurti less than ₹1,500", None, 1500, False),
    ("saree up to 799", None, 799, True),
    ("jacket within 2k", None, 2000, True),
    ("dress at least 300 rupees", 300, None, True),
    ("between 500 and 1000 kurta", 500, 1000, True),
    ("₹500-₹1,000 kurti", 500, 1000, True),
    ("tops 500 to 1000 rs", 500, 1000, True),
    ("shirt above 500 below 1000", 500, 1000, False),
])
def test_price_bounds(query, minimum, maximum, max_inclusive):
    parsed = price(query)
    assert parsed.minimum == minimum and parsed.maximum == maximum
    if maximum is not None:
        assert parsed.max_inclusive is max_inclusive


@pytest.mark.parametrize("query, centre", [("around $100", 100), ("about 1k rupees", 1000), ("roughly ₹ 650 saree", 650)])
def test_approximate_prices_become_a_tolerance_band(query, centre):
    parsed = price(query)
    assert parsed.approximate and parsed.minimum == pytest.approx(centre * 0.8) and parsed.maximum == pytest.approx(centre * 1.2)


def test_strict_and_inclusive_limits_differ_at_the_boundary():
    assert not price("under 100").allows(100) and price("under 100").allows(99)
    assert price("up to 100").allows(100)


@pytest.mark.parametrize("query", [
    "pack of 3 t-shirts", "size 10 shoes", "dress for girls under 10", "kids jacket under 5 years",
    "under tension cotton tee", "durable workout gear", "t-shirt 2020 collection",
])
def test_numbers_that_are_not_prices_are_ignored(query):
    assert price(query) is None


def test_small_amount_with_currency_is_still_a_price():
    assert price("dress under 10 dollars").maximum == 10


def test_price_phrase_is_removed_from_the_semantic_text():
    parsed = parse_query("Comfortable running shoes under $100")
    assert parsed.semantic_text == "comfortable running shoes"
    assert parsed.price.currency == "$" and parsed.price.phrases == ("under $100",)


def test_v2_sample_queries():
    jackets = parse_query("warm winter jackets size M")
    assert jackets.sizes == ("M",) and jackets.price is None and jackets.semantic_text == "warm winter jackets"
    dresses = parse_query("affordable summer dresses for women")
    assert dresses.price_preference == "low" and dresses.audience == WOMEN and dresses.price is None
    wallet = parse_query("black leather wallet for men")
    assert wallet.colors == {"black"} and wallet.materials == {"leather"} and wallet.audience == MEN
    gear = parse_query("durable workout gear")
    assert (gear.price, gear.audience, gear.colors, gear.materials, gear.sizes) == (None, 0, frozenset(), frozenset(), ())


def test_kids_audience_covers_boys_and_girls():
    assert parse_query("kids raincoat").audience & (BOYS | GIRLS) == BOYS | GIRLS


@pytest.mark.parametrize("words, value", [
    ("a hundred", 100), ("one hundred", 100), ("two hundred and fifty", 250), ("fifteen hundred", 1500),
    ("ten thousand", 10000), ("forty five", 45), ("bucks", None),
])
def test_words_to_number(words, value):
    assert words_to_number(words) == value


@pytest.mark.parametrize("query", ["", "   ", None, "x" * 201])
def test_invalid_queries_rejected(query):
    with pytest.raises(ValueError):
        parse_query(query)
