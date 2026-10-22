import numpy as np
import pytest

from app.features import (
    COLOR_FAMILIES, SOURCE_FALLBACK, SOURCE_NONE, SOURCE_PRIMARY, build_catalog, build_text, extract_colors,
    normalize_brand, parse_discount, parse_price, parse_rating, parse_weight_grams, primary_image_url,
)
from tests.conftest import make_record


@pytest.mark.parametrize("raw, expected", [
    ("399.00", 399.0), ("1,299.00", 1299.0), (499, 499.0), ("", None), ("0", None), ("-5", None),
    ("100 - 200", None), (None, None), (True, None), ("abc", None), (float("nan"), None), (float("inf"), None),
])
def test_parse_price(raw, expected):
    assert parse_price(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("200 g", 200.0), ("1.2 Kg", 1200.0), ("250 Grams", 250.0), ("2 Kilograms", 2000.0), ("4.54 g", 4.54),
    ("999999999", None), ("The Slim N Lift slimming shirt", None), ("0.5 g", None), ("25 kg", None),
    ("200 lbs", None), (None, None),
])
def test_parse_weight_grams(raw, expected):
    assert parse_weight_grams(raw) == pytest.approx(expected) if expected else parse_weight_grams(raw) is None


@pytest.mark.parametrize("raw, expected", [("4.5", 4.5), ("1.0", 1.0), ("0", None), ("5.1", None), ("x", None), (None, None), (True, None)])
def test_parse_rating(raw, expected):
    assert parse_rating(raw) == expected


@pytest.mark.parametrize("raw, expected", [("50%", 0.5), ("0%", None), ("100%", None), ("abc", None), (None, None)])
def test_parse_discount(raw, expected):
    assert parse_discount(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("Generic", None), ("#N/A", None), ("N/A", None), ("Unknown", None), ("", None), (None, None),
    (" LEVI'S ", "levis"), ("Levi’s", "levis"), ("Allen Solly Junior", "allen solly junior"),
])
def test_normalize_brand(raw, expected):
    assert normalize_brand(raw) == expected


@pytest.mark.parametrize("text, expected", [
    ("Navy Blue Cotton Kurti", {"blue"}), ("Off-White Saree", {"white"}), ("Tank Top", set()),
    ("Multi Color Dupatta", {"multicolor"}), ("Black and Gold Earrings", {"black", "gold"}),
    ("GOLDEN sandals", {"gold"}), ("Printed Rayon Kurta", set()),
])
def test_extract_colors(text, expected):
    assert extract_colors(text) == expected


def test_build_text_adds_only_new_keyword_tokens():
    record = {"product_name": "Cotton Kurti", "meta_keywords": "cotton kurti for Women Kurti"}
    assert build_text(record) == "Cotton Kurti for Women"


@pytest.mark.parametrize("medium, expected", [
    ("https://images-na.ssl-images-amazon.com/images/I/a.jpg|https://images-na.ssl-images-amazon.com/images/I/b.jpg",
     "https://images-na.ssl-images-amazon.com/images/I/a.jpg"),
    ("http://images-na.ssl-images-amazon.com/images/I/a.jpg", None),
    ("https://evil.example.com/a.jpg", None),
    ("", None),
])
def test_primary_image_url_only_accepts_catalogue_host(medium, expected):
    assert primary_image_url({"medium": medium}) == expected


def test_category_filled_from_reliable_browsenode_only():
    records = [make_record(f"k{i}", f"Kurti {i}", node="N1") for i in range(5)]
    records.append(make_record("fill", "Unlabelled kurti", node="N1", category=None))
    records.append(make_record("none", "Unlabelled thing", node=None, category=None))
    records.append(make_record("weak", "Unlabelled other", node="N2", category=None))
    catalog = build_catalog(records)
    source = dict(zip(catalog.ids, catalog.category_source))
    assert source["k0".ljust(32, "0")] == SOURCE_PRIMARY
    assert source["fill".ljust(32, "0")] == SOURCE_FALLBACK
    assert source["none".ljust(32, "0")] == SOURCE_NONE
    assert source["weak".ljust(32, "0")] == SOURCE_NONE


def test_variant_families_need_brand_and_identical_title():
    records = [
        make_record("a", "Plain Cotton Tee", brand="Acme"),
        make_record("b", "plain  cotton tee", brand="ACME"),
        make_record("c", "Plain Cotton Tee", brand="Other"),
        make_record("d", "Plain Cotton Tee", brand=None),
        make_record("e", "Plain Cotton Tee", brand=None),
    ]
    codes = build_catalog(records).family_codes
    assert codes[0] == codes[1] >= 0
    assert codes[2] == codes[3] == codes[4] == -1


def test_item_weight_used_only_as_fallback():
    record = make_record("w", "Tee")
    record["product_details__k_v_pairs"] = {"Item_Weight": "150 g"}
    catalog = build_catalog([record])
    assert catalog.weight_g[0] == 150.0 and catalog.weight_source[0] == SOURCE_FALLBACK
    assert np.isnan(build_catalog([make_record("x", "Tee")]).weight_g[0])


def test_colour_conflict_is_flagged_and_title_colour_kept():
    conflict = {**make_record("c1", "Black Cotton Kurti"), "meta_keywords": "Red Kurti"}
    keywords_only = {**make_record("c2", "Cotton Kurti"), "meta_keywords": "Red Cotton Kurti"}
    agreeing = {**make_record("c3", "Black Kurti"), "meta_keywords": "black kurti for women"}
    catalog = build_catalog([conflict, keywords_only, agreeing])
    assert catalog.color_conflict.tolist() == [True, False, False]
    assert catalog.color_source.tolist() == [SOURCE_PRIMARY, SOURCE_FALLBACK, SOURCE_PRIMARY]
    black, red = COLOR_FAMILIES.index("black"), COLOR_FAMILIES.index("red")
    assert catalog.color_matrix[0, black] == 1 and catalog.color_matrix[0, red] == 0


def test_derived_regular_price_needs_price_and_valid_discount():
    records = [
        {**make_record("a", "Tee A", price="500.00"), "discount_percentage": "50%"},
        {**make_record("b", "Tee B", price="500.00"), "discount_percentage": "100%"},
        {**make_record("c", "Tee C", price=None), "discount_percentage": "20%"},
    ]
    derived = build_catalog(records).derived_regular_price
    assert derived[0] == 1000.0 and np.isnan(derived[1:]).all()
