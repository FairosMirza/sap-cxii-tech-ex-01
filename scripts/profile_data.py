"""Reproduce the raw data profile and report how each attribute is parsed.

Run from the repository root:  python -m scripts.profile_data
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

from app.data import DEFAULT_DATA_FILE, load_records
from app.features import (
    SOURCE_FALLBACK, SOURCE_NONE, SOURCE_PRIMARY, WEIGHT_PLACEHOLDER, build_catalog, normalize_brand, parse_rating,
)


def _excess(values) -> int:
    return sum(n - 1 for n in Counter(values).values() if n > 1)


def profile(path: str) -> dict:
    records, report = load_records(path)
    catalog = build_catalog(records)
    n = len(records)
    pct = lambda count: round(100 * count / n, 2)  # noqa: E731
    name_key = lambda s: " ".join(unicodedata.normalize("NFKC", s or "").casefold().split())  # noqa: E731

    details = [r.get("product_details__k_v_pairs") or {} for r in records]
    rating_pairs = []
    for r, d in zip(records, details):
        summary = d.get("Customer_Reviews", "")
        if " out of 5 stars" in summary and parse_rating(r.get("rating")) is not None:
            rating_pairs.append(abs(float(summary.split(" out of", 1)[0]) - float(r["rating"])))
    rating_pairs = np.array(rating_pairs)
    families = Counter(int(f) for f in catalog.family_codes if f >= 0)
    raw_brand_present = sum(1 for r in records if (r.get("brand") or "").strip())
    colour_field = [r.get("colour") for r in records if r.get("colour")]

    return {
        "source": {"path": report.path, "sha256": report.sha256, "rows": report.rows,
                   "malformed_lines": len(report.malformed_lines), "invalid_id_lines": len(report.invalid_id_lines),
                   "duplicate_id_lines": len(report.duplicate_id_lines)},
        "identity": {
            "unique_uniq_id": len(set(catalog.ids)),
            "excess_rows_over_unique_asin": _excess(r.get("asin") for r in records),
            "asin_groups_with_duplicates": sum(1 for v in Counter(r.get("asin") for r in records).values() if v > 1),
            "excess_rows_over_unique_normalized_name": _excess(name_key(r.get("product_name")) for r in records),
            "variant_families_brand_and_title": len(families),
            "products_in_variant_families": sum(families.values()),
        },
        "brand": {
            "missing_raw": n - raw_brand_present, "missing_raw_pct": pct(n - raw_brand_present),
            "placeholder_values_treated_missing": raw_brand_present - int((catalog.brand_codes >= 0).sum()),
            "usable": int((catalog.brand_codes >= 0).sum()), "distinct_usable": len(catalog.brands),
            "top_placeholder_examples": [b for b, _ in Counter((r.get("brand") or "").strip() for r in records
                                         if (r.get("brand") or "").strip() and normalize_brand(r.get("brand")) is None).most_common(5)],
        },
        "price": {
            "sales_price_missing": int(np.isnan(catalog.price).sum()), "sales_price_missing_pct": pct(int(np.isnan(catalog.price).sum())),
            "sales_price_min": float(np.nanmin(catalog.price)), "sales_price_median": float(np.nanmedian(catalog.price)),
            "sales_price_max": float(np.nanmax(catalog.price)),
            "discount_percentage_present": sum(1 for r in records if r.get("discount_percentage")),
            "derived_regular_price_available": int(np.isfinite(catalog.derived_regular_price).sum()),
            "note": "Derived regular price = sales_price / (1 - discount) for 0 < discount < 1; kept as unscored metadata to avoid counting price twice.",
        },
        "weight": {
            "placeholder_999999999": sum(1 for r in records if r.get("weight") == WEIGHT_PLACEHOLDER),
            "placeholder_pct": pct(sum(1 for r in records if r.get("weight") == WEIGHT_PLACEHOLDER)),
            "parsed_from_weight_field": int((catalog.weight_source == SOURCE_PRIMARY).sum()),
            "parsed_from_item_weight_fallback": int((catalog.weight_source == SOURCE_FALLBACK).sum()),
            "usable_pct": pct(int(np.isfinite(catalog.weight_g).sum())),
            "rows_with_both_fields": sum(1 for r, d in zip(records, details) if r.get("weight") != WEIGHT_PLACEHOLDER and "Item_Weight" in d),
            "rows_where_both_agree": sum(1 for r, d in zip(records, details) if "Item_Weight" in d and d["Item_Weight"] == r.get("weight")),
        },
        "rating": {
            "usable": int(np.isfinite(catalog.rating).sum()),
            "compared_with_customer_reviews": int(len(rating_pairs)),
            "exact_agreement_pct": round(100 * float((rating_pairs < 0.05).mean()), 2),
            "within_0_5_pct": round(100 * float((rating_pairs <= 0.5).mean()), 2),
            "review_count_usable": int(np.isfinite(catalog.review_count).sum()),
        },
        "color": {
            "from_product_name": int((catalog.color_source == SOURCE_PRIMARY).sum()),
            "from_meta_keywords_fallback": int((catalog.color_source == SOURCE_FALLBACK).sum()),
            "unknown": int((catalog.color_source == SOURCE_NONE).sum()),
            "title_keyword_conflicts_flagged": int(catalog.color_conflict.sum()),
            "multi_family_products": int((catalog.color_matrix.sum(axis=1) > 1).sum()),
            "colour_field_present": len(colour_field),
            "colour_field_with_multiple_values": sum(1 for c in colour_field if "|" in c),
            "note": "The colour field lists a listing's available variant colours, so it is kept but not scored.",
        },
        "category": {
            "browsenode_missing": sum(1 for r in records if not r.get("browsenode")),
            "category_map_missing": sum(1 for r in records if not r.get("parent___child_category__all")),
            "child_category_present": int((catalog.category_source == SOURCE_PRIMARY).sum()),
            "filled_from_browsenode": int((catalog.category_source == SOURCE_FALLBACK).sum()),
            "unknown": int((catalog.category_source == SOURCE_NONE).sum()),
            "distinct_categories": len(catalog.category_names),
            "products_with_multiple_categories": int((catalog.category_matrix.getnnz(axis=1) > 1).sum()),
        },
        "images": {
            "medium_missing": sum(1 for r in records if not r.get("medium")),
            "large_missing": sum(1 for r in records if not r.get("large")),
            "primary_url_usable": sum(1 for u in catalog.image_urls if u),
        },
        "text": {
            "meta_keywords_equal_to_name": sum(1 for r in records if name_key(r.get("meta_keywords")) == name_key(r.get("product_name"))),
            "median_text_tokens": float(np.median([len(t.split()) for t in catalog.texts])),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--out", default="reports/data_profile.json")
    args = parser.parse_args()
    result = profile(args.data)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
