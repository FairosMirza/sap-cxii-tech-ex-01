"""Select the ~30 evaluation queries by stratum, blind to any model output, and split dev / holdout.

Run from the repository root after the image download:  python -m scripts.select_queries
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

from app.data import DEFAULT_DATA_FILE, load_records
from app.features import SOURCE_FALLBACK, SOURCE_NONE, SOURCE_PRIMARY, build_catalog, normalize_key

SEED = 20261021
HOLDOUT_SHARE = 1 / 3
MAX_PER_CATEGORY = 2
STRATA = [  # (name, quota) in selection order
    ("common_category", 9),
    ("rare_category", 5),
    ("missing_category", 3),
    ("missing_brand", 3),
    ("missing_price", 2),
    ("image_failure", 2),
    ("variant_family", 3),
    ("ambiguous_title", 2),
    ("duplicate_asin", 1),
]


def _holdout_counts() -> dict[str, int]:
    """Largest-remainder apportionment of the holdout share across strata."""
    total = round(sum(q for _, q in STRATA) * HOLDOUT_SHARE)
    exact = {name: quota * HOLDOUT_SHARE for name, quota in STRATA}
    counts = {name: math.floor(v) for name, v in exact.items()}
    by_remainder = sorted(STRATA, key=lambda s: -(exact[s[0]] - counts[s[0]]))
    for name, _ in by_remainder[: total - sum(counts.values())]:
        counts[name] += 1
    return counts


def select(data_path: str, image_status_path: Path) -> dict:
    records, report = load_records(data_path)
    c = build_catalog(records)
    n = len(c)
    rng = np.random.default_rng(SEED)

    primary_cat = np.full(n, -1)
    rows_idx, cols_idx = c.category_matrix.nonzero()
    for row, col in zip(rows_idx, cols_idx):
        if primary_cat[row] == -1:
            primary_cat[row] = col
    cat_counts = Counter(int(x) for x in primary_cat if x >= 0)
    top_categories = [cat for cat, _ in cat_counts.most_common(8)]
    family_sizes = Counter(int(f) for f in c.family_codes if f >= 0)
    asin_counts = Counter(c.asins.tolist())
    image_failed: set[str] = set()
    if image_status_path.exists():
        for line in image_status_path.read_text().splitlines():
            row = json.loads(line)
            if not row["status"].startswith("ok"):
                image_failed.add(row["id"])

    predicates = {
        "common_category": lambda i: primary_cat[i] in top_categories,
        "rare_category": lambda i: primary_cat[i] >= 0 and 5 <= cat_counts[int(primary_cat[i])] <= 30,
        "missing_category": lambda i: c.category_source[i] != SOURCE_PRIMARY,
        "missing_brand": lambda i: c.brand_codes[i] < 0,
        "missing_price": lambda i: np.isnan(c.price[i]),
        "variant_family": lambda i: c.family_codes[i] >= 0 and family_sizes[int(c.family_codes[i])] >= 3,
        "ambiguous_title": lambda i: len(normalize_key(c.names[i]).split()) <= 4,
        "duplicate_asin": lambda i: asin_counts[c.asins[i]] > 1,
        "image_failure": lambda i: c.ids[i] in image_failed,
    }
    order = rng.permutation(n)
    picked: list[tuple[str, int]] = []
    used_rows, used_families, used_asins = set(), set(), set()
    category_use: Counter = Counter()
    for stratum, quota in STRATA:
        stratum_picks: list[int] = []
        stratum_categories: set[int] = set()
        for i in order:
            i = int(i)
            if len(stratum_picks) == quota:
                break
            cat = int(primary_cat[i])
            if (i in used_rows or c.asins[i] in used_asins or (c.family_codes[i] >= 0 and int(c.family_codes[i]) in used_families)
                    or not predicates[stratum](i) or (cat >= 0 and category_use[cat] >= MAX_PER_CATEGORY)):
                continue
            # Common and rare strata should each cover distinct categories.
            if stratum in ("common_category", "rare_category") and cat in stratum_categories and len(stratum_categories) < len(top_categories):
                continue
            if stratum == "missing_category":
                wanted = SOURCE_NONE if len(stratum_picks) < 2 else SOURCE_FALLBACK
                if c.category_source[i] != wanted:
                    continue
            stratum_picks.append(i)
            stratum_categories.add(cat)
            used_rows.add(i)
            used_asins.add(c.asins[i])
            if c.family_codes[i] >= 0:
                used_families.add(int(c.family_codes[i]))
            if cat >= 0:
                category_use[cat] += 1
        picked.extend((stratum, i) for i in stratum_picks)

    holdout = _holdout_counts()
    queries = []
    for stratum, _ in STRATA:
        rows = [i for s, i in picked if s == stratum]
        shuffled = [rows[j] for j in rng.permutation(len(rows))]
        for position, i in enumerate(shuffled):
            cat = int(primary_cat[i])
            queries.append({
                "id": c.ids[i],
                "split": "holdout" if position < holdout[stratum] else "dev",
                "stratum": stratum,
                "name": c.names[i],
                "category": c.category_names[cat] if cat >= 0 else None,
                "category_source": {SOURCE_PRIMARY: "rank", SOURCE_FALLBACK: "browsenode", SOURCE_NONE: "none"}[int(c.category_source[i])],
                "brand_known": bool(c.brand_codes[i] >= 0),
                "price": None if np.isnan(c.price[i]) else float(c.price[i]),
            })
    return {
        "created": date.today().isoformat(),
        "seed": SEED,
        "dataset_sha256": report.sha256,
        "image_status_used": image_status_path.exists(),
        "strata": dict(STRATA),
        "split_counts": dict(Counter(q["split"] for q in queries)),
        "note": "Selected by stratum with a fixed seed before any tuning; no model output was used.",
        "queries": queries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--image-status", default="data/images/_status.jsonl")
    parser.add_argument("--out", default="eval/queries.json")
    args = parser.parse_args()
    result = select(args.data, Path(args.image_status))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(result['queries'])} queries {result['split_counts']}")
    for q in result["queries"]:
        print(f"  {q['split']:7} {q['stratum']:16} {(q['category'] or '-')[:24]:24} {q['name'][:70]}")


if __name__ == "__main__":
    main()
