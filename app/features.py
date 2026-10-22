"""Field parsing, normalisation and the columnar product catalogue.

Every parsed attribute is either a valid value or "unavailable" (NaN / -1 / empty);
placeholders and malformed strings never become plausible-looking measurements.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from urllib.parse import urlparse

import numpy as np
from scipy import sparse

WEIGHT_PLACEHOLDER = "999999999"
MIN_WEIGHT_G, MAX_WEIGHT_G = 1.0, 20_000.0
IMAGE_HOSTS = frozenset({"images-na.ssl-images-amazon.com"})
PLACEHOLDER_BRANDS = frozenset({"generic", "unbranded", "no brand", "na", "n a", "none", "unknown", "others"})
# A browse node is only used to fill a missing category when it maps to one category this reliably.
MIN_NODE_SUPPORT, MIN_NODE_PURITY = 5, 0.8

SOURCE_NONE, SOURCE_PRIMARY, SOURCE_FALLBACK = 0, 1, 2

COLOR_FAMILIES = (
    "black", "white", "grey", "silver", "blue", "red", "pink", "purple",
    "green", "yellow", "orange", "brown", "beige", "gold", "multicolor",
)
_COLOR_TERMS = {
    "black": "black", "jet black": "black",
    "white": "white", "off white": "white",
    "grey": "grey", "gray": "grey", "charcoal": "grey",
    "silver": "silver",
    "blue": "blue", "navy": "blue", "navy blue": "blue", "royal blue": "blue", "sky blue": "blue",
    "turquoise": "blue", "teal": "blue", "aqua": "blue", "indigo": "blue",
    "red": "red", "maroon": "red", "wine": "red", "burgundy": "red", "crimson": "red",
    "pink": "pink", "rani pink": "pink", "magenta": "pink", "fuchsia": "pink",
    "purple": "purple", "violet": "purple", "lavender": "purple", "mauve": "purple", "lilac": "purple",
    "green": "green", "olive": "green", "mint": "green", "lime": "green", "mehndi": "green",
    "bottle green": "green", "sea green": "green",
    "yellow": "yellow", "mustard": "yellow", "lemon": "yellow",
    "orange": "orange", "coral": "orange", "peach": "orange", "rust": "orange",
    "brown": "brown", "tan": "brown", "coffee": "brown", "chocolate": "brown", "camel": "brown",
    "beige": "beige", "cream": "beige", "ivory": "beige", "khaki": "beige", "fawn": "beige", "nude": "beige",
    "gold": "gold", "golden": "gold",
    "multicolor": "multicolor", "multicolour": "multicolor", "multicolored": "multicolor",
    "multicoloured": "multicolor", "multi color": "multicolor", "multi colour": "multicolor",
}
_COLOR_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, _COLOR_TERMS), key=len, reverse=True)) + r")\b"
)
_COLOR_INDEX = {name: i for i, name in enumerate(COLOR_FAMILIES)}

MEN, WOMEN, BOYS, GIRLS, BABY = 1, 2, 4, 8, 16
KIDS = BOYS | GIRLS | BABY
AUDIENCE_TERMS = {
    "men": MEN, "mens": MEN, "man": MEN, "male": MEN, "gents": MEN, "gentlemen": MEN,
    "women": WOMEN, "womens": WOMEN, "woman": WOMEN, "female": WOMEN, "ladies": WOMEN, "lady": WOMEN,
    "boy": BOYS, "boys": BOYS, "girl": GIRLS, "girls": GIRLS,
    "kid": KIDS, "kids": KIDS, "child": KIDS, "children": KIDS, "toddler": KIDS,
    "baby": BABY, "babies": BABY, "infant": BABY, "infants": BABY, "newborn": BABY,
    "unisex": MEN | WOMEN,
}
_CATEGORY_AUDIENCE = (("Womens", WOMEN), ("Mens", MEN), ("Boys", BOYS), ("Girls", GIRLS), ("Baby", BABY))

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_WEIGHT = re.compile(r"(\d+(?:\.\d+)?)\s*(g|gm|gms|gram|grams|kg|kgs|kilogram|kilograms)", re.IGNORECASE)
_GRAMS_PER_UNIT = {"g": 1.0, "gm": 1.0, "gms": 1.0, "gram": 1.0, "grams": 1.0,
                   "kg": 1000.0, "kgs": 1000.0, "kilogram": 1000.0, "kilograms": 1000.0}
_DISCOUNT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_REVIEWS = re.compile(r"([\d,]+)\s+customer reviews?", re.IGNORECASE)
_RANK = re.compile(r"#?([\d,]+)")


def clean_text(value: object) -> str:
    """Unicode-normalise and collapse whitespace, keeping case for display and text features."""
    if not isinstance(value, str):
        return ""
    return " ".join(unicodedata.normalize("NFKC", value).split())


def normalize_key(value: object) -> str:
    """Case-, punctuation- and Unicode-insensitive key for equality comparisons."""
    text = clean_text(value).casefold()
    text = re.sub(r"['’`]", "", text)
    return " ".join(re.sub(r"[\W_]+", " ", text).split())


def normalize_brand(value: object) -> str | None:
    key = normalize_key(value)
    return None if not key or key in PLACEHOLDER_BRANDS else key


def parse_price(value: object) -> float | None:
    """Positive finite price; blanks, negatives, ranges and other formats are unavailable."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        price = float(value)
    else:
        text = str(value).replace(",", "").strip()
        numbers = _NUMBER.findall(text)
        if text.startswith("-") or len(numbers) != 1:
            return None
        price = float(numbers[0])
    return price if math.isfinite(price) and price > 0 else None


def parse_discount(value: object) -> float | None:
    match = _DISCOUNT.fullmatch(value.strip()) if isinstance(value, str) else None
    if not match:
        return None
    fraction = float(match.group(1)) / 100
    return fraction if 0 < fraction < 1 else None


def parse_weight_grams(value: object) -> float | None:
    """Weight in grams from strings such as '200 g' or '1.2 Kg'; the 999999999 placeholder is unavailable."""
    if not isinstance(value, str) or value.strip() == WEIGHT_PLACEHOLDER:
        return None
    match = _WEIGHT.fullmatch(value.replace(",", "").strip())
    if not match:
        return None
    grams = float(match.group(1)) * _GRAMS_PER_UNIT[match.group(2).lower()]
    return grams if MIN_WEIGHT_G <= grams <= MAX_WEIGHT_G else None


def parse_rating(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        rating = float(value)
    except (TypeError, ValueError):
        return None
    return rating if 1.0 <= rating <= 5.0 else None


def parse_review_count(record: dict) -> int | None:
    raw = record.get("no__of_reviews")
    if isinstance(raw, str) and raw.replace(",", "").strip().isdigit():
        return int(raw.replace(",", ""))
    summary = (record.get("product_details__k_v_pairs") or {}).get("Customer_Reviews")
    match = _REVIEWS.search(summary) if isinstance(summary, str) else None
    return int(match.group(1).replace(",", "")) if match else None


def extract_colors(text: str) -> set[str]:
    """Colour families named in the text; an empty set means unknown, never 'no colour'."""
    return {_COLOR_TERMS[term] for term in _COLOR_PATTERN.findall(normalize_key(text))}


def audience_from_text(text: object) -> int:
    """Bitmask of audiences named in the text (0 = not stated)."""
    mask = 0
    for token in normalize_key(text).split():
        mask |= AUDIENCE_TERMS.get(token, 0)
    return mask


def audience_from_category(name: str) -> int:
    return next((bits for prefix, bits in _CATEGORY_AUDIENCE if name.startswith(prefix)), 0)


def child_categories(record: dict) -> list[str]:
    """Leaf categories from the sales-rank map, best-ranked first (rank numbers are not used as features)."""
    ranks = record.get("sales_rank_in_child_category") or {}
    if not isinstance(ranks, dict):
        return []

    def rank_of(item: tuple[str, object]) -> int:
        match = _RANK.match(str(item[1]).strip())
        return int(match.group(1).replace(",", "")) if match else 10**12

    return [name for name, _ in sorted(ranks.items(), key=rank_of) if name]


def primary_image_url(record: dict) -> str | None:
    """First catalogue image URL, accepted only over HTTPS from the known Amazon image host."""
    raw = record.get("medium")
    if not isinstance(raw, str) or not raw.strip():
        return None
    url = raw.split("|")[0].strip()
    parsed = urlparse(url)
    return url if parsed.scheme == "https" and parsed.hostname in IMAGE_HOSTS else None


def build_text(record: dict) -> str:
    """Product name plus only those meta-keyword tokens the name does not already contain."""
    name = clean_text(record.get("product_name"))
    seen = set(normalize_key(name).split())
    extra = []
    for token in clean_text(record.get("meta_keywords")).split():
        key = normalize_key(token)
        if key and key not in seen:
            seen.add(key)
            extra.append(token)
    return " ".join([name, *extra]).strip()


@dataclass
class Catalog:
    """Column-oriented view of the catalogue; row i of every array is product ids[i]."""

    ids: list[str]
    index: dict[str, int]
    asins: np.ndarray
    names: list[str]
    texts: list[str]
    brands: list[str]
    brand_codes: np.ndarray
    category_names: list[str]
    category_matrix: sparse.csr_matrix
    category_source: np.ndarray
    color_matrix: np.ndarray
    color_source: np.ndarray
    color_conflict: np.ndarray
    variant_color_matrix: np.ndarray
    audience: np.ndarray
    price: np.ndarray
    derived_regular_price: np.ndarray
    weight_g: np.ndarray
    weight_source: np.ndarray
    rating: np.ndarray
    review_count: np.ndarray
    family_codes: np.ndarray
    title_codes: np.ndarray
    image_urls: list[str | None]
    variant_colours: list[str | None]

    def __len__(self) -> int:
        return len(self.ids)


def _browsenode_to_category(records: list[dict]) -> dict[str, str]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        node = record.get("browsenode")
        cats = child_categories(record)
        if node and cats:
            counts[node][cats[0]] += 1
    mapping = {}
    for node, counter in counts.items():
        category, hits = counter.most_common(1)[0]
        total = sum(counter.values())
        if total >= MIN_NODE_SUPPORT and hits / total >= MIN_NODE_PURITY:
            mapping[node] = category
    return mapping


def build_catalog(records: list[dict]) -> Catalog:
    n = len(records)
    ids = [r["uniq_id"] for r in records]
    node_map = _browsenode_to_category(records)

    brand_vocab: dict[str, int] = {}
    brand_codes = np.full(n, -1, dtype=np.int32)
    category_vocab: dict[str, int] = {}
    cat_rows: list[int] = []
    cat_cols: list[int] = []
    category_source = np.zeros(n, dtype=np.int8)
    color_matrix = np.zeros((n, len(COLOR_FAMILIES)), dtype=np.float32)
    color_source = np.zeros(n, dtype=np.int8)
    color_conflict = np.zeros(n, dtype=bool)
    # "Available in" colours from the variant list; never treated as the item's own colour.
    variant_color_matrix = np.zeros((n, len(COLOR_FAMILIES)), dtype=np.float32)
    audience = np.zeros(n, dtype=np.uint8)
    price = np.full(n, np.nan)
    # Unscored metadata: sales_price / (1 - discount), only for 0 < discount < 1.
    derived_regular_price = np.full(n, np.nan)
    weight_g = np.full(n, np.nan)
    weight_source = np.zeros(n, dtype=np.int8)
    rating = np.full(n, np.nan)
    review_count = np.full(n, np.nan)
    family_keys: list[tuple[str, str] | None] = []
    title_vocab: dict[str, int] = {}
    title_codes = np.zeros(n, dtype=np.int32)

    for i, record in enumerate(records):
        brand = normalize_brand(record.get("brand"))
        if brand is not None:
            brand_codes[i] = brand_vocab.setdefault(brand, len(brand_vocab))

        categories = child_categories(record)
        if categories:
            category_source[i] = SOURCE_PRIMARY
        elif record.get("browsenode") in node_map:
            categories = [node_map[record["browsenode"]]]
            category_source[i] = SOURCE_FALLBACK
        for name in categories:
            cat_rows.append(i)
            cat_cols.append(category_vocab.setdefault(name, len(category_vocab)))
            audience[i] |= audience_from_category(name)
        audience[i] |= audience_from_text(record.get("product_name"))

        # The title is the more specific source; keyword colours are used only when the title has none.
        title_colors = extract_colors(record.get("product_name") or "")
        keyword_colors = extract_colors(record.get("meta_keywords") or "")
        colors = title_colors or keyword_colors
        color_source[i] = SOURCE_PRIMARY if title_colors else SOURCE_FALLBACK if keyword_colors else SOURCE_NONE
        color_conflict[i] = bool(title_colors and keyword_colors and title_colors != keyword_colors)
        for color in colors:
            color_matrix[i, _COLOR_INDEX[color]] = 1.0
        for color in extract_colors(clean_text(record.get("colour")).replace("|", " ")):
            variant_color_matrix[i, _COLOR_INDEX[color]] = 1.0

        p = parse_price(record.get("sales_price"))
        if p is not None:
            price[i] = p
            d = parse_discount(record.get("discount_percentage"))
            if d is not None:
                derived_regular_price[i] = p / (1 - d)

        w = parse_weight_grams(record.get("weight"))
        if w is not None:
            weight_g[i], weight_source[i] = w, SOURCE_PRIMARY
        else:
            w = parse_weight_grams((record.get("product_details__k_v_pairs") or {}).get("Item_Weight"))
            if w is not None:
                weight_g[i], weight_source[i] = w, SOURCE_FALLBACK

        r = parse_rating(record.get("rating"))
        if r is not None:
            rating[i] = r
        c = parse_review_count(record)
        if c is not None:
            review_count[i] = c

        name_key = normalize_key(record.get("product_name"))
        family_keys.append((brand, name_key) if brand and name_key else None)
        title_codes[i] = title_vocab.setdefault(name_key, len(title_vocab)) if name_key else -1

    family_sizes = Counter(k for k in family_keys if k is not None)
    family_vocab: dict[tuple[str, str], int] = {}
    family_codes = np.full(n, -1, dtype=np.int32)
    for i, key in enumerate(family_keys):
        if key is not None and family_sizes[key] > 1:
            family_codes[i] = family_vocab.setdefault(key, len(family_vocab))

    category_matrix = sparse.csr_matrix(
        (np.ones(len(cat_rows), dtype=np.float32), (cat_rows, cat_cols)), shape=(n, len(category_vocab))
    )
    return Catalog(
        ids=ids,
        index={uid: i for i, uid in enumerate(ids)},
        asins=np.array([str(r.get("asin") or f"missing-{r['uniq_id']}") for r in records]),
        names=[clean_text(r.get("product_name")) for r in records],
        texts=[build_text(r) for r in records],
        brands=list(brand_vocab),
        brand_codes=brand_codes,
        category_names=list(category_vocab),
        category_matrix=category_matrix,
        category_source=category_source,
        color_matrix=color_matrix,
        color_source=color_source,
        color_conflict=color_conflict,
        variant_color_matrix=variant_color_matrix,
        audience=audience,
        price=price,
        derived_regular_price=derived_regular_price,
        weight_g=weight_g,
        weight_source=weight_source,
        rating=rating,
        review_count=review_count,
        family_codes=family_codes,
        title_codes=title_codes,
        image_urls=[primary_image_url(r) for r in records],
        variant_colours=[clean_text(r.get("colour")) or None for r in records],
    )
