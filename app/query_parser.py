"""Rule-based extraction of structured constraints from a free-text product query.

Deterministic, free and testable; no external service is needed to run the search. Prices are kept
in the units written: the catalogue is priced in rupees and no currency conversion is applied.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from app.features import AUDIENCE_TERMS, extract_colors, normalize_key

APPROX_TOLERANCE = 0.2
MAX_QUERY_CHARS = 200
_MAX_PRICE_PHRASES = 3
# Without a currency, smaller amounts are more likely ages, sizes or counts ("girls under 10").
MIN_IMPLICIT_PRICE = 50
_NOT_MONEY_AFTER = re.compile(
    r"\s*(?:years?|yrs?|months?|kgs?|g|gms?|grams?|cm|mm|inch(?:es)?|ft|feet|pcs|pieces|pack|packs|ml|l)\b")

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_SCALES = {"thousand": 1_000, "grand": 1_000, "lakh": 100_000}
_NUMBER_WORD = "|".join(sorted([*_UNITS, *_TENS, "hundred", *_SCALES], key=len, reverse=True))
_WORDS = (rf"(?:\b(?:a|an)[\s-]+)?\b(?:{_NUMBER_WORD})\b"
          rf"(?:[\s-]+(?:and[\s-]+)?(?:{_NUMBER_WORD})\b)*")
_CURRENCY = r"(?:\$|₹|\brs\b\.?|\binr\b|\busd\b|\bdollars?\b|\bbucks\b|\brupees?\b)"
_AMOUNT = rf"(?P<NAME>\d+(?:\.\d+)?\s?k\b|\d+(?:\.\d+)?(?![\d.])|{_WORDS})"
_MONEY = rf"(?P<NAME_pre>{_CURRENCY})?\s*{_AMOUNT}(?:\s*(?P<NAME_post>{_CURRENCY}))?"

_MAX_STRICT = r"\b(?:under|below|less\s+than|cheaper\s+than|lower\s+than)\b|<(?!=)"
_MAX_INCLUSIVE = r"\b(?:up\s*to|within|at\s+most|max(?:imum)?|no\s+more\s+than|not\s+more\s+than)\b|<=|≤"
_MIN_STRICT = r"\b(?:over|above|more\s+than|greater\s+than)\b|>(?!=)"
_MIN_INCLUSIVE = r"\b(?:at\s+least|min(?:imum)?|starting\s+(?:at|from))\b|>=|≥"
_APPROX = r"\b(?:around|about|approx(?:imately)?|roughly|nearly|close\s+to)\b|~"


def _money(name: str) -> str:
    return _MONEY.replace("NAME", name)


# (kind, pattern); a bare amount or an unprefixed range counts only when it carries a currency.
_PRICE_PATTERNS = [
    ("range", re.compile(rf"\b(?:between|from)\s+{_money('lo')}\s+(?:and|to|-|–)\s+{_money('hi')}")),
    ("range_bare", re.compile(rf"(?<![\w.]){_money('lo')}\s*(?:-|–|to)\s*{_money('hi')}(?=\W|$)")),
    ("max_strict", re.compile(rf"(?:{_MAX_STRICT})\s*{_money('v')}")),
    ("max_inclusive", re.compile(rf"(?:{_MAX_INCLUSIVE})\s*{_money('v')}")),
    ("min_strict", re.compile(rf"(?:{_MIN_STRICT})\s*{_money('v')}")),
    ("min_inclusive", re.compile(rf"(?:{_MIN_INCLUSIVE})\s*{_money('v')}")),
    ("approx", re.compile(rf"(?:{_APPROX})\s*{_money('v')}")),
    ("approx_bare", re.compile(rf"(?<![\w.]){_money('v')}(?=\W|$)")),
]
_PRICE_LOW = re.compile(r"\b(?:affordable|cheap|cheapest|budget|inexpensive|economical|low[\s-]cost|low[\s-]priced)\b")
_PRICE_HIGH = re.compile(r"\b(?:premium|luxury|luxurious|expensive|high[\s-]end)\b")
_SIZE = re.compile(r"\bsize\s*[:\-]?\s*(xxxl|xxl|xl|xxs|xs|s|m|l|small|medium|large|extra\s+large|free|\d{1,2}(?:\.\d)?)\b")
_SIZE_NAMES = {"small": "S", "medium": "M", "large": "L", "extra large": "XL", "free": "FREE"}
MATERIALS = frozenset({
    "leather", "cotton", "silk", "denim", "wool", "woollen", "woolen", "linen", "polyester", "rayon", "chiffon",
    "georgette", "velvet", "satin", "nylon", "khadi", "jute", "canvas", "suede", "lycra", "spandex", "fleece",
    "crepe", "chanderi", "modal", "viscose",
})


@dataclass(frozen=True)
class PriceConstraint:
    minimum: float | None = None
    maximum: float | None = None
    min_inclusive: bool = True
    max_inclusive: bool = True
    approximate: bool = False
    currency: str | None = None
    phrases: tuple[str, ...] = ()

    def allows(self, price: float) -> bool:
        if self.minimum is not None and (price < self.minimum or (not self.min_inclusive and price == self.minimum)):
            return False
        if self.maximum is not None and (price > self.maximum or (not self.max_inclusive and price == self.maximum)):
            return False
        return True


@dataclass(frozen=True)
class ParsedQuery:
    raw: str
    semantic_text: str
    price: PriceConstraint | None = None
    price_preference: str | None = None
    audience: int = 0
    colors: frozenset[str] = field(default_factory=frozenset)
    materials: frozenset[str] = field(default_factory=frozenset)
    sizes: tuple[str, ...] = ()


def words_to_number(text: str) -> float | None:
    total = current = 0
    seen = False
    for word in re.split(r"[\s-]+", text.strip()):
        if word in ("a", "an"):
            current = current or 1
        elif word == "and":
            continue
        elif word in _UNITS:
            current += _UNITS[word]
        elif word in _TENS:
            current += _TENS[word]
        elif word == "hundred":
            current = (current or 1) * 100
        elif word in _SCALES:
            total += (current or 1) * _SCALES[word]
            current = 0
        else:
            return None
        seen = True
    return float(total + current) if seen else None


def _amount(text: str) -> float | None:
    text = text.strip()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s?k", text)
    if match:
        return float(match.group(1)) * 1000
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    return words_to_number(text)


def _currency(match: re.Match, *names: str) -> str | None:
    for name in names:
        for suffix in ("_pre", "_post"):
            value = match.group(f"{name}{suffix}")
            if value:
                return value.rstrip(".")
    return None


def _first_price_phrase(text: str, tolerance: float) -> tuple[dict, re.Match] | None:
    for kind, pattern in _PRICE_PATTERNS:
        for match in pattern.finditer(text):
            if _NOT_MONEY_AFTER.match(text, match.end()):
                continue
            if kind.startswith("range"):
                lo, hi = _amount(match.group("lo")), _amount(match.group("hi"))
                currency = _currency(match, "lo", "hi")
                if lo is None or hi is None or (currency is None and (kind == "range_bare" or max(lo, hi) < MIN_IMPLICIT_PRICE)):
                    continue
                bounds = {"minimum": min(lo, hi), "maximum": max(lo, hi)}
            else:
                value, currency = _amount(match.group("v")), _currency(match, "v")
                if value is None or value <= 0 or (currency is None and (kind == "approx_bare" or value < MIN_IMPLICIT_PRICE)):
                    continue
                if kind.startswith("approx"):
                    bounds = {"minimum": value * (1 - tolerance), "maximum": value * (1 + tolerance), "approximate": True}
                elif kind.startswith("max"):
                    bounds = {"maximum": value, "max_inclusive": kind == "max_inclusive"}
                else:
                    bounds = {"minimum": value, "min_inclusive": kind == "min_inclusive"}
            return {**bounds, "currency": currency}, match
    return None


def _parse_price(text: str, tolerance: float) -> tuple[PriceConstraint | None, str]:
    """Combine up to three price phrases (e.g. 'above 500 below 1000') into one constraint."""
    merged: dict = {}
    phrases: list[str] = []
    for _ in range(_MAX_PRICE_PHRASES):
        found = _first_price_phrase(text, tolerance)
        if found is None:
            break
        bounds, match = found
        phrases.append(match.group(0).strip())
        text = text[: match.start()] + " " + text[match.end():]
        if bounds.get("minimum") is not None and (merged.get("minimum") is None or bounds["minimum"] > merged["minimum"]):
            merged["minimum"], merged["min_inclusive"] = bounds["minimum"], bounds.get("min_inclusive", True)
        if bounds.get("maximum") is not None and (merged.get("maximum") is None or bounds["maximum"] < merged["maximum"]):
            merged["maximum"], merged["max_inclusive"] = bounds["maximum"], bounds.get("max_inclusive", True)
        merged["approximate"] = merged.get("approximate", False) or bounds.get("approximate", False)
        merged["currency"] = merged.get("currency") or bounds["currency"]
    if not phrases:
        return None, text
    return PriceConstraint(**merged, phrases=tuple(phrases)), text


def parse_query(raw: str, tolerance: float = APPROX_TOLERANCE) -> ParsedQuery:
    """Raises ValueError for empty or over-long queries."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("query must be a non-empty string")
    if len(raw) > MAX_QUERY_CHARS:
        raise ValueError(f"query must be at most {MAX_QUERY_CHARS} characters")
    text = " ".join(unicodedata.normalize("NFKC", raw).casefold().split())
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)

    price, text = _parse_price(text, tolerance)
    preference = "low" if _PRICE_LOW.search(text) else "high" if _PRICE_HIGH.search(text) else None
    text = _PRICE_HIGH.sub(" ", _PRICE_LOW.sub(" ", text))
    sizes = tuple(_SIZE_NAMES.get(re.sub(r"\s+", " ", s), s.upper()) for s in _SIZE.findall(text))
    text = _SIZE.sub(" ", text)

    tokens = normalize_key(text).split()
    audience = 0
    for token in tokens:
        audience |= AUDIENCE_TERMS.get(token, 0)
    materials = frozenset("wool" if t in ("woollen", "woolen") else t for t in tokens if t in MATERIALS)
    semantic = " ".join(text.split()).strip(" -,") or " ".join(raw.split())
    return ParsedQuery(raw=raw, semantic_text=semantic, price=price, price_preference=preference, audience=audience,
                       colors=frozenset(extract_colors(text)), materials=materials, sizes=sizes)
