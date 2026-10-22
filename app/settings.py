"""Scorer configuration loaded from YAML and validated before use."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

BLOCKS = ("text", "image", "attributes")
ATTRIBUTES = ("category", "brand", "color", "price", "weight", "rating")
COMPONENTS = ("text", "image", *ATTRIBUTES)
DEFAULT_CONFIG = Path("configs/default.yaml")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ScoringConfig:
    """Block weights (text / image / attributes) are tuned; attribute sub-weights are fixed by design."""

    block_weights: dict[str, float]
    attribute_weights: dict[str, float]
    min_evidence: float = 0.5
    price_scale: float = math.log(2)
    weight_scale: float = math.log(2)
    exclude_same_asin: bool = True
    exclude_query_family: bool = True
    max_per_family: int = 1
    max_per_title: int | None = None
    max_results: int = 100
    name: str = "default"

    def component_weights(self) -> dict[str, float]:
        attribute_total = sum(self.attribute_weights.values())
        weights = {"text": self.block_weights["text"], "image": self.block_weights["image"]}
        for attribute in ATTRIBUTES:
            share = self.attribute_weights[attribute] / attribute_total if attribute_total else 0.0
            weights[attribute] = self.block_weights["attributes"] * share
        return weights

    def with_blocks(self, **block_weights: float) -> "ScoringConfig":
        merged = {**self.block_weights, **block_weights}
        return _validated(ScoringConfig(**{**self.__dict__, "block_weights": merged}))


def _check_weights(weights: dict[str, Any], expected: tuple[str, ...], label: str) -> dict[str, float]:
    unknown = set(weights) - set(expected)
    if unknown:
        raise ConfigError(f"unknown {label} keys: {sorted(unknown)}")
    clean = {}
    for key in expected:
        value = weights.get(key, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ConfigError(f"{label}.{key} must be a finite non-negative number, got {value!r}")
        clean[key] = float(value)
    return clean


def _validated(config: ScoringConfig) -> ScoringConfig:
    blocks = _check_weights(config.block_weights, BLOCKS, "block_weights")
    attributes = _check_weights(config.attribute_weights, ATTRIBUTES, "attribute_weights")
    if sum(blocks.values()) == 0:
        raise ConfigError("at least one block weight must be positive")
    if blocks["attributes"] > 0 and sum(attributes.values()) == 0:
        raise ConfigError("attributes block is weighted but all attribute weights are zero")
    if not 0.0 <= config.min_evidence <= 1.0:
        raise ConfigError("min_evidence must be within [0, 1]")
    if config.price_scale <= 0 or config.weight_scale <= 0:
        raise ConfigError("price_scale and weight_scale must be positive")
    if config.max_per_family < 1 or config.max_results < 1:
        raise ConfigError("max_per_family and max_results must be at least 1")
    cap = config.max_per_title
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
        raise ConfigError("max_per_title must be null (off) or an integer of at least 1")
    return ScoringConfig(**{**config.__dict__, "block_weights": blocks, "attribute_weights": attributes})


def load_config(path: str | Path = DEFAULT_CONFIG, method: str | None = None) -> ScoringConfig:
    """Read the YAML file; a named method's settings override the defaults."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    scoring = dict(raw.get("scoring") or {})
    scoring.update(raw.get("eligibility") or {})
    if method is not None:
        methods = raw.get("methods") or {}
        if method not in methods:
            raise ConfigError(f"unknown method {method!r}; available: {sorted(methods)}")
        scoring.update(methods[method] or {})
    try:
        config = ScoringConfig(name=method or "default", **scoring)
    except TypeError as exc:
        raise ConfigError(str(exc)) from exc
    return _validated(config)


def method_names(path: str | Path = DEFAULT_CONFIG) -> list[str]:
    return list((yaml.safe_load(Path(path).read_text()) or {}).get("methods") or {})
