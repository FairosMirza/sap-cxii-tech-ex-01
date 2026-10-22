import math

import pytest

from app.settings import ConfigError, ScoringConfig, _validated, load_config, method_names


def make(blocks=None, attributes=None, **kwargs) -> ScoringConfig:
    return _validated(ScoringConfig(
        block_weights=blocks or {"text": 0.5, "image": 0.0, "attributes": 0.5},
        attribute_weights=attributes or {"category": 1.0},
        **kwargs,
    ))


@pytest.mark.parametrize("blocks", [
    {"text": -0.1, "image": 0, "attributes": 1},
    {"text": math.nan, "image": 0, "attributes": 1},
    {"text": 0, "image": 0, "attributes": 0},
    {"text": 1, "image": 0, "attributes": 0, "audio": 1},
    {"text": "1", "image": 0, "attributes": 0},
])
def test_invalid_block_weights_rejected(blocks):
    with pytest.raises(ConfigError):
        make(blocks=blocks)


def test_weighted_attributes_block_needs_attribute_weights():
    with pytest.raises(ConfigError):
        make(attributes={"category": 0.0})


@pytest.mark.parametrize("kwargs", [
    {"min_evidence": 1.5}, {"price_scale": 0}, {"max_per_family": 0}, {"max_results": 0},
    {"max_per_title": 0}, {"max_per_title": 2.5}, {"max_per_title": True},
])
def test_invalid_scalars_rejected(kwargs):
    with pytest.raises(ConfigError):
        make(**kwargs)


@pytest.mark.parametrize("cap", [None, 1, 2])
def test_title_cap_accepts_off_or_positive_integer(cap):
    assert make(max_per_title=cap).max_per_title == cap


def test_component_weights_split_attribute_block_by_fixed_shares():
    weights = make(attributes={"category": 3.0, "price": 1.0}).component_weights()
    assert weights["category"] == pytest.approx(0.375) and weights["price"] == pytest.approx(0.125)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_repository_config_and_methods_load():
    assert load_config().name == "default"
    for method in method_names():
        assert sum(load_config(method=method).component_weights().values()) > 0
    with pytest.raises(ConfigError):
        load_config(method="does-not-exist")
