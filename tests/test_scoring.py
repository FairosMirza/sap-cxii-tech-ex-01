import numpy as np
import pytest

from app.retrieval import ProductNotFoundError
from app.scoring import component_scores, score_all
from tests.conftest import config, engine_for, make_record


def ids(*shorts: str) -> list[str]:
    return [s.ljust(32, "0") for s in shorts]


def test_excludes_query_and_same_asin_records():
    records = [
        make_record("q", "Black Cotton Kurti", asin="X"),
        make_record("dup", "Black Cotton Kurti", asin="X", brand="Other"),
        make_record("c1", "Black Rayon Kurti", brand="Other"),
        make_record("c2", "Red Cotton Kurti", brand="Other"),
    ]
    result = engine_for(records).find_similar("q".ljust(32, "0"), 3)
    assert "q".ljust(32, "0") not in result and "dup".ljust(32, "0") not in result
    assert len(result) == len(set(result)) == 2


def test_query_family_excluded_and_other_families_capped():
    records = [
        make_record("q", "Plain Tee", brand="Acme"),
        make_record("qsize", "Plain Tee", brand="Acme"),
        *[make_record(f"f{i}", "Striped Tee", brand="Zed") for i in range(3)],
        make_record("solo", "Polo Tee", brand="Other"),
    ]
    result = engine_for(records).find_similar("q".ljust(32, "0"), 5)
    assert "qsize".ljust(32, "0") not in result
    assert sum(r.startswith("f") for r in result) == 1
    assert len(result) == 2  # documented shortfall: only two eligible products remain


def test_returns_exactly_k_when_enough_candidates():
    records = [make_record(f"p{i:02d}", f"Kurti {i}", brand=f"B{i}") for i in range(15)]
    engine = engine_for(records)
    for k in (1, 5, 14):
        assert len(engine.find_similar("p00".ljust(32, "0"), k)) == k


def test_ties_break_by_rating_then_reviews_then_id_and_are_stable():
    records = [
        make_record("q", "Kurti", brand="Acme"),
        make_record("b", "Kurti B", brand="Other", rating="4.0", reviews=5),
        make_record("a", "Kurti A", brand="Other", rating="4.0", reviews=5),
        make_record("c", "Kurti C", brand="Other", rating="4.0", reviews=50),
        make_record("d", "Kurti D", brand="Other", rating="4.8", reviews=1),
    ]
    engine = engine_for(records)
    first = engine.find_similar("q".ljust(32, "0"), 4)
    assert first == ids("d", "c", "a", "b")
    assert engine.find_similar("q".ljust(32, "0"), 4) == first


def test_missing_brand_on_both_sides_is_not_a_match():
    records = [make_record("q", "Kurti", brand=None), make_record("x", "Kurti X", brand=None), make_record("y", "Kurti Y")]
    engine = engine_for(records)
    scores = component_scores(engine.store, 0, engine.config, {"brand"})
    assert not scores.masks["brand"].any() and not scores.sims["brand"].any()


def test_evidence_guard_stops_a_single_weak_match_from_winning():
    records = [
        make_record("q", "Kurti", brand="Acme", price="500.00"),
        make_record("thin", "Thing", brand="Acme", price=None, category=None),
        make_record("rich", "Kurti Two", brand="Other", price="500.00"),
    ]
    guarded = engine_for(records, config(min_evidence=0.5)).find_similar("q".ljust(32, "0"), 2)
    unguarded = engine_for(records, config(min_evidence=0.0)).find_similar("q".ljust(32, "0"), 2)
    assert guarded[0] == "rich".ljust(32, "0")
    assert unguarded[0] == "thin".ljust(32, "0")


def test_scores_are_finite_with_sparse_metadata_and_text():
    records = [make_record(f"s{i}", "x", brand=None, price=None, category=None, reviews=None) for i in range(4)]
    records += [make_record("full1", "Black Cotton Kurti"), make_record("full2", "Black Cotton Kurta")]
    engine = engine_for(records, config(text=0.5, attributes=0.5), text=True)
    assert engine.store.modalities["tfidf"].available.tolist() == [False] * 4 + [True] * 2
    for row in range(len(records)):
        assert np.isfinite(score_all(engine.store, row, engine.config)).all()
    assert len(engine.find_similar("s0".ljust(32, "0"), 4)) == 4


@pytest.mark.parametrize("k", [0, -1, 101, 2.5, True, "5", None])
def test_invalid_k_rejected(k):
    engine = engine_for([make_record("q", "Kurti"), make_record("x", "Kurti X")])
    with pytest.raises(ValueError):
        engine.find_similar("q".ljust(32, "0"), k)


@pytest.mark.parametrize("product_id", ["missing".ljust(32, "0"), "", None, 123])
def test_unknown_id_raises_lookup_error(product_id):
    engine = engine_for([make_record("q", "Kurti"), make_record("x", "Kurti X")])
    with pytest.raises(ProductNotFoundError):
        engine.find_similar(product_id, 1)


def _title_cap_records() -> list[dict]:
    query = make_record("q", "Straight Kurta", brand=None)
    same_title = [make_record(f"t{i}", "Juniper Straight Kurta", brand=None) for i in range(4)]
    others = [make_record(f"o{i}", f"Anarkali Kurta {i}", brand=None, price="650.00") for i in range(3)]
    return [query, *same_title, *others]


def test_title_cap_is_off_by_default():
    result = engine_for(_title_cap_records()).search(0, 7)
    assert [row for row, _ in result.hits][:4] == [1, 2, 3, 4]
    assert result.title_cap_skipped == result.title_cap_shortfall == 0


def test_title_cap_keeps_relevance_order_and_fills_from_lower_ranks():
    result = engine_for(_title_cap_records(), config(max_per_title=2)).search(0, 4)
    assert [row for row, _ in result.hits] == [1, 2, 5, 6]
    assert result.title_cap_skipped == 2 and result.title_cap_shortfall == 0


@pytest.mark.parametrize("k, expected_shortfall", [(5, 0), (6, 1), (7, 2)])
def test_title_cap_records_result_shortfall(k, expected_shortfall):
    result = engine_for(_title_cap_records(), config(max_per_title=2)).search(0, k)
    assert len(result.hits) == min(k, 5)
    assert result.title_cap_shortfall == expected_shortfall
