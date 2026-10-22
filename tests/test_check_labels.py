import json

import pytest

from scripts.build_label_pools import supplement_queries
from scripts.check_labels import check

Q, A, B, C, D = (s.ljust(32, "0") for s in ("a1", "b2", "c3", "d4", "e5"))
HOLDOUT = "f6".ljust(32, "0")
NO_HOLDOUT = {"similar": {HOLDOUT}, "semantic": {"secret holdout query"}}


def label(candidate=A, grade=2, query=Q, task="similar", split="dev", at="2026-10-22T10:00:00+00:00"):
    return json.dumps({"task": task, "split": split, "query": query, "candidate": candidate, "grade": grade,
                       "note": "", "labelled_at": at})


@pytest.fixture
def run(tmp_path):
    (tmp_path / "similar_dev.json").write_text(json.dumps({"queries": [{"query": Q, "candidates": [A, B]}]}))
    (tmp_path / "similar_dev_membership.json").write_text(json.dumps({"queries": {}}))
    (tmp_path / "similar_dev_supplement.json").write_text(json.dumps({"queries": [{"query": Q, "candidates": [C]}]}))

    def go(*lines, latest_wins=False):
        (tmp_path / "labels.jsonl").write_text("".join(line + "\n" for line in lines))
        return check(tmp_path / "labels.jsonl", tmp_path, "dev", NO_HOLDOUT, latest_wins)
    return go


def test_complete_labels_pass_and_merge_pool_with_supplement(run):
    report, merged = run(label(A, 3), label(B, 0), label(C, 2))
    assert report["failures"] == {} and report["expected_pairs"] == report["labelled_pairs"] == 3
    assert report["pool_files"] == ["similar_dev.json", "similar_dev_supplement.json"]
    assert merged["similar"] == {Q: {A: 3, B: 0, C: 2}}


def test_missing_supplemental_label_fails(run):
    report, _ = run(label(A), label(B))
    assert report["failures"] == {"missing_labels": [["similar", Q, C]]}


def test_identical_repeat_is_reported_not_failed(run):
    report, _ = run(label(A), label(A), label(B), label(C))
    assert report["failures"] == {} and report["repeats_with_same_grade"] == 1


def test_conflicting_grades_fail_unless_latest_wins(run):
    lines = (label(A, 1), label(B), label(C), label(A, 3, at="2026-10-22T11:00:00+00:00"))
    report, _ = run(*lines)
    assert report["failures"] == {"conflicts": [["similar", Q, A, [1, 3]]]}
    report, merged = run(*lines, latest_wins=True)
    assert report["failures"] == {} and report["regrades_accepted"] and merged["similar"][Q][A] == 3


@pytest.mark.parametrize("bad", [label(grade=4), label(grade=True), label(grade="2"), label(candidate="x"),
                                 label(task="other"), label(at="yesterday"), "{not json", json.dumps({"task": "similar"})])
def test_malformed_lines_fail(run, bad):
    report, _ = run(label(A), label(B), label(C), bad)
    assert len(report["failures"]["invalid_lines"]) == 1


def test_unpooled_pair_fails(run):
    report, _ = run(label(A), label(B), label(C), label(D))
    assert report["failures"] == {"unexpected_pairs": [["similar", Q, D]]}


@pytest.mark.parametrize("touch", [label(D, query=HOLDOUT), label(split="holdout"),
                                   label("abc".ljust(32, "0"), task="semantic", query="secret holdout query")])
def test_any_holdout_label_fails(run, touch):
    report, _ = run(label(A), label(B), label(C), touch)
    assert list(report["failures"]) == ["holdout_labels"]


def test_pair_in_pool_and_supplement_fails(run, tmp_path):
    (tmp_path / "similar_dev_supplement.json").write_text(json.dumps({"queries": [{"query": Q, "candidates": [A, C]}]}))
    report, _ = run(label(A), label(B), label(C))
    assert report["failures"] == {"pooled_twice": [["similar", Q, A]]}


def test_supplement_holds_only_unpooled_pairs_and_hides_systems():
    rankings = {"q1": {"S1": ["a", "b"], "S2": ["b", "c"]}, "q2": {"S1": ["d"], "S2": ["e"]}, "q3": {"S1": ["f"]}}
    pooled = {("q1", "a"), ("q1", "b"), ("q2", "d"), ("q3", "f")}
    pool, membership = supplement_queries(rankings, pooled)
    assert {(q["query"], c) for q in pool for c in q["candidates"]} == {("q1", "c"), ("q2", "e")}
    assert all(set(q) == {"query", "candidates"} for q in pool)
    assert membership == {"q1": {"S1": [], "S2": ["c"]}, "q2": {"S1": [], "S2": ["e"]}}
    assert supplement_queries(rankings, pooled) == (pool, membership)
    assert supplement_queries(rankings, pooled | {("q1", "c"), ("q2", "e")}) == ([], {})


def test_variant_rankers_change_only_their_setting():
    from scripts.build_label_pools import is_similar_ranker, ranker_config

    base, capped, blind = ranker_config("C_bge"), ranker_config("C_bge+title_cap"), ranker_config("C_bge+no_category")
    assert base.max_per_title is None and capped.max_per_title == 2
    assert blind.attribute_weights["category"] == 0 and blind.attribute_weights["price"] == base.attribute_weights["price"]
    assert capped.text_model == blind.text_model == base.text_model == "bge-small"
    assert is_similar_ranker("A_structured+title_cap") and not is_similar_ranker("C_bge+unknown")
    assert not is_similar_ranker("bge-small")
