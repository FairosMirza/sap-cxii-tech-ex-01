import math

import numpy as np
import pytest

from scripts.evaluate import choose, choose_text_weight, compare_paths, evaluate_task, query_metrics, title_repeats


@pytest.mark.parametrize("sweep, chosen", [
    ({0.4: {"mean": 0.01, "se": 0.02}, 0.6: {"mean": -0.03, "se": 0.01}}, 0.5),
    ({0.3: {"mean": 0.08, "se": 0.02}, 0.6: {"mean": 0.03, "se": 0.01}}, 0.6),
    ({0.4: {"mean": 0.03, "se": 0.01}, 0.6: {"mean": 0.05, "se": 0.01}}, 0.6),
])
def test_text_weight_moves_only_for_a_gain_beyond_one_standard_error(sweep, chosen):
    assert choose_text_weight(0.5, sweep) == chosen


def test_ndcg_is_one_for_the_ideal_order_and_lower_otherwise():
    judged = {"a": 3, "b": 1, "c": 0}
    assert query_metrics(["a", "b", "c"], judged)["ndcg"] == pytest.approx(1.0)
    worse = query_metrics(["c", "b", "a"], judged)["ndcg"]
    expected = (1 / math.log2(3) + 7 / math.log2(4)) / (7 + 1 / math.log2(3))
    assert worse == pytest.approx(expected)


def test_ideal_ranking_uses_every_judged_product_of_the_query():
    # The ranker found only the weak match; a strong match exists among the judged products.
    assert query_metrics(["b"], {"a": 3, "b": 1})["ndcg"] == pytest.approx(1 / (7 + 1 / math.log2(3)))


def test_precision_counts_grade_two_and_up_over_five_even_if_fewer_returned():
    metrics = query_metrics(["a", "b", "c"], {"a": 3, "b": 2, "c": 1})
    assert metrics["p5"] == pytest.approx(2 / 5)


def test_query_without_any_relevant_product_has_no_metrics():
    assert query_metrics(["a"], {"a": 0, "b": 0}) is None and query_metrics([], {}) is None


def test_a_list_with_an_unjudged_result_gets_no_metrics():
    metrics = query_metrics(["a", "x"], {"a": 2})
    assert metrics["returned"] == 2 and metrics["judged"] == 1
    assert metrics["ndcg"] is None and metrics["ndcg_linear"] is None and metrics["p5"] is None


def test_one_standard_error_rule_prefers_the_simpler_system_within_noise():
    base = np.linspace(0.4, 0.8, 20)
    scores = {"simple": base, "complex": base + 0.01 + np.tile([0.05, -0.05], 10)}  # +0.01 mean, SE about 0.0115
    result = choose(("simple", "complex"), scores)
    assert result["lead"] == "complex" and result["choice"] == "simple"


def test_one_standard_error_rule_keeps_a_clearly_better_system():
    base = np.linspace(0.3, 0.6, 20)
    result = choose(("simple", "complex"), {"simple": base, "complex": base + 0.2})
    assert result["lead"] == result["choice"] == "complex"


def test_title_repeats_ignore_untitled_products():
    titles = {"a": 1, "b": 1, "c": 2, "d": -1, "e": -1}
    assert title_repeats(["a", "b", "c", "d", "e"], titles) == 1


def test_rankers_with_unjudged_results_are_not_compared():
    rankings = {"q1": {"A_structured": ["a", "b"], "B_tfidf": ["a", "x"]}, "q2": {"A_structured": ["c"], "B_tfidf": ["c"]}}
    judgements = {"q1": {"a": 3, "b": 1}, "q2": {"c": 2}}
    result = evaluate_task("similar", rankings, judgements, {pid: -1 for pid in "abcx"})
    assert result["rankers"]["B_tfidf"]["complete"] is False and "ndcg" not in result["rankers"]["B_tfidf"]
    assert result["choice"]["lead"] == result["choice"]["choice"] == "A_structured"


def test_path_comparison_counts_unjudged_results_apart():
    # Reference a b c d; the path keeps a, drops b (relevant), c (grade 0), d (unjudged), adds e (grade 1), f (unjudged).
    rankings = {"q": {"exact": ["a", "b", "c", "d"], "deployed": ["a", "e", "f"]}}
    judgements = {"q": {"a": 3, "b": 2, "c": 0, "e": 1}}
    result = compare_paths(rankings, judgements, "deployed", "exact", comparable={})
    assert result["dropped"] == {"relevant": 1, "grade_1": 0, "grade_0": 1, "unjudged": 1}
    assert result["added"] == {"relevant": 0, "grade_1": 1, "grade_0": 0, "unjudged": 1}
    assert result["coverage"] == {"mean": 0.25, "min": 0.25, "fully_covered": 0}
    assert result["queries_below_full_coverage"][0]["dropped"] == {"relevant": 1, "grade_0": 1, "unjudged": 1}
    assert "ndcg_difference" not in result and "unjudged" in result["paired_differences"]


def test_frozen_paths_are_compared_but_never_chosen():
    rankings = {"q1": {"A_structured": ["a", "b"], "C_bge": ["a", "b"], "served_exact": ["a", "b"], "served": ["a", "c"]},
                "q2": {"A_structured": ["d"], "C_bge": ["e"], "served_exact": ["e"], "served": ["e"]}}
    judgements = {"q1": {"a": 3, "b": 2, "c": 0}, "q2": {"d": 1, "e": 3}}
    result = evaluate_task("similar", rankings, judgements, {pid: -1 for pid in "abcde"}, configured="C_bge")
    assert set(result["choice"]["versus_lead"]) | {result["choice"]["lead"]} == {"A_structured", "C_bge"}
    deployed, exact = result["deployed_vs_exact"], result["exact_vs_base"]
    assert deployed["dropped"]["relevant"] == 1 and deployed["added"]["grade_0"] == 1
    assert deployed["ndcg_difference"]["mean"] < 0 and exact["coverage"]["mean"] == 1.0
    assert result["rankers"]["served"]["grade_0_results_shown"] == 1 and result["rankers"]["served"]["mean_results_shown"] == 1.5
