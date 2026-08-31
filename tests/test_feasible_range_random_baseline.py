"""
Tests for random baseline over feasible_range_ratio.
"""

from __future__ import annotations

from embedding_models_eval.metrics.feasible_range_random_baseline import (
    compute_feasible_ratio_for_votes,
    compute_observed_macro,
    compute_random_baseline_summary,
    extract_prompt_votes_from_detailed_rows,
    parse_k_values,
    parse_percentiles,
)


def test_compute_feasible_ratio_for_votes_basic_case() -> None:
    # predicted order votes: [60, 40, 80]
    rmax, rmean, rmin = compute_feasible_ratio_for_votes([60.0, 40.0, 80.0], k=2)
    assert rmax == 0.0
    assert rmean == 0.0
    assert rmin == 0.0


def test_extract_prompt_votes_excludes_winner_and_missing_rank() -> None:
    rows = [
        {"prompt_id": "p1", "rank_in_prompt": 1, "rank_pred": None, "likes": 100},
        {"prompt_id": "p1", "rank_in_prompt": 2, "rank_pred": 2, "likes": 50},
        {"prompt_id": "p1", "rank_in_prompt": 3, "rank_pred": 1, "likes": 70},
        {"prompt_id": "p2", "rank_in_prompt": 2, "rank_pred": 1, "likes": 20},
    ]
    out = extract_prompt_votes_from_detailed_rows(rows)
    assert set(out.keys()) == {"p1", "p2"}
    assert out["p1"] == [70.0, 50.0]
    assert out["p2"] == [20.0]


def test_random_baseline_summary_is_reproducible() -> None:
    prompt_votes = {
        "p1": [60.0, 40.0, 80.0],
        "p2": [10.0, 10.0, 20.0],
    }
    k_values = [1, 3]
    s1 = compute_random_baseline_summary(
        prompt_votes=prompt_votes,
        k_values=k_values,
        num_permutations=50,
        seed=123,
        percentiles=[90, 95],
    )
    s2 = compute_random_baseline_summary(
        prompt_votes=prompt_votes,
        k_values=k_values,
        num_permutations=50,
        seed=123,
        percentiles=[90, 95],
    )
    assert s1 == s2
    assert "ratio_mean@1" in s1
    assert "p95" in s1["ratio_mean@1"]


def test_compute_observed_macro_has_expected_keys() -> None:
    prompt_votes = {"p1": [60.0, 40.0, 80.0]}
    out = compute_observed_macro(prompt_votes, [1, 2, 3])
    assert "ratio_max@1_mean" in out
    assert "ratio_mean@2_mean" in out
    assert "ratio_min@3_mean" in out


def test_parse_helpers_validate_ranges() -> None:
    assert parse_k_values("1,3,7") == [1, 3, 7]
    assert parse_percentiles("90,95,99") == [90, 95, 99]

