"""
Unit tests for H2 metrics helpers.
"""

from __future__ import annotations

from embedding_models_eval.h2_metrics import (
    aggregate_prompt_metrics,
    build_h2_detailed_rows,
    build_macro_row,
    build_story_votes_map,
    compute_prompt_h2_metrics,
    compute_prompt_h2_topn_real_ratio,
    group_rows_by_prompt,
)


def test_group_rows_by_prompt_and_story_votes_map() -> None:
    rows = [
        {
            "contest_number": "1",
            "context_prompt_url": "pA",
            "story_url": "s1",
            "likes": 10,
        },
        {
            "contest_number": "1",
            "context_prompt_url": "pA",
            "story_url": "s2",
            "likes": 20,
        },
        {
            "contest_number": "2",
            "context_prompt_url": "pB",
            "story_url": "s3",
            "likes": 30,
        },
    ]
    grouped = group_rows_by_prompt(rows)
    assert set(grouped.keys()) == {"1::pA", "2::pB"}
    votes_map = build_story_votes_map(grouped["1::pA"])
    assert votes_map == {"s1": 10.0, "s2": 20.0}


def test_compute_prompt_h2_metrics_with_winner_exclusion() -> None:
    story_votes = {"s1": 100.0, "s2": 80.0, "s3": 60.0, "s4": 40.0}
    predicted = ["s4", "s3", "s2", "s1"]  # reversed quality
    out = compute_prompt_h2_metrics(
        predicted_story_ids=predicted,
        story_votes=story_votes,
        k_values=[1, 2],
        exclude_winner=True,
    )
    assert "ratio_max@1" in out
    assert "ratio_mean@2" in out
    assert "ratio_min@2" in out
    assert "votes_top1_ratio" in out
    assert 0.0 <= out["votes_top1_ratio"] <= 1.0


def test_compute_prompt_h2_topn_real_ratio_with_winner_exclusion() -> None:
    story_votes = {"s1": 100.0, "s2": 80.0, "s3": 60.0, "s4": 40.0}
    predicted = ["s4", "s3", "s2", "s1"]  # reversed quality
    out = compute_prompt_h2_topn_real_ratio(
        predicted_story_ids=predicted,
        story_votes=story_votes,
        k_values=[1, 2],
        exclude_winner=True,
    )
    assert "ratio_max@1" in out
    assert "ratio_mean@2" in out
    assert "ratio_min@2" in out
    assert 0.0 <= out["ratio_max@1"] <= 1.0


def test_aggregate_prompt_metrics_and_macro_row() -> None:
    prompt_metrics = [
        {"ratio_max@1": 0.5, "ratio_mean@1": 0.4, "ratio_min@1": 0.3, "votes_top1_ratio": 0.8},
        {"ratio_max@1": 0.7, "ratio_mean@1": 0.6, "ratio_min@1": 0.2, "votes_top1_ratio": 1.0},
    ]
    macro = aggregate_prompt_metrics(prompt_metrics)
    assert "ratio_max@1_mean" in macro
    assert "ratio_max@1_std" in macro
    assert "votes_top1_ratio_mean" in macro
    row = build_macro_row("model_h2", macro)
    assert row["Modelo"] == "model_h2"


def test_build_h2_detailed_rows_has_required_fields() -> None:
    prompt_rows = [
        {"story_url": "s1", "likes": 100},
        {"story_url": "s2", "likes": 80},
        {"story_url": "s3", "likes": 60},
    ]
    predicted = ["s3", "s2", "s1"]
    rows = build_h2_detailed_rows("p1", prompt_rows, predicted)
    assert len(rows) == 3
    first = rows[0]
    for key in ("prompt_id", "story_url", "likes", "rank_in_prompt", "rank_pred"):
        assert key in first
    winner = [r for r in rows if r["story_url"] == "s1"][0]
    assert winner["rank_in_prompt"] == 1
    assert winner["rank_pred"] == 3
