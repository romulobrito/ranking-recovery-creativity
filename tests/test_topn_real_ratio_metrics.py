"""
Unit tests for topn_real_ratio metric.
"""

from __future__ import annotations

import pandas as pd

from embedding_models_eval.metrics.base import get_metric, list_metrics
from embedding_models_eval.metrics.topn_real_ratio import TopNRealRatioMetrics


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"prompt_id": "p1", "rank_in_prompt": 1, "rank_pred": None, "likes": 120},
            {"prompt_id": "p1", "rank_in_prompt": 2, "rank_pred": 2, "likes": 95},
            {"prompt_id": "p1", "rank_in_prompt": 3, "rank_pred": 3, "likes": 75},
            {"prompt_id": "p1", "rank_in_prompt": 4, "rank_pred": 1, "likes": 40},
            {"prompt_id": "p1", "rank_in_prompt": 5, "rank_pred": 4, "likes": 20},
        ]
    )


def test_metric_is_registered() -> None:
    assert "topn_real_ratio" in list_metrics()
    metric = get_metric("topn_real_ratio", k_values=[1, 3])
    assert isinstance(metric, TopNRealRatioMetrics)


def test_topn_real_ratio_values_are_direct_pred_over_real() -> None:
    df = _toy_df()
    metric = TopNRealRatioMetrics(k_values=[1, 3], exclude_winner=True)
    result = metric.compute(df)
    per_prompt = result["per_prompt"].set_index("prompt_id")

    # After winner exclusion, real order votes: [95, 75, 40, 20]
    # Pred order votes: [40, 95, 75, 20]
    # k=1: max/mean/min = 40 over 95
    assert round(float(per_prompt.loc["p1", "ratio_max@1"]), 6) == round(40.0 / 95.0, 6)
    assert round(float(per_prompt.loc["p1", "ratio_mean@1"]), 6) == round(40.0 / 95.0, 6)
    assert round(float(per_prompt.loc["p1", "ratio_min@1"]), 6) == round(40.0 / 95.0, 6)

    # k=3:
    # pred top3 [40,95,75] => max=95 mean=70 min=40
    # real top3 [95,75,40] => max=95 mean=70 min=40
    assert per_prompt.loc["p1", "ratio_max@3"] == 1.0
    assert per_prompt.loc["p1", "ratio_mean@3"] == 1.0
    assert per_prompt.loc["p1", "ratio_min@3"] == 1.0


def test_macro_contains_mean_and_std_keys() -> None:
    df = _toy_df()
    metric = TopNRealRatioMetrics(k_values=[1, 2], exclude_winner=True)
    result = metric.compute(df)
    macro = result["macro"]

    assert "ratio_max@1_mean" in macro
    assert "ratio_max@1_std" in macro
    assert "ratio_mean@2_mean" in macro
    assert "ratio_min@2_std" in macro

