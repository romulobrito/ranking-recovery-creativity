"""
Top-N direct ratio metrics against real top-N references.

This metric computes prompt-level ratios comparing predicted top-N vote
statistics to the corresponding real (gold) top-N statistics:
- ratio_max@k = max(pred_top_k_votes) / max(real_top_k_votes)
- ratio_mean@k = mean(pred_top_k_votes) / mean(real_top_k_votes)
- ratio_min@k = min(pred_top_k_votes) / min(real_top_k_votes)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .base import Metric, register_metric


class TopNRealRatioMetrics(Metric):
    """
    Computes direct top-N ratios against real top-N references.
    """

    def __init__(
        self,
        k_values: List[int] | None = None,
        votes_col: str = "likes",
        rank_pred_col: str = "rank_pred",
        prompt_id_col: str = "prompt_id",
        rank_gold_col: str = "rank_in_prompt",
        winner_rank_col: str = "rank_in_prompt",
        winner_rank_value: int = 1,
        exclude_winner: bool = True,
        clip_0_1: bool = True,
    ) -> None:
        if k_values is None:
            k_values = [1, 2, 3, 4, 5, 6, 7]
        super().__init__("topn_real_ratio", k_values)
        self.votes_col = votes_col
        self.rank_pred_col = rank_pred_col
        self.prompt_id_col = prompt_id_col
        self.rank_gold_col = rank_gold_col
        self.winner_rank_col = winner_rank_col
        self.winner_rank_value = winner_rank_value
        self.exclude_winner = exclude_winner
        self.clip_0_1 = clip_0_1

    def _safe_div(self, observed: float, reference: float) -> float:
        """
        Safe observed/reference ratio with optional clipping to [0, 1].
        """
        if reference == 0:
            return 1.0 if observed == 0 else 0.0
        ratio = observed / reference
        if self.clip_0_1:
            return float(np.clip(ratio, 0.0, 1.0))
        return float(ratio)

    def compute(
        self,
        df: pd.DataFrame,
        rank_pred_col: str = "rank_pred",
        rank_gold_col: str = "rank_gold",
        prompt_id_col: str = "prompt_id",
    ) -> Dict:
        """
        Computes per-prompt and macro top-N direct ratios.
        """
        required_cols = {
            self.prompt_id_col,
            self.rank_pred_col,
            self.rank_gold_col,
            self.votes_col,
        }
        if self.exclude_winner and self.winner_rank_col in df.columns:
            required_cols.add(self.winner_rank_col)

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for topn_real_ratio: {missing}")

        per_prompt_rows: List[Dict[str, float | str]] = []

        for pid, group in df.groupby(self.prompt_id_col):
            work = group.copy()
            if self.exclude_winner and self.winner_rank_col in work.columns:
                work = work[work[self.winner_rank_col] != self.winner_rank_value]

            work = work[
                work[self.rank_pred_col].notna()
                & work[self.rank_gold_col].notna()
                & work[self.votes_col].notna()
            ]
            if work.empty:
                continue

            pred_sorted = work.sort_values(self.rank_pred_col, ascending=True, kind="mergesort")
            gold_sorted = work.sort_values(self.rank_gold_col, ascending=True, kind="mergesort")
            total = len(pred_sorted)
            if total == 0:
                continue

            row: Dict[str, float | str] = {"prompt_id": pid}

            pred_votes = pred_sorted[self.votes_col].astype(float).tolist()
            gold_votes = gold_sorted[self.votes_col].astype(float).tolist()

            for k in self.k_values:
                n_eff = min(int(k), total)
                top_pred = pred_votes[:n_eff]
                top_gold = gold_votes[:n_eff]

                obs_max = float(max(top_pred))
                obs_mean = float(sum(top_pred) / n_eff)
                obs_min = float(min(top_pred))

                ref_max = float(max(top_gold))
                ref_mean = float(sum(top_gold) / n_eff)
                ref_min = float(min(top_gold))

                row[f"ratio_max@{k}"] = self._safe_div(obs_max, ref_max)
                row[f"ratio_mean@{k}"] = self._safe_div(obs_mean, ref_mean)
                row[f"ratio_min@{k}"] = self._safe_div(obs_min, ref_min)

            per_prompt_rows.append(row)

        if not per_prompt_rows:
            return {"per_prompt": pd.DataFrame(), "macro": {}}

        per_prompt_df = pd.DataFrame(per_prompt_rows)
        macro_results: Dict[str, float] = {}
        for k in self.k_values:
            for metric_name in (f"ratio_max@{k}", f"ratio_mean@{k}", f"ratio_min@{k}"):
                vals = per_prompt_df[metric_name].dropna().astype(float).values
                macro_results[f"{metric_name}_mean"] = float(np.mean(vals)) if len(vals) else 0.0
                macro_results[f"{metric_name}_std"] = (
                    float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                )

        return {"per_prompt": per_prompt_df, "macro": macro_results}


register_metric("topn_real_ratio", TopNRealRatioMetrics)
register_metric("topn_real_ratio_metrics", TopNRealRatioMetrics)

