"""
Tests for H1+H2 consolidation utility.
"""

from __future__ import annotations

import pandas as pd

from embedding_models_eval.h1_h2_comparison import consolidate_h1_h2


def test_consolidate_h1_h2_generates_expected_outputs(tmp_path) -> None:
    h1_dir = tmp_path / "h1"
    h2_dir = tmp_path / "h2"
    out_dir = tmp_path / "out"
    h1_dir.mkdir(parents=True, exist_ok=True)
    h2_dir.mkdir(parents=True, exist_ok=True)

    h1_df = pd.DataFrame(
        [
            {
                "Modelo": "minilm",
                "ratio_max@1_mean": 0.11,
                "ratio_mean@1_mean": 0.08,
            },
            {
                "Modelo": "openai_large",
                "ratio_max@1_mean": 0.52,
                "ratio_mean@1_mean": 0.30,
            },
        ]
    )
    h2_df = pd.DataFrame(
        [
            {
                "Modelo": "minilm",
                "votes_top1_ratio_mean": 0.41,
                "votes_top1_ratio_std": 0.12,
            },
            {
                "Modelo": "openai_large",
                "votes_top1_ratio_mean": 0.63,
                "votes_top1_ratio_std": 0.10,
            },
        ]
    )

    h1_csv = h1_dir / "comparacao_modelos_macro.csv"
    h2_csv = h2_dir / "comparacao_modelos_macro.csv"
    h1_df.to_csv(h1_csv, index=False)
    h2_df.to_csv(h2_csv, index=False)

    outputs = consolidate_h1_h2(
        h1_path=str(h1_dir),
        h2_path=str(h2_dir),
        out_dir=str(out_dir),
    )

    assert outputs["wide_csv"].exists()
    assert outputs["wide_json"].exists()
    assert outputs["long_csv"].exists()
    assert outputs["long_json"].exists()

    long_df = pd.read_csv(outputs["long_csv"])
    wide_df = pd.read_csv(outputs["wide_csv"])

    assert set(long_df["hypothesis"]) == {"H1", "H2"}
    assert {"hypothesis", "model"}.issubset(set(wide_df.columns))
    assert "ratio_max@1_mean" in set(wide_df.columns)
    assert "votes_top1_ratio_mean" in set(wide_df.columns)

    h1_ratio = long_df[
        (long_df["hypothesis"] == "H1")
        & (long_df["model"] == "minilm")
        & (long_df["metric_key"] == "ratio_max@1_mean")
    ]
    assert len(h1_ratio) == 1
    assert float(h1_ratio.iloc[0]["n"]) == 1.0
    assert h1_ratio.iloc[0]["stat"] == "mean"
