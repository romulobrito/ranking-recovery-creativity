"""
Unit tests for per-prompt table generation helpers.
"""

from __future__ import annotations

from embedding_models_eval.prompt_tables import (
    PromptTableInput,
    compute_prompt_rows,
    validate_prompt_tables_config,
)


def _valid_cfg() -> dict:
    return {
        "protocol": {"k_values": [1, 3], "exclude_winner": True},
        "output": {"out_dir": "results/prompt_tables_test"},
        "inputs": [
            {"hypothesis": "H1", "model": "m1", "detailed_json": "a.json"},
            {"hypothesis": "H2", "model": "m2", "detailed_json": "b.json"},
        ],
    }


def test_validate_prompt_tables_config_ok() -> None:
    cfg = _valid_cfg()
    validate_prompt_tables_config(cfg)


def test_validate_prompt_tables_config_rejects_unknown_hypothesis() -> None:
    cfg = _valid_cfg()
    cfg["inputs"][0]["hypothesis"] = "H3"
    try:
        validate_prompt_tables_config(cfg)
    except ValueError as exc:
        assert "must be H1 or H2" in str(exc)
    else:
        raise AssertionError("Expected validation to fail for unknown hypothesis")


def test_compute_prompt_rows_generates_required_columns() -> None:
    item = PromptTableInput(hypothesis="H2", model="m2", detailed_json="dummy.json")
    rows = [
        {"prompt_id": "p1", "story_url": "s1", "likes": 100, "rank_in_prompt": 1, "rank_pred": 3},
        {"prompt_id": "p1", "story_url": "s2", "likes": 80, "rank_in_prompt": 2, "rank_pred": 1},
        {"prompt_id": "p1", "story_url": "s3", "likes": 60, "rank_in_prompt": 3, "rank_pred": 2},
    ]
    out = compute_prompt_rows(item, rows, k_values=[1, 2], exclude_winner=True)
    assert len(out) == 2
    first = out[0]
    for key in ("TOPN_max", "TOPN_media", "TOPN_min", "Real", "Votos_Real", "Votos_TOP1_Best"):
        assert key in first
    assert first["Real"] == 80.0
    assert first["Votos_Real"] == 1.0
