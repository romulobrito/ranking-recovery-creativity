"""
Unit tests for H2 random baseline config helpers.
"""

from __future__ import annotations

from embedding_models_eval.h2_random_baseline import (
    build_h2_baseline_inputs,
    validate_h2_random_baseline_config,
)


def _valid_cfg() -> dict:
    return {
        "protocol": {
            "k_values": [1, 2, 3],
            "num_permutations": 10,
            "percentiles": [95],
            "seed": 42,
        },
        "dataset": {
            "output_csv": "results/x.csv",
            "output_json": "results/x.json",
        },
        "models": [
            {"name": "m1", "detailed_json": "results/m1/resultados_h2_detalhado.json"},
            {"name": "m2", "detailed_json": "results/m2/resultados_h2_detalhado.json"},
        ],
    }


def test_validate_h2_random_baseline_config_ok() -> None:
    cfg = _valid_cfg()
    validate_h2_random_baseline_config(cfg)


def test_validate_h2_random_baseline_config_duplicate_model() -> None:
    cfg = _valid_cfg()
    cfg["models"] = [
        {"name": "m1", "detailed_json": "a.json"},
        {"name": "m1", "detailed_json": "b.json"},
    ]
    try:
        validate_h2_random_baseline_config(cfg)
    except ValueError as exc:
        assert "Duplicate model name" in str(exc)
    else:
        raise AssertionError("Expected duplicate model validation error")


def test_build_h2_baseline_inputs() -> None:
    cfg = _valid_cfg()
    items = build_h2_baseline_inputs(cfg)
    assert len(items) == 2
    assert items[0].model_name == "m1"
    assert items[1].detailed_json.endswith("resultados_h2_detalhado.json")
