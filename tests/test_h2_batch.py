"""
Tests for H2 batch config validation and job building.
"""

from __future__ import annotations

import pytest

from embedding_models_eval.h2_batch import (
    build_h2_jobs,
    validate_h2_batch_config,
)


def _base_cfg() -> dict:
    return {
        "dataset": {
            "input_json": "saida_final.json",
            "output_root": "results/experimento2_h2",
        },
        "protocol": {
            "k_values": [1, 2, 3],
            "exclude_winner": True,
            "required_models": ["m1", "m2"],
        },
        "models": [
            {"name": "m1", "config": "configs/a.yaml", "output_subdir": "m1"},
            {"name": "m2", "config": "configs/b.yaml", "output_subdir": "m2"},
        ],
    }


def test_validate_h2_batch_config_accepts_matching_model_set() -> None:
    cfg = _base_cfg()
    validate_h2_batch_config(cfg)
    jobs = build_h2_jobs(cfg)
    assert len(jobs) == 2
    assert jobs[0].model_name == "m1"


def test_validate_h2_batch_config_rejects_model_mismatch() -> None:
    cfg = _base_cfg()
    cfg["protocol"]["required_models"] = ["m1", "m2", "m3"]
    with pytest.raises(ValueError, match="do not match required_models exactly"):
        validate_h2_batch_config(cfg)


def test_validate_h2_batch_config_enforces_exclude_winner() -> None:
    cfg = _base_cfg()
    cfg["protocol"]["exclude_winner"] = False
    with pytest.raises(ValueError, match="exclude_winner must be true"):
        validate_h2_batch_config(cfg)
