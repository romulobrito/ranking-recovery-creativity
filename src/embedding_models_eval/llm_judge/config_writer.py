"""
Write YAML configs for LLM-judge random baselines (one per protocol x metric).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

from embedding_models_eval.llm_judge.geval_loader import PROTOCOL_SPECS


def build_baseline_config(
    protocol_id: str,
    metric: str,
    judge_slugs: List[str],
    output_root: Path,
    num_permutations: int = 2500,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Build a YAML-serializable config for one protocol and metric family.

    metric: "feasible" or "topn"
    """
    if metric not in {"feasible", "topn"}:
        raise ValueError(f"Unknown metric family: {metric}")

    base_dir = output_root / "by_protocol" / protocol_id
    if metric == "feasible":
        out_dir = base_dir / "random_baseline_feasible_2500"
    else:
        out_dir = base_dir / "random_baseline_topn_2500"

    models = []
    for slug in judge_slugs:
        models.append(
            {
                "name": slug,
                "detailed_json": str(
                    base_dir / "judges" / slug / "detalhado.json"
                ),
            }
        )

    return {
        "protocol": {
            "k_values": [1, 2, 3, 4, 5, 6, 7],
            "num_permutations": int(num_permutations),
            "percentiles": [95],
            "seed": int(seed),
            "protocol_id": protocol_id,
            "metric_family": metric,
        },
        "dataset": {
            "output_csv": str(out_dir / "summary.csv"),
            "output_json": str(out_dir / "summary.json"),
        },
        "models": models,
    }


def write_all_baseline_configs(
    output_root: Path,
    configs_dir: Path,
    run_summary: Dict[str, Any],
    num_permutations: int = 2500,
    seed: int = 42,
) -> List[Path]:
    """
    Write YAML configs under configs/llm_judge_normalized/ for each protocol.
    """
    configs_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    protocols = run_summary.get("protocols", {})
    for spec in PROTOCOL_SPECS:
        proto = protocols.get(spec.protocol_id)
        if not proto:
            continue
        judges = list(proto.get("judges", []))
        for metric in ("feasible", "topn"):
            cfg = build_baseline_config(
                protocol_id=spec.protocol_id,
                metric=metric,
                judge_slugs=judges,
                output_root=output_root,
                num_permutations=num_permutations,
                seed=seed,
            )
            path = configs_dir / f"{spec.protocol_id}_{metric}_baseline_2500.yaml"
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
            written.append(path)
    return written
