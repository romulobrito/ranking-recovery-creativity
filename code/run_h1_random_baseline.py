"""
Run H1 random baseline from a YAML config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from embedding_models_eval.h2_random_baseline import (
    build_h2_baseline_inputs,
    load_h2_random_baseline_config,
    validate_h2_random_baseline_config,
)
from embedding_models_eval.metrics.feasible_range_random_baseline import (
    summarize_model,
    write_csv,
)


def run_h1_random_baseline_from_config(cfg: Dict) -> List[Dict[str, object]]:
    """Execute H1 random baseline for all models in config."""
    protocol = cfg["protocol"]
    k_values = [int(k) for k in protocol["k_values"]]
    num_permutations = int(protocol["num_permutations"])
    seed = int(protocol["seed"])
    percentiles = [int(p) for p in protocol["percentiles"]]

    rows: List[Dict[str, object]] = []
    for idx, model_input in enumerate(build_h2_baseline_inputs(cfg)):
        model_rows = summarize_model(
            model_name=model_input.model_name,
            detailed_json_path=model_input.detailed_json,
            k_values=k_values,
            num_permutations=num_permutations,
            seed=seed + idx,
            percentiles=percentiles,
        )
        rows.extend(model_rows)
    return rows


def main() -> None:
    """CLI entrypoint for H1 random baseline using YAML config."""
    parser = argparse.ArgumentParser(description="Run H1 random baseline from YAML config")
    parser.add_argument("--config", required=True, help="H1 random baseline YAML config")
    args = parser.parse_args()

    cfg = load_h2_random_baseline_config(args.config)
    validate_h2_random_baseline_config(cfg)
    rows = run_h1_random_baseline_from_config(cfg)

    output_csv = str(cfg["dataset"]["output_csv"])
    output_json = str(cfg["dataset"]["output_json"])

    write_csv(rows, output_csv)
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=True)

    print(f"CSV gerado: {output_csv}")
    print(f"JSON gerado: {output_json}")
    print(f"Linhas: {len(rows)}")


if __name__ == "__main__":
    main()
