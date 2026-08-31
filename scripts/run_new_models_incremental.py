"""
Run H1 + H2 (feasible_range_ratio + topn_real_ratio) for new models only,
placing results into the SAME directory structure used by the existing 8 models.
Then reconsolidate all 10 models.

Usage:
    python scripts/run_new_models_incremental.py
    python scripts/run_new_models_incremental.py --dry-run
    python scripts/run_new_models_incremental.py --models qwen3-embedding-8b
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]


NEW_MODELS_H1_FEASIBLE = [
    {
        "name": "gemini-embedding",
        "provider": "openai",
        "config": {
            "model_name": "google/gemini-embedding-001",
            "api_key": "${OPENROUTER_API_KEY}",
            "base_url": "https://openrouter.ai/api/v1",
            "batch_size": 100,
            "normalize": True,
            "max_retries": 3,
        },
    },
    {
        "name": "qwen3-embedding-8b",
        "provider": "openai",
        "config": {
            "model_name": "qwen/qwen3-embedding-8b",
            "api_key": "${OPENROUTER_API_KEY}",
            "base_url": "https://openrouter.ai/api/v1",
            "batch_size": 100,
            "normalize": True,
            "max_retries": 3,
        },
    },
]


H2_CONFIGS = {
    "gemini-embedding": "configs/h2_eval_embeddinggemma.yaml",
    "qwen3-embedding-8b": "configs/h2_eval_qwen3_openrouter.yaml",
}

H1_FEASIBLE_DIR = "results/experimento2_h1_selected"
H1_TOPN_DIR = "results/experimento2_h1_topn_real_ratio"
H2_FEASIBLE_DIR = "results/experimento2_h2"
H2_TOPN_DIR = "results/experimento2_h2_topn_real_ratio"
H1_H2_FEASIBLE_OUT = "results/experimento2_h1_h2"
H1_H2_TOPN_OUT = "results/experimento2_h1_h2_topn_real_ratio"


def _run(cmd: List[str], dry_run: bool, label: str = "") -> None:
    """Print and execute a subprocess command."""
    printable = " ".join(shlex.quote(x) for x in cmd)
    tag = f"[{label}] " if label else ""
    print(f"{tag}CMD: {printable}")
    if dry_run:
        return
    result = subprocess.run(cmd, cwd=str(ROOT_DIR))
    if result.returncode != 0:
        print(f"  ERROR: command failed with exit code {result.returncode}")
        sys.exit(1)


def _append_rows_to_csv(target_csv: Path, source_csv: Path) -> None:
    """Append data rows (skip header) from source to target CSV."""
    if not source_csv.is_file():
        print(f"  WARNING: source not found: {source_csv}")
        return
    with source_csv.open("r", encoding="utf-8") as sf:
        reader = csv.reader(sf)
        header = next(reader, None)
        new_rows = list(reader)
    if not new_rows:
        return

    if not target_csv.is_file():
        target_csv.parent.mkdir(parents=True, exist_ok=True)
        with target_csv.open("w", encoding="utf-8", newline="") as tf:
            writer = csv.writer(tf)
            if header:
                writer.writerow(header)
            writer.writerows(new_rows)
        return

    with target_csv.open("r", encoding="utf-8") as tf:
        existing_reader = csv.reader(tf)
        existing_header = next(existing_reader, None)
        existing_rows = list(existing_reader)

    existing_models = {row[0] for row in existing_rows if row}
    rows_to_add = [r for r in new_rows if r and r[0] not in existing_models]
    if not rows_to_add:
        print(f"  (models already present in {target_csv.name}, skipping)")
        return

    with target_csv.open("w", encoding="utf-8", newline="") as tf:
        writer = csv.writer(tf)
        if existing_header:
            writer.writerow(existing_header)
        writer.writerows(existing_rows)
        writer.writerows(rows_to_add)
    print(f"  Appended {len(rows_to_add)} row(s) to {target_csv}")


def run_h1_feasible(model_names: List[str], dry_run: bool) -> None:
    """Run H1 feasible_range_ratio for selected new models."""
    print("\n" + "=" * 70)
    print("STEP 1: H1 feasible_range_ratio (new models)")
    print("=" * 70)

    for model_def in NEW_MODELS_H1_FEASIBLE:
        name = model_def["name"]
        if name not in model_names:
            continue
        out_dir = ROOT_DIR / H1_FEASIBLE_DIR / name
        if (out_dir / "comparacao_modelos_macro.csv").is_file():
            print(f"  [{name}] Already exists, skipping. Delete to re-run.")
            continue

        print(f"\n  [{name}] Running H1 feasible_range_ratio...")
        _run(
            [
                sys.executable,
                str(ROOT_DIR / "run_pipeline.py"),
                "--config", "configs/h1_paper_new_models_only.yaml",
                "--models", name,
                "--output-dir", str(out_dir),
            ],
            dry_run=dry_run,
            label=f"H1-feasible {name}",
        )


def run_h1_topn(model_names: List[str], dry_run: bool) -> None:
    """Run H1 topn_real_ratio for selected new models and append to existing CSV."""
    print("\n" + "=" * 70)
    print("STEP 2: H1 topn_real_ratio (new models)")
    print("=" * 70)

    tmp_dir = ROOT_DIR / "results" / "_tmp_h1_topn_new_models"

    for model_def in NEW_MODELS_H1_FEASIBLE:
        name = model_def["name"]
        if name not in model_names:
            continue

        target_csv = ROOT_DIR / H1_TOPN_DIR / "comparacao_modelos_macro.csv"
        if target_csv.is_file():
            with target_csv.open("r", encoding="utf-8") as f:
                content = f.read()
            if name in content:
                print(f"  [{name}] Already in topn CSV, skipping.")
                continue

        model_tmp = tmp_dir / name
        print(f"\n  [{name}] Running H1 topn_real_ratio...")
        _run(
            [
                sys.executable,
                str(ROOT_DIR / "run_pipeline.py"),
                "--config", "configs/h1_paper_feasible_plus_topn.yaml",
                "--models", name,
                "--output-dir", str(model_tmp),
            ],
            dry_run=dry_run,
            label=f"H1-topn {name}",
        )

        if not dry_run:
            source_csv = model_tmp / "comparacao_modelos_macro.csv"
            _append_rows_to_csv(target_csv, source_csv)


def run_h2_feasible(model_names: List[str], dry_run: bool) -> None:
    """Run H2 feasible_range_ratio for new models."""
    print("\n" + "=" * 70)
    print("STEP 3: H2 feasible_range_ratio (new models)")
    print("=" * 70)

    for name in model_names:
        h2_cfg = H2_CONFIGS.get(name)
        if not h2_cfg:
            print(f"  [{name}] No H2 config found, skipping.")
            continue
        out_dir = ROOT_DIR / H2_FEASIBLE_DIR / name
        if (out_dir / "comparacao_modelos_macro.csv").is_file():
            print(f"  [{name}] Already exists, skipping.")
            continue

        print(f"\n  [{name}] Running H2 feasible_range_ratio...")
        _run(
            [
                sys.executable,
                str(ROOT_DIR / "scripts" / "run_h2_eval.py"),
                "--config", h2_cfg,
                "--input-json", "saida_final.json",
                "--output-dir", str(out_dir),
                "--model-name", name,
                "--k-values", "1,2,3,4,5,6,7",
                "--metric-mode", "feasible_range_ratio",
                "--exclude-winner",
            ],
            dry_run=dry_run,
            label=f"H2-feasible {name}",
        )


def run_h2_topn(model_names: List[str], dry_run: bool) -> None:
    """Run H2 topn_real_ratio for new models."""
    print("\n" + "=" * 70)
    print("STEP 4: H2 topn_real_ratio (new models)")
    print("=" * 70)

    for name in model_names:
        h2_cfg = H2_CONFIGS.get(name)
        if not h2_cfg:
            print(f"  [{name}] No H2 config found, skipping.")
            continue
        out_dir = ROOT_DIR / H2_TOPN_DIR / name
        if (out_dir / "comparacao_modelos_macro.csv").is_file():
            print(f"  [{name}] Already exists, skipping.")
            continue

        print(f"\n  [{name}] Running H2 topn_real_ratio...")
        _run(
            [
                sys.executable,
                str(ROOT_DIR / "scripts" / "run_h2_eval.py"),
                "--config", h2_cfg,
                "--input-json", "saida_final.json",
                "--output-dir", str(out_dir),
                "--model-name", name,
                "--k-values", "1,2,3,4,5,6,7",
                "--metric-mode", "topn_real_ratio",
                "--exclude-winner",
            ],
            dry_run=dry_run,
            label=f"H2-topn {name}",
        )


def reconsolidate(dry_run: bool) -> None:
    """Re-run H1+H2 consolidation for both metrics (all 10 models)."""
    print("\n" + "=" * 70)
    print("STEP 5: Reconsolidate H1+H2 (10 models)")
    print("=" * 70)

    comparison_script = ROOT_DIR / "scripts" / "build_h1_h2_comparison.py"

    print("\n  [feasible_range_ratio] Consolidating...")
    _run(
        [
            sys.executable,
            str(comparison_script),
            "--h1-path", str(ROOT_DIR / H1_FEASIBLE_DIR),
            "--h2-path", str(ROOT_DIR / H2_FEASIBLE_DIR),
            "--out-dir", str(ROOT_DIR / H1_H2_FEASIBLE_OUT),
        ],
        dry_run=dry_run,
        label="consolidate-feasible",
    )

    print("\n  [topn_real_ratio] Consolidating...")
    _run(
        [
            sys.executable,
            str(comparison_script),
            "--h1-path", str(ROOT_DIR / H1_TOPN_DIR),
            "--h2-path", str(ROOT_DIR / H2_TOPN_DIR),
            "--out-dir", str(ROOT_DIR / H1_H2_TOPN_OUT),
        ],
        dry_run=dry_run,
        label="consolidate-topn",
    )


def smoke_test(model_names: List[str]) -> List[str]:
    """Quick validation that required credentials are available."""
    print("=" * 70)
    print("PRE-FLIGHT: Checking credentials and model access")
    print("=" * 70)

    from dotenv import load_dotenv
    env_path = ROOT_DIR.parent / "experimento_convergencia_visualizacao_metricas" / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)

    ready: List[str] = []
    blocked: List[str] = []

    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        print("  OPENROUTER_API_KEY: MISSING - cannot proceed")
        return ready

    for name in model_names:
        print(f"  [{name}] OPENROUTER_API_KEY: OK")
        ready.append(name)

    if blocked:
        print(f"\n  BLOCKED models: {blocked}")
        print("  Will proceed with ready models only.")
    print(f"  READY models: {ready}\n")
    return ready


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run H1+H2 for new models incrementally and reconsolidate"
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Print commands without executing",
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["gemini-embedding", "qwen3-embedding-8b"],
        help="Which new models to run (default: both)",
    )
    parser.add_argument(
        "--skip-preflight", action="store_true", default=False,
        help="Skip credential checks",
    )
    parser.add_argument(
        "--only-reconsolidate", action="store_true", default=False,
        help="Skip model runs, only reconsolidate existing results",
    )
    args = parser.parse_args()

    if args.only_reconsolidate:
        reconsolidate(args.dry_run)
        print("\nDone (reconsolidate only).")
        return

    if args.skip_preflight:
        model_names = args.models
    else:
        model_names = smoke_test(args.models)
        if not model_names:
            print("No models ready. Fix credentials and retry.")
            sys.exit(1)

    run_h1_feasible(model_names, args.dry_run)
    run_h1_topn(model_names, args.dry_run)
    run_h2_feasible(model_names, args.dry_run)
    run_h2_topn(model_names, args.dry_run)
    reconsolidate(args.dry_run)

    print("\n" + "=" * 70)
    print("ALL DONE. Results are in:")
    print(f"  H1 feasible: {H1_FEASIBLE_DIR}/{{model}}/")
    print(f"  H1 topn:     {H1_TOPN_DIR}/comparacao_modelos_macro.csv")
    print(f"  H2 feasible: {H2_FEASIBLE_DIR}/{{model}}/")
    print(f"  H2 topn:     {H2_TOPN_DIR}/{{model}}/")
    print(f"  Consolidated feasible: {H1_H2_FEASIBLE_OUT}/")
    print(f"  Consolidated topn:     {H1_H2_TOPN_OUT}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
