"""
Run full Experimento 2 artifact generation from a single YAML config.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

import yaml


def _run_step(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    """Print and optionally execute one subprocess command."""
    printable = " ".join(shlex.quote(x) for x in cmd)
    print(f"CMD: {printable}")
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(cwd))


def main() -> None:
    """CLI entrypoint to run full H1/H2 artifact pipeline."""
    parser = argparse.ArgumentParser(description="Run full Experimento 2 artifact pipeline")
    parser.add_argument("--config", required=True, help="Full pipeline YAML config")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print commands without executing",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Full pipeline config root must be a mapping")

    root_dir = Path(__file__).resolve().parents[1]
    py_exec = "python"

    h2_batch_cfg = str(cfg.get("h2_batch", {}).get("config") or "").strip()
    h2_random_cfg = str(cfg.get("h2_random_baseline", {}).get("config") or "").strip()
    prompt_tables_cfg = str(cfg.get("prompt_tables", {}).get("config") or "").strip()
    if not h2_batch_cfg or not h2_random_cfg or not prompt_tables_cfg:
        raise ValueError("Config must include h2_batch.config, h2_random_baseline.config, and prompt_tables.config")

    print("[1/3] H2 batch")
    _run_step(
        [py_exec, str(root_dir / "scripts" / "run_h2_batch.py"), "--config", h2_batch_cfg],
        cwd=root_dir,
        dry_run=bool(args.dry_run),
    )

    print("[2/3] H2 random baseline")
    _run_step(
        [py_exec, str(root_dir / "scripts" / "run_h2_random_baseline.py"), "--config", h2_random_cfg],
        cwd=root_dir,
        dry_run=bool(args.dry_run),
    )

    print("[3/3] Per-prompt tables H1/H2")
    _run_step(
        [py_exec, str(root_dir / "scripts" / "build_prompt_tables.py"), "--config", prompt_tables_cfg],
        cwd=root_dir,
        dry_run=bool(args.dry_run),
    )

    print("Experimento 2 full pipeline completed.")


if __name__ == "__main__":
    main()
