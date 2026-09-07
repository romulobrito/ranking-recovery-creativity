"""
CLI: normalize GVALD LLM-judge rankings under the embedding H1/H2 protocol.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _default_project_root() -> Path:
    """Resolve embedding_models_eval root from this script location."""
    return Path(__file__).resolve().parents[1]


def main() -> None:
    """Parse CLI args and run the normalization pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Normalize GVALD LLM-judge rankings with feasible_range_ratio, "
            "topn_real_ratio, and matched random baselines (N=1..7)."
        )
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(_default_project_root()),
        help="Path to embedding_models_eval project root",
    )
    parser.add_argument(
        "--geval-root",
        type=str,
        default=None,
        help="Path to LLM-as-judge-data-geval (default: <project-root>/LLM-as-judge-data-geval)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Output directory (default: <project-root>/results/llm_judge_normalized)",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="all",
        help=(
            "Protocol id or 'all'. Valid: tournament_description, with_anchor_no_desc, "
            "with_anchor_with_desc, no_anchor_no_desc, no_anchor_with_desc"
        ),
    )
    parser.add_argument(
        "--num-permutations",
        type=int,
        default=2500,
        help="Random baseline permutations (default: 2500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed (default: 42)",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip random baselines (smoke / observed-only)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation gates on existing artifacts",
    )
    args = parser.parse_args()

    # Ensure src is importable when run as a script.
    project_root = Path(args.project_root).resolve()
    src = project_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from embedding_models_eval.llm_judge.pipeline import run_pipeline, validate_pipeline

    if args.validate_only:
        report = validate_pipeline(project_root=project_root)
        print(json_dumps(report))
        if not report.get("ok", False):
            raise SystemExit(1)
        return

    protocol_filter = None
    if args.protocol != "all":
        protocol_filter = [args.protocol]

    summary = run_pipeline(
        project_root=project_root,
        geval_root=Path(args.geval_root) if args.geval_root else None,
        output_root=Path(args.output_root) if args.output_root else None,
        protocol_filter=protocol_filter,
        num_permutations=int(args.num_permutations),
        seed=int(args.seed),
        skip_baseline=bool(args.skip_baseline),
    )
    print(json_dumps(summary))

    report = validate_pipeline(project_root=project_root)
    print(json_dumps(report))
    if not report.get("ok", False):
        raise SystemExit(1)


def json_dumps(payload: object) -> str:
    """Serialize payload as ASCII JSON for console output."""
    import json

    return json.dumps(payload, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
