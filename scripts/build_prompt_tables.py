"""
Build per-prompt H1/H2 tables from detailed JSON files via YAML config.
"""

from __future__ import annotations

import argparse

from embedding_models_eval.prompt_tables import (
    build_prompt_tables_from_config,
    load_prompt_tables_config,
    validate_prompt_tables_config,
)


def main() -> None:
    """CLI entrypoint for per-prompt table generation."""
    parser = argparse.ArgumentParser(description="Build per-prompt H1/H2 tables from YAML config")
    parser.add_argument("--config", required=True, help="Prompt table YAML config")
    args = parser.parse_args()

    cfg = load_prompt_tables_config(args.config)
    validate_prompt_tables_config(cfg)
    outputs = build_prompt_tables_from_config(cfg)

    print("Per-prompt tables generated.")
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
