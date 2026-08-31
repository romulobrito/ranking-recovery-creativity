"""
LLM-as-judge (GVALD) loaders and normalization under the embedding protocol.
"""

from .geval_loader import (
    PROTOCOL_SPECS,
    build_manifest,
    judge_slug_from_parquet,
    load_parquet_as_detailed_rows,
    resolve_score_column,
)

__all__ = [
    "PROTOCOL_SPECS",
    "build_manifest",
    "judge_slug_from_parquet",
    "load_parquet_as_detailed_rows",
    "resolve_score_column",
]
