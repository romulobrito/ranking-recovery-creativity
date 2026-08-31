"""
Load GVALD aggregated parquet files and convert scores into rank_pred rows.

Score rule: prefer creativity_score_mean when present; else creativity_score.
Winner rows are kept in detalhado.json and excluded later by metric code.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

# Root-relative default for GVALD dumps inside embedding_models_eval.
DEFAULT_GVALD_ROOT = Path("LLM-as-judge-data-geval")

# Timestamp suffix on aggregated parquet filenames: _YYYYMMDD_HHMMSS_aggregated
_PARQUET_SUFFIX_RE = re.compile(r"_\d{8}_\d{6}_aggregated\.parquet$")


@dataclass(frozen=True)
class ProtocolSpec:
    """One experimental prompt-condition cell."""

    protocol_id: str
    role: str
    # Substrings that must appear in the posix relative path (all required).
    path_must_contain: Tuple[str, ...]
    # Substrings that must NOT appear (any match rejects).
    path_must_not_contain: Tuple[str, ...] = ()


PROTOCOL_SPECS: Tuple[ProtocolSpec, ...] = (
    ProtocolSpec(
        protocol_id="tournament_description",
        role="primary",
        path_must_contain=("com descricao do torneio", "com descricao"),
        path_must_not_contain=("/experiments/",),
    ),
    ProtocolSpec(
        protocol_id="with_anchor_no_desc",
        role="ablation",
        path_must_contain=("com ancora", "sem descricao"),
        path_must_not_contain=("com descricao do torneio", "/experiments/"),
    ),
    ProtocolSpec(
        protocol_id="with_anchor_with_desc",
        role="ablation",
        path_must_contain=("com ancora", "com descricao"),
        path_must_not_contain=("com descricao do torneio", "sem descricao", "/experiments/"),
    ),
    ProtocolSpec(
        protocol_id="no_anchor_no_desc",
        role="ablation",
        path_must_contain=("sem ancora", "sem descricao"),
        path_must_not_contain=("com descricao do torneio", "/experiments/"),
    ),
    ProtocolSpec(
        protocol_id="no_anchor_with_desc",
        role="ablation",
        path_must_contain=("sem ancora", "com descricao"),
        path_must_not_contain=("com descricao do torneio", "sem descricao", "/experiments/"),
    ),
)


def judge_slug_from_parquet(path: Path) -> str:
    """
    Derive a stable judge slug from an aggregated parquet filename.

    Example: x-ai-grok-4.3_20260627_074741_aggregated.parquet -> x-ai-grok-4.3
    """
    name = path.name
    cleaned = _PARQUET_SUFFIX_RE.sub("", name)
    if cleaned.endswith(".parquet"):
        cleaned = cleaned[: -len(".parquet")]
    if not cleaned:
        raise ValueError(f"Cannot derive judge slug from {path}")
    return cleaned


def resolve_score_column(columns: Sequence[str]) -> str:
    """
    Prefer creativity_score_mean; fall back to creativity_score.

    Raises:
        ValueError: if neither column exists.
    """
    col_set = set(columns)
    if "creativity_score_mean" in col_set:
        return "creativity_score_mean"
    if "creativity_score" in col_set:
        return "creativity_score"
    raise ValueError(
        "Parquet must contain creativity_score_mean or creativity_score; "
        f"got columns={list(columns)}"
    )


def _path_matches_protocol(rel_posix: str, spec: ProtocolSpec) -> bool:
    """Return True if relative path matches a protocol cell."""
    for needle in spec.path_must_contain:
        if needle not in rel_posix:
            return False
    for banned in spec.path_must_not_contain:
        if banned in rel_posix:
            return False
    return True


def discover_protocol_parquets(
    geval_root: Path,
) -> Dict[str, List[Path]]:
    """
    Discover aggregated parquet files for each protocol_id.

    Prefers non-experiments paths via PROTOCOL_SPECS filters.
    Falls back to experiments/ copies only when a protocol has zero hits.
    """
    if not geval_root.is_dir():
        raise FileNotFoundError(f"GVALD root not found: {geval_root}")

    all_parquets = sorted(geval_root.rglob("*_aggregated.parquet"))
    by_protocol: Dict[str, List[Path]] = {spec.protocol_id: [] for spec in PROTOCOL_SPECS}

    for spec in PROTOCOL_SPECS:
        for pq in all_parquets:
            rel = pq.relative_to(geval_root).as_posix()
            if _path_matches_protocol(rel, spec):
                by_protocol[spec.protocol_id].append(pq)

    # Fallback: allow experiments/ if a cell is empty.
    for spec in PROTOCOL_SPECS:
        if by_protocol[spec.protocol_id]:
            continue
        relaxed = ProtocolSpec(
            protocol_id=spec.protocol_id,
            role=spec.role,
            path_must_contain=spec.path_must_contain,
            path_must_not_contain=tuple(
                b for b in spec.path_must_not_contain if b != "/experiments/"
            ),
        )
        for pq in all_parquets:
            rel = pq.relative_to(geval_root).as_posix()
            if _path_matches_protocol(rel, relaxed):
                by_protocol[spec.protocol_id].append(pq)

    # Deduplicate by judge slug (keep first / non-experiments preferred by sort).
    for protocol_id, paths in by_protocol.items():
        seen: Dict[str, Path] = {}
        for pq in paths:
            slug = judge_slug_from_parquet(pq)
            # Prefer paths without /experiments/
            if slug not in seen:
                seen[slug] = pq
            elif "/experiments/" in seen[slug].as_posix() and "/experiments/" not in pq.as_posix():
                seen[slug] = pq
        by_protocol[protocol_id] = sorted(seen.values(), key=lambda p: judge_slug_from_parquet(p))

    return by_protocol


def assign_rank_pred(
    df: pd.DataFrame,
    score_col: str,
    prompt_id_col: str = "prompt_id",
    story_url_col: str = "story_url",
    winner_rank_col: str = "rank_in_prompt",
    winner_rank_value: int = 1,
) -> pd.DataFrame:
    """
    Assign rank_pred within each prompt by descending score.

    Tie-break: story_url ascending (stable, deterministic).

    Rows with missing scores are allowed only for the human winner
    (rank_in_prompt == 1), which is common in with-anchor GVALD runs where
    the winner is used as reference and not scored. Those rows get
    rank_pred = NA and are excluded later by metric code.
    """
    if score_col not in df.columns:
        raise ValueError(f"Missing score column: {score_col}")
    required = {prompt_id_col, story_url_col, winner_rank_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work[winner_rank_col] = pd.to_numeric(work[winner_rank_col], errors="coerce")

    missing_score = work[score_col].isna()
    is_winner = work[winner_rank_col] == winner_rank_value
    illegal_missing = missing_score & ~is_winner
    if illegal_missing.any():
        n_bad = int(illegal_missing.sum())
        raise ValueError(
            f"Found {n_bad} non-winner rows with non-numeric/missing {score_col}"
        )

    work["rank_pred"] = pd.NA
    scored = work.loc[~missing_score].copy()
    scored = scored.sort_values(
        by=[prompt_id_col, score_col, story_url_col],
        ascending=[True, False, True],
        kind="mergesort",
    )
    scored["rank_pred"] = scored.groupby(prompt_id_col, sort=False).cumcount() + 1
    work.loc[scored.index, "rank_pred"] = scored["rank_pred"].astype("Int64")
    return work


def load_parquet_as_detailed_rows(
    parquet_path: Path,
    protocol_id: str,
    judge_model: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load one aggregated parquet and return detalhado rows plus metadata.

    Returns:
        (rows, metadata) where rows match the H2 detalhado schema minimum.
        Unscored winners have rank_pred = null.
    """
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    score_col = resolve_score_column(list(df.columns))
    ranked = assign_rank_pred(df, score_col=score_col)

    required = {"prompt_id", "story_url", "likes", "rank_in_prompt", "rank_pred"}
    missing = sorted(required - set(ranked.columns))
    if missing:
        raise ValueError(f"Parquet missing columns after ranking: {missing}")

    slug = judge_model or judge_slug_from_parquet(parquet_path)
    rows: List[Dict[str, Any]] = []
    n_null_rank_pred = 0
    for _, r in ranked.iterrows():
        rank_pred_val = r["rank_pred"]
        if pd.isna(rank_pred_val):
            rank_pred_out: Optional[int] = None
            n_null_rank_pred += 1
        else:
            rank_pred_out = int(rank_pred_val)
        rows.append(
            {
                "prompt_id": str(r["prompt_id"]),
                "story_url": str(r["story_url"]),
                "likes": float(r["likes"]),
                "rank_in_prompt": int(r["rank_in_prompt"]),
                "rank_pred": rank_pred_out,
            }
        )

    metadata: Dict[str, Any] = {
        "protocol_id": protocol_id,
        "judge_model": slug,
        "score_column": score_col,
        "source_parquet": str(parquet_path),
        "n_rows": len(rows),
        "n_prompts": int(ranked["prompt_id"].nunique()),
        "n_winners": int((ranked["rank_in_prompt"] == 1).sum()),
        "n_null_rank_pred": n_null_rank_pred,
        "has_creativity_score_mean": "creativity_score_mean" in df.columns,
        "has_creativity_score": "creativity_score" in df.columns,
        "n_runs_unique": (
            sorted(pd.to_numeric(df["n_runs"], errors="coerce").dropna().unique().tolist())
            if "n_runs" in df.columns
            else []
        ),
    }
    return rows, metadata


def build_manifest(
    geval_root: Path,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Build a versioned inventory of protocol cells and judge parquets.
    """
    by_protocol = discover_protocol_parquets(geval_root)
    role_by_id = {spec.protocol_id: spec.role for spec in PROTOCOL_SPECS}

    protocols: List[Dict[str, Any]] = []
    for protocol_id, paths in by_protocol.items():
        judges: List[Dict[str, Any]] = []
        for pq in paths:
            try:
                df = pd.read_parquet(pq, columns=None)
                score_col = resolve_score_column(list(df.columns))
                n_prompts = int(df["prompt_id"].nunique()) if "prompt_id" in df.columns else 0
                n_stories = int(len(df))
            except Exception as exc:  # noqa: BLE001 - inventory must continue
                judges.append(
                    {
                        "judge_model": judge_slug_from_parquet(pq),
                        "source_parquet": str(pq),
                        "error": str(exc),
                    }
                )
                continue
            judges.append(
                {
                    "judge_model": judge_slug_from_parquet(pq),
                    "source_parquet": str(pq),
                    "score_column": score_col,
                    "n_prompts": n_prompts,
                    "n_stories": n_stories,
                }
            )
        protocols.append(
            {
                "protocol_id": protocol_id,
                "role": role_by_id[protocol_id],
                "n_judges": len(judges),
                "judges": judges,
            }
        )

    return {
        "geval_root": str(geval_root),
        "project_root": str(project_root) if project_root is not None else None,
        "k_values": [1, 2, 3, 4, 5, 6, 7],
        "num_permutations": 2500,
        "seed": 42,
        "score_rule": "creativity_score_mean if present else creativity_score",
        "exclude_winner": True,
        "protocols": protocols,
    }


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with ASCII-only escaping disabled for paths; ensure parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def write_detailed_json(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write detalhado.json rows."""
    write_json(path, list(rows))


def protocol_registry_rows(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten manifest protocols into registry CSV rows."""
    rows: List[Dict[str, Any]] = []
    for proto in manifest.get("protocols", []):
        rows.append(
            {
                "protocol_id": proto["protocol_id"],
                "role": proto["role"],
                "n_judges": proto["n_judges"],
                "judge_models": "|".join(
                    j["judge_model"] for j in proto.get("judges", []) if "judge_model" in j
                ),
            }
        )
    return rows
