"""
Build per-prompt tables for H1 and H2 under a unified schema.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd
import yaml


@dataclass(frozen=True)
class PromptTableInput:
    """Detailed input file for one hypothesis/model pair."""

    hypothesis: str
    model: str
    detailed_json: str


def load_prompt_tables_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration for per-prompt table generation."""
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Prompt table config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Prompt table config root must be a mapping")
    cfg["_config_dir"] = str(cfg_path.parent)
    return cfg


def validate_prompt_tables_config(cfg: Dict[str, Any]) -> None:
    """Validate required schema for per-prompt table generation."""
    protocol = cfg.get("protocol")
    inputs = cfg.get("inputs")
    output = cfg.get("output")

    if not isinstance(protocol, dict):
        raise ValueError("protocol section is required")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("inputs section must be a non-empty list")
    if not isinstance(output, dict):
        raise ValueError("output section is required")

    k_values = protocol.get("k_values")
    if not isinstance(k_values, list) or not k_values:
        raise ValueError("protocol.k_values must be a non-empty list")
    if any(int(k) <= 0 for k in k_values):
        raise ValueError("protocol.k_values must contain positive integers")

    if not bool(protocol.get("exclude_winner", True)):
        raise ValueError("protocol.exclude_winner must be true for H1/H2 comparability")

    output_dir = str(output.get("out_dir") or "").strip()
    if not output_dir:
        raise ValueError("output.out_dir is required")

    for idx, item in enumerate(inputs):
        if not isinstance(item, dict):
            raise ValueError(f"inputs[{idx}] must be a mapping")
        hypothesis = str(item.get("hypothesis") or "").strip()
        model = str(item.get("model") or "").strip()
        detailed_json = str(item.get("detailed_json") or "").strip()
        if hypothesis not in {"H1", "H2"}:
            raise ValueError(f"inputs[{idx}].hypothesis must be H1 or H2")
        if not model:
            raise ValueError(f"inputs[{idx}].model is required")
        if not detailed_json:
            raise ValueError(f"inputs[{idx}].detailed_json is required")


def build_prompt_table_inputs(cfg: Dict[str, Any]) -> List[PromptTableInput]:
    """Convert config input items to PromptTableInput dataclasses."""
    out: List[PromptTableInput] = []
    for item in cfg["inputs"]:
        out.append(
            PromptTableInput(
                hypothesis=str(item["hypothesis"]).strip(),
                model=str(item["model"]).strip(),
                detailed_json=str(item["detailed_json"]).strip(),
            )
        )
    return out


def _coerce_float(value: Any) -> float | None:
    """Safely parse numeric values from row fields."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _coerce_int(value: Any) -> int | None:
    """Safely parse integer values from row fields."""
    parsed = _coerce_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _row_prompt_id(row: Dict[str, Any]) -> str:
    """Resolve prompt id from row, with fallback to contest/prompt URL tuple."""
    prompt_id = str(row.get("prompt_id") or "").strip()
    if prompt_id:
        return prompt_id
    contest_number = str(row.get("contest_number") or "").strip()
    context_prompt_url = str(row.get("context_prompt_url") or "").strip()
    return f"{contest_number}::{context_prompt_url}"


def _group_rows_by_prompt(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group detailed rows by prompt id."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        pid = _row_prompt_id(row)
        grouped.setdefault(pid, []).append(row)
    return grouped


def _eligible_rows(rows: Sequence[Dict[str, Any]], exclude_winner: bool) -> List[Dict[str, Any]]:
    """Filter rows to valid candidates for top-N stats."""
    out: List[Dict[str, Any]] = []
    for row in rows:
        if exclude_winner and _coerce_int(row.get("rank_in_prompt")) == 1:
            continue
        rank_pred = _coerce_int(row.get("rank_pred"))
        likes = _coerce_float(row.get("likes"))
        if rank_pred is None or likes is None:
            continue
        out.append(row)
    out.sort(key=lambda r: (_coerce_int(r.get("rank_pred")) or 10**9, str(r.get("story_url") or "")))
    return out


def _reference_real_votes(rows: Sequence[Dict[str, Any]], exclude_winner: bool) -> float:
    """Compute reference 'Real' votes (best remaining under winner exclusion)."""
    votes: List[float] = []
    for row in rows:
        if exclude_winner and _coerce_int(row.get("rank_in_prompt")) == 1:
            continue
        likes = _coerce_float(row.get("likes"))
        if likes is None:
            continue
        votes.append(likes)
    if not votes:
        return 0.0
    return float(max(votes))


def compute_prompt_rows(
    input_item: PromptTableInput,
    rows: Sequence[Dict[str, Any]],
    k_values: Sequence[int],
    exclude_winner: bool,
) -> List[Dict[str, object]]:
    """Build long-format per-prompt rows for one model/hypothesis detailed file."""
    grouped = _group_rows_by_prompt(rows)
    out: List[Dict[str, object]] = []

    for prompt_id, prompt_rows in grouped.items():
        eligible = _eligible_rows(prompt_rows, exclude_winner=exclude_winner)
        if not eligible:
            continue

        real_votes = _reference_real_votes(prompt_rows, exclude_winner=exclude_winner)
        for k in k_values:
            k_eff = min(int(k), len(eligible))
            top_rows = eligible[:k_eff]
            top_votes = [float(_coerce_float(r.get("likes")) or 0.0) for r in top_rows]
            top1_votes = top_votes[0] if top_votes else 0.0
            top_max = max(top_votes) if top_votes else 0.0
            top_mean = (sum(top_votes) / len(top_votes)) if top_votes else 0.0
            top_min = min(top_votes) if top_votes else 0.0
            votos_real = (top1_votes / real_votes) if real_votes > 0.0 else 0.0

            first = prompt_rows[0]
            out.append(
                {
                    "hypothesis": input_item.hypothesis,
                    "model": input_item.model,
                    "prompt_id": prompt_id,
                    "contest_number": str(first.get("contest_number") or ""),
                    "context_prompt_url": str(first.get("context_prompt_url") or ""),
                    "N": int(k),
                    "TOPN_max": float(top_max),
                    "TOPN_media": float(top_mean),
                    "TOPN_min": float(top_min),
                    "Real": float(real_votes),
                    "Votos_Real": float(votos_real),
                    "Votos_TOP1_Best": float(votos_real),  # explicit alias for H2 checklist item
                }
            )
    return out


def build_prompt_tables_from_config(cfg: Dict[str, Any]) -> Dict[str, Path]:
    """Generate per-prompt tables for all configured inputs."""
    protocol = cfg["protocol"]
    k_values = [int(v) for v in protocol["k_values"]]
    exclude_winner = bool(protocol.get("exclude_winner", True))

    config_dir = Path(str(cfg.get("_config_dir") or ".")).expanduser().resolve()
    # Most project configs live under <project_root>/configs; prefer project root for relative paths.
    project_root = config_dir.parent if config_dir.name == "configs" else config_dir

    inputs = build_prompt_table_inputs(cfg)
    all_rows: List[Dict[str, object]] = []
    for item in inputs:
        detailed_path = Path(item.detailed_json).expanduser()
        if not detailed_path.is_absolute():
            detailed_path = (project_root / detailed_path).resolve()
        else:
            detailed_path = detailed_path.resolve()
        if not detailed_path.is_file():
            raise FileNotFoundError(f"Detailed file not found: {detailed_path}")
        with detailed_path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f"Detailed file must contain a list of rows: {detailed_path}")
        all_rows.extend(
            compute_prompt_rows(
                input_item=item,
                rows=rows,
                k_values=k_values,
                exclude_winner=exclude_winner,
            )
        )

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise ValueError("No per-prompt rows were generated")

    out_dir = Path(str(cfg["output"]["out_dir"])).expanduser()
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()
    else:
        out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_csv = out_dir / "tabelas_por_prompt_h1_h2.csv"
    h1_csv = out_dir / "tabelas_por_prompt_h1.csv"
    h2_csv = out_dir / "tabelas_por_prompt_h2.csv"
    h2_top1_csv = out_dir / "tabelas_por_prompt_h2_top1_best.csv"

    df.to_csv(combined_csv, index=False)
    df[df["hypothesis"] == "H1"].to_csv(h1_csv, index=False)
    df[df["hypothesis"] == "H2"].to_csv(h2_csv, index=False)
    df[df["hypothesis"] == "H2"][
        [
            "hypothesis",
            "model",
            "prompt_id",
            "contest_number",
            "context_prompt_url",
            "N",
            "Votos_TOP1_Best",
        ]
    ].to_csv(h2_top1_csv, index=False)

    return {
        "combined_csv": combined_csv,
        "h1_csv": h1_csv,
        "h2_csv": h2_csv,
        "h2_top1_csv": h2_top1_csv,
    }
