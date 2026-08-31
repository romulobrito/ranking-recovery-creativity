"""
Random permutation baseline for topn_real_ratio.

Reads *_detalhado.json outputs, recomputes observed top-N direct ratios,
and estimates a matched random baseline by shuffling predicted vote order
per prompt (same protocol as feasible_range_random_baseline).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

RatioTriple = Tuple[float, float, float]
PromptVotes = Dict[str, List[float]]


def _safe_div(observed: float, reference: float, clip_0_1: bool = True) -> float:
    """Safe observed/reference ratio with optional clipping to [0, 1]."""
    if reference == 0:
        return 1.0 if observed == 0 else 0.0
    ratio = observed / reference
    if clip_0_1:
        return float(np.clip(ratio, 0.0, 1.0))
    return float(ratio)


def compute_topn_real_ratio_for_votes(
    votes_pred_order: Sequence[float],
    k: int,
    *,
    clip_0_1: bool = True,
) -> RatioTriple:
    """
    Compute (ratio_max, ratio_mean, ratio_min) for one prompt at cutoff k.

    Real top-N reference is the descending vote order from the same pool.
    """
    if len(votes_pred_order) == 0:
        raise ValueError("votes_pred_order cannot be empty")

    votes_real_order = sorted(votes_pred_order, reverse=True)
    total = len(votes_pred_order)
    n_eff = min(int(k), total)

    top_pred = list(votes_pred_order[:n_eff])
    top_real = votes_real_order[:n_eff]

    obs_max = float(max(top_pred))
    obs_mean = float(sum(top_pred) / n_eff)
    obs_min = float(min(top_pred))

    ref_max = float(max(top_real))
    ref_mean = float(sum(top_real) / n_eff)
    ref_min = float(min(top_real))

    ratio_max = _safe_div(obs_max, ref_max, clip_0_1=clip_0_1)
    ratio_mean = _safe_div(obs_mean, ref_mean, clip_0_1=clip_0_1)
    ratio_min = _safe_div(obs_min, ref_min, clip_0_1=clip_0_1)
    return ratio_max, ratio_mean, ratio_min


def extract_prompt_votes_from_detailed_rows(
    rows: Iterable[dict],
    prompt_id_col: str = "prompt_id",
    votes_col: str = "likes",
    rank_pred_col: str = "rank_pred",
    winner_rank_col: str = "rank_in_prompt",
    winner_rank_value: int = 1,
) -> PromptVotes:
    """
    Extract predicted-order vote lists per prompt, excluding the winner.
    """
    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        prompt_id = row.get(prompt_id_col)
        if prompt_id is None:
            continue
        grouped.setdefault(str(prompt_id), []).append(row)

    prompt_votes: PromptVotes = {}
    for pid, group in grouped.items():
        filtered: List[dict] = []
        for row in group:
            if winner_rank_col in row and row.get(winner_rank_col) == winner_rank_value:
                continue
            if row.get(rank_pred_col) is None:
                continue
            if row.get(votes_col) is None:
                continue
            filtered.append(row)

        if not filtered:
            continue

        filtered.sort(key=lambda x: float(x[rank_pred_col]))
        votes = [float(row[votes_col]) for row in filtered]
        if votes:
            prompt_votes[pid] = votes

    return prompt_votes


def compute_observed_macro(prompt_votes: PromptVotes, k_values: Sequence[int]) -> Dict[str, float]:
    """Compute macro means of topn_real_ratio metrics."""
    acc: Dict[str, List[float]] = {}

    for votes in prompt_votes.values():
        for k in k_values:
            rmax, rmean, rmin = compute_topn_real_ratio_for_votes(votes, int(k))
            acc.setdefault(f"ratio_max@{k}", []).append(rmax)
            acc.setdefault(f"ratio_mean@{k}", []).append(rmean)
            acc.setdefault(f"ratio_min@{k}", []).append(rmin)

    out: Dict[str, float] = {}
    for key, vals in acc.items():
        out[f"{key}_mean"] = float(np.mean(vals)) if vals else 0.0
    return out


def compute_random_baseline_summary(
    prompt_votes: PromptVotes,
    k_values: Sequence[int],
    num_permutations: int,
    seed: int,
    percentiles: Sequence[int],
) -> Dict[str, Dict[str, float]]:
    """Estimate random baseline distribution via per-prompt vote shuffles."""
    rng = random.Random(seed)
    runs: Dict[str, List[float]] = {}

    prompts = list(prompt_votes.values())
    for _ in range(int(num_permutations)):
        per_run_acc: Dict[str, List[float]] = {}
        for votes in prompts:
            shuffled = list(votes)
            rng.shuffle(shuffled)
            for k in k_values:
                rmax, rmean, rmin = compute_topn_real_ratio_for_votes(shuffled, int(k))
                per_run_acc.setdefault(f"ratio_max@{k}", []).append(rmax)
                per_run_acc.setdefault(f"ratio_mean@{k}", []).append(rmean)
                per_run_acc.setdefault(f"ratio_min@{k}", []).append(rmin)

        for metric_key, vals in per_run_acc.items():
            runs.setdefault(metric_key, []).append(float(np.mean(vals)))

    summary: Dict[str, Dict[str, float]] = {}
    for metric_key, vals in runs.items():
        arr = np.array(vals, dtype=float)
        row: Dict[str, float] = {
            "random_mean": float(np.mean(arr)),
            "random_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        }
        for p in percentiles:
            row[f"p{int(p)}"] = float(np.percentile(arr, float(p)))
        summary[metric_key] = row
    return summary


def summarize_model(
    model_name: str,
    detailed_json_path: str,
    k_values: Sequence[int],
    num_permutations: int,
    seed: int,
    percentiles: Sequence[int],
) -> List[Dict[str, object]]:
    """Generate long-format summary rows for one model."""
    with open(detailed_json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    prompt_votes = extract_prompt_votes_from_detailed_rows(rows)
    observed = compute_observed_macro(prompt_votes, k_values)
    random_summary = compute_random_baseline_summary(
        prompt_votes=prompt_votes,
        k_values=k_values,
        num_permutations=num_permutations,
        seed=seed,
        percentiles=percentiles,
    )

    out_rows: List[Dict[str, object]] = []
    for k in k_values:
        for metric_prefix in ("ratio_max", "ratio_mean", "ratio_min"):
            metric_key = f"{metric_prefix}@{k}"
            obs_key = f"{metric_key}_mean"
            rand = random_summary[metric_key]

            row: Dict[str, object] = {
                "model": model_name,
                "k": int(k),
                "metric": metric_prefix,
                "observed_mean": float(observed[obs_key]),
                "random_mean": float(rand["random_mean"]),
                "random_std": float(rand["random_std"]),
                "n_prompts": len(prompt_votes),
                "num_permutations": int(num_permutations),
                "seed": int(seed),
            }
            for p in percentiles:
                p_key = f"p{int(p)}"
                row[p_key] = float(rand[p_key])
            if "p95" in row:
                row["above_p95"] = bool(float(row["observed_mean"]) > float(row["p95"]))
            out_rows.append(row)
    return out_rows


def write_csv(rows: Sequence[Dict[str, object]], output_csv: str) -> None:
    """Write summary rows to CSV."""
    if not rows:
        raise ValueError("No rows to write to CSV")
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_percentiles(raw: str) -> List[int]:
    """Parse comma-separated percentile list."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    vals = [int(p) for p in parts]
    for p in vals:
        if p < 0 or p > 100:
            raise ValueError("Percentiles must be in [0, 100]")
    return vals


def parse_k_values(raw: str) -> List[int]:
    """Parse comma-separated k values."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    vals = [int(p) for p in parts]
    if not vals:
        raise ValueError("k_values cannot be empty")
    if any(v <= 0 for v in vals):
        raise ValueError("k_values must be positive")
    return vals


def main() -> None:
    """CLI entrypoint for topn_real_ratio random baseline."""
    parser = argparse.ArgumentParser(
        description="Random permutation baseline for topn_real_ratio"
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Path to *_detalhado.json (repeat per model)",
    )
    parser.add_argument(
        "--model-name",
        action="append",
        default=[],
        help="Model name for each --input (same order)",
    )
    parser.add_argument(
        "--k-values",
        default="1,2,3,4,5,6,7",
        help="Comma-separated cutoffs",
    )
    parser.add_argument(
        "--num-permutations",
        type=int,
        default=2500,
        help="Number of random permutations per model",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global seed")
    parser.add_argument(
        "--percentiles",
        default="95",
        help="Comma-separated percentiles",
    )
    parser.add_argument("--output-csv", required=True, help="Output CSV path")
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional JSON output (defaults to CSV with .json suffix)",
    )
    args = parser.parse_args()

    k_values = parse_k_values(args.k_values)
    percentiles = parse_percentiles(args.percentiles)

    input_paths = args.input
    model_names = args.model_name
    if model_names and len(model_names) != len(input_paths):
        raise ValueError("--model-name must match --input count")

    rows: List[Dict[str, object]] = []
    for idx, input_path in enumerate(input_paths):
        if model_names:
            model_name = model_names[idx]
        else:
            stem = Path(input_path).stem
            model_name = stem.replace("_detalhado", "").replace("resultados_h2_detalhado", "")

        model_rows = summarize_model(
            model_name=model_name,
            detailed_json_path=input_path,
            k_values=k_values,
            num_permutations=int(args.num_permutations),
            seed=int(args.seed) + idx,
            percentiles=percentiles,
        )
        rows.extend(model_rows)

    write_csv(rows, args.output_csv)

    json_output = args.output_json or str(Path(args.output_csv).with_suffix(".json"))
    Path(json_output).parent.mkdir(parents=True, exist_ok=True)
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=True)

    print(f"CSV gerado: {args.output_csv}")
    print(f"JSON gerado: {json_output}")
    print(f"Linhas: {len(rows)}")


if __name__ == "__main__":
    main()
