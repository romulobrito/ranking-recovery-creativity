"""
End-to-end normalization pipeline for GVALD LLM judges.

Produces observed metrics, random baselines, and latex_ready tables under
results/llm_judge_normalized/.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from embedding_models_eval.llm_judge.config_writer import write_all_baseline_configs
from embedding_models_eval.llm_judge.geval_loader import (
    PROTOCOL_SPECS,
    build_manifest,
    discover_protocol_parquets,
    judge_slug_from_parquet,
    load_parquet_as_detailed_rows,
    protocol_registry_rows,
    write_detailed_json,
    write_json,
)
from embedding_models_eval.metrics.feasible_range_random_baseline import (
    summarize_model as summarize_feasible_baseline,
)
from embedding_models_eval.metrics.feasible_range_ratio import FeasibleRangeRatioMetrics
from embedding_models_eval.metrics.topn_real_random_baseline import (
    summarize_model as summarize_topn_baseline,
)
from embedding_models_eval.metrics.topn_real_ratio import TopNRealRatioMetrics

K_VALUES = [1, 2, 3, 4, 5, 6, 7]
DEFAULT_PERMUTATIONS = 2500
DEFAULT_SEED = 42
DEFAULT_PERCENTILES = [95]
PRIMARY_PROTOCOL = "tournament_description"


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write list-of-dict rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _macro_to_long_rows(
    model_name: str,
    macro: Dict[str, float],
    k_values: Sequence[int],
) -> List[Dict[str, Any]]:
    """Convert metric macro dict into long-format rows."""
    rows: List[Dict[str, Any]] = []
    for k in k_values:
        for prefix in ("ratio_max", "ratio_mean", "ratio_min"):
            mean_key = f"{prefix}@{k}_mean"
            std_key = f"{prefix}@{k}_std"
            if mean_key not in macro:
                continue
            rows.append(
                {
                    "model": model_name,
                    "k": int(k),
                    "metric": prefix,
                    "observed_mean": float(macro[mean_key]),
                    "observed_std": float(macro.get(std_key, 0.0)),
                }
            )
    return rows


def compute_observed_for_detailed(
    detailed_json: Path,
    model_name: str,
    k_values: Sequence[int] = K_VALUES,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """
    Compute feasible_range and topn_real observed metrics from detalhado.json.

    Returns:
        feasible_long, topn_long, feasible_per_prompt, topn_per_prompt
    """
    with open(detailed_json, "r", encoding="utf-8") as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)

    feas = FeasibleRangeRatioMetrics(k_values=list(k_values)).compute(df)
    topn = TopNRealRatioMetrics(k_values=list(k_values), exclude_winner=True).compute(df)

    feas_long = _macro_to_long_rows(model_name, feas["macro"], k_values)
    topn_long = _macro_to_long_rows(model_name, topn["macro"], k_values)
    return feas_long, topn_long, feas["per_prompt"], topn["per_prompt"]


def _parse_bool(value: object) -> bool:
    """Parse bool from CSV/JSON-ish values without treating 'False' as True."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _above_p95_table(baseline_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build per-N table of models above P95 on ratio_mean.
    """
    by_k: Dict[int, List[str]] = {}
    all_models: set[str] = set()
    for row in baseline_rows:
        if str(row.get("metric")) != "ratio_mean":
            continue
        model = str(row["model"])
        all_models.add(model)
        k = int(row["k"])
        above = _parse_bool(row.get("above_p95", False))
        by_k.setdefault(k, [])
        if above:
            by_k[k].append(model)

    n_models = len(all_models)
    out: List[Dict[str, Any]] = []
    for k in sorted(by_k.keys()) or K_VALUES:
        models = sorted(set(by_k.get(k, [])))
        out.append(
            {
                "k": k,
                "n_above": len(models),
                "n_models": n_models,
                "count_label": f"{len(models)}/{n_models}",
                "models": "|".join(models) if models else "(none)",
            }
        )
    # Ensure all k present even if empty.
    present = {int(r["k"]) for r in out}
    for k in K_VALUES:
        if k not in present:
            out.append(
                {
                    "k": k,
                    "n_above": 0,
                    "n_models": n_models,
                    "count_label": f"0/{n_models}",
                    "models": "(none)",
                }
            )
    out.sort(key=lambda r: int(r["k"]))
    return out


def _chance_by_n_table(baseline_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate chance-calibration summary across models for ratio_mean by N.
    """
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    for row in baseline_rows:
        if str(row.get("metric")) != "ratio_mean":
            continue
        buckets.setdefault(int(row["k"]), []).append(row)

    out: List[Dict[str, Any]] = []
    for k in sorted(buckets.keys()):
        group = buckets[k]
        n_models = len(group)
        n_above = sum(1 for r in group if _parse_bool(r.get("above_p95", False)))
        out.append(
            {
                "k": k,
                "obs_mean": float(sum(float(r["observed_mean"]) for r in group) / n_models),
                "rand_mean": float(sum(float(r["random_mean"]) for r in group) / n_models),
                "p95_mean": float(sum(float(r["p95"]) for r in group) / n_models),
                "models_above_p95": f"{n_above}/{n_models}",
            }
        )
    return out


def _n3_calibrated_table(baseline_rows: Sequence[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
    """Build obs/rand/P95/delta table at a fixed k for ratio_mean."""
    out: List[Dict[str, Any]] = []
    for row in baseline_rows:
        if str(row.get("metric")) != "ratio_mean":
            continue
        if int(row["k"]) != int(k):
            continue
        obs = float(row["observed_mean"])
        p95 = float(row["p95"])
        out.append(
            {
                "model": row["model"],
                "observed_mean": obs,
                "random_mean": float(row["random_mean"]),
                "p95": p95,
                "delta_to_p95": obs - p95,
                "above_p95": _parse_bool(row.get("above_p95", False)),
            }
        )
    out.sort(key=lambda r: str(r["model"]))
    return out


def run_pipeline(
    project_root: Path,
    geval_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
    protocol_filter: Optional[Sequence[str]] = None,
    num_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    skip_baseline: bool = False,
    embed_h2_baseline_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run full GVALD normalization pipeline.

    Args:
        project_root: embedding_models_eval root.
        geval_root: path to LLM-as-judge-data-geval.
        output_root: results/llm_judge_normalized.
        protocol_filter: optional subset of protocol_ids.
        num_permutations: random baseline permutations.
        seed: base RNG seed.
        skip_baseline: if True, skip expensive random baselines.
        embed_h2_baseline_csv: optional H2 embedding baseline for cross-family table.

    Returns:
        Summary dict with paths and counts.
    """
    project_root = project_root.resolve()
    if geval_root is None:
        geval_root = project_root / "LLM-as-judge-data-geval"
    else:
        geval_root = geval_root.resolve()
    if output_root is None:
        output_root = project_root / "results" / "llm_judge_normalized"
    else:
        output_root = output_root.resolve()

    allowed = {spec.protocol_id for spec in PROTOCOL_SPECS}
    if protocol_filter is not None:
        wanted = set(protocol_filter)
        unknown = wanted - allowed
        if unknown:
            raise ValueError(f"Unknown protocol_id(s): {sorted(unknown)}")
    else:
        wanted = allowed

    output_root.mkdir(parents=True, exist_ok=True)

    # --- Manifest / registry ---
    manifest = build_manifest(geval_root=geval_root, project_root=project_root)
    write_json(output_root / "manifest.json", manifest)
    _write_csv(output_root / "protocol_registry.csv", protocol_registry_rows(manifest))

    readme = "\n".join(
        [
            "LLM judge normalization under embedding H1/H2 protocol",
            "",
            "Score: creativity_score_mean if present else creativity_score",
            "Winner exclusion: rank_in_prompt == 1 (at metric time)",
            f"k_values: {K_VALUES}",
            f"num_permutations: {num_permutations}",
            f"seed: {seed}",
            f"primary_protocol: {PRIMARY_PROTOCOL}",
            "",
            "Regenerate:",
            "  python code/run_llm_judge_normalized_eval.py --project-root .",
            "",
        ]
    )
    (output_root / "README.md").write_text(readme, encoding="utf-8")

    by_protocol = discover_protocol_parquets(geval_root)
    model_index = 0
    summary: Dict[str, Any] = {"protocols": {}, "output_root": str(output_root)}

    for protocol_id in [s.protocol_id for s in PROTOCOL_SPECS]:
        if protocol_id not in wanted:
            continue
        paths = by_protocol.get(protocol_id, [])
        proto_dir = output_root / "by_protocol" / protocol_id
        judges_dir = proto_dir / "judges"
        judges_dir.mkdir(parents=True, exist_ok=True)

        observed_feas_all: List[Dict[str, Any]] = []
        observed_topn_all: List[Dict[str, Any]] = []
        baseline_feas_all: List[Dict[str, Any]] = []
        baseline_topn_all: List[Dict[str, Any]] = []

        for pq in paths:
            slug = judge_slug_from_parquet(pq)
            judge_dir = judges_dir / slug
            judge_dir.mkdir(parents=True, exist_ok=True)

            rows, meta = load_parquet_as_detailed_rows(pq, protocol_id=protocol_id, judge_model=slug)
            detailed_path = judge_dir / "detalhado.json"
            write_detailed_json(detailed_path, rows)
            write_json(judge_dir / "metadata.json", meta)

            feas_long, topn_long, feas_pp, topn_pp = compute_observed_for_detailed(
                detailed_path, model_name=slug, k_values=K_VALUES
            )
            _write_csv(judge_dir / "observed_feasible_range.csv", feas_long)
            _write_csv(judge_dir / "observed_topn_real_ratio.csv", topn_long)
            feas_pp.to_csv(judge_dir / "per_prompt_feasible_range.csv", index=False)
            topn_pp.to_csv(judge_dir / "per_prompt_topn_real_ratio.csv", index=False)

            observed_feas_all.extend(feas_long)
            observed_topn_all.extend(topn_long)

            if not skip_baseline:
                feas_base = summarize_feasible_baseline(
                    model_name=slug,
                    detailed_json_path=str(detailed_path),
                    k_values=K_VALUES,
                    num_permutations=num_permutations,
                    seed=seed + model_index,
                    percentiles=DEFAULT_PERCENTILES,
                )
                topn_base = summarize_topn_baseline(
                    model_name=slug,
                    detailed_json_path=str(detailed_path),
                    k_values=K_VALUES,
                    num_permutations=num_permutations,
                    seed=seed + model_index,
                    percentiles=DEFAULT_PERCENTILES,
                )
                baseline_feas_all.extend(feas_base)
                baseline_topn_all.extend(topn_base)
            model_index += 1

        _write_csv(proto_dir / "summary_observed_feasible_range.csv", observed_feas_all)
        _write_csv(proto_dir / "summary_observed_topn_real_ratio.csv", observed_topn_all)

        if not skip_baseline:
            feas_base_dir = proto_dir / "random_baseline_feasible_2500"
            topn_base_dir = proto_dir / "random_baseline_topn_2500"
            feas_base_dir.mkdir(parents=True, exist_ok=True)
            topn_base_dir.mkdir(parents=True, exist_ok=True)
            _write_csv(feas_base_dir / "summary.csv", baseline_feas_all)
            write_json(feas_base_dir / "summary.json", baseline_feas_all)
            _write_csv(topn_base_dir / "summary.csv", baseline_topn_all)
            write_json(topn_base_dir / "summary.json", baseline_topn_all)

            above_feas = _above_p95_table(baseline_feas_all)
            above_topn = _above_p95_table(baseline_topn_all)
            _write_csv(proto_dir / "above_p95_models_feasible.csv", above_feas)
            _write_csv(proto_dir / "above_p95_models_topn.csv", above_topn)

        summary["protocols"][protocol_id] = {
            "n_judges": len(paths),
            "judges": [judge_slug_from_parquet(p) for p in paths],
            "skip_baseline": skip_baseline,
        }

    # --- latex_ready tables ---
    if not skip_baseline:
        _build_latex_ready(
            output_root=output_root,
            embed_h2_baseline_csv=embed_h2_baseline_csv
            or (project_root / "results" / "experimento2_h2_random_baseline_2500" / "summary.csv"),
        )

    write_json(output_root / "run_summary.json", summary)

    # Persist YAML configs for reproducibility (even when baselines were skipped).
    configs_dir = project_root / "configs" / "llm_judge_normalized"
    written_cfgs = write_all_baseline_configs(
        output_root=output_root,
        configs_dir=configs_dir,
        run_summary=summary,
        num_permutations=num_permutations,
        seed=seed,
    )
    summary["baseline_configs"] = [str(p) for p in written_cfgs]
    write_json(output_root / "run_summary.json", summary)
    return summary


def _build_latex_ready(output_root: Path, embed_h2_baseline_csv: Path) -> None:
    """Assemble latex_ready CSVs from by_protocol baselines."""
    primary = PRIMARY_PROTOCOL
    primary_dir = output_root / "by_protocol" / primary
    latex_primary = output_root / "latex_ready" / "primary_tournament_description"
    latex_ablation = output_root / "latex_ready" / "ablation_prompt_conditions"
    latex_cross = output_root / "latex_ready" / "cross_family"
    latex_primary.mkdir(parents=True, exist_ok=True)
    latex_ablation.mkdir(parents=True, exist_ok=True)
    latex_cross.mkdir(parents=True, exist_ok=True)

    feas_rows = _read_csv_dicts(primary_dir / "random_baseline_feasible_2500" / "summary.csv")
    topn_rows = _read_csv_dicts(primary_dir / "random_baseline_topn_2500" / "summary.csv")

    _write_csv(latex_primary / "tab_llm_feasible_n3.csv", _n3_calibrated_table(feas_rows, k=3))
    _write_csv(latex_primary / "tab_llm_topn_n3.csv", _n3_calibrated_table(topn_rows, k=3))
    _write_csv(latex_primary / "tab_llm_chance_by_n_feasible.csv", _chance_by_n_table(feas_rows))
    _write_csv(latex_primary / "tab_llm_chance_by_n_topn.csv", _chance_by_n_table(topn_rows))
    _write_csv(
        latex_primary / "tab_llm_above_p95_feasible.csv",
        _read_csv_dicts(primary_dir / "above_p95_models_feasible.csv"),
    )
    _write_csv(
        latex_primary / "tab_llm_above_p95_topn.csv",
        _read_csv_dicts(primary_dir / "above_p95_models_topn.csv"),
    )

    # Ablation: ratio_mean@3 across protocols
    ablation_rows: List[Dict[str, Any]] = []
    ablation_counts: List[Dict[str, Any]] = []
    for spec in PROTOCOL_SPECS:
        proto_dir = output_root / "by_protocol" / spec.protocol_id
        base_csv = proto_dir / "random_baseline_feasible_2500" / "summary.csv"
        if not base_csv.is_file():
            continue
        rows = _read_csv_dicts(base_csv)
        for row in rows:
            if str(row.get("metric")) != "ratio_mean" or int(row["k"]) != 3:
                continue
            ablation_rows.append(
                {
                    "protocol_id": spec.protocol_id,
                    "role": spec.role,
                    "model": row["model"],
                    "observed_mean": float(row["observed_mean"]),
                    "random_mean": float(row["random_mean"]),
                    "p95": float(row["p95"]),
                    "delta_to_p95": float(row["observed_mean"]) - float(row["p95"]),
                    "above_p95": _parse_bool(row.get("above_p95", False)),
                }
            )
        above_csv = proto_dir / "above_p95_models_feasible.csv"
        if above_csv.is_file():
            for row in _read_csv_dicts(above_csv):
                ablation_counts.append(
                    {
                        "protocol_id": spec.protocol_id,
                        "role": spec.role,
                        "k": int(row["k"]),
                        "count_label": row["count_label"],
                        "models": row["models"],
                    }
                )
    _write_csv(latex_ablation / "tab_ablation_ratio_mean_n3.csv", ablation_rows)
    _write_csv(latex_ablation / "tab_ablation_above_p95_counts.csv", ablation_counts)

    # Cross-family: embedding H2 vs LLM primary at N=3 ratio_mean
    cross_rows: List[Dict[str, Any]] = []
    if embed_h2_baseline_csv.is_file():
        for row in _read_csv_dicts(embed_h2_baseline_csv):
            if str(row.get("metric")) != "ratio_mean" or int(row["k"]) != 3:
                continue
            cross_rows.append(
                {
                    "family": "embedding_h2",
                    "model": row["model"],
                    "observed_mean": float(row["observed_mean"]),
                    "random_mean": float(row["random_mean"]),
                    "p95": float(row["p95"]),
                    "delta_to_p95": float(row["observed_mean"]) - float(row["p95"]),
                    "above_p95": _parse_bool(row.get("above_p95", False)),
                }
            )
    for row in _n3_calibrated_table(feas_rows, k=3):
        cross_rows.append(
            {
                "family": "llm_judge_tournament_description",
                "model": row["model"],
                "observed_mean": row["observed_mean"],
                "random_mean": row["random_mean"],
                "p95": row["p95"],
                "delta_to_p95": row["delta_to_p95"],
                "above_p95": row["above_p95"],
            }
        )
    _write_csv(latex_cross / "tab_embed_vs_llm_n3.csv", cross_rows)


def _read_csv_dicts(path: Path) -> List[Dict[str, Any]]:
    """Read CSV into list of dicts; empty if missing."""
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def validate_pipeline(
    project_root: Path,
    output_root: Optional[Path] = None,
    reference_detailed_json: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run scientific validation gates on generated artifacts.

    Gates:
    1. 20 prompts match reference H2 partition when available
    2. Winner exclusion in metric inputs (rank_in_prompt==1 filtered)
    3. Manifest coverage
    """
    project_root = project_root.resolve()
    if output_root is None:
        output_root = project_root / "results" / "llm_judge_normalized"
    output_root = output_root.resolve()

    report: Dict[str, Any] = {"ok": True, "checks": []}

    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        report["ok"] = False
        report["checks"].append({"name": "manifest_exists", "ok": False, "detail": "missing"})
        return report

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Coverage: all 5 protocols present
    proto_ids = {p["protocol_id"] for p in manifest.get("protocols", [])}
    expected = {s.protocol_id for s in PROTOCOL_SPECS}
    cov_ok = expected <= proto_ids
    report["checks"].append(
        {
            "name": "manifest_protocols",
            "ok": cov_ok,
            "detail": f"found={sorted(proto_ids)} expected={sorted(expected)}",
        }
    )
    if not cov_ok:
        report["ok"] = False

    # Reference prompts from H2 detalhado
    if reference_detailed_json is None:
        reference_detailed_json = (
            project_root
            / "results"
            / "experimento2_h2"
            / "qwen3-embedding-8b"
            / "resultados_h2_detalhado.json"
        )
    ref_prompts: Optional[set[str]] = None
    if reference_detailed_json.is_file():
        with open(reference_detailed_json, "r", encoding="utf-8") as f:
            ref_rows = json.load(f)
        ref_prompts = {str(r["prompt_id"]) for r in ref_rows}
        report["checks"].append(
            {
                "name": "reference_prompt_count",
                "ok": len(ref_prompts) == 20,
                "detail": f"n={len(ref_prompts)}",
            }
        )
        if len(ref_prompts) != 20:
            report["ok"] = False

    # Per-judge detalhado checks
    for proto_dir in sorted((output_root / "by_protocol").glob("*")):
        if not proto_dir.is_dir():
            continue
        for judge_dir in sorted((proto_dir / "judges").glob("*")):
            detalhado = judge_dir / "detalhado.json"
            if not detalhado.is_file():
                report["ok"] = False
                report["checks"].append(
                    {
                        "name": "detalhado_missing",
                        "ok": False,
                        "detail": str(detalhado),
                    }
                )
                continue
            with open(detalhado, "r", encoding="utf-8") as f:
                rows = json.load(f)
            prompts = {str(r["prompt_id"]) for r in rows}
            winners = [r for r in rows if int(r["rank_in_prompt"]) == 1]
            # Winner rows must exist in detalhado (20) but be excluded by metric extractors
            from embedding_models_eval.metrics.feasible_range_random_baseline import (
                extract_prompt_votes_from_detailed_rows,
            )

            votes = extract_prompt_votes_from_detailed_rows(rows)
            prompt_ok = ref_prompts is None or prompts == ref_prompts
            winner_ok = len(winners) == 20
            extract_ok = len(votes) == 20
            check_ok = prompt_ok and winner_ok and extract_ok
            if not check_ok:
                report["ok"] = False
            report["checks"].append(
                {
                    "name": f"partition::{proto_dir.name}::{judge_dir.name}",
                    "ok": check_ok,
                    "detail": (
                        f"n_prompts={len(prompts)} n_winners={len(winners)} "
                        f"n_extracted={len(votes)} prompt_match={prompt_ok}"
                    ),
                }
            )
            # Ensure extracted pools are non-empty after winner exclusion.
            if any(len(v) < 1 for v in votes.values()):
                report["ok"] = False
                report["checks"].append(
                    {
                        "name": f"empty_pool::{judge_dir.name}",
                        "ok": False,
                        "detail": "empty vote pool after winner exclusion",
                    }
                )

    # latex_ready consistency: above_p95 must match observed > p95
    n3_path = (
        output_root
        / "latex_ready"
        / "primary_tournament_description"
        / "tab_llm_feasible_n3.csv"
    )
    if n3_path.is_file() and n3_path.stat().st_size > 0:
        n3_rows = _read_csv_dicts(n3_path)
        bad = []
        for row in n3_rows:
            obs = float(row["observed_mean"])
            p95 = float(row["p95"])
            flag = _parse_bool(row["above_p95"])
            expected = obs > p95
            if flag != expected:
                bad.append(str(row["model"]))
        ok_n3 = len(bad) == 0
        if not ok_n3:
            report["ok"] = False
        report["checks"].append(
            {
                "name": "latex_ready_n3_above_p95_consistency",
                "ok": ok_n3,
                "detail": "ok" if ok_n3 else f"mismatched models={bad}",
            }
        )

    write_json(output_root / "validation_report.json", report)
    return report
