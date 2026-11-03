"""Build a Markdown report (+ figures) from Val3 evaluation outputs.

Reads tables already written under ``<run_root>/evaluation/<direction>/`` and renders summaries /
figures via ``val3.viz`` (pandas merges only; no inference re-run).
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd
import seaborn as sns  # noqa: E402

plt.rcParams.update({"svg.fonttype": "path"})

from . import posthoc, viz
from .decoy import (
    TARGET_DECOY_NULL_TYPE,
    TARGET_DECOY_SMD_ACCEPTANCE_THRESHOLD,
    build_target_decoy_matching_acceptance,
)
from .streaming import (
    ALLOWED_FINAL_NULL_TYPES,
    assert_distinguishable_null_types,
    prune_non_v23_null_types,
    read_parquet_any_engine,
    write_parquet_zstd,
)

# docs/val3/val3_triplet_methodology_v2.4.md の最上行と揃える。
VAL3_TRIPLET_REVISED_METHODOLOGY_VERSION = "v2.4"
VAL3_REPORT_FILENAME = "val3_report.md"


def _slug(s: str) -> str:
    x = re.sub(r"[^\w.\-]+", "_", str(s), flags=re.UNICODE).strip("_")
    return x[:120] if x else "unknown"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _run_decoy_selection(meta: Mapping[str, Any] | None) -> str | None:
    """Target-Decoy selection rule of this run, or ``None`` when it is not recorded.

    An older run has no ``target_decoy_model.decoy_selection``; its figures then keep
    the unqualified ``Target-Decoy`` label rather than claiming a random selection.
    """
    model = (meta or {}).get("target_decoy_model")
    if not isinstance(model, Mapping):
        return None
    value = model.get("decoy_selection")
    return str(value) if value else None


def _read_parquet(path: Path) -> pd.DataFrame:
    """Prefer fastparquet when pyarrow cannot read a file (seen in some Val3 outputs)."""
    try:
        return read_parquet_any_engine(path)
    except Exception:
        return pd.DataFrame()


def _markdown_table(df: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if df.empty:
        return "_（空）_\n"
    view = df if max_rows is None else df.head(int(max_rows))
    try:
        return view.to_markdown(index=False) + "\n"
    except Exception:
        return "```text\n" + view.to_csv(index=False, sep="\t") + "```\n"


# The report uses two-sided paired Wilcoxon signed-rank tests with Holm correction
# throughout. Other test families and legacy reference columns remain in source
# tables from older runs but are not rendered in the markdown.
NON_PRIMARY_TEST_COLUMNS: frozenset[str] = frozenset({
    "p_sign_greater", "p_sign_holm", "sign_test_status",
    "p_t_two_sided", "p_t_holm", "t_statistic", "t_test_status",
    "p_wilcoxon_greater", "wilcoxon_status",
    "p_resampled", "p_finite_null", "p_empirical", "p_empirical_floor", "n_null_ge_observed",
})
REFERENCE_COLUMN_PREFIX = "ref_"


def _is_nonprimary_column(name: Any) -> bool:
    """Whether a column is a non-primary p value or a ``ref_*`` reference value.

    Matching is by prefix so that scope-suffixed variants written by earlier runs
    (``p_finite_null_all_edges``) are recognised when such a run's report is
    regenerated. The current ``p_wilcoxon_two_sided`` columns never match.
    """
    column = str(name)
    if column.startswith(REFERENCE_COLUMN_PREFIX):
        return True
    return any(column == base or column.startswith(f"{base}_") for base in NON_PRIMARY_TEST_COLUMNS)


def _drop_nonprimary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-primary p-value and ``ref_*`` columns from a report-facing table copy.

    Only the rendered markdown is filtered; the underlying TSV/parquet keeps every column.
    """
    drop = [c for c in df.columns if _is_nonprimary_column(c)]
    return df.drop(columns=drop) if drop else df


def _display_metric_names(
    df: pd.DataFrame,
    *,
    decoy_selection: str | None = None,
) -> pd.DataFrame:
    """Return a report-facing copy with metric/statistic values relabeled."""
    out = _drop_nonprimary_columns(df).copy()
    if "metric" in out.columns:
        out["metric"] = out["metric"].map(viz.format_metric_label)
    if "statistic" in out.columns:
        out["statistic"] = out["statistic"].map(viz.format_metric_label)
    if "null_type" in out.columns:
        out["null_type"] = out["null_type"].map(
            lambda value: viz.format_null_type_label(
                value, decoy_selection=decoy_selection,
            )
        )
    return out


def _format_p_value(value: Any) -> str:
    try:
        p = float(value)
    except Exception:
        return ""
    if not np.isfinite(p):
        return ""
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _format_p_label(value: Any) -> str:
    try:
        p = float(value)
    except Exception:
        return ""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "p < 0.001"
    return f"p={_format_p_value(p)}"


def _holm_adjust(p_values: list[float]) -> list[float]:
    return posthoc.holm_adjust(p_values)


def _jaccard_paired_wilcoxon_holm_table(
    dfs: Mapping[str, pd.DataFrame],
    *,
    eval_type: str | None = None,
) -> pd.DataFrame:
    """Jaccard-only two-sided Wilcoxon tests against the per-triplet null median."""
    columns = [
        "null_type",
        "evaluable_set_type",
        "n_triplets",
        "observed_mean",
        "baseline_representative",
        "null_median_mean",
        "delta_mean",
        "delta_median",
        "n_positive_delta",
        "n_negative_delta",
        "n_zero_delta",
        "n_nonzero_delta",
        "wilcoxon_two_sided_statistic",
        "p_wilcoxon_two_sided",
        "p_wilcoxon_two_sided_holm",
        "wilcoxon_two_sided_status",
    ]
    true_df = dfs.get("global_joint_true", pd.DataFrame()).copy()
    base_df = dfs.get("global_joint_baseline", pd.DataFrame()).copy()
    required_true = {"evaluable_set_type", "triplet_id", "joint_jaccard"}
    required_base = {"null_type", "evaluable_set_type", "triplet_id", "joint_jaccard"}
    if true_df.empty or base_df.empty or not required_true <= set(true_df.columns) or not required_base <= set(base_df.columns):
        return pd.DataFrame(columns=columns)

    true = true_df[["evaluable_set_type", "triplet_id", "joint_jaccard"]].copy()
    true["observed"] = pd.to_numeric(true["joint_jaccard"], errors="coerce")
    true = true.drop(columns=["joint_jaccard"]).dropna(subset=["observed"])
    base = prune_non_v23_null_types(base_df)
    if eval_type is not None:
        true = true.loc[true["evaluable_set_type"].astype(str).eq(str(eval_type))]
        base = base.loc[base["evaluable_set_type"].astype(str).eq(str(eval_type))]
    if true.empty or base.empty:
        return pd.DataFrame(columns=columns)

    base = base[["null_type", "evaluable_set_type", "triplet_id", "joint_jaccard"]].copy()
    base["null_value"] = pd.to_numeric(base["joint_jaccard"], errors="coerce")
    base = base.drop(columns=["joint_jaccard"]).dropna(subset=["null_value"])
    null_rep = (
        base.groupby(["null_type", "evaluable_set_type", "triplet_id"], dropna=False, sort=False)["null_value"]
        .median()
        .reset_index(name="null_median")
    )
    paired = null_rep.merge(true, on=["evaluable_set_type", "triplet_id"], how="inner")
    if paired.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for (null_type, evaluable_set_type), sub in paired.groupby(["null_type", "evaluable_set_type"], dropna=False, sort=False):
        valid = sub.dropna(subset=["observed", "null_median"]).copy()
        valid["delta"] = (
            pd.to_numeric(valid["observed"], errors="coerce")
            - pd.to_numeric(valid["null_median"], errors="coerce")
        )
        valid = valid.dropna(subset=["delta"])
        delta = valid["delta"].to_numpy(dtype=float)
        delta = delta[np.isfinite(delta)]
        n = int(len(delta))
        n_positive = int(np.sum(delta > 0))
        n_negative = int(np.sum(delta < 0))
        n_zero = int(np.sum(delta == 0))
        n_nonzero = n_positive + n_negative
        wilcoxon_statistic, p_wilcoxon, wilcoxon_status = (
            posthoc.wilcoxon_signed_rank_two_sided(delta)
        )
        rows.append({
            "null_type": str(null_type),
            "evaluable_set_type": str(evaluable_set_type),
            "n_triplets": n,
            "observed_mean": float(valid["observed"].mean()) if n else float("nan"),
            "baseline_representative": "median",
            "null_median_mean": float(valid["null_median"].mean()) if n else float("nan"),
            "delta_mean": float(np.mean(delta)) if n else float("nan"),
            "delta_median": float(np.median(delta)) if n else float("nan"),
            "n_positive_delta": n_positive,
            "n_negative_delta": n_negative,
            "n_zero_delta": n_zero,
            "n_nonzero_delta": n_nonzero,
            "wilcoxon_two_sided_statistic": wilcoxon_statistic,
            "p_wilcoxon_two_sided": p_wilcoxon,
            "p_wilcoxon_two_sided_holm": float("nan"),
            "wilcoxon_two_sided_status": wilcoxon_status,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=columns)
    out["p_wilcoxon_two_sided_holm"] = _holm_adjust([
        float(x) if pd.notna(x) else float("nan") for x in out["p_wilcoxon_two_sided"]
    ])
    out = out.sort_values("null_type", key=lambda s: s.astype(str).map(viz.null_type_sort_key))
    return out[columns].reset_index(drop=True)


def _load_evaluation_artifacts(run_root: Path, direction: str) -> dict[str, pd.DataFrame]:
    """Load standard evaluation parquet/TSV files into a fixed key → DataFrame map."""
    evaluation_root = run_root / "evaluation" / direction
    expected_outputs = {
        "triplet_true": evaluation_root / "triplet_metrics_true.parquet",
        "diag_true": evaluation_root / "cellpair_filter_diagnostics_true.parquet",
        "global_evaluable_triplets": evaluation_root / "global_evaluable_triplets.tsv",
        "global_joint_true": evaluation_root / "global_joint_triplet_metrics_true.parquet",
        "global_joint_baseline": evaluation_root / "global_joint_triplet_metrics_baseline.parquet",
        "global_joint_triplet_comparison": evaluation_root / "global_joint_triplet_comparison.tsv",
        "global_joint_triplet_comparison_sets": evaluation_root / "global_joint_triplet_comparison_sets.tsv",
        "global_effect_size": evaluation_root / "global_effect_size.tsv",
        # Target-Decoy matching / evaluability diagnostics; absent for the other nulls.
        # ``decoy_matching_balance`` / ``decoy_matching_acceptance`` are also absent for a
        # random decoy draw, which has no matching to judge.
        "decoy_matching_balance": evaluation_root / "decoy_matching_balance.tsv",
        "decoy_matching_acceptance": evaluation_root / "decoy_matching_acceptance.tsv",
        "decoy_evaluability_summary": evaluation_root / "decoy_evaluability_summary.tsv",
        "decoy_triplet_evaluability": evaluation_root / "decoy_triplet_evaluability.tsv",
        "decoy_delta_by_n_evaluable_decoys_raw": evaluation_root / "decoy_delta_by_n_evaluable_decoys.tsv",
        # First-edge exclusion sensitivity analysis; absent when the scope is disabled.
        "first_edge_exclusion_summary_raw": evaluation_root / "first_edge_exclusion_summary.tsv",
        "first_edge_excluded_comparison": (
            evaluation_root / "first_edge_excluded" / "global_joint_triplet_comparison.tsv"
        ),
    }
    dfs: dict[str, pd.DataFrame] = {}
    for key, path in expected_outputs.items():
        if not path.exists():
            dfs[key] = pd.DataFrame()
            continue
        try:
            if path.suffix == ".tsv":
                dfs[key] = pd.read_csv(path, sep="\t")
            else:
                dfs[key] = _read_parquet(path)
        except Exception:
            dfs[key] = pd.DataFrame()
    for key in ["cellpair_true", "cellpair_baseline", "triplet_baseline", "diag_baseline"]:
        dfs[key] = pd.DataFrame()
    return dfs


def _median_first_edge_exclusion_summary(
    dfs: Mapping[str, pd.DataFrame],
    *,
    direction: str = "",
    seed: int = 0,
    n_bootstrap: int = posthoc.BOOTSTRAP_N_RESAMPLES,
) -> pd.DataFrame:
    """Rebuild first-edge effects with per-triplet null medians and bootstrap CIs."""

    keys = ["null_type", "evaluable_set_type", "statistic"]

    def _scope_summary(comparison: pd.DataFrame, suffix: str) -> pd.DataFrame:
        required = {*keys, "triplet_id", "observed_value", "baseline_median"}
        if comparison.empty or not required <= set(comparison.columns):
            return pd.DataFrame()
        frame = prune_non_v23_null_types(comparison).copy()
        frame["observed_value"] = pd.to_numeric(frame["observed_value"], errors="coerce")
        frame["baseline_median"] = pd.to_numeric(frame["baseline_median"], errors="coerce")
        frame["delta"] = frame["observed_value"] - frame["baseline_median"]
        frame = frame.loc[frame["statistic"].astype(str).eq(posthoc.JACCARD_STATISTIC)].copy()
        frame = frame.dropna(subset=["observed_value", "baseline_median", "delta"])
        if frame.empty:
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        for group_key, sub in frame.groupby(keys, dropna=False, sort=False):
            triplet_level = (
                sub[["triplet_id", "observed_value", "baseline_median", "delta"]]
                .groupby("triplet_id", dropna=False, sort=True)
                .agg(
                    observed_value=("observed_value", "mean"),
                    baseline_median=("baseline_median", "mean"),
                    delta=("delta", "mean"),
                )
                .reset_index()
            )
            point, ci025, ci975, group_seed = posthoc.bootstrap_mean_ci(
                triplet_level["delta"],
                seed=seed,
                group_parts=(direction, *group_key, suffix, "all_cell_pairs"),
                n_bootstrap=n_bootstrap,
            )
            rows.append({
                **dict(zip(keys, [str(value) for value in group_key])),
                "n_comparison_triplets": int(len(triplet_level)),
                f"_triplet_ids_{suffix}": tuple(
                    sorted(triplet_level["triplet_id"].astype(str))
                ),
                f"observed_mean_{suffix}": float(triplet_level["observed_value"].mean()),
                f"null_median_{suffix}": float(triplet_level["baseline_median"].mean()),
                f"delta_mean_{suffix}": point,
                f"delta_mean_ci025_{suffix}": ci025,
                f"delta_mean_ci975_{suffix}": ci975,
                f"bootstrap_seed_{suffix}": group_seed,
                f"n_bootstrap_{suffix}": int(n_bootstrap),
            })
        return pd.DataFrame(rows)

    all_edges = _scope_summary(
        dfs.get("global_joint_triplet_comparison", pd.DataFrame()), "all_edges",
    )
    excluded = _scope_summary(
        dfs.get("first_edge_excluded_comparison", pd.DataFrame()), "first_edge_excluded",
    )
    if all_edges.empty or excluded.empty:
        return pd.DataFrame()
    excluded = excluded.drop(columns=["n_comparison_triplets"], errors="ignore")
    out = all_edges.merge(excluded, on=keys, how="inner")
    mismatched = out.loc[
        out["_triplet_ids_all_edges"] != out["_triplet_ids_first_edge_excluded"]
    ]
    if not mismatched.empty:
        raise AssertionError(
            "First-edge exclusion changed the comparison triplet set: "
            f"{mismatched[keys].to_dict(orient='records')[:5]}"
        )
    out = out.drop(
        columns=["_triplet_ids_all_edges", "_triplet_ids_first_edge_excluded"]
    )
    out["estimand"] = posthoc.PRIMARY_ESTIMAND
    out["resampling_unit"] = "triplet_id"
    out["ci_method"] = posthoc.BOOTSTRAP_CI_METHOD
    out["ci_level"] = posthoc.BOOTSTRAP_CI_LEVEL
    denominator = pd.to_numeric(out["delta_mean_all_edges"], errors="coerce")
    numerator = pd.to_numeric(out["delta_mean_first_edge_excluded"], errors="coerce")
    out["delta_mean_ratio"] = numerator / denominator.replace(0.0, np.nan)

    diagnostics = dfs.get("first_edge_exclusion_summary_raw", pd.DataFrame())
    diagnostic_columns = [
        *keys,
        "n_pairs_empty_after_exclusion",
        "n_edges_removed_median_drug",
        "n_edges_removed_median_pert",
    ]
    if not diagnostics.empty and set(keys) <= set(diagnostics.columns):
        available = [column for column in diagnostic_columns if column in diagnostics.columns]
        out = out.merge(diagnostics[available].drop_duplicates(keys), on=keys, how="left")
    return out.sort_values(
        keys,
        key=lambda s: (
            s.astype(str).map(viz.null_type_sort_key)
            if s.name == "null_type"
            else s.astype(str)
        ),
    ).reset_index(drop=True)


def _median_target_decoy_delta_strata(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Stratify Target-Decoy deltas after recomputing them from null medians."""
    columns = [
        "null_type",
        "evaluable_set_type",
        "statistic",
        "n_evaluable_decoys",
        "n_triplets",
        "delta_mean",
        "delta_sd",
    ]
    required = {
        "null_type", "evaluable_set_type", "statistic", "triplet_id",
        "observed_value", "baseline_median", "n_decoy_anchors",
    }
    if comparison.empty or not required <= set(comparison.columns):
        return pd.DataFrame(columns=columns)
    frame = comparison.loc[
        comparison["null_type"].astype(str).eq(TARGET_DECOY_NULL_TYPE)
    ].copy()
    frame["observed_value"] = pd.to_numeric(frame["observed_value"], errors="coerce")
    frame["baseline_median"] = pd.to_numeric(frame["baseline_median"], errors="coerce")
    frame["n_evaluable_decoys"] = pd.to_numeric(frame["n_decoy_anchors"], errors="coerce")
    frame["delta"] = frame["observed_value"] - frame["baseline_median"]
    frame = frame.dropna(subset=["delta", "n_evaluable_decoys"])
    if frame.empty:
        return pd.DataFrame(columns=columns)
    out = (
        frame.groupby(
            ["null_type", "evaluable_set_type", "statistic", "n_evaluable_decoys"],
            dropna=False,
            sort=True,
        )
        .agg(
            n_triplets=("triplet_id", "nunique"),
            delta_mean=("delta", "mean"),
            delta_sd=("delta", "std"),
        )
        .reset_index()
    )
    out["n_evaluable_decoys"] = out["n_evaluable_decoys"].astype(int)
    return out[columns]



def _load_report_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        if path.suffix == ".parquet":
            return _read_parquet(path)
        return pd.read_csv(
            path,
            sep="\t",
            keep_default_na=False,
            na_values=["", "NaN", "nan"],
        )
    except Exception:
        return pd.DataFrame()

def _inventory_table(run_root: Path, direction: str, dfs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    evaluation_root = run_root / "evaluation" / direction
    expected_outputs = {
        "triplet_true": evaluation_root / "triplet_metrics_true.parquet",
        "diag_true": evaluation_root / "cellpair_filter_diagnostics_true.parquet",
        "global_evaluable_triplets": evaluation_root / "global_evaluable_triplets.tsv",
        "global_joint_true": evaluation_root / "global_joint_triplet_metrics_true.parquet",
        "global_joint_baseline": evaluation_root / "global_joint_triplet_metrics_baseline.parquet",
        "global_joint_triplet_comparison": evaluation_root / "global_joint_triplet_comparison.tsv",
        "global_effect_size": evaluation_root / "global_effect_size.tsv",
    }
    rows: list[dict[str, Any]] = []
    for key, path in expected_outputs.items():
        df = dfs.get(key, pd.DataFrame())
        err: str | None = None
        loaded = bool(path.exists()) and not df.empty
        if path.exists() and df.empty:
            err = "read_failed_or_empty"
        rows.append(
            {
                "key": key,
                "exists": bool(path.exists()),
                "loaded": loaded,
                "n_rows": int(len(df)),
                "n_cols": int(len(df.columns)),
                "error": err,
                "path": str(path),
            }
        )
    return pd.DataFrame(rows)


def _overview_rows_core(dfs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    overview_rows: list[dict[str, Any]] = []
    for key in ["triplet_true", "diag_true"]:
        df = dfs.get(key, pd.DataFrame())
        row: dict[str, Any] = {"table": key, "n_rows": len(df), "n_cols": len(df.columns)}
        for col in ["triplet_id", "iteration", "null_type", "evaluable_set_type"]:
            if col in df.columns:
                row[f"n_{col}"] = int(df[col].nunique(dropna=True))
        overview_rows.append(row)
    return pd.DataFrame(overview_rows)


def _overview_rows_global(dfs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    global_keys = [
        "global_evaluable_triplets",
        "global_joint_true",
        "global_joint_baseline",
        "global_joint_triplet_comparison",
        "global_effect_size",
    ]
    global_overview: list[dict[str, Any]] = []
    for key in global_keys:
        df = dfs.get(key, pd.DataFrame())
        row: dict[str, Any] = {"table": key, "n_rows": len(df), "n_cols": len(df.columns)}
        for col in ["null_type", "evaluable_set_type", "statistic", "iteration"]:
            if col in df.columns:
                row[f"n_{col}"] = int(df[col].nunique(dropna=True))
        global_overview.append(row)
    return pd.DataFrame(global_overview)


def _global_summary_view(dfs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Paired global effect sizes keyed by (null_type, evaluable_set_type, statistic)."""
    global_effect = dfs.get("global_effect_size", pd.DataFrame())
    if global_effect.empty:
        return pd.DataFrame()
    key_cols = ["null_type", "evaluable_set_type", "statistic"]
    cols = [
        c for c in key_cols + ["n_triplets", "paired_delta_mean", "paired_delta_median"]
        if c in global_effect.columns
    ]
    out = _drop_nonprimary_columns(global_effect[cols].copy())
    sort_cols = [c for c in key_cols if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True) if sort_cols else out.reset_index(drop=True)


def _make_global_metric_long(
    dfs: Mapping[str, pd.DataFrame],
    *,
    metrics: tuple[str, ...] = ("jaccard",),
) -> pd.DataFrame:
    """Triplet-level observed and null-median values for global joint plots."""
    true_df = dfs.get("global_joint_true", pd.DataFrame()).copy()
    base_df = dfs.get("global_joint_baseline", pd.DataFrame()).copy()
    parts: list[pd.DataFrame] = []

    required_true = {"evaluable_set_type", "triplet_id"}
    required_base = {"null_type", "evaluable_set_type", "triplet_id"}
    if true_df.empty or base_df.empty or not required_true <= set(true_df.columns) or not required_base <= set(base_df.columns):
        return pd.DataFrame()

    for metric in metrics:
        col = f"joint_{metric}"
        if col not in true_df.columns or col not in base_df.columns:
            continue
        obs = true_df[["evaluable_set_type", "triplet_id", col]].copy()
        obs["observed_value"] = pd.to_numeric(obs[col], errors="coerce")
        obs = obs.drop(columns=[col]).dropna(subset=["observed_value"])

        base = base_df[["null_type", "evaluable_set_type", "triplet_id", col]].copy()
        base["null_value"] = pd.to_numeric(base[col], errors="coerce")
        null_rep = (
            base.drop(columns=[col])
            .dropna(subset=["null_value"])
            .groupby(["null_type", "evaluable_set_type", "triplet_id"], dropna=False, sort=False, as_index=False)
            .agg(value=("null_value", "median"))
        )
        merged = null_rep.merge(obs, on=["evaluable_set_type", "triplet_id"], how="inner")
        if merged.empty:
            continue

        common_cols = ["null_type", "evaluable_set_type", "triplet_id"]
        obs_part = merged[common_cols + ["observed_value"]].rename(columns={"observed_value": "value"})
        obs_part["source"] = "observed"
        null_part = merged[common_cols + ["value"]].copy()
        null_part["source"] = "null median"
        metric_df = pd.concat([obs_part, null_part], ignore_index=True)
        metric_df["iteration"] = pd.NA
        metric_df["metric"] = metric
        parts.append(metric_df)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _build_jaccard_observed_vs_null_summary(
    global_metric_long: pd.DataFrame,
    paired_test_summary: pd.DataFrame,
    *,
    seed: int = 0,
    n_bootstrap: int = posthoc.BOOTSTRAP_N_RESAMPLES,
) -> pd.DataFrame:
    """Paper-facing means and paired-triplet bootstrap CIs for observed versus null."""
    columns = [
        "null_type",
        "evaluable_set_type",
        "n_triplets",
        "n_bootstrap",
        "bootstrap_seed",
        "resampling_unit",
        "ci_method",
        "ci_level",
        "observed_mean",
        "observed_mean_ci025",
        "observed_mean_ci975",
        "null_median_mean",
        "null_median_mean_ci025",
        "null_median_mean_ci975",
        "p_wilcoxon_two_sided_holm",
    ]
    required = {
        "null_type",
        "evaluable_set_type",
        "triplet_id",
        "metric",
        "source",
        "value",
    }
    if global_metric_long.empty or not required <= set(global_metric_long.columns):
        return pd.DataFrame(columns=columns)
    metric_long = global_metric_long.loc[
        global_metric_long["metric"].astype(str).eq("jaccard")
    ].copy()
    metric_long["value"] = pd.to_numeric(metric_long["value"], errors="coerce")
    metric_long = metric_long.dropna(subset=["value"])
    if metric_long.empty:
        return pd.DataFrame(columns=columns)

    test_lookup = paired_test_summary.copy()
    if not test_lookup.empty:
        test_lookup = test_lookup.set_index(["null_type", "evaluable_set_type"])
    rows: list[dict[str, Any]] = []
    for (null_type, eval_type), group in metric_long.groupby(
        ["null_type", "evaluable_set_type"],
        dropna=False,
        sort=False,
    ):
        paired = (
            group.pivot_table(
                index="triplet_id",
                columns="source",
                values="value",
                aggfunc="mean",
            )
            .reindex(columns=["observed", "null median"])
            .dropna()
            .sort_index()
        )
        observed = paired["observed"].to_numpy(dtype=float)
        null_median = paired["null median"].to_numpy(dtype=float)
        n = int(len(paired))
        n_boot = int(n_bootstrap)
        if n_boot <= 0:
            raise ValueError("n_bootstrap must be positive.")
        bootstrap_seed = posthoc.stable_bootstrap_seed(
            seed,
            "jaccard_observed_vs_null",
            null_type,
            eval_type,
        )
        observed_mean = float(np.mean(observed)) if n else float("nan")
        null_median_mean = float(np.mean(null_median)) if n else float("nan")
        if n == 0:
            observed_ci025 = observed_ci975 = float("nan")
            null_median_ci025 = null_median_ci975 = float("nan")
        elif n == 1:
            observed_ci025 = observed_ci975 = observed_mean
            null_median_ci025 = null_median_ci975 = null_median_mean
        else:
            rng = np.random.default_rng(bootstrap_seed)
            sample_idx = rng.integers(
                0,
                n,
                size=(n_boot, n),
                endpoint=False,
            )
            observed_bootstrap_mean = np.mean(observed[sample_idx], axis=1)
            null_median_bootstrap_mean = np.mean(
                null_median[sample_idx],
                axis=1,
            )
            alpha = (1.0 - posthoc.BOOTSTRAP_CI_LEVEL) / 2.0
            observed_ci025, observed_ci975 = np.quantile(
                observed_bootstrap_mean,
                [alpha, 1.0 - alpha],
            )
            null_median_ci025, null_median_ci975 = np.quantile(
                null_median_bootstrap_mean,
                [alpha, 1.0 - alpha],
            )

        p_holm = float("nan")
        key = (str(null_type), str(eval_type))
        if not test_lookup.empty and key in test_lookup.index:
            test_row = test_lookup.loc[key]
            if isinstance(test_row, pd.DataFrame):
                test_row = test_row.iloc[0]
            p_holm = float(test_row.get("p_wilcoxon_two_sided_holm", float("nan")))
        rows.append({
            "null_type": str(null_type),
            "evaluable_set_type": str(eval_type),
            "n_triplets": n,
            "n_bootstrap": n_boot,
            "bootstrap_seed": bootstrap_seed,
            "resampling_unit": "triplet_id",
            "ci_method": posthoc.BOOTSTRAP_CI_METHOD,
            "ci_level": posthoc.BOOTSTRAP_CI_LEVEL,
            "observed_mean": observed_mean,
            "observed_mean_ci025": float(observed_ci025),
            "observed_mean_ci975": float(observed_ci975),
            "null_median_mean": null_median_mean,
            "null_median_mean_ci025": float(null_median_ci025),
            "null_median_mean_ci975": float(null_median_ci975),
            "p_wilcoxon_two_sided_holm": p_holm,
        })
    out = pd.DataFrame(rows, columns=columns)
    if out.empty:
        return out
    out["_null_order"] = out["null_type"].map(viz.null_type_sort_key)
    return (
        out.sort_values(
            ["evaluable_set_type", "_null_order"],
            kind="stable",
        )
        .drop(columns=["_null_order"])
        .reset_index(drop=True)
    )


def _make_global_effect_long(
    dfs: Mapping[str, pd.DataFrame],
    *,
    metrics: tuple[str, ...] = ("jaccard",),
) -> pd.DataFrame:
    """Per-triplet observed − baseline deltas on joint metrics, repeated per baseline iteration."""
    true_df = dfs.get("global_joint_true", pd.DataFrame()).copy()
    base_df = dfs.get("global_joint_baseline", pd.DataFrame()).copy()
    if true_df.empty or base_df.empty:
        return pd.DataFrame()

    key_cols = [c for c in ["evaluable_set_type", "triplet_id"] if c in true_df.columns and c in base_df.columns]
    if not key_cols:
        return pd.DataFrame()

    parts: list[pd.DataFrame] = []
    for metric in metrics:
        col = f"joint_{metric}"
        if col not in true_df.columns or col not in base_df.columns:
            continue
        t = true_df[key_cols + [col]].rename(columns={col: "observed_value"})
        b_cols = [c for c in ["null_type", "iteration"] if c in base_df.columns]
        b = base_df[key_cols + b_cols + [col]].rename(columns={col: "baseline_value"})
        merged = b.merge(t, on=key_cols, how="inner")
        merged["metric"] = metric
        merged["delta"] = pd.to_numeric(merged["observed_value"], errors="coerce") - pd.to_numeric(
            merged["baseline_value"], errors="coerce",
        )
        parts.append(merged)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def build_global_joint_triplet_delta_bootstrap(
    comparison_df: pd.DataFrame,
    *,
    direction: str,
    seed: int,
    n_bootstrap: int = posthoc.BOOTSTRAP_N_RESAMPLES,
) -> pd.DataFrame:
    """Post-hoc triplet-level paired-delta bootstrap from global_joint_triplet_comparison.tsv.

    The input table is already paired at triplet level.  The delta is always
    recomputed as ``observed_value - baseline_median`` so a mean-based legacy
    ``delta`` column cannot leak into the median-based report.
    Resampling is performed over unique ``triplet_id`` values within each
    (direction, null_type, evaluable_set_type, statistic) group.
    """
    columns = [
        "direction", "null_type", "evaluable_set_type", "statistic", "estimand",
        "n_triplets", "n_bootstrap", "bootstrap_seed", "resampling_unit",
        "ci_method", "ci_level",
        "delta_mean_hat", "delta_mean_ci025", "delta_mean_ci975",
    ]
    if comparison_df.empty:
        return pd.DataFrame(columns=columns)

    required = {"triplet_id", "null_type", "evaluable_set_type", "statistic"}
    missing = required - set(comparison_df.columns)
    if missing:
        raise ValueError(
            "global_joint_triplet_comparison.tsv missing required columns for paired delta bootstrap: "
            f"{sorted(missing)}"
        )

    df = comparison_df.copy()
    if "direction" not in df.columns:
        df["direction"] = str(direction)
    df["direction"] = df["direction"].fillna(str(direction)).astype(str)

    if not {"observed_value", "baseline_median"} <= set(df.columns):
        raise ValueError(
            "global_joint_triplet_comparison.tsv requires both "
            "'observed_value' and 'baseline_median'."
        )
    df["_delta"] = pd.to_numeric(df["observed_value"], errors="coerce") - pd.to_numeric(
        df["baseline_median"], errors="coerce"
    )
    df = df.loc[df["statistic"].astype(str).eq(posthoc.JACCARD_STATISTIC)].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    group_cols = ["direction", "null_type", "evaluable_set_type", "statistic"]
    rows: list[dict[str, Any]] = []
    n_boot = int(n_bootstrap)
    if n_boot <= 0:
        raise ValueError("n_bootstrap must be positive.")

    for group_key, sub in df.groupby(group_cols, dropna=False, sort=True):
        triplet_delta = (
            sub[["triplet_id", "_delta"]]
            .dropna(subset=["_delta"])
            .groupby("triplet_id", dropna=False, sort=True)["_delta"]
            .mean()
            .dropna()
            .to_numpy(dtype=float)
        )
        triplet_delta = triplet_delta[np.isfinite(triplet_delta)]
        n = int(len(triplet_delta))
        if n <= 0:
            continue

        direction_s, null_type_s, eval_type_s, stat_s = [str(x) for x in group_key]
        point, ci025, ci975, group_seed = posthoc.bootstrap_mean_ci(
            triplet_delta,
            seed=seed,
            group_parts=(
                direction_s, null_type_s, eval_type_s, stat_s,
                "all_edges", "all_cell_pairs",
            ),
            n_bootstrap=n_boot,
        )
        rows.append({
            "direction": direction_s,
            "null_type": null_type_s,
            "evaluable_set_type": eval_type_s,
            "statistic": stat_s,
            "estimand": posthoc.PRIMARY_ESTIMAND,
            "n_triplets": n,
            "n_bootstrap": n_boot,
            "bootstrap_seed": group_seed,
            "resampling_unit": "triplet_id",
            "ci_method": posthoc.BOOTSTRAP_CI_METHOD,
            "ci_level": posthoc.BOOTSTRAP_CI_LEVEL,
            "delta_mean_hat": point,
            "delta_mean_ci025": ci025,
            "delta_mean_ci975": ci975,
        })

    return pd.DataFrame(rows, columns=columns)


def _savefig_png_pdf(fig: Any, png_path: Path) -> tuple[Path, Path | None]:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    pdf_path: Path | None = png_path.with_suffix(".pdf")
    try:
        fig.savefig(pdf_path, bbox_inches="tight")
    except Exception:
        pdf_path = None
    try:
        fig.savefig(png_path.with_suffix(".svg"), bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)
    return png_path, pdf_path


def _plot_global_joint_boxplots(
    global_metric_long: pd.DataFrame,
    figures_dir: Path,
    *,
    decoy_selection: str | None = None,
) -> list[tuple[str, str, Path, Path | None]]:
    """Boxplots of observed vs per-triplet null-median joint metrics."""
    out_paths: list[tuple[str, str, Path, Path | None]] = []
    if global_metric_long.empty:
        return out_paths
    base_settings = global_metric_long.loc[
        global_metric_long["source"].eq("null median"), ["null_type", "evaluable_set_type"],
    ].drop_duplicates()
    if base_settings.empty:
        return out_paths

    idx = 0
    for null_type, eval_type in base_settings.itertuples(index=False, name=None):
        base_plot = global_metric_long[
            global_metric_long["source"].eq("null median")
            & global_metric_long["null_type"].eq(null_type)
            & global_metric_long["evaluable_set_type"].eq(eval_type)
        ]
        obs_plot = global_metric_long[
            global_metric_long["source"].eq("observed")
            & global_metric_long["null_type"].eq(null_type)
            & global_metric_long["evaluable_set_type"].eq(eval_type)
        ].copy()
        plot_df = pd.concat([obs_plot, base_plot], ignore_index=True).dropna(subset=["value"])
        if plot_df.empty:
            continue

        n_by_box = plot_df.groupby(["metric", "source"], dropna=False).size()
        metric_order = [m for m in ["jaccard"] if m in set(plot_df["metric"])]
        metric_labels = {
            metric: (
                f"{viz.format_metric_label(metric)}\n"
                f"Observed n={int(n_by_box.get((metric, 'observed'), 0))}\n"
                f"Null n={int(n_by_box.get((metric, 'null median'), 0))}"
            )
            for metric in metric_order
        }
        plot_df = plot_df.copy()
        plot_df["metric_label"] = plot_df["metric"].map(metric_labels)
        plot_df["source_label"] = plot_df["source"].map(
            {"observed": "Observed", "null median": "Null"}
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(
            data=plot_df,
            x="metric_label",
            y="value",
            hue="source_label",
            hue_order=["Observed", "Null"],
            order=[metric_labels[m] for m in metric_order],
            showfliers=False,
            ax=ax,
        )
        ax.set_xlabel("metric")
        ax.set_ylabel("joint metric value")
        ax.grid(axis="y", alpha=0.25)

        fname = f"fig_global_joint_obs_vs_null_median_{idx:02d}__{_slug(null_type)}__{_slug(eval_type)}.png"
        png_path, pdf_path = _savefig_png_pdf(fig, figures_dir / fname)
        rel = Path("figures") / fname
        rel_pdf = Path("figures") / png_path.with_suffix(".pdf").name if pdf_path is not None else None
        out_paths.append((str(null_type), str(eval_type), rel, rel_pdf))
        idx += 1
    return out_paths


def _plot_global_joint_jaccard_null_median_histograms(
    global_metric_long: pd.DataFrame,
    figures_dir: Path,
) -> list[tuple[str, str, Path, Path | None]]:
    """Histograms of per-triplet null-median Jaccard values, faceted by null model."""
    out_paths: list[tuple[str, str, Path, Path | None]] = []
    if global_metric_long.empty:
        return out_paths
    plot_base = global_metric_long[
        global_metric_long["source"].eq("null median")
        & global_metric_long["metric"].eq("jaccard")
    ].dropna(subset=["value"]).copy()
    if plot_base.empty:
        return out_paths

    for eval_type in plot_base["evaluable_set_type"].astype(str).drop_duplicates():
        plot_df = plot_base.loc[plot_base["evaluable_set_type"].astype(str).eq(eval_type)].copy()
        nulls = sorted(plot_df["null_type"].astype(str).drop_duplicates(), key=viz.null_type_sort_key)
        if not nulls:
            continue
        fig, axes = plt.subplots(
            1,
            len(nulls),
            figsize=(7.2, 2.4),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for ax, null_type in zip(axes[0], nulls):
            values = pd.to_numeric(
                plot_df.loc[plot_df["null_type"].astype(str).eq(null_type), "value"],
                errors="coerce",
            ).dropna()
            ax.hist(values, bins=30, range=(0.0, 1.0), color="#5B8DEF", alpha=0.82, edgecolor="white", linewidth=0.4)
            ax.set_xlabel("Null Jaccard")
            ax.grid(axis="y", alpha=0.25)
        axes[0, 0].set_ylabel("Count")
        fig.tight_layout()

        fname = f"fig_global_joint_null_median_jaccard_hist__{_slug(eval_type)}.png"
        png_path, pdf_path = _savefig_png_pdf(fig, figures_dir / fname)
        rel = Path("figures") / fname
        rel_pdf = Path("figures") / png_path.with_suffix(".pdf").name if pdf_path is not None else None
        out_paths.append(("null median histogram", eval_type, rel, rel_pdf))
    return out_paths


def _plot_jaccard_observed_vs_null(
    summary_df: pd.DataFrame,
    figures_dir: Path,
    *,
    decoy_selection: str | None = None,
) -> tuple[
    list[tuple[str, Path, Path | None]],
    list[tuple[str, Path, Path | None]],
]:
    """Canonical Fig. 4a source: observed Jaccard versus three structured nulls."""
    out: list[tuple[str, Path, Path | None]] = []
    compact_out: list[tuple[str, Path, Path | None]] = []
    if summary_df.empty or "evaluable_set_type" not in summary_df.columns:
        return out, compact_out
    for eval_type in summary_df["evaluable_set_type"].astype(str).drop_duplicates():
        fig = viz.plot_jaccard_observed_vs_null(
            summary_df,
            evaluable_set_type=eval_type,
            decoy_selection=decoy_selection,
        )
        fname = f"fig_jaccard_mean_se__{_slug(eval_type)}.png"
        png_path, pdf_path = _savefig_png_pdf(fig, figures_dir / fname)
        rel_png = Path("figures") / fname
        rel_pdf = (
            Path("figures") / png_path.with_suffix(".pdf").name
            if pdf_path is not None
            else None
        )
        out.append((str(eval_type), rel_png, rel_pdf))

        compact_fig = viz.plot_jaccard_observed_vs_null(
            summary_df,
            evaluable_set_type=eval_type,
            decoy_selection=decoy_selection,
            figwidth=3.7555,
        )
        compact_fname = (
            "fig_jaccard_observed_vs_null_mean_bootstrap_ci_compact"
            f"__{_slug(eval_type)}.png"
        )
        compact_png, compact_pdf = _savefig_png_pdf(
            compact_fig,
            figures_dir / compact_fname,
        )
        compact_rel_png = Path("figures") / compact_fname
        compact_rel_pdf = (
            Path("figures") / compact_png.with_suffix(".pdf").name
            if compact_pdf is not None
            else None
        )
        compact_out.append(
            (str(eval_type), compact_rel_png, compact_rel_pdf)
        )
    return out, compact_out


def _plot_global_effect_boxplots(
    global_effect_long: pd.DataFrame,
    figures_dir: Path,
    *,
    decoy_selection: str | None = None,
) -> list[tuple[str, str, Path, Path | None]]:
    """Boxplots of per-triplet deltas (observed − baseline)."""
    out_paths: list[tuple[str, str, Path, Path | None]] = []
    if global_effect_long.empty:
        return out_paths
    idx = 0
    for null_type, eval_type in global_effect_long[["null_type", "evaluable_set_type"]].drop_duplicates().itertuples(index=False, name=None):
        plot_df = global_effect_long[
            global_effect_long["null_type"].eq(null_type)
            & global_effect_long["evaluable_set_type"].eq(eval_type)
        ].dropna(subset=["delta"])
        if plot_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        metric_order = [m for m in ["jaccard"] if m in set(plot_df["metric"])]
        metric_labels = {metric: viz.format_metric_label(metric) for metric in metric_order}
        plot_df = plot_df.copy()
        plot_df["metric_label"] = plot_df["metric"].map(metric_labels)
        sns.boxplot(
            data=plot_df,
            x="metric_label",
            y="delta",
            order=[metric_labels[m] for m in metric_order],
            showfliers=False,
            ax=ax,
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("metric")
        ax.set_ylabel("delta")
        ax.grid(axis="y", alpha=0.25)

        fname = f"fig_global_effect_delta_{idx:02d}__{_slug(null_type)}__{_slug(eval_type)}.png"
        png_path, pdf_path = _savefig_png_pdf(fig, figures_dir / fname)
        rel = Path("figures") / fname
        rel_pdf = Path("figures") / png_path.with_suffix(".pdf").name if pdf_path is not None else None
        out_paths.append((str(null_type), str(eval_type), rel, rel_pdf))
        idx += 1
    return out_paths


def _plot_global_joint_delta_bootstrap(
    delta_bootstrap: pd.DataFrame,
    *,
    direction: str,
    figures_dir: Path,
    decoy_selection: str | None = None,
) -> list[tuple[str, str, Path, Path | None]]:
    """Plot the primary mean paired-delta estimate with its 95% bootstrap CI."""
    out: list[tuple[str, str, Path, Path | None]] = []
    if delta_bootstrap.empty:
        return out
    eval_types = list(delta_bootstrap["evaluable_set_type"].astype(str).drop_duplicates())
    for eval_type in eval_types:
        fig = viz.plot_global_joint_triplet_delta_bootstrap(
            delta_bootstrap,
            estimator="mean",
            evaluable_set_type=eval_type,
            statistics=[posthoc.JACCARD_STATISTIC],
            decoy_selection=decoy_selection,
        )
        fname = (
            "fig_jaccard_observed_minus_null_median_mean_bootstrap_ci__"
            f"{_slug(eval_type)}.png"
        )
        png_path, pdf_path = _savefig_png_pdf(fig, figures_dir / fname)
        rel_png = Path("figures") / fname
        rel_pdf = Path("figures") / fname.replace(".png", ".pdf") if pdf_path is not None else None
        out.append((str(eval_type), "mean", rel_png, rel_pdf))
    return out


PATH_SUPPORT_POSTHOC_KEYS: tuple[str, ...] = (
    "evaluated_scope_summary",
    "triplet_delta_by_triplet",
    "triplet_delta_distribution",
    "null_repeat_stability",
    "delta_cluster_bootstrap",
    "null_evaluability_summary",
    "null_evaluability_by_triplet",
    "null_evaluability_sensitivity",
    "same_cell_triplet_metrics",
    "same_cell_delta_by_triplet",
    "same_cell_delta_summary",
)


def build_path_support_posthoc_frames(
    *,
    run_root: Path,
    direction: str,
    dfs: Mapping[str, pd.DataFrame],
    seed: int,
    n_bootstrap: int = posthoc.BOOTSTRAP_N_RESAMPLES,
) -> dict[str, pd.DataFrame]:
    """Compute the post-hoc path-support null analyses from completed evaluation outputs.

    Nothing here re-runs inference: the triplet-wise paired delta, null-repeat
    stability, drug/target/disease cluster bootstrap, null-evaluability diagnostics,
    and the same-cell-line subset are all derived from ``evaluation/<direction>/``
    tables and the cell-pair debug shards.
    """
    group_map = posthoc.load_triplet_group_map(run_root, direction)
    triplet_delta = posthoc.build_triplet_delta_table(
        dfs.get("global_joint_triplet_comparison", pd.DataFrame()),
        group_map=group_map,
    )
    scope_summary = posthoc.build_evaluated_scope_summary(
        true_joint=dfs.get("global_joint_true", pd.DataFrame()),
        baseline_joint=dfs.get("global_joint_baseline", pd.DataFrame()),
        comparison_sets=dfs.get("global_joint_triplet_comparison_sets", pd.DataFrame()),
        group_map=group_map,
    )
    cluster_bootstrap = posthoc.build_delta_cluster_bootstrap(
        triplet_delta,
        direction=direction,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    null_repeat_stability = posthoc.build_null_repeat_stability(
        true_joint=dfs.get("global_joint_true", pd.DataFrame()),
        baseline_joint=dfs.get("global_joint_baseline", pd.DataFrame()),
        direction=direction,
    )
    evaluability_summary, evaluability_by_triplet = posthoc.build_null_evaluability(
        true_joint=dfs.get("global_joint_true", pd.DataFrame()),
        baseline_joint=dfs.get("global_joint_baseline", pd.DataFrame()),
    )
    evaluability_sensitivity = posthoc.build_null_evaluability_sensitivity(
        triplet_delta,
        evaluability_by_triplet,
    )
    cell_scope_metrics = posthoc.build_cell_scope_triplet_metrics(run_root, direction)
    decoy_map = posthoc.load_decoy_pseudo_triplet_map(run_root, direction)
    same_cell_delta, same_cell_summary = posthoc.build_cell_scope_delta(
        cell_scope_metrics,
        decoy_triplet_map=decoy_map,
        direction=direction,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    return {
        "evaluated_scope_summary": scope_summary,
        "triplet_delta_by_triplet": triplet_delta,
        "triplet_delta_distribution": posthoc.summarize_triplet_delta_distribution(triplet_delta),
        "null_repeat_stability": null_repeat_stability,
        "delta_cluster_bootstrap": cluster_bootstrap,
        "null_evaluability_summary": evaluability_summary,
        "null_evaluability_by_triplet": evaluability_by_triplet,
        "null_evaluability_sensitivity": evaluability_sensitivity,
        "same_cell_triplet_metrics": cell_scope_metrics,
        "same_cell_delta_by_triplet": same_cell_delta,
        "same_cell_delta_summary": same_cell_summary,
    }


def write_path_support_posthoc_tables(
    frames: Mapping[str, pd.DataFrame],
    *,
    tables_dir: Path,
) -> dict[str, str]:
    """Write the post-hoc tables; large per-row tables go to zstd parquet like other Val3 tables."""
    parquet_keys = {
        "triplet_delta_by_triplet",
        "null_evaluability_by_triplet",
        "same_cell_triplet_metrics",
        "same_cell_delta_by_triplet",
    }
    outputs: dict[str, str] = {}
    tables_dir.mkdir(parents=True, exist_ok=True)
    for key in PATH_SUPPORT_POSTHOC_KEYS:
        df = frames.get(key, pd.DataFrame())
        if key in parquet_keys:
            outputs[key] = write_parquet_zstd(df, tables_dir / f"{key}.parquet")
        else:
            path = tables_dir / f"{key}.tsv"
            df.to_csv(path, sep="\t", index=False)
            outputs[key] = str(path)
    return outputs


def _plot_path_support_posthoc(
    frames: Mapping[str, pd.DataFrame],
    *,
    figures_dir: Path,
    statistic: str = posthoc.JACCARD_STATISTIC,
    decoy_selection: str | None = None,
) -> dict[str, list[tuple[str, Path, Path | None]]]:
    """Figures for the post-hoc path-support analyses, one set per evaluable set type."""
    triplet_delta = frames.get("triplet_delta_by_triplet", pd.DataFrame())
    eval_types = (
        list(triplet_delta["evaluable_set_type"].astype(str).drop_duplicates())
        if not triplet_delta.empty and "evaluable_set_type" in triplet_delta.columns
        else []
    )
    out: dict[str, list[tuple[str, Path, Path | None]]] = {
        "triplet_delta": [],
        "null_repeat_stability": [],
        "delta_cluster_bootstrap": [],
        "null_evaluability": [],
        "same_cell_delta": [],
    }
    if not eval_types:
        return out

    metric_slug = _slug(viz.format_metric_label(statistic).lower())
    for eval_type in eval_types:
        specs = [
            (
                "triplet_delta",
                f"fig_triplet_delta_{metric_slug}_hist__{_slug(eval_type)}.png",
                lambda et=eval_type: viz.plot_triplet_delta_distribution(
                    triplet_delta, statistic=statistic, evaluable_set_type=et,
                    decoy_selection=decoy_selection,
                ),
            ),
            (
                "null_repeat_stability",
                f"fig_null_repeat_stability__{_slug(eval_type)}.png",
                lambda et=eval_type: viz.plot_null_repeat_stability(
                    frames.get("null_repeat_stability", pd.DataFrame()),
                    statistic=statistic,
                    evaluable_set_type=et,
                    decoy_selection=decoy_selection,
                ),
            ),
            (
                "delta_cluster_bootstrap",
                f"fig_delta_cluster_bootstrap__{_slug(eval_type)}.png",
                lambda et=eval_type: viz.plot_delta_cluster_bootstrap(
                    frames.get("delta_cluster_bootstrap", pd.DataFrame()),
                    statistic=statistic,
                    evaluable_set_type=et,
                    decoy_selection=decoy_selection,
                ),
            ),
            (
                "null_evaluability",
                f"fig_null_evaluability__{_slug(eval_type)}.png",
                lambda et=eval_type: viz.plot_null_evaluability(
                    frames.get("null_evaluability_summary", pd.DataFrame()),
                    frames.get("null_evaluability_by_triplet", pd.DataFrame()),
                    evaluable_set_type=et,
                    decoy_selection=decoy_selection,
                ),
            ),
            (
                "same_cell_delta",
                f"fig_same_cell_delta_{metric_slug}__{_slug(eval_type)}.png",
                lambda et=eval_type: viz.plot_cell_scope_delta(
                    frames.get("same_cell_delta_summary", pd.DataFrame()),
                    statistic=statistic,
                    evaluable_set_type=et,
                    decoy_selection=decoy_selection,
                ),
            ),
        ]
        for key, fname, make_fig in specs:
            fig = make_fig()
            png_path, pdf_path = _savefig_png_pdf(fig, figures_dir / fname)
            rel = Path("figures") / fname
            rel_pdf = Path("figures") / png_path.with_suffix(".pdf").name if pdf_path is not None else None
            out[key].append((str(eval_type), rel, rel_pdf))
    return out


def _plot_first_edge_exclusion(
    dfs: Mapping[str, pd.DataFrame],
    figures_dir: Path,
    *,
    statistic: str = "joint_jaccard",
    decoy_selection: str | None = None,
) -> list[tuple[str, Path, Path | None]]:
    """First-edge exclusion delta figure, one per evaluable set."""
    summary = dfs.get("first_edge_exclusion_summary", pd.DataFrame())
    if summary.empty or "evaluable_set_type" not in summary.columns:
        return []
    eval_types = sorted(summary["evaluable_set_type"].astype(str).drop_duplicates())
    out: list[tuple[str, Path, Path | None]] = []
    for eval_type in eval_types:
        fig = viz.plot_first_edge_exclusion_delta(
            summary,
            statistic=statistic,
            evaluable_set_type=eval_type,
            decoy_selection=decoy_selection,
        )
        fname = f"fig_first_edge_exclusion_delta__{_slug(eval_type)}.png"
        png_path, pdf_path = _savefig_png_pdf(fig, figures_dir / fname)
        rel = Path("figures") / fname
        rel_pdf = Path("figures") / png_path.with_suffix(".pdf").name if pdf_path is not None else None
        out.append((str(eval_type), rel, rel_pdf))
    return out


def _first_edge_exclusion_markdown_lines(
    dfs: Mapping[str, pd.DataFrame],
    figure_relpaths: Mapping[str, list[tuple[Any, ...]]],
    *,
    decoy_selection: str | None = None,
) -> list[str]:
    """Markdown for the first-edge exclusion sensitivity analysis."""
    summary = dfs.get("first_edge_exclusion_summary", pd.DataFrame())
    if summary.empty:
        return []
    lines: list[str] = []
    lines.append("## First-edge exclusion sensitivity\n\n")
    lines.append(
        "- 共有標的直下 edge を除いても observed − null の差が保たれるかを見る感度解析であり、"
        "評価 triplet 集合と cell pair 集合は主解析と同一です。\n"
    )
    lines.append(
        "- drug 側・pert 側の両方について、path の root（`pert_id`）を起点とする signed edge をすべて除いた "
        "edge 集合で overlap を測り直します。下流 edge の符号は初期極性で決まるため再推論は不要です。\n"
    )
    if "path_count_length_matched_structural_resampling" not in set(summary["null_type"].astype(str)):
        lines.append(
            "- Path-shuffle（structural resampling）は primary config では無効です。有効な run では、"
            "同一の再標本 path 集合から両 scope を導出するため標本抽出は変わりません。\n"
        )
    lines.append(
        "- columns: `delta_mean_all_edges` / `delta_mean_first_edge_excluded` は各 scope の triplet 平均 delta、"
        "`delta_mean_ci025_*` / `delta_mean_ci975_*` は各 scope の triplet bootstrap 95% CI、"
        "`delta_mean_ratio` はその比、"
        "`n_pairs_empty_after_exclusion` は除外後に片側以上が空集合になった cell pair 数、"
        "`n_edges_removed_median_*` は 1 pair あたり除去された edge 数の中央値です。\n\n"
    )
    lines.append(_markdown_table(
        _display_metric_names(summary.copy(), decoy_selection=decoy_selection)
    ))
    lines.append("\n")
    lines.extend(_figure_lines(
        figure_relpaths,
        "first_edge_exclusion",
        alt="first-edge exclusion paired delta",
        description=[
            "- x 軸は null model、y 軸は Mean delta Jaccard です。棒は全 edge と first-edge 除外後を示します。\n",
            "- エラーバーは triplet bootstrap の 95% CI、棒の上の `n=` は比較 triplet 数で、両 scope で同一です。\n\n",
        ],
    ))
    return lines


def _plot_target_decoy_matching_diagnostics(
    dfs: Mapping[str, pd.DataFrame],
    figures_dir: Path,
    *,
    decoy_selection: str | None = None,
) -> list[tuple[str, Path, Path | None]]:
    """Supplementary Target-Decoy matching / evaluability figure, one per evaluable set.

    A random decoy draw writes no balance table, so nothing is plotted for it.
    """
    balance = dfs.get("decoy_matching_balance", pd.DataFrame())
    triplet_evaluability = dfs.get("decoy_triplet_evaluability", pd.DataFrame())
    if balance.empty:
        return []
    eval_types = (
        sorted(
            triplet_evaluability["evaluable_set_type"].astype(str).drop_duplicates(),
        )
        if not triplet_evaluability.empty and "evaluable_set_type" in triplet_evaluability.columns
        else []
    )
    if not eval_types:
        eval_types = sorted(
            balance.loc[balance["evaluable_set_type"].notna(), "evaluable_set_type"].astype(str).drop_duplicates()
        ) if "evaluable_set_type" in balance.columns else []
    if not eval_types:
        return []

    out: list[tuple[str, Path, Path | None]] = []
    for eval_type in eval_types:
        fig = viz.plot_target_decoy_matching_diagnostics(
            balance,
            triplet_evaluability,
            evaluable_set_type=eval_type,
            decoy_selection=decoy_selection,
            smd_threshold=TARGET_DECOY_SMD_ACCEPTANCE_THRESHOLD,
        )
        fname = f"fig_target_decoy_matching_diagnostics__{_slug(eval_type)}.png"
        png_path, pdf_path = _savefig_png_pdf(fig, figures_dir / fname)
        rel = Path("figures") / fname
        rel_pdf = Path("figures") / png_path.with_suffix(".pdf").name if pdf_path is not None else None
        out.append((str(eval_type), rel, rel_pdf))
    return out


def _target_decoy_matching_markdown_lines(
    dfs: Mapping[str, pd.DataFrame],
    figure_relpaths: Mapping[str, list[tuple[Any, ...]]],
    *,
    decoy_selection: str | None = None,
) -> list[str]:
    """Supplementary markdown for the Target-Decoy matching / evaluability diagnostics.

    A random decoy draw has no matching balance / SMD acceptance to report, so those
    tables, figures and paragraphs are skipped and only the evaluability part is shown.
    """
    acceptance = dfs.get("decoy_matching_acceptance", pd.DataFrame())
    balance = dfs.get("decoy_matching_balance", pd.DataFrame())
    evaluability = dfs.get("decoy_evaluability_summary", pd.DataFrame())
    delta_strata = dfs.get("decoy_delta_by_n_evaluable_decoys", pd.DataFrame())
    if all(df.empty for df in [acceptance, balance, evaluability, delta_strata]):
        return []
    has_matching = not (acceptance.empty and balance.empty)

    lines: list[str] = []
    lines.append("## Target-Decoy matching diagnostics（supplementary）\n\n")
    if has_matching:
        lines.append(
            "- topology- and profile-matched decoy target control の matching balance と "
            "decoy 評価可能性の診断です。主要結果の図には含めず、supplementary 用の表と図としてのみ示します。\n\n"
        )
    else:
        lines.append(
            "- decoy 評価可能性の診断です。主要結果の図には含めず、supplementary 用の表としてのみ示します。\n\n"
        )
    if str(decoy_selection) == "random":
        lines.append(
            "- decoy selection: random（同一 candidate universe からの一様ランダム抽出。"
            "matching balance / SMD acceptance は定義上評価しません）\n\n"
        )
    elif decoy_selection is not None:
        lines.append(
            f"- decoy selection: {decoy_selection}（1:N percentile-rank nearest-neighbour matching）\n\n"
        )

    if not acceptance.empty:
        lines.append("### Matching balance acceptance\n\n")
        lines.append(
            f"- primary acceptance は `analysis_scope = primary_path_evaluable`、`stage = pre_inference_selected`、"
            f"`weighting = triplet_weighted` の matching 3 変数について |SMD| < {TARGET_DECOY_SMD_ACCEPTANCE_THRESHOLD:g} で判定します。\n"
        )
        lines.append(
            "- target-weighted、post-evaluability、total degree の行は `decoy_matching_balance.tsv` に残しますが、"
            "acceptance の判定には使いません。\n"
        )
        lines.append(
            "- columns: `variable` は matching 変数、`smd` / `abs_smd` は observed − decoy の標準化平均差、"
            "`abs_smd_threshold` は判定閾値、`is_within_threshold` は変数単位の判定、"
            "`max_abs_smd` / `all_matching_variables_within_threshold` は evaluable set 単位の判定結果です。\n\n"
        )
        lines.append(_markdown_table(
            _display_metric_names(acceptance.copy(), decoy_selection=decoy_selection)
        ))
        lines.append("\n")

    if not evaluability.empty:
        lines.append("### Decoy candidate evaluability\n\n")
        lines.append(
            "- `candidate_level_evaluable_rate` の分母は選択した decoy candidate 全体（`n_selected_decoys_total`）で、"
            "comparison 行数ではありません。\n"
        )
        lines.append(
            "- columns: `n_selected_decoys_total` は 1 triplet あたり 10 candidate の合計、"
            "`n_evaluable_decoys_total` はそのうち推論後に評価可能だった数、`triplet_level_coverage` は "
            "observed path-evaluable triplet のうち評価可能 decoy を 1 つ以上持つ割合、"
            "`n_evaluable_decoys_histogram` は |E_i| の分布、`n_comparison_triplets` は Target-Decoy 比較に入った triplet 数です。\n\n"
        )
        lines.append(_markdown_table(
            _display_metric_names(evaluability.copy(), decoy_selection=decoy_selection)
        ))
        lines.append("\n")

    if not delta_strata.empty:
        lines.append("### Triplet delta by |E_i|\n\n")
        lines.append(
            "- triplet-wise delta は評価可能 decoy Jaccard の median から計算します。|E_i| による系統差がないかを確認する診断表です。\n"
        )
        lines.append(
            "- columns: `n_evaluable_decoys` は |E_i|、`n_triplets` はその層の triplet 数、"
            "`delta_mean` / `delta_sd` は層内の delta の平均・標準偏差です。\n\n"
        )
        lines.append(_markdown_table(
            _display_metric_names(delta_strata.copy(), decoy_selection=decoy_selection)
        ))
        lines.append("\n")

    lines.extend(_figure_lines(
        figure_relpaths,
        "target_decoy_matching",
        alt="Target-Decoy matching balance and evaluable decoy counts",
        description=[
            "- 左パネル: y 軸は matching 変数、x 軸は observed − decoy の SMD です。濃い棒が matching 変数、"
            "薄い棒は診断のみの total degree で、破線は |SMD| 閾値です。\n",
            "- 右パネル: x 軸は triplet ごとの評価可能 decoy 数 |E_i|、y 軸は triplet 数です。|E_i| = 0 の triplet は "
            "Target-Decoy 比較からのみ除外されます。\n\n",
        ],
    ))
    return lines


def _plot_triplet_and_pert_sections(
    dfs: Mapping[str, pd.DataFrame],
    figures_dir: Path,
) -> list[tuple[str, Path, Path | None]]:
    """Observed triplet / target distribution figures; returns (title, relative_path)."""
    out: list[tuple[str, Path, Path | None]] = []
    triplet_true = dfs.get("triplet_true", pd.DataFrame())
    pert_col = "target_id" if "target_id" in triplet_true.columns else "pert_id"

    if triplet_true.empty:
        return out

    viz.apply_val3_paper_style()

    for metric in viz.DEFAULT_METRICS:
        if metric not in triplet_true.columns:
            continue
        metric_label = viz.format_metric_label(metric)
        fig = viz.plot_observed_triplet_metric_box_hist_pair(
            triplet_true,
            metric,
            title=metric_label,
        )
        rel = Path("figures") / f"fig_observed_triplet_{metric}_box_hist.png"
        _, pdf_path = _savefig_png_pdf(fig, figures_dir / rel.name)
        rel_pdf = Path("figures") / rel.with_suffix(".pdf").name if pdf_path is not None else None
        out.append((f"Observed triplet {metric_label} (box + hist)", rel, rel_pdf))

    if pert_col not in triplet_true.columns:
        return out

    fig = viz.plot_observed_pert_metrics_box_hist_faceted(triplet_true, pert_col=pert_col)
    rel2 = Path("figures") / "fig_observed_target_metrics_box_hist.png"
    _, pdf_path2 = _savefig_png_pdf(fig, figures_dir / rel2.name)
    rel_pdf2 = Path("figures") / rel2.with_suffix(".pdf").name if pdf_path2 is not None else None
    out.append((f"Observed target metrics aggregated by `{pert_col}` (box + hist)", rel2, rel_pdf2))
    return out


def _figure_lines(
    figure_relpaths: Mapping[str, list[tuple[Any, ...]]],
    key: str,
    *,
    alt: str,
    description: list[str],
) -> list[str]:
    lines: list[str] = []
    for entry in figure_relpaths.get(key, []):
        eval_type, rel, rel_pdf = entry[0], entry[1], entry[2]
        lines.append(f"#### Figure: {alt} ({eval_type})\n\n")
        lines.extend(description)
        lines.append(f"![{alt}]({rel.as_posix()})\n\n")
        if rel_pdf is not None:
            lines.append(f"PDF: `{rel_pdf.as_posix()}`\n\n")
    return lines


def _path_support_posthoc_markdown_lines(
    dfs: Mapping[str, pd.DataFrame],
    figure_relpaths: Mapping[str, list[tuple[Any, ...]]],
    *,
    decoy_selection: str | None = None,
) -> list[str]:
    """Markdown for the post-hoc path-support null analyses (existing outputs only)."""
    frames = {key: dfs.get(key, pd.DataFrame()) for key in PATH_SUPPORT_POSTHOC_KEYS}
    if all(df.empty for df in frames.values()):
        return []

    lines: list[str] = []
    lines.append("## Path support の追加 null 解析\n\n")
    lines.append("- 既存の `evaluation/<direction>/` 出力と cell-pair debug shard のみから計算した追加解析です。推論の再実行は行っていません。\n")
    lines.append("- Observed − null median の triplet 分布、null-repeat stability、drug / target / disease cluster bootstrap、null evaluability、same-cell-line subset を扱います。\n\n")

    scope = frames["evaluated_scope_summary"]
    if not scope.empty:
        lines.append("### Evaluated scope\n\n")
        lines.append("- observed と各 null model について、評価対象の triplet 数と unique drug / target / disease 数、null 側の評価可能率をまとめます。\n")
        lines.append("- columns: `scope` は observed/null、`n_triplets` は評価対象 triplet 数、`n_unique_*` は該当 ID のユニーク数、`n_null_rows` は null draw（shuffle iteration または decoy anchor）の行数、`n_null_rows_evaluable` はそのうち score が得られた行数、`null_evaluable_rate` はその割合（分母は `null_evaluable_rate_denominator` に明記、Target-Decoy の candidate-level 率は `decoy_evaluability_summary.tsv`）、`n_comparison_triplets` は比較に用いた triplet 数、`n_triplets_without_null_score` は observed 側で評価可能だが null score が 1 つも無い triplet 数です。\n\n")
        lines.append(_markdown_table(
            _display_metric_names(scope.copy(), decoy_selection=decoy_selection)
        ))
        lines.append("\n")

    distribution = frames["triplet_delta_distribution"]
    if not distribution.empty:
        lines.append("### Triplet-wise Observed − null median\n\n")
        lines.append("- `global_joint_triplet_comparison.tsv` の triplet 単位 paired delta の分布要約です。集約値ではなく triplet ごとのばらつきを確認します。\n")
        lines.append("- columns: `delta_mean` / `delta_median` は paired delta の平均・中央値、`delta_q25` / `delta_q75` は四分位点、`n_positive_delta` / `n_negative_delta` は observed が null median より大きい／小さい triplet 数、`fraction_positive_delta` はその割合です。`p_wilcoxon_two_sided` は paired delta を 0 と比較する両側 Wilcoxon 符号付順位検定です。\n\n")
        table = _display_metric_names(
            distribution.copy(), decoy_selection=decoy_selection,
        )
        if "p_wilcoxon_two_sided" in table.columns:
            table["p_wilcoxon_two_sided"] = table["p_wilcoxon_two_sided"].map(
                _format_p_value
            )
        lines.append(_markdown_table(table))
        lines.append("\n")
        lines.extend(_figure_lines(
            figure_relpaths,
            "triplet_delta",
            alt="triplet-wise observed − null median distribution",
            description=[
                "- x 軸: triplet ごとの Observed − null median、y 軸: triplet 数です。パネルは null model ごとに分かれています。\n",
                "- 破線は 0、実線は中央値です。パネル内に triplet 数、中央値、delta > 0 の割合を記載しています。\n\n",
            ],
        ))

    repeat_stability = frames["null_repeat_stability"]
    if not repeat_stability.empty:
        lines.append("### Null-repeat stability\n\n")
        lines.append("- Gene-node shuffle と Path-shuffle について、反復数を段階的に増やし、各時点で triplet 別 null median と Mean delta Jaccard を再計算した収束診断です。Target-Decoy は反復型 null ではないため含めません。\n")
        lines.append("- columns: `n_repeats` は累積 null 反復数、`n_triplets` はその時点で有限 null median を持つ triplet 数、`delta_mean` は Mean delta Jaccard、`abs_delta_mean_change_from_full` は全反復推定値からの絶対差です。\n\n")
        lines.append(_markdown_table(
            _display_metric_names(
                repeat_stability.copy(),
                decoy_selection=decoy_selection,
            )
        ))
        lines.append("\n")
        lines.extend(_figure_lines(
            figure_relpaths,
            "null_repeat_stability",
            alt="null-repeat stability",
            description=[
                "- x 軸は累積 null 反復数、y 軸は Mean delta Jaccard です。\n",
                "- 反復数の増加に伴う主推定値の収束を示します。\n\n",
            ],
        ))

    cluster = frames["delta_cluster_bootstrap"]
    if not cluster.empty:
        lines.append("### Cluster bootstrap（drug / target / disease）\n\n")
        lines.append("- triplet を独立とみなさず、drug / target / disease を cluster として復元抽出した bootstrap です。triplet cluster は既存の triplet-level bootstrap と同じ水準の参照になります。\n")
        lines.append("- columns: `cluster_level` は resampling 単位、`n_clusters` はその単位数、`n_triplets` は triplet 数、`delta_mean_hat` は点推定、`delta_mean_ci025` / `delta_mean_ci975` は 95% percentile CI です。\n\n")
        lines.append(_markdown_table(
            _display_metric_names(cluster.copy(), decoy_selection=decoy_selection)
        ))
        lines.append("\n")
        lines.extend(_figure_lines(
            figure_relpaths,
            "delta_cluster_bootstrap",
            alt="cluster bootstrap of observed − null median",
            description=[
                "- x 軸は null model、y 軸は Mean delta Jaccard です。棒の色は resampling 単位（triplet / drug / target / disease）を示します。\n",
                "- エラーバーは cluster bootstrap の 95% CI です。CI が 0 をまたぐかどうかを確認します。\n\n",
            ],
        ))

    evaluability = frames["null_evaluability_summary"]
    sensitivity = frames["null_evaluability_sensitivity"]
    if not evaluability.empty or not sensitivity.empty:
        lines.append("### Null evaluability\n\n")
        lines.append("- null 側で score が得られなかった draw の割合と、その脱落が observed 値や cell-pair 支持と相関していないかを確認します。\n\n")
        if not evaluability.empty:
            lines.append("- columns: `null_evaluable_rate` は comparison 表に入っている null draw 行（`n_null_rows`、`null_evaluable_rate_denominator` に明記）を分母とする評価可能率、`n_triplets_all_draws_evaluable` / `n_triplets_partially_evaluable` / `n_triplets_no_null_score` は triplet 単位の内訳、`spearman_evaluable_fraction_vs_observed` は triplet ごとの評価可能割合と observed 値の Spearman 相関、`spearman_evaluable_fraction_vs_n_cell_pairs_used` は評価に使えた cell-pair 数との相関、`observed_mean_triplets_with_null_score` / `observed_mean_triplets_without_null_score` は null score の有無別に見た observed 平均です。\n\n")
            lines.append("- Target-Decoy の comparison 表には評価可能な decoy candidate だけが入るため、この列は定義上 1.0 になります。選択した 10 candidate を分母とする candidate-level 評価可能率は `candidate_level_evaluable_rate_source`（`decoy_evaluability_summary.tsv`）を参照してください。\n\n")
            lines.append(_markdown_table(
                _display_metric_names(
                    evaluability.copy(), decoy_selection=decoy_selection,
                )
            ))
            lines.append("\n")
        if not sensitivity.empty:
            lines.append("- null score が 1 つも無い triplet を delta = 0 として加えた保守的な感度解析です。\n")
            lines.append("- columns: `delta_mean_primary` / `delta_median_primary` は現行の推定、`n_triplets_without_null_score` は 0 補完した triplet 数、`delta_mean_zero_imputed` / `delta_median_zero_imputed` は 0 補完後の推定です。\n\n")
            lines.append(_markdown_table(
                _display_metric_names(
                    sensitivity.copy(), decoy_selection=decoy_selection,
                )
            ))
            lines.append("\n")
        lines.extend(_figure_lines(
            figure_relpaths,
            "null_evaluability",
            alt="triplet-level null-score coverage",
            description=[
                "- x 軸は null model、y 軸は共通の observed-evaluable triplet のうち少なくとも 1 つ null score が得られた割合です。\n",
                "- 全 null model で同じ 1,570 observed triplet を分母とし、null draw 行を分母とする `null_evaluable_rate` とは区別します。\n\n",
            ],
        ))

    same_cell = frames["same_cell_delta_summary"]
    if not same_cell.empty:
        lines.append("### Same-cell-line subset\n\n")
        lines.append("- cell-pair debug shard から triplet 指標を再構成し、`cell_id_drug == cell_id_pert` の cell-pair のみに限定した部分集合と、全 cell-pair 集約を同一 triplet 上で比較します。\n")
        lines.append("- Target-Decoy は decoy candidate（pseudo triplet）単位の cell-pair debug shard から same-cell subset を再構成し、original triplet ごとに evaluable decoy の median を null 代表値としています。\n")
        lines.append("- columns: `cell_scope` は `all_cell_pairs` / `same_cell`、`n_triplets` は比較 triplet 数、`n_cell_pairs_used_observed_total` は observed 側で使えた cell-pair 数、`baseline_median_mean` は triplet 別 null median の triplet 間平均、`delta_mean` は主推定量、`delta_mean_ci025` / `delta_mean_ci975` は triplet bootstrap 95% CI です。\n")
        lines.append("- `p_wilcoxon_two_sided` は同じ triplet の `observed - null median` を 0 と比較する両側 Wilcoxon 符号付順位検定、`p_wilcoxon_two_sided_holm` は cell scope ごとに null model 間で Holm 補正した値です。zero delta は `zero_method=\"wilcox\"` により順位計算から除外します。\n\n")
        table = _display_metric_names(
            same_cell.copy(), decoy_selection=decoy_selection,
        )
        for col in ["p_wilcoxon_two_sided", "p_wilcoxon_two_sided_holm"]:
            if col in table.columns:
                table[col] = table[col].map(_format_p_value)
        lines.append(_markdown_table(table))
        lines.append("\n")
        lines.extend(_figure_lines(
            figure_relpaths,
            "same_cell_delta",
            alt="same-cell-line subset paired delta",
            description=[
                "- x 軸は null model、y 軸は Mean delta Jaccard です。棒は全 cell-pair と同一細胞株 subset を示します。\n",
                "- エラーバーは triplet bootstrap の 95% CI、棒の上の `n=` は比較 triplet 数です。\n\n",
            ],
        ))
    return lines


def build_report_markdown(
    *,
    run_root: Path,
    report_dir: Path,
    direction: str,
    dfs: Mapping[str, pd.DataFrame],
    figure_relpaths: Mapping[str, list[tuple[Any, ...]]],
    decoy_selection: str | None = None,
) -> str:
    """Assemble markdown string (figures referenced with paths relative to the md file)."""
    evaluation_root = run_root / "evaluation" / direction
    meta = _read_json(run_root / "metadata.json") or {}
    if decoy_selection is None:
        decoy_selection = _run_decoy_selection(meta)
    run_summary = _read_json(run_root / "run_summary.json") or {}

    lines: list[str] = []
    lines.append(f"{VAL3_TRIPLET_REVISED_METHODOLOGY_VERSION}\n\n")
    lines.append(f"# Val3 triplet visualization report (`{run_root.name}`)\n\n")
    lines.append("evaluation 出力（`evaluation/<direction>/`）から、triplet 評価の要約表と確認用の図をまとめたレポートです。\n\n")
    lines.append(f"- `run_root`: `{run_root}`\n")
    lines.append(f"- `evaluation_root`: `{evaluation_root}`\n\n")

    lines.append("## 実験設定\n\n")
    dirs = meta.get("analysis", {}) if isinstance(meta.get("analysis"), dict) else {}
    directions = dirs.get("directions") if isinstance(dirs, dict) else None
    direction_guess = ""
    if isinstance(directions, list) and directions:
        direction_guess = str(directions[0])
    lines.append("- この run の方向・path 探索条件・baseline 条件を確認するための最小情報です。\n")
    lines.append("- columns: `key` は設定項目名、`value` はこの run で使われた値です。\n\n")
    lines.append(_markdown_table(pd.DataFrame([
        {"key": "config_path", "value": str(meta.get("config_path", ""))},
        {"key": "seed", "value": str(meta.get("seed", ""))},
        {"key": "direction", "value": str(run_summary.get("direction") or direction_guess or direction)},
        {"key": "drug_mode", "value": str(run_summary.get("drug_mode", ""))},
        {"key": "pert_mode", "value": str(run_summary.get("pert_mode", ""))},
        {"key": "optimal_effective_distance_k", "value": str((meta.get("analysis") or {}).get("k_star", ""))},
        {"key": "gap", "value": str((meta.get("analysis") or {}).get("gap", ""))},
        {"key": "cell_pair_mode", "value": str((meta.get("analysis") or {}).get("cell_pair_mode", ""))},
        {"key": "baseline.n_shuffle", "value": str((meta.get("baseline") or {}).get("n_shuffle", ""))},
        {"key": "baseline.is_primary_spec", "value": str((meta.get("baseline") or {}).get("is_primary_spec", ""))},
    ])))
    lines.append("\n")

    lines.append("## Observed core tables\n\n")
    lines.append("- 最小保存する observed triplet 指標と diagnostics の行数・主要 ID 数です。\n")
    lines.append("- columns: `table` は評価出力名、`n_rows` / `n_cols` は行数・列数、`n_*` は該当 ID やカテゴリのユニーク数です。\n\n")
    lines.append(_markdown_table(_overview_rows_core(dfs)))
    lines.append("\n")

    tables_dir = report_dir / "tables"
    coverage_summary = _load_report_table(tables_dir / "coverage_summary_by_null.tsv")
    null_diagnostics = _load_report_table(tables_dir / "null_diagnostics_summary.tsv")
    delta_bootstrap = dfs.get("global_joint_triplet_delta_bootstrap", pd.DataFrame())
    if delta_bootstrap.empty:
        delta_bootstrap = _load_report_table(tables_dir / "global_joint_triplet_delta_bootstrap.tsv")
    if not coverage_summary.empty or not null_diagnostics.empty:
        lines.append("## Report summary tables\n\n")
        lines.append("- `report/tables/` 配下に保存された、coverage と null diagnostics の要約です。\n\n")
        if not coverage_summary.empty:
            lines.append("### Coverage by null model\n\n")
            lines.append("- 各 null model と evaluable set ごとの評価対象 coverage を確認します。\n")
            lines.append("- columns: `source` は observed/null、`null_type` は null model、`evaluable_set_type` は評価対象セット、`n_rows` / `n_triplets` / `n_iterations` は件数、`n_cell_pairs_*` は cell-pair 数、`path_recoverability_mean` は path が存在した raw pair の平均割合です。\n\n")
            lines.append(_markdown_table(
                _display_metric_names(
                    coverage_summary, decoy_selection=decoy_selection,
                )
            ))
            lines.append("\n")
        if not null_diagnostics.empty:
            lines.append("### Null diagnostics\n\n")
            lines.append("- null model のサンプリングや候補集合が想定どおりかを確認する表です。\n")
            lines.append("- columns: `null_type` は null model、`diagnostic` は診断対象、`n_rows` / `n_triplets` は診断表の規模、`summary_value` は `summary_name` が示す代表値です。\n\n")
            lines.append(_markdown_table(
                _display_metric_names(
                    null_diagnostics, decoy_selection=decoy_selection,
                )
            ))
            lines.append("\n")

    lines.append("## Observed triplet / target distributions\n\n")
    lines.append("- observed の triplet 単位指標と、`target_id`（なければ `pert_id`）で集約した target 単位指標です。\n")
    lines.append("- Val3 の statistic は Jaccard のみに統一します。\n\n")
    triplet_true = dfs.get("triplet_true", pd.DataFrame())
    if triplet_true.empty:
        lines.append("_`triplet_true` が空のためスキップ_\n\n")
    else:
        triplet_summary = viz.summarize_observed_triplet_metrics(triplet_true)
        lines.append("### Triplet-level observed summary\n\n")
        lines.append("- 1 行 = 1 triplet の observed Jaccard 分布要約です。\n\n")
        lines.append("- columns: `metric` は指標名、`n` は有効値数、`mean` / `median` / `std` は平均・中央値・標準偏差、`q25` / `q75` は四分位点、`min` / `max` は最小・最大です。\n\n")
        lines.append(_markdown_table(_display_metric_names(triplet_summary)))
        lines.append("\n")
        pert_col = "target_id" if "target_id" in triplet_true.columns else "pert_id"
        if pert_col in triplet_true.columns:
            pert_agg = viz.aggregate_observed_metrics_by_pert(triplet_true, pert_col=pert_col)
            pert_summary = viz.summarize_observed_triplet_metrics(pert_agg)
            lines.append("### Target-level observed summary\n\n")
            lines.append(f"- `{pert_col}` ごとに triplet 指標を平均した分布要約です。\n\n")
            lines.append("- columns: `metric` は指標名、`n` は target 数、`mean` / `median` / `std` は平均・中央値・標準偏差、`q25` / `q75` は四分位点、`min` / `max` は最小・最大です。\n\n")
            lines.append(_markdown_table(_display_metric_names(pert_summary)))
            lines.append("\n")

    for title, rel, rel_pdf in figure_relpaths.get("triplet_pert", []):
        lines.append(f"### Figure: {title}\n\n")
        if title.startswith("Observed triplet "):
            lines.append("- 1 つの評価指標について、triplet 単位の observed 分布を示します。\n")
            lines.append("- 左パネル: x 軸は observed group、y 軸は metric 値です。箱は四分位範囲、中央線は中央値、点は個別 triplet です。\n")
        else:
            lines.append("- Jaccard の target 単位分布を示します。\n")
            lines.append("- 左パネル: x 軸は observed group、y 軸は metric 値です。箱は四分位範囲、中央線は中央値、点は個別 target です。\n")
        lines.append("- 右パネル: x 軸は metric 値、y 軸は count です。値がどの範囲に集中しているかを見ます。\n\n")
        lines.append(f"![{title}]({rel.as_posix()})\n\n")
        if rel_pdf is not None:
            lines.append(f"PDF: `{rel_pdf.as_posix()}`\n\n")

    lines.append("## Edge / cell-pair diagnostics\n\n")
    lines.append("- path 数・edge 数と、cell-pair filter による評価可能ペア数を確認する診断表です。\n")
    lines.append("- observed 行が主な確認対象です。baseline 行は該当する要約入力がある場合のみ表示されます。\n\n")
    edge_diag = viz.summarize_edge_count_diagnostics(
        dfs.get("triplet_true", pd.DataFrame()),
        dfs.get("triplet_baseline", pd.DataFrame()),
    )
    pair_diag = viz.summarize_evaluable_pair_diagnostics(
        dfs.get("diag_true", pd.DataFrame()),
        dfs.get("diag_baseline", pd.DataFrame()),
    )
    lines.append("### Edge/path count diagnostics\n\n")
    lines.append("- columns: `source` は observed/baseline、`quantity` は path/edge 数の種類、`n` は有効値数、`mean` / `median` は代表値です。\n\n")
    lines.append(_markdown_table(edge_diag))
    lines.append("\n")
    lines.append("### Cell-pair filter diagnostics\n\n")
    lines.append("- columns: `source` は observed/baseline、`n_cell_pairs_raw` は候補 cell-pair 数、`n_cell_pairs_evaluable` は評価可能数、`fraction_evaluable` はその割合、`n_excluded_*` は除外理由別の件数です。\n\n")
    lines.append(_markdown_table(pair_diag))
    lines.append("\n")

    lines.append("## Global result tables\n\n")
    lines.append("- global joint / paired delta などの出力規模を確認する表です。\n\n")
    lines.append("- columns: `table` は出力名、`n_rows` / `n_cols` は行数・列数、`n_null_type` / `n_evaluable_set_type` / `n_statistic` / `n_iteration` は各カテゴリのユニーク数です。\n\n")
    lines.append(_markdown_table(_overview_rows_global(dfs)))
    lines.append("\n")

    lines.append("## Global null summaries\n\n")
    lines.append("global null 要約では、triplet ごとの null 代表値を median に統一し、Mean (Observed − null median) と triplet bootstrap 95% CI を主結果とします。統計検定は両側 Wilcoxon 符号付順位検定だけを行い、null model 間で Holm 補正します。\n\n")
    observed_vs_null = dfs.get(
        "jaccard_observed_vs_null_summary",
        pd.DataFrame(),
    )
    if not observed_vs_null.empty:
        lines.append("### Observed Jaccard versus structured nulls\n\n")
        lines.append("- Observed Jaccard と3種類の structured null の triplet-specific median を同じ尺度で比較します。\n")
        lines.append("- columns: `observed_mean` / `null_median_mean` は triplet 間平均、`*_mean_ci025` / `*_mean_ci975` は同一tripletを対で再標本化した percentile bootstrap 95% CI、`p_wilcoxon_two_sided_holm` は paired ΔJaccard の両側 Wilcoxon 検定を3 null 間で Holm 補正した値です。\n\n")
        table = _display_metric_names(
            observed_vs_null.copy(),
            decoy_selection=decoy_selection,
        )
        if "p_wilcoxon_two_sided_holm" in table.columns:
            table["p_wilcoxon_two_sided_holm"] = table[
                "p_wilcoxon_two_sided_holm"
            ].map(_format_p_value)
        lines.append(_markdown_table(table))
        lines.append("\n")
        lines.extend(_figure_lines(
            figure_relpaths,
            "observed_vs_null",
            alt="observed Jaccard versus structured nulls",
            description=[
                "- x 軸は null model、y 軸は Mean Jaccard です。\n",
                "- 濃色は observed、淡色は triplet-specific null median、エラーバーは paired triplet bootstrap 95% CI です。\n\n",
            ],
        ))

    if delta_bootstrap.empty:
        lines.append("### Triplet-level paired delta bootstrap\n\n_`report/tables/global_joint_triplet_delta_bootstrap.tsv` が空または未生成のためスキップ_\n\n")
    else:
        cols = [c for c in [
            "direction", "null_type", "evaluable_set_type", "statistic", "estimand",
            "n_triplets", "n_bootstrap", "bootstrap_seed", "resampling_unit", "ci_method", "ci_level",
            "delta_mean_hat", "delta_mean_ci025", "delta_mean_ci975",
        ] if c in delta_bootstrap.columns]
        lines.append("### Triplet-level paired delta bootstrap\n\n")
        lines.append("- `global_joint_triplet_comparison.tsv` の triplet-level paired delta を用いた post-hoc bootstrap です。\n")
        lines.append("- 各 group 内で `triplet_id` を復元抽出し、mean(Observed − null median) の 95% percentile CI を計算します。\n")
        lines.append("- columns: `delta_mean_hat` は主点推定、`delta_mean_ci025` / `delta_mean_ci975` は triplet bootstrap 95% CI、`n_bootstrap` は再標本化回数です。\n\n")
        table = _display_metric_names(
            delta_bootstrap[cols].copy(), decoy_selection=decoy_selection,
        )
        lines.append(_markdown_table(table))
        lines.append("\n")
        for eval_type, estimator, rel_png, rel_pdf in figure_relpaths.get("delta_bootstrap", []):
            lines.append(f"#### Figure: paired delta bootstrap ({eval_type}, {estimator})\n\n")
            lines.append("- x 軸は null model、y 軸は Mean delta Jaccard です。0 より上なら observed が null/control より大きいことを示します。\n")
            lines.append("- 棒は Jaccard paired delta の平均、エラーバーは triplet bootstrap 95% CI です。\n\n")
            lines.append(f"![paired delta bootstrap {estimator}]({rel_png.as_posix()})\n\n")
            if rel_pdf is not None:
                lines.append(f"PDF: `{rel_pdf.as_posix()}`\n\n")

    lines.append("## Null-median Jaccard diagnostics\n\n")
    lines.append("- path-existing pairs に限定し、triplet 別 null median Jaccard の分布を null model ごとに確認します。\n\n")
    for kind, eval_type, rel, rel_pdf in figure_relpaths.get("global_joint_jaccard", []):
        if kind == "null median histogram":
            lines.append(f"### Figure: null median distribution ({eval_type}, Jaccard)\n\n")
            lines.append("- x 軸: triplet ごとの null median Jaccard、y 軸: count です。\n")
            lines.append("- 3 つの null model を横並びにし、null median がどの範囲に集中するかを確認します。\n\n")
            alt = "global joint null median Jaccard histogram"
        else:
            continue
        lines.append(f"![{alt}]({rel.as_posix()})\n\n")
        if rel_pdf is not None:
            lines.append(f"PDF: `{rel_pdf.as_posix()}`\n\n")

    lines.extend(_first_edge_exclusion_markdown_lines(
        dfs, figure_relpaths, decoy_selection=decoy_selection,
    ))
    lines.extend(
        _target_decoy_matching_markdown_lines(
            dfs, figure_relpaths, decoy_selection=decoy_selection,
        )
    )
    lines.extend(_path_support_posthoc_markdown_lines(
        dfs, figure_relpaths, decoy_selection=decoy_selection,
    ))
    return "".join(lines)

def write_val3_triplet_visualization_report(
    *,
    run_root: Path,
    direction: str,
    output_md: Path | None = None,
    figures_dir: Path | None = None,
    config: Mapping[str, Any] | None = None,
    evaluation_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write Val3 tables, figures, then the Markdown report."""
    run_root = run_root.expanduser().resolve()
    if output_md is None:
        output_md = run_root / "report" / VAL3_REPORT_FILENAME
    else:
        output_md = output_md.expanduser().resolve()
    report_dir = output_md.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = report_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    if figures_dir is None:
        figures_dir = report_dir / "figures"
    else:
        figures_dir = figures_dir.expanduser().resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)

    dfs = _load_evaluation_artifacts(run_root, direction)
    meta = _read_json(run_root / "metadata.json") or {}
    seed = _safe_int(meta.get("seed", 0), default=0)
    n_bootstrap = posthoc.BOOTSTRAP_N_RESAMPLES
    dfs["first_edge_exclusion_summary"] = _median_first_edge_exclusion_summary(
        dfs,
        direction=direction,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    dfs["decoy_delta_by_n_evaluable_decoys"] = _median_target_decoy_delta_strata(
        dfs.get("global_joint_triplet_comparison", pd.DataFrame()),
    )
    report_tables = build_v3_report_tables(
        run_root=run_root,
        direction=direction,
        config=config,
        evaluation_summary=evaluation_summary,
        report_dir=report_dir,
    )
    first_edge_report_summary = dfs.get("first_edge_exclusion_summary", pd.DataFrame())
    if not first_edge_report_summary.empty:
        first_edge_report_path = tables_dir / "first_edge_exclusion_summary.tsv"
        first_edge_report_summary.to_csv(first_edge_report_path, sep="\t", index=False)
        report_tables["first_edge_exclusion_summary"] = str(first_edge_report_path)
    # The Target-Decoy null_type string is identical for both selection modes, so the
    # figure label has to come from the run metadata and is read once here.
    decoy_selection = _run_decoy_selection(meta)
    delta_bootstrap = build_global_joint_triplet_delta_bootstrap(
        dfs.get("global_joint_triplet_comparison", pd.DataFrame()),
        direction=direction,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    delta_bootstrap_path = tables_dir / "global_joint_triplet_delta_bootstrap.tsv"
    delta_bootstrap.to_csv(delta_bootstrap_path, sep="\t", index=False)
    report_tables["global_joint_triplet_delta_bootstrap"] = str(delta_bootstrap_path)
    dfs = dict(dfs)
    dfs["global_joint_triplet_delta_bootstrap"] = delta_bootstrap

    posthoc_frames = build_path_support_posthoc_frames(
        run_root=run_root,
        direction=direction,
        dfs=dfs,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    report_tables.update(write_path_support_posthoc_tables(posthoc_frames, tables_dir=tables_dir))
    dfs.update(posthoc_frames)

    jaccard_paired_wilcoxon_holm = _jaccard_paired_wilcoxon_holm_table(dfs)
    jaccard_paired_wilcoxon_holm_path = (
        tables_dir / "jaccard_paired_wilcoxon_holm.tsv"
    )
    jaccard_paired_wilcoxon_holm.to_csv(
        jaccard_paired_wilcoxon_holm_path, sep="\t", index=False,
    )
    report_tables["jaccard_paired_wilcoxon_holm"] = str(
        jaccard_paired_wilcoxon_holm_path
    )

    global_metric_long = _make_global_metric_long(dfs)
    observed_vs_null_summary = _build_jaccard_observed_vs_null_summary(
        global_metric_long,
        jaccard_paired_wilcoxon_holm,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    observed_vs_null_summary_path = (
        tables_dir / "jaccard_observed_vs_null_summary.tsv"
    )
    observed_vs_null_summary.to_csv(
        observed_vs_null_summary_path, sep="\t", index=False,
    )
    report_tables["jaccard_observed_vs_null_summary"] = str(
        observed_vs_null_summary_path
    )
    dfs["jaccard_observed_vs_null_summary"] = observed_vs_null_summary

    viz.apply_val3_paper_style()

    triplet_pert_figs = _plot_triplet_and_pert_sections(dfs, figures_dir)
    global_effect_long = _make_global_effect_long(dfs)
    observed_vs_null_figs, observed_vs_null_compact_figs = (
        _plot_jaccard_observed_vs_null(
        observed_vs_null_summary,
        figures_dir,
        decoy_selection=decoy_selection,
        )
    )
    global_joint_jaccard_figs = _plot_global_joint_jaccard_null_median_histograms(
        global_metric_long, figures_dir,
    )
    delta_bootstrap_figs = _plot_global_joint_delta_bootstrap(
        delta_bootstrap, direction=direction, figures_dir=figures_dir, decoy_selection=decoy_selection,
    )
    posthoc_figs = _plot_path_support_posthoc(
        posthoc_frames, figures_dir=figures_dir, decoy_selection=decoy_selection,
    )
    target_decoy_figs = _plot_target_decoy_matching_diagnostics(
        dfs, figures_dir, decoy_selection=decoy_selection,
    )
    first_edge_figs = _plot_first_edge_exclusion(
        dfs, figures_dir, decoy_selection=decoy_selection,
    )

    figure_relpaths: dict[str, Any] = {
        "triplet_pert": triplet_pert_figs,
        "observed_vs_null": observed_vs_null_figs,
        "observed_vs_null_compact": observed_vs_null_compact_figs,
        "global_joint_jaccard": global_joint_jaccard_figs,
        "delta_bootstrap": delta_bootstrap_figs,
        "target_decoy_matching": target_decoy_figs,
        "first_edge_exclusion": first_edge_figs,
        **posthoc_figs,
    }

    md = build_report_markdown(
        run_root=run_root,
        report_dir=report_dir,
        direction=direction,
        dfs=dfs,
        figure_relpaths=figure_relpaths,
        decoy_selection=decoy_selection,
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(md, encoding="utf-8")

    return {
        "output_md": str(output_md),
        "figures_dir": str(figures_dir),
        "tables_dir": str(tables_dir),
        "report_tables": report_tables,
        "n_global_metric_long_rows": int(len(global_metric_long)),
        "n_global_effect_long_rows": int(len(global_effect_long)),
        "global_joint_triplet_delta_bootstrap": str(delta_bootstrap_path),
        "n_global_joint_triplet_delta_bootstrap_rows": int(len(delta_bootstrap)),
        "jaccard_observed_vs_null_summary": str(
            observed_vs_null_summary_path
        ),
        "n_jaccard_observed_vs_null_summary_rows": int(
            len(observed_vs_null_summary)
        ),
        "jaccard_observed_vs_null_compact_figures": [
            {
                "evaluable_set_type": eval_type,
                "png": str(report_dir / rel_png),
                "pdf": (
                    str(report_dir / rel_pdf)
                    if rel_pdf is not None
                    else None
                ),
            }
            for eval_type, rel_png, rel_pdf in observed_vs_null_compact_figs
        ],
        "path_support_posthoc_rows": {
            key: int(len(posthoc_frames.get(key, pd.DataFrame()))) for key in PATH_SUPPORT_POSTHOC_KEYS
        },
    }


def parse_report_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate a Val3 Markdown report from completed outputs.")
    p.add_argument("--run-root", type=Path, required=True, help="Run root containing evaluation/<direction>/ outputs.")
    p.add_argument("--direction", default="inhibitory", help="Evaluation subdirectory name.")
    p.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help=f"Output markdown (default: <run_root>/report/{VAL3_REPORT_FILENAME}).",
    )
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Figures directory (default: parent of output-md / figures).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_report_args(argv)
    outputs = write_val3_triplet_visualization_report(
        run_root=args.run_root,
        direction=str(args.direction),
        output_md=args.output_md,
        figures_dir=args.figures_dir,
    )
    run_summary_path = args.run_root.expanduser().resolve() / "run_summary.json"
    run_summary = _read_json(run_summary_path)
    if run_summary:
        prior_report_outputs = run_summary.get("report_outputs")
        merged_report_outputs = dict(outputs)
        if isinstance(prior_report_outputs, Mapping):
            cleanup_after_report = prior_report_outputs.get("cleanup_after_report")
            if cleanup_after_report is not None:
                merged_report_outputs["cleanup_after_report"] = cleanup_after_report
        run_summary["report_outputs"] = merged_report_outputs
        run_summary_path.write_text(
            json.dumps(run_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return outputs


# ---------------------------------------------------------------------------
# Val3 v3 final report table helpers
# ---------------------------------------------------------------------------

def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return _read_parquet(path)
    try:
        return pd.read_csv(path, sep="\t")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _metric_cols(df: pd.DataFrame, *, prefix: str = "") -> list[str]:
    names = []
    for m in ["jaccard"]:
        col = f"{prefix}{m}"
        if col in df.columns:
            names.append(col)
    return names


def build_metric_triplet_long(true_joint: pd.DataFrame, baseline_joint: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not true_joint.empty:
        for rec in true_joint.to_dict(orient="records"):
            for metric in ["jaccard"]:
                col = f"joint_{metric}"
                if col not in rec:
                    continue
                rows.append({
                    "source": "observed",
                    "null_type": "observed",
                    "iteration": None,
                    "evaluable_set_type": rec.get("evaluable_set_type"),
                    "triplet_id": rec.get("triplet_id"),
                    "drug_id": rec.get("drug_id"),
                    "target_id": rec.get("target_id"),
                    "direction": rec.get("direction"),
                    "metric": metric,
                    "value": rec.get(col),
                    "n_cell_pairs_raw": rec.get("n_cell_pairs_raw"),
                    "n_cell_pairs_used": rec.get("n_cell_pairs_used"),
                    "path_recoverability": rec.get("path_recoverability"),
                })
    if not baseline_joint.empty:
        baseline_joint = prune_non_v23_null_types(baseline_joint)
        for rec in baseline_joint.to_dict(orient="records"):
            for metric in ["jaccard"]:
                col = f"joint_{metric}"
                if col not in rec:
                    continue
                rows.append({
                    "source": "null",
                    "null_type": rec.get("null_type"),
                    "iteration": rec.get("iteration"),
                    "evaluable_set_type": rec.get("evaluable_set_type"),
                    "triplet_id": rec.get("triplet_id"),
                    "drug_id": rec.get("drug_id"),
                    "target_id": rec.get("target_id"),
                    "direction": rec.get("direction"),
                    "metric": metric,
                    "value": rec.get(col),
                    "n_cell_pairs_raw": rec.get("n_cell_pairs_raw"),
                    "n_cell_pairs_used": rec.get("n_cell_pairs_used"),
                    "path_recoverability": rec.get("path_recoverability"),
                })
    return pd.DataFrame(rows)


def build_metric_pert_level_long(metric_triplet_long: pd.DataFrame) -> pd.DataFrame:
    if metric_triplet_long.empty:
        return pd.DataFrame(columns=["source", "null_type", "iteration", "evaluable_set_type", "target_id", "direction", "metric", "value", "n_triplets"])
    group_cols = [c for c in ["source", "null_type", "iteration", "evaluable_set_type", "target_id", "direction", "metric"] if c in metric_triplet_long.columns]
    return (
        metric_triplet_long.groupby(group_cols, dropna=False)
        .agg(value=("value", "mean"), n_triplets=("triplet_id", "nunique"))
        .reset_index()
    )


def build_coverage_triplet_summary(true_joint: pd.DataFrame, baseline_joint: pd.DataFrame) -> pd.DataFrame:
    keep = ["source", "null_type", "iteration", "evaluable_set_type", "triplet_id", "drug_id", "target_id", "direction", "n_cell_pairs_raw", "n_cell_pairs_used", "path_recoverability"]
    parts: list[pd.DataFrame] = []
    if not true_joint.empty:
        obs = true_joint.copy()
        obs["source"] = "observed"
        obs["null_type"] = "observed"
        obs["iteration"] = None
        parts.append(obs[[c for c in keep if c in obs.columns]])
    if not baseline_joint.empty:
        base = prune_non_v23_null_types(baseline_joint)
        base = base.copy()
        base["source"] = "null"
        parts.append(base[[c for c in keep if c in base.columns]])
    if not parts:
        return pd.DataFrame(columns=keep)
    out = pd.concat(parts, ignore_index=True)
    for col in keep:
        if col not in out.columns:
            out[col] = None
    return out[keep]


def build_coverage_summary_by_null(coverage: pd.DataFrame) -> pd.DataFrame:
    if coverage.empty:
        return pd.DataFrame(columns=["source", "null_type", "evaluable_set_type", "n_rows", "n_triplets", "n_iterations", "n_cell_pairs_raw", "n_cell_pairs_used", "path_recoverability_mean"])
    return (
        coverage.groupby(["source", "null_type", "evaluable_set_type"], dropna=False)
        .agg(
            n_rows=("triplet_id", "size"),
            n_triplets=("triplet_id", "nunique"),
            n_iterations=("iteration", lambda s: int(pd.Series(s).dropna().nunique())),
            n_cell_pairs_raw=("n_cell_pairs_raw", "sum"),
            n_cell_pairs_used=("n_cell_pairs_used", "sum"),
            path_recoverability_mean=("path_recoverability", "mean"),
        )
        .reset_index()
    )


def build_null_diagnostics_summary(eval_dir: str | Path) -> pd.DataFrame:
    root = Path(eval_dir)
    rows = []
    null1 = _read_table(root / "null1_local_shuffle_diagnostics.tsv")
    if not null1.empty:
        rows.append({
            "null_type": "target_khop_restricted_paired_gene_label_permutation",
            "diagnostic": "local_shuffle_background",
            "n_rows": int(len(null1)),
            "n_triplets": int(null1.get("triplet_id", pd.Series(dtype=str)).astype(str).nunique()) if "triplet_id" in null1 else None,
            "summary_value": float(pd.to_numeric(null1.get("n_local_bg_genes", pd.Series(dtype=float)), errors="coerce").median()),
            "summary_name": "median_n_local_bg_genes",
        })
    null2 = _read_table(root / "global_structural_pool_diagnostics.tsv")
    if not null2.empty:
        rows.append({
            "null_type": "path_count_length_matched_structural_resampling",
            "diagnostic": "target_khop_path_pool",
            "n_rows": int(len(null2)),
            "n_triplets": int(null2.get("target_id", pd.Series(dtype=str)).astype(str).nunique()) if "target_id" in null2 else None,
            "summary_value": float(pd.to_numeric(null2.get("n_observed_paths_missing_from_pool", pd.Series(dtype=float)), errors="coerce").sum()),
            "summary_name": "sum_n_observed_paths_missing_from_pool",
        })
    null3 = _read_table(root / "decoy_anchor_diagnostics.tsv")
    if not null3.empty:
        null3_types = null3.get("null_type", pd.Series(dtype=str)).astype(str)
        rows.append({
            "null_type": str(null3_types.iloc[0]) if not null3_types.empty else TARGET_DECOY_NULL_TYPE,
            "diagnostic": "final_evaluable_decoy_anchors",
            "n_rows": int(len(null3)),
            "n_triplets": int(null3.get("triplet_id", pd.Series(dtype=str)).astype(str).nunique()) if "triplet_id" in null3 else None,
            "summary_value": float(pd.to_numeric(null3.get("n_decoy_anchors", pd.Series(dtype=float)), errors="coerce").median()),
            "summary_name": "median_n_decoy_anchors",
        })
    target_decoy_eval = _read_table(root / "decoy_evaluability_summary.tsv")
    if not target_decoy_eval.empty:
        rows.append({
            "null_type": TARGET_DECOY_NULL_TYPE,
            "diagnostic": "matched_decoy_candidate_evaluability",
            "n_rows": int(pd.to_numeric(target_decoy_eval.get("n_selected_decoys_total", pd.Series(dtype=float)), errors="coerce").sum()),
            "n_triplets": int(pd.to_numeric(target_decoy_eval.get("n_triplets_with_evaluable_decoy", pd.Series(dtype=float)), errors="coerce").max()),
            "summary_value": float(pd.to_numeric(target_decoy_eval.get("candidate_level_evaluable_rate", pd.Series(dtype=float)), errors="coerce").median()),
            "summary_name": "candidate_level_evaluable_rate",
        })
    # Primary acceptance is judged on primary_path_evaluable / pre_inference_selected /
    # triplet_weighted only, so the acceptance table is used rather than the full balance
    # table, whose other scopes and weightings are diagnostics.
    acceptance = _read_table(root / "decoy_matching_acceptance.tsv")
    if acceptance.empty:
        balance = _read_table(root / "decoy_matching_balance.tsv")
        acceptance = build_target_decoy_matching_acceptance(balance)
    if not acceptance.empty and "abs_smd" in acceptance.columns:
        abs_smd = pd.to_numeric(acceptance["abs_smd"], errors="coerce")
        within = acceptance["is_within_threshold"].astype(str).str.lower().isin({"true", "1", "1.0"})
        rows.append({
            "null_type": TARGET_DECOY_NULL_TYPE,
            "diagnostic": "matched_decoy_balance_acceptance",
            "n_rows": int(len(acceptance)),
            "n_triplets": int(pd.to_numeric(acceptance.get("n_observed_rows", pd.Series(dtype=float)), errors="coerce").max()),
            "summary_value": float(abs_smd.max()),
            "summary_name": (
                "max_abs_smd_matching_variables_primary_path_evaluable"
                + ("_within_threshold" if bool(within.all()) else "_threshold_exceeded")
            ),
        })
    out = pd.DataFrame(rows)
    # Assembled from per-null diagnostic files rather than a pruned table, so the
    # Target-Decoy label collision is re-checked here as well.
    assert_distinguishable_null_types(out)
    return out


def build_output_inventory(run_root: str | Path) -> pd.DataFrame:
    root = Path(run_root)
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rec = {"path": str(path.relative_to(root)), "bytes": int(path.stat().st_size), "suffix": path.suffix, "n_rows": None, "n_cols": None}
        try:
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
                rec["n_rows"] = int(len(df))
                rec["n_cols"] = int(len(df.columns))
            elif path.suffix in {".tsv", ".csv"}:
                sep = "\t" if path.suffix == ".tsv" else ","
                df = pd.read_csv(path, sep=sep)
                rec["n_rows"] = int(len(df))
                rec["n_cols"] = int(len(df.columns))
        except Exception:
            pass
        rows.append(rec)
    return pd.DataFrame(rows)


def build_v3_report_tables(
    *,
    run_root: str | Path,
    direction: str,
    config: Mapping[str, Any] | None = None,
    evaluation_summary: Mapping[str, Any] | None = None,
    report_dir: str | Path | None = None,
) -> dict[str, str]:
    run_root = Path(run_root)
    eval_dir = run_root / "evaluation" / direction
    report_dir = Path(report_dir) if report_dir is not None else run_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = report_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    true_joint = _read_table(eval_dir / "global_joint_triplet_metrics_true.parquet")
    baseline_joint = _read_table(eval_dir / "global_joint_triplet_metrics_baseline.parquet")

    outputs: dict[str, str] = {}
    settings_rows = []
    if config is not None:
        settings_rows.append({"key": "config_json", "value": json.dumps(config, ensure_ascii=False, sort_keys=True, default=str)})
    if evaluation_summary is not None:
        settings_rows.append({"key": "evaluation_summary_json", "value": json.dumps(evaluation_summary, ensure_ascii=False, sort_keys=True, default=str)})
    settings_rows.append({"key": "direction", "value": direction})
    settings = pd.DataFrame(settings_rows)
    path = tables_dir / "report_run_settings.tsv"
    settings.to_csv(path, sep="\t", index=False)
    outputs["report_run_settings"] = str(path)

    coverage = build_coverage_triplet_summary(true_joint, baseline_joint)
    outputs["coverage_triplet_summary"] = write_parquet_zstd(coverage, tables_dir / "coverage_triplet_summary.parquet")
    coverage_summary = build_coverage_summary_by_null(coverage)
    path = tables_dir / "coverage_summary_by_null.tsv"
    coverage_summary.to_csv(path, sep="\t", index=False)
    outputs["coverage_summary_by_null"] = str(path)

    metric_triplet = build_metric_triplet_long(true_joint, baseline_joint)
    outputs["metric_triplet_long"] = write_parquet_zstd(metric_triplet, tables_dir / "metric_triplet_long.parquet")
    metric_pert = build_metric_pert_level_long(metric_triplet)
    outputs["metric_pert_level_long"] = write_parquet_zstd(metric_pert, tables_dir / "metric_pert_level_long.parquet")

    diag = build_null_diagnostics_summary(eval_dir)
    path = tables_dir / "null_diagnostics_summary.tsv"
    diag.to_csv(path, sep="\t", index=False)
    outputs["null_diagnostics_summary"] = str(path)

    return outputs


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, ensure_ascii=False))
