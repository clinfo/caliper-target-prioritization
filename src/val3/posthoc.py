"""Post-hoc path-support null analyses computed from completed Val3 outputs.

Every function here reads tables that ``evaluate_direction`` already wrote under
``<run_root>/evaluation/<direction>/`` (pandas only, no inference re-run), so the whole
module can be applied to an existing run:

* triplet-wise ``Observed - null median`` distributions (per triplet, drug, target, disease),
* null-repeat stability of the primary paired Jaccard estimand,
* evaluated scope reporting (triplets / drugs / targets / diseases / null evaluable rate),
* cluster bootstrap of the paired delta over drug / target / disease clusters,
* null-evaluability diagnostics and a zero-imputed sensitivity for non-evaluable triplets,
* the same-cell-line cell-pair subset rebuilt from ``cellpair_metrics_debug``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

from .decoy import DECOY_NULL_TYPES, TARGET_DECOY_NULL_TYPE
from .metrics import _spearman_or_nan
from .streaming import prune_non_v23_null_types, read_parquet_any_engine
from .viz import null_type_sort_key


DEFAULT_STATISTICS: tuple[str, ...] = ("joint_jaccard",)
JACCARD_STATISTIC = "joint_jaccard"
CELLPAIR_METRICS: tuple[str, ...] = ("jaccard",)
CLUSTER_LEVELS: tuple[str, ...] = ("triplet", "drug", "target", "disease")
CLUSTER_LEVEL_COLUMNS: dict[str, str] = {
    "triplet": "triplet_id",
    "drug": "drug_id",
    "target": "target_id",
    "disease": "disease_id",
}
CELL_SCOPES: tuple[str, ...] = ("all_cell_pairs", "same_cell")
OBSERVED_SOURCE = "observed"

CELLPAIR_DEBUG_DIRNAME = "cellpair_metrics_debug"
# Target-Decoy cell-pair shards are keyed by pseudo triplet id, so they are folded back
# onto the original triplet through this assignment table before being paired with observed.
DECOY_ASSIGNMENTS_FILENAME = "decoy_matching_assignments.tsv"
PRIMARY_ESTIMAND = "mean_observed_minus_triplet_null_median"
BOOTSTRAP_N_RESAMPLES = 1000
BOOTSTRAP_CI_LEVEL = 0.95
BOOTSTRAP_CI_METHOD = "percentile"
NULL_REPEAT_STABILITY_CHECKPOINTS: tuple[int, ...] = (5, 10, 20, 50, 100)


def stable_bootstrap_seed(seed: int, *group_parts: Any) -> int:
    """Derive a deterministic per-group RNG seed independent of row/group order."""
    payload = "\x1f".join([str(int(seed)), *[str(part) for part in group_parts]])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray | pd.Series,
    *,
    seed: int,
    group_parts: Sequence[Any],
    n_bootstrap: int = BOOTSTRAP_N_RESAMPLES,
) -> tuple[float, float, float, int]:
    """Mean and 95% percentile triplet-bootstrap CI for finite paired deltas."""
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    # Values are exchangeable bootstrap units. Canonicalizing their order makes the
    # finite Monte Carlo result invariant to DataFrame row order as well.
    arr = np.sort(arr)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), stable_bootstrap_seed(seed, *group_parts)
    n_boot = int(n_bootstrap)
    if n_boot <= 0:
        raise ValueError("n_bootstrap must be positive.")
    group_seed = stable_bootstrap_seed(seed, *group_parts)
    point = float(np.mean(arr))
    if arr.size == 1:
        return point, point, point, group_seed
    rng = np.random.default_rng(group_seed)
    sample_idx = rng.integers(0, arr.size, size=(n_boot, arr.size), endpoint=False)
    boot_mean = np.mean(arr[sample_idx], axis=1)
    alpha = (1.0 - BOOTSTRAP_CI_LEVEL) / 2.0
    return (
        point,
        float(np.quantile(boot_mean, alpha)),
        float(np.quantile(boot_mean, 1.0 - alpha)),
        group_seed,
    )


def wilcoxon_signed_rank_two_sided(
    values: Sequence[float] | np.ndarray | pd.Series,
) -> tuple[float, float, str]:
    """Two-sided Wilcoxon signed-rank test of finite paired deltas against zero."""
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), "no_finite_delta"
    if not bool(np.any(arr != 0)):
        return float("nan"), float("nan"), "no_nonzero_delta"
    try:
        from scipy.stats import wilcoxon  # type: ignore

        try:
            result = wilcoxon(
                arr,
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
                method="auto",
            )
        except TypeError:
            # Compatibility with SciPy versions predating the ``method`` keyword.
            result = wilcoxon(
                arr,
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
            )
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        if not (np.isfinite(statistic) and np.isfinite(p_value)):
            return float("nan"), float("nan"), "nonfinite_result"
        return statistic, p_value, "computed"
    except Exception:
        # Do not substitute another hypothesis test when Wilcoxon cannot be computed.
        return float("nan"), float("nan"), "not_computed"


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjustment of a family of p values; non-finite entries stay ``nan``."""
    values = [float(p) if p is not None and pd.notna(p) else float("nan") for p in p_values]
    m = len(values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: (np.inf if not np.isfinite(values[i]) else values[i]))
    adjusted = [float("nan")] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        p = values[idx]
        if not np.isfinite(p):
            adjusted[idx] = float("nan")
            continue
        running_max = max(running_max, (m - rank) * p)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def _evaluation_root(run_root: str | Path, direction: str) -> Path:
    return Path(run_root) / "evaluation" / str(direction)


def _read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        if p.suffix == ".parquet":
            return read_parquet_any_engine(p)
        return pd.read_csv(p, sep="\t")
    except Exception:
        return pd.DataFrame()


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _unique_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return int(df[column].dropna().astype(str).nunique())


def _sort_canonical(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Sort report-facing tables, keeping Val3 category columns in their canonical order."""
    if df.empty:
        return df.reset_index(drop=True)
    keys: dict[str, pd.Series] = {}
    for column in columns:
        if column not in df.columns:
            continue
        values = df[column].astype(str)
        if column == "null_type":
            keys[f"_{column}"] = values.map(null_type_sort_key)
        elif column == "scope":
            keys[f"_{column}"] = values.ne(OBSERVED_SOURCE).astype(int)
        elif column == "cluster_level":
            keys[f"_{column}"] = values.map(lambda v: (CLUSTER_LEVELS.index(v) if v in CLUSTER_LEVELS else len(CLUSTER_LEVELS), v))
        elif column == "statistic":
            keys[f"_{column}"] = values.map(lambda v: (DEFAULT_STATISTICS.index(v) if v in DEFAULT_STATISTICS else len(DEFAULT_STATISTICS), v))
        elif column == "cell_scope":
            keys[f"_{column}"] = values.map(lambda v: (CELL_SCOPES.index(v) if v in CELL_SCOPES else len(CELL_SCOPES), v))
        else:
            keys[f"_{column}"] = values
    if not keys:
        return df.reset_index(drop=True)
    out = df.assign(**keys)
    return out.sort_values(list(keys), kind="stable").drop(columns=list(keys)).reset_index(drop=True)


# ---------------------------------------------------------------------------
# triplet -> (drug, target, disease) map
# ---------------------------------------------------------------------------

def load_triplet_group_map(run_root: str | Path, direction: str) -> pd.DataFrame:
    """Return ``triplet_id`` -> ``drug_id`` / ``target_id`` / ``disease_id``.

    ``global_joint_triplet_metrics_true.parquet`` carries no ``disease_id``, so the disease
    cluster level is recovered from the prepared observed triplet table, falling back to the
    decoy anchor diagnostics and finally to the cell-pair debug shards.
    """
    columns = ["triplet_id", "drug_id", "target_id", "disease_id"]
    evaluation_root = _evaluation_root(run_root, direction)
    candidates: list[pd.DataFrame] = [
        _read_table(Path(run_root) / "prepared_inputs" / "observed" / str(direction) / "profile_supported_observed_triplets.tsv"),
        _read_table(evaluation_root / "decoy_anchor_diagnostics.tsv"),
    ]
    for path, _meta in iter_cellpair_debug_parts(run_root, direction, source=OBSERVED_SOURCE):
        try:
            candidates.append(read_parquet_any_engine(path, columns=columns))
        except Exception:
            continue

    parts: list[pd.DataFrame] = []
    for candidate in candidates:
        if candidate.empty or "triplet_id" not in candidate.columns:
            continue
        part = candidate.copy()
        for col in columns:
            if col not in part.columns:
                part[col] = pd.NA
        part = part[columns].astype("string")
        part = part.mask(part.eq(""))
        parts.append(part.drop_duplicates("triplet_id"))
    if not parts:
        return pd.DataFrame(columns=columns)
    # Sources are ordered by preference; ``first()`` keeps the first non-missing id per triplet.
    return pd.concat(parts, ignore_index=True).groupby("triplet_id", as_index=False, sort=True).first()[columns]


def attach_group_ids(df: pd.DataFrame, group_map: pd.DataFrame | None) -> pd.DataFrame:
    """Fill missing ``drug_id`` / ``target_id`` / ``disease_id`` columns from ``group_map``."""
    out = df.copy()
    if "triplet_id" not in out.columns:
        return out
    out["triplet_id"] = out["triplet_id"].astype(str)
    if group_map is None or group_map.empty:
        for col in ["drug_id", "target_id", "disease_id"]:
            if col not in out.columns:
                out[col] = pd.NA
        return out
    gm = group_map.copy()
    gm["triplet_id"] = gm["triplet_id"].astype(str)
    gm = gm.drop_duplicates("triplet_id").set_index("triplet_id")
    for col in ["drug_id", "target_id", "disease_id"]:
        mapped = out["triplet_id"].map(gm[col]) if col in gm.columns else pd.Series(pd.NA, index=out.index)
        if col in out.columns:
            out[col] = out[col].where(out[col].notna() & out[col].astype(str).ne(""), mapped)
        else:
            out[col] = mapped
    return out


# ---------------------------------------------------------------------------
# triplet-wise paired delta (Observed - null median)
# ---------------------------------------------------------------------------

def build_triplet_delta_table(
    comparison_df: pd.DataFrame,
    *,
    group_map: pd.DataFrame | None = None,
    statistics: Sequence[str] = DEFAULT_STATISTICS,
) -> pd.DataFrame:
    """Return per-triplet paired deltas from ``global_joint_triplet_comparison.tsv``.

    ``delta`` is recomputed as ``observed - median(null draws)`` for every null model.
    This deliberately ignores a mean-based ``delta`` column in an evaluation table
    produced before the report switched to the median representative.
    """
    columns = [
        "null_type", "evaluable_set_type", "statistic", "triplet_id",
        "drug_id", "target_id", "disease_id",
        "observed_value", "baseline_median", "delta",
    ]
    if comparison_df.empty:
        return pd.DataFrame(columns=columns)
    required = {"null_type", "evaluable_set_type", "statistic", "triplet_id"}
    missing = required - set(comparison_df.columns)
    if missing:
        raise ValueError(
            f"global_joint_triplet_comparison.tsv missing columns for triplet delta: {sorted(missing)}"
        )

    df = prune_non_v23_null_types(comparison_df)
    if df.empty:
        return pd.DataFrame(columns=columns)
    wanted = {str(s) for s in statistics}
    df = df.loc[df["statistic"].astype(str).isin(wanted)].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    for col in ["observed_value", "baseline_median"]:
        df[col] = _numeric(df[col]) if col in df.columns else np.nan
    df["delta"] = df["observed_value"] - df["baseline_median"]
    if "ref_delta_vs_null_median" in df.columns:
        ref_delta = _numeric(df["ref_delta_vs_null_median"])
        df["delta"] = df["delta"].where(df["delta"].notna(), ref_delta)

    df = attach_group_ids(df, group_map)
    out = df[columns].dropna(subset=["delta"]).copy()
    out["null_type"] = out["null_type"].astype(str)
    out["evaluable_set_type"] = out["evaluable_set_type"].astype(str)
    out["statistic"] = out["statistic"].astype(str)
    sort_cols = ["null_type", "evaluable_set_type", "statistic", "triplet_id"]
    return out.sort_values(sort_cols).reset_index(drop=True)


def summarize_triplet_delta_distribution(delta_df: pd.DataFrame) -> pd.DataFrame:
    """Distribution summary of the triplet-wise paired delta, per null model and statistic.

    The delta is ``observed - null median`` for every null model.
    """
    columns = [
        "null_type", "evaluable_set_type", "statistic", "n_triplets",
        "n_drugs", "n_targets", "n_diseases",
        "delta_mean", "delta_median", "delta_sd", "delta_q25", "delta_q75",
        "delta_min", "delta_max",
        "n_positive_delta", "n_negative_delta", "n_zero_delta",
        "fraction_positive_delta",
        "wilcoxon_two_sided_statistic", "p_wilcoxon_two_sided",
        "wilcoxon_two_sided_status",
    ]
    if delta_df.empty:
        return pd.DataFrame(columns=columns)
    delta_df = delta_df.loc[
        delta_df["statistic"].astype(str).eq(JACCARD_STATISTIC)
    ].copy()
    if delta_df.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    group_cols = ["null_type", "evaluable_set_type", "statistic"]
    for keys, sub in delta_df.groupby(group_cols, dropna=False, sort=False):
        delta = _numeric(sub["delta"]).dropna()
        n = int(len(delta))
        n_pos = int((delta > 0).sum())
        n_neg = int((delta < 0).sum())
        wilcoxon_statistic, p_wilcoxon, wilcoxon_status = wilcoxon_signed_rank_two_sided(delta)
        rows.append({
            "null_type": str(keys[0]),
            "evaluable_set_type": str(keys[1]),
            "statistic": str(keys[2]),
            "n_triplets": int(sub["triplet_id"].astype(str).nunique()),
            "n_drugs": _unique_count(sub, "drug_id"),
            "n_targets": _unique_count(sub, "target_id"),
            "n_diseases": _unique_count(sub, "disease_id"),
            "delta_mean": float(delta.mean()) if n else float("nan"),
            "delta_median": float(delta.median()) if n else float("nan"),
            "delta_sd": float(delta.std(ddof=1)) if n >= 2 else float("nan"),
            "delta_q25": float(delta.quantile(0.25)) if n else float("nan"),
            "delta_q75": float(delta.quantile(0.75)) if n else float("nan"),
            "delta_min": float(delta.min()) if n else float("nan"),
            "delta_max": float(delta.max()) if n else float("nan"),
            "n_positive_delta": n_pos,
            "n_negative_delta": n_neg,
            "n_zero_delta": int((delta == 0).sum()),
            "fraction_positive_delta": float(n_pos / n) if n else float("nan"),
            "wilcoxon_two_sided_statistic": wilcoxon_statistic,
            "p_wilcoxon_two_sided": p_wilcoxon,
            "wilcoxon_two_sided_status": wilcoxon_status,
        })
    return _sort_canonical(pd.DataFrame(rows, columns=columns), ["evaluable_set_type", "statistic", "null_type"])


# ---------------------------------------------------------------------------
# null-repeat stability
# ---------------------------------------------------------------------------

def build_null_repeat_stability(
    *,
    true_joint: pd.DataFrame,
    baseline_joint: pd.DataFrame,
    direction: str,
    statistic: str = JACCARD_STATISTIC,
    checkpoints: Sequence[int] = NULL_REPEAT_STABILITY_CHECKPOINTS,
) -> pd.DataFrame:
    """Cumulative stability of the primary paired delta across repeat-driven nulls.

    For each checkpoint, the null representative is recomputed as the per-triplet
    median over finite draws from iterations ``1..checkpoint``.  The point estimate
    is then the mean paired ``Observed - null median`` across evaluable triplets,
    matching the primary Val3 estimand without rerunning inference.

    Target-Decoy is intentionally absent because it uses matched decoy candidates
    rather than numbered null repeats.
    """
    columns = [
        "direction",
        "null_type",
        "evaluable_set_type",
        "statistic",
        "n_repeats",
        "n_triplets",
        "delta_mean",
        "delta_median",
        "full_n_repeats",
        "delta_mean_full",
        "abs_delta_mean_change_from_full",
    ]
    value_col = str(statistic)
    required_true = {"evaluable_set_type", "triplet_id", value_col}
    required_base = {
        "null_type",
        "evaluable_set_type",
        "iteration",
        "triplet_id",
        value_col,
    }
    if (
        true_joint.empty
        or baseline_joint.empty
        or not required_true <= set(true_joint.columns)
        or not required_base <= set(baseline_joint.columns)
    ):
        return pd.DataFrame(columns=columns)

    observed = true_joint[
        ["evaluable_set_type", "triplet_id", value_col]
    ].copy()
    observed["observed_value"] = _numeric(observed[value_col])
    observed = observed.drop(columns=[value_col]).dropna(subset=["observed_value"])
    observed = observed.drop_duplicates(["evaluable_set_type", "triplet_id"])

    baseline = prune_non_v23_null_types(baseline_joint)
    baseline = baseline[
        ["null_type", "evaluable_set_type", "iteration", "triplet_id", value_col]
    ].copy()
    baseline["iteration"] = _numeric(baseline["iteration"])
    baseline["null_value"] = _numeric(baseline[value_col])
    baseline = baseline.drop(columns=[value_col]).dropna(subset=["iteration"])
    baseline = baseline.loc[baseline["iteration"] > 0].copy()
    if baseline.empty:
        return pd.DataFrame(columns=columns)
    baseline["iteration"] = baseline["iteration"].astype(int)

    requested = sorted({int(value) for value in checkpoints if int(value) > 0})
    rows: list[dict[str, Any]] = []
    group_cols = ["null_type", "evaluable_set_type"]
    for (null_type, eval_type), group in baseline.groupby(
        group_cols, dropna=False, sort=False,
    ):
        full_n_repeats = int(group["iteration"].max())
        group_checkpoints = [value for value in requested if value <= full_n_repeats]
        if full_n_repeats not in group_checkpoints:
            group_checkpoints.append(full_n_repeats)
        group_checkpoints = sorted(set(group_checkpoints))
        observed_group = observed.loc[
            observed["evaluable_set_type"].astype(str).eq(str(eval_type))
        ]
        group_rows: list[dict[str, Any]] = []
        for n_repeats in group_checkpoints:
            prefix = group.loc[group["iteration"] <= int(n_repeats)]
            null_median = (
                prefix.dropna(subset=["null_value"])
                .groupby("triplet_id", sort=False, as_index=False)
                .agg(null_median=("null_value", "median"))
            )
            paired = null_median.merge(
                observed_group[["triplet_id", "observed_value"]],
                on="triplet_id",
                how="inner",
            )
            delta = (
                _numeric(paired["observed_value"])
                - _numeric(paired["null_median"])
            ).dropna()
            group_rows.append({
                "direction": str(direction),
                "null_type": str(null_type),
                "evaluable_set_type": str(eval_type),
                "statistic": value_col,
                "n_repeats": int(n_repeats),
                "n_triplets": int(len(delta)),
                "delta_mean": float(delta.mean()) if len(delta) else float("nan"),
                "delta_median": float(delta.median()) if len(delta) else float("nan"),
            })
        full_delta_mean = float(group_rows[-1]["delta_mean"])
        for row in group_rows:
            row["full_n_repeats"] = full_n_repeats
            row["delta_mean_full"] = full_delta_mean
            current = float(row["delta_mean"])
            row["abs_delta_mean_change_from_full"] = (
                abs(current - full_delta_mean)
                if np.isfinite(current) and np.isfinite(full_delta_mean)
                else float("nan")
            )
            rows.append(row)

    out = pd.DataFrame(rows, columns=columns)
    return _sort_canonical(
        out,
        ["evaluable_set_type", "statistic", "null_type"],
    )


# ---------------------------------------------------------------------------
# evaluated scope reporting
# ---------------------------------------------------------------------------

def build_evaluated_scope_summary(
    *,
    true_joint: pd.DataFrame,
    baseline_joint: pd.DataFrame,
    comparison_sets: pd.DataFrame,
    group_map: pd.DataFrame | None = None,
    statistic: str = JACCARD_STATISTIC,
) -> pd.DataFrame:
    """Report evaluated triplets / unique drugs / targets / diseases / null evaluable rate."""
    columns = [
        "scope", "null_type", "evaluable_set_type", "n_triplets",
        "n_unique_drugs", "n_unique_targets", "n_unique_diseases",
        "n_cell_pairs_used_total",
        "n_null_rows", "n_null_rows_evaluable", "null_evaluable_rate",
        "null_evaluable_rate_denominator",
        "n_comparison_triplets", "n_triplets_with_null_score", "n_triplets_without_null_score",
    ]
    rows: list[dict[str, Any]] = []

    observed = attach_group_ids(true_joint, group_map) if not true_joint.empty else pd.DataFrame()
    if not observed.empty and statistic in observed.columns:
        observed = observed.loc[_numeric(observed[statistic]).notna()].copy()
    if observed.empty or "evaluable_set_type" not in observed.columns:
        observed = pd.DataFrame()
    observed_groups = list(observed.groupby("evaluable_set_type", dropna=False, sort=True)) if not observed.empty else []
    for eval_type, sub in observed_groups:
        rows.append({
            "scope": OBSERVED_SOURCE,
            "null_type": OBSERVED_SOURCE,
            "evaluable_set_type": str(eval_type),
            "n_triplets": int(sub["triplet_id"].astype(str).nunique()),
            "n_unique_drugs": _unique_count(sub, "drug_id"),
            "n_unique_targets": _unique_count(sub, "target_id"),
            "n_unique_diseases": _unique_count(sub, "disease_id"),
            "n_cell_pairs_used_total": int(_numeric(sub.get("n_cell_pairs_used", pd.Series(dtype=float))).fillna(0).sum()),
            "n_null_rows": pd.NA,
            "n_null_rows_evaluable": pd.NA,
            "null_evaluable_rate": float("nan"),
            "null_evaluable_rate_denominator": pd.NA,
            "n_comparison_triplets": pd.NA,
            "n_triplets_with_null_score": pd.NA,
            "n_triplets_without_null_score": pd.NA,
        })

    if not baseline_joint.empty:
        base = attach_group_ids(prune_non_v23_null_types(baseline_joint), group_map)
        observed_ids_by_set: dict[str, set[str]] = {}
        if not observed.empty and "evaluable_set_type" in observed.columns:
            observed_ids_by_set = {
                str(k): set(v["triplet_id"].astype(str))
                for k, v in observed.groupby("evaluable_set_type", dropna=False)
            }
        for keys, sub in base.groupby(["null_type", "evaluable_set_type"], dropna=False, sort=True):
            null_type, eval_type = str(keys[0]), str(keys[1])
            values = _numeric(sub[statistic]) if statistic in sub.columns else pd.Series(dtype=float)
            evaluable = sub.loc[values.notna()] if not values.empty else sub.iloc[0:0]
            with_score = set(evaluable["triplet_id"].astype(str)) if not evaluable.empty else set()
            observed_ids = observed_ids_by_set.get(eval_type, set())
            n_rows = int(len(sub))
            rows.append({
                "scope": "null",
                "null_type": null_type,
                "evaluable_set_type": eval_type,
                "n_triplets": int(sub["triplet_id"].astype(str).nunique()),
                "n_unique_drugs": _unique_count(evaluable if not evaluable.empty else sub, "drug_id"),
                "n_unique_targets": _unique_count(evaluable if not evaluable.empty else sub, "target_id"),
                "n_unique_diseases": _unique_count(evaluable if not evaluable.empty else sub, "disease_id"),
                "n_cell_pairs_used_total": int(_numeric(sub.get("n_cell_pairs_used", pd.Series(dtype=float))).fillna(0).sum()),
                "n_null_rows": n_rows,
                "n_null_rows_evaluable": int(len(evaluable)),
                "null_evaluable_rate": float(len(evaluable) / n_rows) if n_rows else float("nan"),
                # Comparison-row denominator, not the number of selected decoy candidates.
                # Candidate-level Target-Decoy rates live in decoy_evaluability_summary.tsv.
                "null_evaluable_rate_denominator": "n_null_rows_in_comparison_table",
                "n_comparison_triplets": _count_comparison_triplets(comparison_sets, null_type=null_type, eval_type=eval_type),
                "n_triplets_with_null_score": int(len(with_score)),
                "n_triplets_without_null_score": int(len(observed_ids - with_score)) if observed_ids else pd.NA,
            })

    return _sort_canonical(pd.DataFrame(rows, columns=columns), ["evaluable_set_type", "scope", "null_type"])


def _count_comparison_triplets(comparison_sets: pd.DataFrame, *, null_type: str, eval_type: str) -> Any:
    if comparison_sets.empty or "is_comparison_triplet" not in comparison_sets.columns:
        return pd.NA
    sub = comparison_sets.loc[
        comparison_sets["null_type"].astype(str).eq(str(null_type))
        & comparison_sets["evaluable_set_type"].astype(str).eq(str(eval_type))
    ]
    if sub.empty:
        return pd.NA
    flag = sub["is_comparison_triplet"].astype(str).str.lower().isin({"true", "1", "1.0"})
    return int(sub.loc[flag, "triplet_id"].astype(str).nunique())


# ---------------------------------------------------------------------------
# cluster bootstrap over drug / target / disease
# ---------------------------------------------------------------------------

def build_delta_cluster_bootstrap(
    delta_df: pd.DataFrame,
    *,
    direction: str,
    seed: int,
    n_bootstrap: int = BOOTSTRAP_N_RESAMPLES,
    cluster_levels: Sequence[str] = CLUSTER_LEVELS,
) -> pd.DataFrame:
    """Cluster bootstrap of the triplet-level paired delta over triplet/drug/target/disease.

    Clusters are resampled with replacement; the statistic is recomputed on the pooled
    triplet deltas of the sampled clusters, so triplets that share a drug, target or
    disease are no longer treated as independent.  The input ``delta`` is the
    ``observed - null median`` paired delta of every null model.
    """
    columns = [
        "direction", "null_type", "evaluable_set_type", "statistic", "cluster_level",
        "estimand", "n_clusters", "n_triplets", "n_bootstrap", "bootstrap_seed",
        "resampling_unit", "ci_method", "ci_level",
        "delta_mean_hat", "delta_mean_ci025", "delta_mean_ci975",
    ]
    if delta_df.empty:
        return pd.DataFrame(columns=columns)
    delta_df = delta_df.loc[
        delta_df["statistic"].astype(str).eq(JACCARD_STATISTIC)
    ].copy()
    if delta_df.empty:
        return pd.DataFrame(columns=columns)
    n_boot = int(n_bootstrap)
    if n_boot <= 0:
        raise ValueError("n_bootstrap must be positive.")
    unknown = sorted({str(level) for level in cluster_levels} - set(CLUSTER_LEVEL_COLUMNS))
    if unknown:
        raise ValueError(f"Unsupported cluster level(s): {unknown}. valid={sorted(CLUSTER_LEVEL_COLUMNS)}")

    rows: list[dict[str, Any]] = []
    group_cols = ["null_type", "evaluable_set_type", "statistic"]
    for keys, sub in delta_df.groupby(group_cols, dropna=False, sort=True):
        triplet_delta = (
            sub[["triplet_id", "delta"]]
            .assign(delta=lambda d: _numeric(d["delta"]))
            .dropna(subset=["delta"])
            .groupby("triplet_id", dropna=False, sort=True)["delta"]
            .mean()
        )
        if triplet_delta.empty:
            continue
        for level in cluster_levels:
            key_col = CLUSTER_LEVEL_COLUMNS[str(level)]
            if key_col == "triplet_id":
                keys_by_triplet = pd.Series(triplet_delta.index, index=triplet_delta.index, dtype="object")
            elif key_col in sub.columns:
                keys_by_triplet = (
                    sub[["triplet_id", key_col]]
                    .dropna(subset=[key_col])
                    .astype({"triplet_id": str, key_col: str})
                    .drop_duplicates("triplet_id")
                    .set_index("triplet_id")[key_col]
                )
            else:
                continue
            paired = pd.DataFrame({"delta": triplet_delta}).join(keys_by_triplet.rename("cluster"), how="inner")
            paired = paired.dropna(subset=["cluster"])
            if paired.empty:
                continue

            cluster_arrays = [
                grp["delta"].to_numpy(dtype=float)
                for _, grp in paired.groupby("cluster", dropna=False, sort=True)
            ]
            all_delta = paired["delta"].to_numpy(dtype=float)
            n_clusters = len(cluster_arrays)
            group_seed = stable_bootstrap_seed(
                seed, direction, keys[0], keys[1], keys[2], "cluster", level,
            )
            rng = np.random.default_rng(group_seed)
            boot_mean = np.empty(n_boot, dtype=float)
            for r in range(n_boot):
                idx = rng.integers(0, n_clusters, size=n_clusters)
                sampled = np.concatenate([cluster_arrays[int(i)] for i in idx])
                boot_mean[r] = float(np.mean(sampled))

            rows.append({
                "direction": str(direction),
                "null_type": str(keys[0]),
                "evaluable_set_type": str(keys[1]),
                "statistic": str(keys[2]),
                "cluster_level": str(level),
                "estimand": PRIMARY_ESTIMAND,
                "n_clusters": int(n_clusters),
                "n_triplets": int(len(all_delta)),
                "n_bootstrap": n_boot,
                "bootstrap_seed": group_seed,
                "resampling_unit": f"{level}_cluster",
                "ci_method": BOOTSTRAP_CI_METHOD,
                "ci_level": BOOTSTRAP_CI_LEVEL,
                "delta_mean_hat": float(np.mean(all_delta)),
                "delta_mean_ci025": float(np.quantile(boot_mean, 0.025)),
                "delta_mean_ci975": float(np.quantile(boot_mean, 0.975)),
            })
    return _sort_canonical(
        pd.DataFrame(rows, columns=columns),
        ["evaluable_set_type", "statistic", "null_type", "cluster_level"],
    )


# ---------------------------------------------------------------------------
# null evaluability
# ---------------------------------------------------------------------------

def build_null_evaluability(
    *,
    true_joint: pd.DataFrame,
    baseline_joint: pd.DataFrame,
    statistic: str = JACCARD_STATISTIC,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-triplet and per-null-model evaluability of the null/control draws.

    Returns ``(summary, by_triplet)``.  ``by_triplet`` counts, for every observed-evaluable
    triplet, how many null draws (shuffle iterations or decoy anchors) produced a score, so a
    non-random dropout can be checked against the observed value and the cell-pair support.
    """
    summary_columns = [
        "null_type", "evaluable_set_type", "statistic",
        "n_observed_evaluable_triplets", "n_null_rows", "n_null_rows_evaluable", "null_evaluable_rate",
        "null_evaluable_rate_denominator", "candidate_level_evaluable_rate_source",
        "n_triplets_all_draws_evaluable", "n_triplets_partially_evaluable", "n_triplets_no_null_score",
        "mean_null_evaluable_fraction", "median_n_evaluable_draws",
        "spearman_evaluable_fraction_vs_observed",
        "spearman_evaluable_fraction_vs_n_cell_pairs_used",
        "observed_mean_triplets_with_null_score", "observed_mean_triplets_without_null_score",
    ]
    by_triplet_columns = [
        "null_type", "evaluable_set_type", "triplet_id", "n_null_draws", "n_null_draws_evaluable",
        "null_evaluable_fraction", "has_null_score", "observed_value", "n_cell_pairs_used_observed",
    ]
    if true_joint.empty or baseline_joint.empty or statistic not in baseline_joint.columns:
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=by_triplet_columns)

    observed = true_joint.copy()
    observed["triplet_id"] = observed["triplet_id"].astype(str)
    observed["observed_value"] = _numeric(observed[statistic]) if statistic in observed.columns else np.nan
    observed["n_cell_pairs_used_observed"] = _numeric(observed.get("n_cell_pairs_used", pd.Series(dtype=float)))
    observed = observed.loc[observed["observed_value"].notna()]
    observed = observed[["evaluable_set_type", "triplet_id", "observed_value", "n_cell_pairs_used_observed"]]

    base = prune_non_v23_null_types(baseline_joint).copy()
    base["triplet_id"] = base["triplet_id"].astype(str)
    base["_value"] = _numeric(base[statistic])

    draw_counts = (
        base.groupby(["null_type", "evaluable_set_type", "triplet_id"], dropna=False, sort=True)
        .agg(n_null_draws=("_value", "size"), n_null_draws_evaluable=("_value", lambda s: int(s.notna().sum())))
        .reset_index()
    )
    by_triplet_parts: list[pd.DataFrame] = []
    for null_type in draw_counts["null_type"].astype(str).drop_duplicates():
        sub = draw_counts.loc[draw_counts["null_type"].astype(str).eq(null_type)]
        merged = observed.merge(
            sub.drop(columns=["null_type"]),
            on=["evaluable_set_type", "triplet_id"],
            how="left",
        )
        merged["null_type"] = null_type
        by_triplet_parts.append(merged)
    by_triplet = pd.concat(by_triplet_parts, ignore_index=True) if by_triplet_parts else pd.DataFrame()
    if by_triplet.empty:
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=by_triplet_columns)

    by_triplet["n_null_draws"] = _numeric(by_triplet["n_null_draws"]).fillna(0).astype(int)
    by_triplet["n_null_draws_evaluable"] = _numeric(by_triplet["n_null_draws_evaluable"]).fillna(0).astype(int)
    by_triplet["null_evaluable_fraction"] = np.where(
        by_triplet["n_null_draws"] > 0,
        by_triplet["n_null_draws_evaluable"] / by_triplet["n_null_draws"].replace(0, np.nan),
        np.nan,
    )
    by_triplet["has_null_score"] = by_triplet["n_null_draws_evaluable"] > 0
    by_triplet = by_triplet[by_triplet_columns]

    rows: list[dict[str, Any]] = []
    for keys, sub in by_triplet.groupby(["null_type", "evaluable_set_type"], dropna=False, sort=True):
        null_type, eval_type = str(keys[0]), str(keys[1])
        base_sub = base.loc[
            base["null_type"].astype(str).eq(null_type)
            & base["evaluable_set_type"].astype(str).eq(eval_type)
        ]
        n_rows = int(len(base_sub))
        n_rows_evaluable = int(base_sub["_value"].notna().sum())
        with_score = sub.loc[sub["has_null_score"]]
        without_score = sub.loc[~sub["has_null_score"]]
        all_evaluable = sub.loc[
            (sub["n_null_draws"] > 0) & (sub["n_null_draws_evaluable"] == sub["n_null_draws"])
        ]
        rows.append({
            "null_type": null_type,
            "evaluable_set_type": eval_type,
            "statistic": str(statistic),
            "n_observed_evaluable_triplets": int(sub["triplet_id"].nunique()),
            "n_null_rows": n_rows,
            "n_null_rows_evaluable": n_rows_evaluable,
            "null_evaluable_rate": float(n_rows_evaluable / n_rows) if n_rows else float("nan"),
            # Comparison-row denominator. For the decoy controls the comparison table only
            # holds evaluable candidates, so this rate is 1.0 by construction and must not
            # be read as a candidate-level evaluable rate.
            "null_evaluable_rate_denominator": "n_null_rows_in_comparison_table",
            "candidate_level_evaluable_rate_source": (
                "decoy_evaluability_summary.tsv" if null_type in DECOY_NULL_TYPES else pd.NA
            ),
            "n_triplets_all_draws_evaluable": int(len(all_evaluable)),
            "n_triplets_partially_evaluable": int(len(with_score) - len(all_evaluable)),
            "n_triplets_no_null_score": int(len(without_score)),
            "mean_null_evaluable_fraction": float(sub["null_evaluable_fraction"].mean(skipna=True)),
            "median_n_evaluable_draws": float(sub["n_null_draws_evaluable"].median()),
            "spearman_evaluable_fraction_vs_observed": _spearman_or_nan(
                sub["null_evaluable_fraction"], sub["observed_value"],
            ),
            "spearman_evaluable_fraction_vs_n_cell_pairs_used": _spearman_or_nan(
                sub["null_evaluable_fraction"], sub["n_cell_pairs_used_observed"],
            ),
            "observed_mean_triplets_with_null_score": float(with_score["observed_value"].mean()) if not with_score.empty else float("nan"),
            "observed_mean_triplets_without_null_score": float(without_score["observed_value"].mean()) if not without_score.empty else float("nan"),
        })
    summary = _sort_canonical(pd.DataFrame(rows, columns=summary_columns), ["evaluable_set_type", "null_type"])
    return summary, by_triplet


def build_null_evaluability_sensitivity(
    delta_df: pd.DataFrame,
    by_triplet: pd.DataFrame,
    *,
    statistic: str = JACCARD_STATISTIC,
) -> pd.DataFrame:
    """Zero-imputed sensitivity for triplets that have no evaluable null/control score.

    The primary estimate only uses triplets with at least one null score.  The conservative
    variant adds every remaining observed-evaluable triplet with ``delta = 0``, i.e. assumes no
    advantage over the null wherever the null could not be evaluated.
    """
    columns = [
        "null_type", "evaluable_set_type", "statistic",
        "n_triplets_primary", "delta_mean_primary", "delta_median_primary",
        "n_triplets_without_null_score",
        "n_triplets_zero_imputed", "delta_mean_zero_imputed", "delta_median_zero_imputed",
        "wilcoxon_two_sided_statistic_primary", "p_wilcoxon_two_sided_primary",
        "wilcoxon_two_sided_status_primary",
    ]
    if delta_df.empty:
        return pd.DataFrame(columns=columns)
    sub_delta = delta_df.loc[delta_df["statistic"].astype(str).eq(str(statistic))]
    if sub_delta.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for keys, sub in sub_delta.groupby(["null_type", "evaluable_set_type"], dropna=False, sort=True):
        null_type, eval_type = str(keys[0]), str(keys[1])
        delta = _numeric(sub["delta"]).dropna()
        n_missing = 0
        if not by_triplet.empty:
            missing = by_triplet.loc[
                by_triplet["null_type"].astype(str).eq(null_type)
                & by_triplet["evaluable_set_type"].astype(str).eq(eval_type)
                & (~by_triplet["has_null_score"].astype(bool))
            ]
            n_missing = int(missing["triplet_id"].astype(str).nunique())
        imputed = pd.concat([delta, pd.Series(np.zeros(n_missing, dtype=float))], ignore_index=True)
        wilcoxon_statistic, p_wilcoxon, wilcoxon_status = wilcoxon_signed_rank_two_sided(delta)
        rows.append({
            "null_type": null_type,
            "evaluable_set_type": eval_type,
            "statistic": str(statistic),
            "n_triplets_primary": int(len(delta)),
            "delta_mean_primary": float(delta.mean()) if len(delta) else float("nan"),
            "delta_median_primary": float(delta.median()) if len(delta) else float("nan"),
            "n_triplets_without_null_score": n_missing,
            "n_triplets_zero_imputed": int(len(imputed)),
            "delta_mean_zero_imputed": float(imputed.mean()) if len(imputed) else float("nan"),
            "delta_median_zero_imputed": float(imputed.median()) if len(imputed) else float("nan"),
            "wilcoxon_two_sided_statistic_primary": wilcoxon_statistic,
            "p_wilcoxon_two_sided_primary": p_wilcoxon,
            "wilcoxon_two_sided_status_primary": wilcoxon_status,
        })
    return _sort_canonical(pd.DataFrame(rows, columns=columns), ["evaluable_set_type", "null_type"])


# ---------------------------------------------------------------------------
# same-cell-line cell-pair subset
# ---------------------------------------------------------------------------

def iter_cellpair_debug_parts(
    run_root: str | Path,
    direction: str,
    *,
    source: str | None = None,
    evaluable_set_type: str | None = None,
) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Yield ``(part_path, partition_values)`` for the cell-pair debug shards of a run."""
    root = _evaluation_root(run_root, direction) / CELLPAIR_DEBUG_DIRNAME
    if not root.is_dir():
        return
    for path in sorted(root.glob("source=*/null_type=*/evaluable_set_type=*/iteration=*/*.parquet")):
        meta: dict[str, Any] = {}
        for part in path.parts:
            if "=" in part:
                key, _, value = part.partition("=")
                meta[key] = value
        if source is not None and str(meta.get("source")) != str(source):
            continue
        if evaluable_set_type is not None and str(meta.get("evaluable_set_type")) != str(evaluable_set_type):
            continue
        iteration = meta.get("iteration")
        meta["iteration"] = None if iteration in {None, "NA", ""} else int(iteration)
        yield path, meta


def load_decoy_pseudo_triplet_map(run_root: str | Path, direction: str) -> pd.DataFrame:
    """Read ``pseudo_triplet_id`` -> ``original_triplet_id`` from the assignment table.

    Returns an empty frame for a run without the Target-Decoy control, where
    ``decoy_matching_assignments.tsv`` does not exist.
    """
    columns = ["pseudo_triplet_id", "original_triplet_id", "decoy_target_id"]
    table = _read_table(_evaluation_root(run_root, direction) / DECOY_ASSIGNMENTS_FILENAME)
    if table.empty or not {"pseudo_triplet_id", "original_triplet_id"} <= set(table.columns):
        return pd.DataFrame(columns=columns)
    out = table.copy()
    if "decoy_target_id" not in out.columns:
        out["decoy_target_id"] = pd.NA
    for col in columns:
        out[col] = out[col].astype("string")
    return out[columns].drop_duplicates("pseudo_triplet_id").reset_index(drop=True)


def build_cell_scope_triplet_metrics(
    run_root: str | Path,
    direction: str,
    *,
    evaluable_set_type: str | None = "path_existing_pairs",
    metrics: Sequence[str] = CELLPAIR_METRICS,
) -> pd.DataFrame:
    """Rebuild triplet-level joint metrics for all cell pairs and for the same-cell subset.

    ``cellpair_metrics_debug`` stores the cell pairs that entered the triplet aggregate, so the
    triplet score is reproduced exactly as the mean over used cell pairs (a non-finite cell-pair
    value contributes 0, as in ``metrics._aggregate_joint_pairs``).  Restricting the same mean to
    ``cell_id_drug == cell_id_pert`` yields the same-cell subset without re-running inference.
    """
    columns = [
        "cell_scope", "source", "null_type", "iteration", "evaluable_set_type",
        "triplet_id", "statistic", "value", "n_cell_pairs_used",
    ]
    group_cols = ["cell_scope", "source", "null_type", "iteration", "evaluable_set_type", "triplet_id", "statistic"]
    read_columns = ["triplet_id", "cell_id_drug", "cell_id_pert", *[str(m) for m in metrics]]
    parts: list[pd.DataFrame] = []
    for path, meta in iter_cellpair_debug_parts(run_root, direction, evaluable_set_type=evaluable_set_type):
        try:
            pairs = read_parquet_any_engine(path, columns=read_columns)
        except Exception:
            continue
        if pairs.empty:
            continue
        pairs = pairs.copy()
        pairs["triplet_id"] = pairs["triplet_id"].astype(str)
        same_cell = (
            pairs["cell_id_drug"].notna()
            & pairs["cell_id_pert"].notna()
            & pairs["cell_id_drug"].astype(str).eq(pairs["cell_id_pert"].astype(str))
        )
        for scope in CELL_SCOPES:
            scoped = pairs if scope == "all_cell_pairs" else pairs.loc[same_cell]
            if scoped.empty:
                continue
            agg = _sum_cellpair_values(scoped, metrics=metrics)
            agg["cell_scope"] = scope
            agg["source"] = str(meta.get("source", ""))
            agg["null_type"] = str(meta.get("null_type", ""))
            agg["iteration"] = meta.get("iteration")
            agg["evaluable_set_type"] = str(meta.get("evaluable_set_type", ""))
            parts.append(agg)
    if not parts:
        return pd.DataFrame(columns=columns)
    # Shards of the same (source, null_type, iteration) are summed before dividing, so a
    # part-split debug output gives the same triplet mean as a single shard.
    out = (
        pd.concat(parts, ignore_index=True)
        .groupby(group_cols, dropna=False, sort=True)
        .agg(value_sum=("value_sum", "sum"), n_cell_pairs_used=("n_cell_pairs_used", "sum"))
        .reset_index()
    )
    out["value"] = out["value_sum"] / out["n_cell_pairs_used"].replace(0, np.nan)
    return out[columns].reset_index(drop=True)


def _sum_cellpair_values(pairs: pd.DataFrame, *, metrics: Sequence[str]) -> pd.DataFrame:
    """Per-triplet sum and count of cell-pair metric values (non-finite values count as 0)."""
    frames: list[pd.DataFrame] = []
    counts = pairs.groupby("triplet_id", dropna=False, sort=True).size().rename("n_cell_pairs_used")
    for metric in metrics:
        if str(metric) not in pairs.columns:
            continue
        values = _numeric(pairs[str(metric)]).fillna(0.0)
        agg = values.groupby(pairs["triplet_id"], dropna=False, sort=True).sum().rename("value_sum").to_frame()
        agg = agg.join(counts, how="left").reset_index()
        agg["statistic"] = f"joint_{metric}"
        frames.append(agg)
    if not frames:
        return pd.DataFrame(columns=["triplet_id", "value_sum", "n_cell_pairs_used", "statistic"])
    return pd.concat(frames, ignore_index=True)


def _fold_decoy_pseudo_triplets(
    nulls: pd.DataFrame,
    decoy_triplet_map: pd.DataFrame | None,
) -> pd.DataFrame:
    """Rewrite Target-Decoy pseudo triplet ids to their original triplet id.

    Pseudo ids missing from the assignment table are dropped rather than zero-filled:
    a decoy that cannot be traced back to its observed triplet has no partner to be
    paired against.
    """
    if nulls.empty:
        return nulls
    is_decoy = nulls["null_type"].astype(str).eq(TARGET_DECOY_NULL_TYPE)
    if not bool(is_decoy.any()):
        return nulls
    mapping: dict[str, str] = {}
    if decoy_triplet_map is not None and not decoy_triplet_map.empty:
        mapping = dict(
            zip(
                decoy_triplet_map["pseudo_triplet_id"].astype(str),
                decoy_triplet_map["original_triplet_id"].astype(str),
            )
        )
    decoy_rows = nulls.loc[is_decoy].copy()
    decoy_rows["triplet_id"] = decoy_rows["triplet_id"].astype(str).map(mapping)
    decoy_rows = decoy_rows.loc[decoy_rows["triplet_id"].notna()]
    return pd.concat([nulls.loc[~is_decoy], decoy_rows], ignore_index=True)


def build_cell_scope_delta(
    cell_scope_metrics: pd.DataFrame,
    *,
    decoy_triplet_map: pd.DataFrame | None = None,
    direction: str = "",
    seed: int = 0,
    n_bootstrap: int = BOOTSTRAP_N_RESAMPLES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair observed vs the per-triplet null median within each cell scope.

    Returns ``(by_triplet, summary)``.  The comparison is restricted to triplets that are
    evaluable in both the observed and the null side of the same cell scope, so the same-cell
    subset is compared like-for-like against the all-cell-pairs aggregate.

    Target-Decoy cell-pair shards are written per decoy candidate, so their pseudo
    triplet ids are folded back onto the original triplet with ``decoy_triplet_map``
    first; the null representative is then the median over that triplet's decoys whose
    same-cell subset was non-empty, exactly as for the repeat-driven nulls.

    The summary carries the two-sided Wilcoxon signed-rank test of each cell scope
    (``p_wilcoxon_two_sided`` / ``p_wilcoxon_two_sided_holm``), Holm-corrected across the null
    models within a ``(cell_scope, evaluable_set_type, statistic)`` family, so the same-cell
    subset is tested on its own triplets rather than inheriting the all-cell-pairs test.
    """
    delta_columns = [
        "cell_scope", "null_type", "evaluable_set_type", "statistic", "triplet_id",
        "observed_value", "baseline_median", "delta", "n_null_draws",
        "n_cell_pairs_used_observed",
    ]
    summary_columns = [
        "cell_scope", "null_type", "evaluable_set_type", "statistic", "n_triplets",
        "n_cell_pairs_used_observed_total", "observed_mean", "baseline_median_mean",
        "estimand", "n_bootstrap", "bootstrap_seed", "resampling_unit", "ci_method", "ci_level",
        "delta_mean", "delta_mean_ci025", "delta_mean_ci975", "delta_median",
        "n_positive_delta", "n_negative_delta", "n_zero_delta", "n_nonzero_delta",
        "wilcoxon_two_sided_statistic", "p_wilcoxon_two_sided",
        "p_wilcoxon_two_sided_holm", "wilcoxon_two_sided_status",
    ]
    if cell_scope_metrics.empty:
        return pd.DataFrame(columns=delta_columns), pd.DataFrame(columns=summary_columns)

    df = cell_scope_metrics.copy()
    df = df.loc[df["statistic"].astype(str).eq(JACCARD_STATISTIC)].copy()
    if df.empty:
        return pd.DataFrame(columns=delta_columns), pd.DataFrame(columns=summary_columns)
    df["value"] = _numeric(df["value"])
    observed = df.loc[df["source"].astype(str).eq(OBSERVED_SOURCE)].copy()
    nulls = df.loc[~df["source"].astype(str).eq(OBSERVED_SOURCE)].copy()
    if observed.empty or nulls.empty:
        return pd.DataFrame(columns=delta_columns), pd.DataFrame(columns=summary_columns)
    nulls = _fold_decoy_pseudo_triplets(nulls, decoy_triplet_map)
    if nulls.empty:
        return pd.DataFrame(columns=delta_columns), pd.DataFrame(columns=summary_columns)

    observed = (
        observed.groupby(["cell_scope", "evaluable_set_type", "statistic", "triplet_id"], dropna=False, sort=True)
        .agg(observed_value=("value", "mean"), n_cell_pairs_used_observed=("n_cell_pairs_used", "sum"))
        .reset_index()
    )
    null_rep = (
        nulls.groupby(["cell_scope", "null_type", "evaluable_set_type", "statistic", "triplet_id"], dropna=False, sort=True)
        .agg(
            baseline_median=("value", "median"),
            n_null_draws=("value", "size"),
        )
        .reset_index()
    )
    by_triplet = null_rep.merge(
        observed, on=["cell_scope", "evaluable_set_type", "statistic", "triplet_id"], how="inner",
    )
    if by_triplet.empty:
        return pd.DataFrame(columns=delta_columns), pd.DataFrame(columns=summary_columns)
    by_triplet["delta"] = by_triplet["observed_value"] - by_triplet["baseline_median"]
    by_triplet = by_triplet[delta_columns].dropna(subset=["delta"]).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for keys, sub in by_triplet.groupby(["cell_scope", "null_type", "evaluable_set_type", "statistic"], dropna=False, sort=True):
        delta = _numeric(sub["delta"]).dropna()
        n = int(len(delta))
        n_pos = int((delta > 0).sum())
        n_neg = int((delta < 0).sum())
        n_nonzero = n_pos + n_neg
        delta_mean, ci025, ci975, group_seed = bootstrap_mean_ci(
            delta,
            seed=seed,
            group_parts=(direction, keys[1], keys[2], keys[3], "all_edges", keys[0]),
            n_bootstrap=n_bootstrap,
        )
        wilcoxon_statistic, p_wilcoxon, wilcoxon_status = wilcoxon_signed_rank_two_sided(delta)
        rows.append({
            "cell_scope": str(keys[0]),
            "null_type": str(keys[1]),
            "evaluable_set_type": str(keys[2]),
            "statistic": str(keys[3]),
            "n_triplets": int(sub["triplet_id"].astype(str).nunique()),
            "n_cell_pairs_used_observed_total": int(_numeric(sub["n_cell_pairs_used_observed"]).fillna(0).sum()),
            "observed_mean": float(_numeric(sub["observed_value"]).mean()),
            "baseline_median_mean": float(_numeric(sub["baseline_median"]).mean()),
            "estimand": PRIMARY_ESTIMAND,
            "n_bootstrap": int(n_bootstrap),
            "bootstrap_seed": group_seed,
            "resampling_unit": "triplet_id",
            "ci_method": BOOTSTRAP_CI_METHOD,
            "ci_level": BOOTSTRAP_CI_LEVEL,
            "delta_mean": delta_mean,
            "delta_mean_ci025": ci025,
            "delta_mean_ci975": ci975,
            "delta_median": float(delta.median()) if n else float("nan"),
            "n_positive_delta": n_pos,
            "n_negative_delta": n_neg,
            "n_zero_delta": int((delta == 0).sum()),
            "n_nonzero_delta": n_nonzero,
            "wilcoxon_two_sided_statistic": wilcoxon_statistic,
            "p_wilcoxon_two_sided": p_wilcoxon,
            "p_wilcoxon_two_sided_holm": float("nan"),
            "wilcoxon_two_sided_status": wilcoxon_status,
        })
    summary = pd.DataFrame(rows, columns=summary_columns)
    # Holm is applied across the null models of one (cell_scope, evaluable set, statistic),
    # i.e. the same family size as the global Jaccard Wilcoxon test, so the same-cell subset
    # and the all-cell-pairs aggregate each carry their own corrected p values.
    if not summary.empty:
        for _, sub in summary.groupby(["cell_scope", "evaluable_set_type", "statistic"], dropna=False, sort=False):
            summary.loc[sub.index, "p_wilcoxon_two_sided_holm"] = holm_adjust(
                list(sub["p_wilcoxon_two_sided"])
            )
    summary = _sort_canonical(
        summary, ["evaluable_set_type", "statistic", "cell_scope", "null_type"],
    )
    return by_triplet, summary
