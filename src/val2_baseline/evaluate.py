"""Ranking evaluation and statistical comparison (methodology §8).

Reuses the val2 tie-aware metrics (expected AP / expected P@K) so all four
methods share an identical tie policy (validation #12). Computes per-disease
AP and Lift@K for each K in ``K_list``, paired CALIPER-minus-baseline deltas,
and paired bootstrap CIs.

Significance testing is the two-sided paired t-test of the per-disease Δ
against 0, Holm-corrected across baselines within a metric, and is restricted
to CALIPER at the primary distance ``k=4``. Both the test and the correction
are imported from :mod:`val2.stats` so val2 and val2_baseline report the same
statistic.

Also supports supplementary Lift@K_eval macro-averages over D_eval(K) where
E[h_d,K] >= lift_eval_min_expected_hits.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from val2.legacy import (
    expected_average_precision_with_ties,
    expected_precision_at_k_with_ties,
    average_precision,
    precision_at_k,
)
from val2.stats import CORRECTION, holm_adjust, paired_ttest

from .caliper import detect_caliper_methods, primary_caliper_methods

METHODS = ["CALIPER", "NCS", "proximity", "RWR"]
BASELINES = ["NCS", "proximity", "RWR"]
PRIMARY_METRICS = ["AP", "Lift@10"]
_LIFT_COL_RE = re.compile(r"^Lift@(\d+)$")


def normalize_k_list(K_list: List[int] | None, *, K_eff_cap: int = 10) -> List[int]:
    if K_list:
        out = sorted(dict.fromkeys(int(k) for k in K_list))
    else:
        out = [int(K_eff_cap)]
    if not out or any(k <= 0 for k in out):
        raise ValueError(f"[evaluate] K_list must be a non-empty list of positive integers. got {out!r}")
    return out


def lift_col(k: int) -> str:
    return f"Lift@{int(k)}"


def lift_eval_col(k: int) -> str:
    return f"Lift@{int(k)}_eval"


def p_at_k_col(k: int) -> str:
    return f"P@{int(k)}"


def delta_lift_col(k: int) -> str:
    return f"delta_Lift@{int(k)}"


def in_lift_eval_col(k: int) -> str:
    return f"in_lift_eval_{int(k)}"


def e_random_hits_col(k: int) -> str:
    return f"E_random_hits_{int(k)}"


def boundary_tie_col(k: int) -> str:
    return f"top{int(k)}_boundary_tie_size"


def default_stats_metrics(k_list: List[int]) -> List[str]:
    metrics = ["AP"]
    for k in k_list:
        metrics.append(lift_col(k))
        metrics.append(lift_eval_col(k))
    return metrics


def build_delta_col_map(k_list: List[int]) -> Dict[str, str]:
    col_map = {"AP": "delta_AP"}
    for k in k_list:
        col_map[lift_col(k)] = delta_lift_col(k)
        col_map[lift_eval_col(k)] = delta_lift_col(k)
    return col_map


def lift_columns_in_df(df: pd.DataFrame) -> List[int]:
    ks: List[int] = []
    for col in df.columns:
        m = _LIFT_COL_RE.match(str(col))
        if m:
            ks.append(int(m.group(1)))
    return sorted(set(ks))


def expected_random_hits(n_pos: int, n_cand: int, k_cap: int) -> float:
    """Expected positives in top-K_eff under random ranking (method-independent)."""
    if n_cand <= 0:
        return float("nan")
    k_eff = min(int(k_cap), int(n_cand))
    return float(k_eff) * float(n_pos) / float(n_cand)


def build_lift_eval_cohorts(
    metrics_df: pd.DataFrame,
    *,
    k_list: List[int],
    min_expected_hits: float = 1.0,
) -> pd.DataFrame:
    """Per-disease cohort flags for each K in ``k_list`` (long format)."""
    if metrics_df.empty:
        return pd.DataFrame(
            columns=["disease_id", "K", "n_pos", "n_candidate", "K_eff", "E_random_hits", "in_lift_eval"]
        )
    dis = (
        metrics_df.groupby("disease_id", as_index=False)
        .agg(n_pos=("n_pos", "first"), n_candidate=("n_candidate", "first"))
        .copy()
    )
    frames: List[pd.DataFrame] = []
    for k in k_list:
        part = dis.copy()
        part["K"] = int(k)
        part["K_eff"] = part["n_candidate"].apply(lambda n, kk=k: min(int(kk), int(n)))
        part["E_random_hits"] = part.apply(
            lambda r, kk=k: expected_random_hits(int(r["n_pos"]), int(r["n_candidate"]), int(kk)),
            axis=1,
        )
        part["in_lift_eval"] = part["E_random_hits"] >= float(min_expected_hits)
        frames.append(part)
    return pd.concat(frames, ignore_index=True).sort_values(["K", "disease_id"]).reset_index(drop=True)


def build_lift_eval_cohort(
    metrics_df: pd.DataFrame,
    *,
    k_eff_cap: int,
    min_expected_hits: float = 1.0,
) -> pd.DataFrame:
    """Backward-compatible wrapper for a single K (primary cohort table)."""
    return build_lift_eval_cohorts(
        metrics_df,
        k_list=[int(k_eff_cap)],
        min_expected_hits=min_expected_hits,
    ).rename(columns={"K": "K_cap"})


def annotate_lift_eval_flags(metrics_df: pd.DataFrame, cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Attach per-K ``E_random_hits_K`` / ``in_lift_eval_K`` columns."""
    out = metrics_df.copy()
    if cohort_df.empty:
        return out
    if "K" in cohort_df.columns:
        for k in sorted(cohort_df["K"].astype(int).unique()):
            sub = cohort_df[cohort_df["K"].astype(int) == int(k)][
                ["disease_id", "E_random_hits", "in_lift_eval"]
            ].rename(
                columns={
                    "E_random_hits": e_random_hits_col(k),
                    "in_lift_eval": in_lift_eval_col(k),
                }
            )
            out = out.merge(sub, on="disease_id", how="left")
        primary_k = int(sorted(cohort_df["K"].astype(int).unique())[0])
        out["E_random_hits"] = out[e_random_hits_col(primary_k)]
        out["in_lift_eval"] = out[in_lift_eval_col(primary_k)]
        return out
    out = out.merge(
        cohort_df[["disease_id", "E_random_hits", "in_lift_eval"]],
        on="disease_id",
        how="left",
    )
    return _backfill_per_k_lift_flags(out)


def _backfill_per_k_lift_flags(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Copy legacy ``in_lift_eval`` / ``E_random_hits`` to per-K columns when absent."""
    out = metrics_df.copy()
    if "in_lift_eval" not in out.columns:
        return out
    for k in lift_columns_in_df(out):
        flag_col = in_lift_eval_col(k)
        hits_col = e_random_hits_col(k)
        if flag_col not in out.columns:
            out[flag_col] = out["in_lift_eval"]
        if hits_col not in out.columns and "E_random_hits" in out.columns:
            out[hits_col] = out["E_random_hits"]
    return out


def _parse_lift_eval_k(metric: str) -> int | None:
    prefix = "Lift@"
    suffix = "_eval"
    if str(metric).startswith(prefix) and str(metric).endswith(suffix):
        body = str(metric)[len(prefix) : -len(suffix)]
        if body.isdigit():
            return int(body)
    return None


def _filter_deltas_for_metric(deltas_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    k_eval = _parse_lift_eval_k(metric)
    if k_eval is None:
        return deltas_df
    flag_col = in_lift_eval_col(k_eval)
    if flag_col not in deltas_df.columns:
        raise ValueError(f"[evaluate] deltas_df missing {flag_col} for {metric}")
    return deltas_df[deltas_df[flag_col].astype(bool)].copy()


def apply_fallback_floor(
    method_maps: Dict[str, Dict[Tuple[str, str], float]],
    fallback_keys_by_method: Dict[str, Set[Tuple[str, str]]],
    common_candidates_df: pd.DataFrame,
) -> None:
    """Floor fallback-pair scores below every valid score, per disease/method.

    "Fallback" pairs (no local evidence: empty local seed set, empty local
    feature universe, etc.) must not sit at an arbitrary neutral point in a
    method's score distribution (methodology fairness fix): for each disease
    and each method with at least one fallback pair, every fallback entry is
    overwritten with ``min(valid_scores_in_disease) - 1.0``, so "no local
    evidence" always ranks strictly below every valid score for that method
    in that disease. When a disease has no valid score at all for a method
    (every candidate is a fallback), all of its entries for that method are
    set to the same constant (0.0) -- a tie at chance level, since there is
    no valid score to floor below. CALIPER has no fallback concept and must
    not appear in ``fallback_keys_by_method``.

    Mutates ``method_maps`` in place; returns ``None``.
    """
    disease_targets: Dict[str, List[str]] = {}
    for r in common_candidates_df.itertuples(index=False):
        disease_targets.setdefault(str(r.disease_id), []).append(str(r.target_gene))

    for method, fallback_keys in fallback_keys_by_method.items():
        if not fallback_keys:
            continue
        if method not in method_maps:
            continue
        score_map = method_maps[method]
        for disease_id, targets in disease_targets.items():
            keys = [(disease_id, t) for t in targets]
            fb_keys_here = [k for k in keys if k in fallback_keys]
            if not fb_keys_here:
                continue
            valid_scores = [score_map[k] for k in keys if k in score_map and k not in fallback_keys]
            floor = float(min(valid_scores)) - 1.0 if valid_scores else 0.0
            for k in fb_keys_here:
                if k in score_map:
                    score_map[k] = floor


def assert_method_consistency(long_df: pd.DataFrame, methods: List[str] = METHODS) -> None:
    """Validation #6, #7, #12: identical candidate set per disease across methods,
    and all scores finite."""
    if not np.isfinite(long_df["score"].to_numpy(dtype=float)).all():
        bad = long_df[~np.isfinite(long_df["score"].astype(float))]
        raise ValueError(f"[evaluate] non-finite scores after common-set fix: {len(bad)} rows.")
    for disease_id, g in long_df.groupby("disease_id"):
        target_sets = {}
        for method in methods:
            sub = g[g["method"] == method]
            target_sets[method] = tuple(sorted(sub["target_gene"].astype(str)))
        ref = target_sets[methods[0]]
        for method in methods[1:]:
            if target_sets[method] != ref:
                raise ValueError(
                    f"[evaluate] disease {disease_id}: method '{method}' candidate set "
                    f"differs from '{methods[0]}' (#7 violation)."
                )


def compute_metrics_by_disease(
    long_df: pd.DataFrame,
    *,
    methods: List[str] = METHODS,
    K_list: List[int] | None = None,
    K_eff_cap: int = 10,
    tie_policy: str = "expected",
) -> pd.DataFrame:
    k_list = normalize_k_list(K_list, K_eff_cap=K_eff_cap)
    primary_k = k_list[0]
    rows: List[Dict[str, object]] = []
    for (disease_id, method), g in long_df.groupby(["disease_id", "method"]):
        y_true = g["label"].astype(int).tolist()
        y_score = g["score"].astype(float).tolist()
        n_candidate = len(y_true)
        n_pos = int(sum(y_true))
        if n_pos == 0:
            continue
        chance = n_pos / n_candidate if n_candidate else float("nan")
        if tie_policy == "legacy":
            ap = average_precision(y_true, y_score)
            n_score_groups = np.nan
            n_tie_groups = np.nan
            largest_tie_group = np.nan
        else:
            ap, n_score_groups, n_tie_groups, largest_tie_group = expected_average_precision_with_ties(
                y_true, y_score
            )
        row: Dict[str, object] = {
            "disease_id": str(disease_id),
            "method": str(method),
            "AP": float(ap),
            "n_pos": n_pos,
            "n_candidate": int(n_candidate),
            "chance": float(chance),
            "n_score_groups": n_score_groups,
            "n_tie_groups": n_tie_groups,
            "largest_tie_group": largest_tie_group,
        }
        for k in k_list:
            k_eff = min(int(k), n_candidate)
            if tie_policy == "legacy":
                p_at_k = precision_at_k(y_true, y_score, k_eff)
                boundary_tie = np.nan
            else:
                p_at_k, boundary_tie = expected_precision_at_k_with_ties(y_true, y_score, k_eff)
            lift = (p_at_k / chance) if (chance and chance > 0 and np.isfinite(p_at_k)) else float("nan")
            row[p_at_k_col(k)] = float(p_at_k)
            row[lift_col(k)] = float(lift)
            row[boundary_tie_col(k)] = boundary_tie
        row["K_eff"] = int(min(int(primary_k), n_candidate))
        row["P@10"] = row.get(p_at_k_col(10), row[p_at_k_col(primary_k)])
        row["Lift@10"] = row.get(lift_col(10), row[lift_col(primary_k)])
        row["top10_boundary_tie_size"] = row.get(boundary_tie_col(10), row[boundary_tie_col(primary_k)])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["disease_id", "method"]).reset_index(drop=True)


def compute_method_summary(
    metrics_df: pd.DataFrame,
    methods: List[str] = METHODS,
    *,
    k_list: List[int] | None = None,
) -> pd.DataFrame:
    k_list = k_list or lift_columns_in_df(metrics_df)
    if not k_list:
        k_list = [10]
    rows = []
    for method in methods:
        sub = metrics_df[metrics_df["method"] == method]
        row: Dict[str, object] = {
            "method": method,
            "n_diseases": int(len(sub)),
            "mAP": float(sub["AP"].mean()) if len(sub) else float("nan"),
            "median_AP": float(sub["AP"].median()) if len(sub) else float("nan"),
        }
        for k in k_list:
            lcol = lift_col(k)
            flag_col = in_lift_eval_col(k)
            sub_eval = sub[sub[flag_col].astype(bool)] if flag_col in sub.columns else sub.iloc[0:0]
            row[f"mean_{lcol}"] = float(sub[lcol].mean()) if len(sub) and lcol in sub.columns else float("nan")
            row[f"median_{lcol}"] = float(sub[lcol].median()) if len(sub) and lcol in sub.columns else float("nan")
            row[f"n_diseases_lift_eval_{k}"] = int(len(sub_eval))
            row[f"mean_{lift_eval_col(k)}"] = (
                float(sub_eval[lcol].mean()) if len(sub_eval) and lcol in sub_eval.columns else float("nan")
            )
            row[f"median_{lift_eval_col(k)}"] = (
                float(sub_eval[lcol].median()) if len(sub_eval) and lcol in sub_eval.columns else float("nan")
            )
        if k_list == [10] or 10 in k_list:
            row["mean_Lift@10"] = row.get("mean_Lift@10", float("nan"))
            row["median_Lift@10"] = row.get("median_Lift@10", float("nan"))
            row["n_diseases_lift_eval"] = row.get("n_diseases_lift_eval_10", 0)
            row["mean_Lift@10_eval"] = row.get("mean_Lift@10_eval", float("nan"))
            row["median_Lift@10_eval"] = row.get("median_Lift@10_eval", float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def _resolve_caliper_methods(
    metrics_df: pd.DataFrame | None = None,
    deltas_df: pd.DataFrame | None = None,
    *,
    caliper_methods: List[str] | None = None,
) -> List[str]:
    if caliper_methods is not None:
        return list(caliper_methods)
    if deltas_df is not None and "caliper_method" in deltas_df.columns:
        return sorted(dict.fromkeys(deltas_df["caliper_method"].astype(str)))
    if metrics_df is not None and "method" in metrics_df.columns:
        detected = detect_caliper_methods(metrics_df["method"].astype(str))
        if detected:
            return detected
    return ["CALIPER"]


def _delta_groups(deltas_df: pd.DataFrame, caliper_methods: List[str]) -> List[tuple[str | None, pd.DataFrame]]:
    if "caliper_method" in deltas_df.columns and len(caliper_methods) > 1:
        return [
            (str(cm), deltas_df[deltas_df["caliper_method"].astype(str) == str(cm)].copy())
            for cm in caliper_methods
        ]
    return [(None, deltas_df)]


def compute_paired_deltas(
    metrics_df: pd.DataFrame,
    baselines: List[str] = BASELINES,
    *,
    caliper_methods: List[str] | None = None,
    k_list: List[int] | None = None,
) -> pd.DataFrame:
    k_list = k_list or lift_columns_in_df(metrics_df)
    if not k_list:
        k_list = [10]
    caliper_methods = _resolve_caliper_methods(metrics_df, caliper_methods=caliper_methods)
    cohort_cols = [c for c in metrics_df.columns if str(c).startswith("in_lift_eval")]
    cohort = (
        metrics_df.groupby("disease_id", as_index=False)[cohort_cols].first()
        if cohort_cols
        else None
    )
    rows: List[Dict[str, object]] = []
    for caliper_method in caliper_methods:
        cal_sub = metrics_df[metrics_df["method"].astype(str) == str(caliper_method)]
        pivot_ap = cal_sub.pivot(index="disease_id", columns="method", values="AP")
        for disease_id in pivot_ap.index:
            for baseline in baselines:
                base_ap = float(
                    metrics_df[
                        (metrics_df["disease_id"].astype(str) == str(disease_id))
                        & (metrics_df["method"].astype(str) == str(baseline))
                    ]["AP"].iloc[0]
                )
                row: Dict[str, object] = {
                    "disease_id": str(disease_id),
                    "caliper_method": str(caliper_method),
                    "baseline": baseline,
                    "delta_AP": float(pivot_ap.loc[disease_id, caliper_method] - base_ap),
                }
                for k in k_list:
                    pivot_lift = cal_sub.pivot(index="disease_id", columns="method", values=lift_col(k))
                    base_lift = float(
                        metrics_df[
                            (metrics_df["disease_id"].astype(str) == str(disease_id))
                            & (metrics_df["method"].astype(str) == str(baseline))
                        ][lift_col(k)].iloc[0]
                    )
                    row[delta_lift_col(k)] = float(pivot_lift.loc[disease_id, caliper_method] - base_lift)
                rows.append(row)
    out = pd.DataFrame(rows).sort_values(["caliper_method", "baseline", "disease_id"]).reset_index(drop=True)
    if cohort is not None:
        out = out.merge(cohort, on="disease_id", how="left")
    if "in_lift_eval_10" in out.columns:
        out["in_lift_eval"] = out["in_lift_eval_10"]
    return out


def bootstrap_median_ci(values: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float, float]:
    """95% percentile bootstrap CI of the *median* via disease-resampling.

    Draws exactly one ``rng.integers(0, n, size=n)`` per iteration so that, given
    the same ``(values, n_boot, seed)``, it reproduces the median CI computed
    alongside the mean CI in :func:`_paired_bootstrap_ci` (the resampling indices
    are identical). Used as the figure-layer fallback when the persisted
    ``bootstrap_summary.tsv`` predates the median-CI columns.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    boot_medians = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        boot_medians[b] = np.median(values[rng.integers(0, n, size=n)])
    med_low, med_high = np.percentile(boot_medians, [2.5, 97.5])
    return float(np.median(values)), float(med_low), float(med_high)


def _paired_bootstrap_ci(
    values: np.ndarray, n_boot: int, seed: int
) -> Tuple[float, float, float, float, float, float]:
    """Bootstrap the paired Δ via disease-resampling.

    Returns ``(mean, median, ci_low, ci_high, median_ci_low, median_ci_high)``
    where ``ci_low/ci_high`` is the 95% CI of the *mean* and
    ``median_ci_low/median_ci_high`` is the 95% CI of the *median*. Both CIs come
    from the same resampling draws (one ``rng.integers`` call per iteration), so
    the median CI matches :func:`bootstrap_median_ci` for the same seed.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return (float("nan"),) * 6
    rng = np.random.default_rng(int(seed))
    boot_means = np.empty(int(n_boot), dtype=float)
    boot_medians = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        sample = values[rng.integers(0, n, size=n)]
        boot_means[b] = sample.mean()
        boot_medians[b] = np.median(sample)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    med_low, med_high = np.percentile(boot_medians, [2.5, 97.5])
    return (
        float(values.mean()),
        float(np.median(values)),
        float(ci_low),
        float(ci_high),
        float(med_low),
        float(med_high),
    )


def compute_bootstrap_summary(
    deltas_df: pd.DataFrame,
    *,
    baselines: List[str] = BASELINES,
    metrics: List[str] | None = None,
    caliper_methods: List[str] | None = None,
    k_list: List[int] | None = None,
    n_boot: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    k_list = k_list or lift_columns_in_df(deltas_df)
    if metrics is None:
        metrics = default_stats_metrics(k_list or [10])
    col_map = build_delta_col_map(k_list or [10])
    caliper_methods = _resolve_caliper_methods(deltas_df=deltas_df, caliper_methods=caliper_methods)
    rows: List[Dict[str, object]] = []
    for caliper_method, group_df in _delta_groups(deltas_df, caliper_methods):
        for baseline in baselines:
            sub = group_df[group_df["baseline"].astype(str) == str(baseline)]
            for metric in metrics:
                sub_m = _filter_deltas_for_metric(sub, metric)
                vals = sub_m[col_map[metric]].to_numpy(dtype=float)
                mean, median, ci_low, ci_high, median_ci_low, median_ci_high = _paired_bootstrap_ci(
                    vals, n_boot, seed
                )
                finite = vals[np.isfinite(vals)]
                frac_pos = float(np.mean(finite > 0)) if len(finite) else float("nan")
                row = {
                    "baseline": baseline,
                    "metric": metric,
                    "mean_delta": mean,
                    "median_delta": median,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "median_ci_low": median_ci_low,
                    "median_ci_high": median_ci_high,
                    "win_rate": frac_pos,
                    "frac_disease_positive": frac_pos,
                    "n_boot": int(n_boot),
                    "bootstrap_seed": int(seed),
                }
                if caliper_method is not None:
                    row["caliper_method"] = caliper_method
                rows.append(row)
    return pd.DataFrame(rows)


# The Holm step-down is shared with val2 so both validations correct the same
# way; ``degree_support`` imports it from here under the historical name.
_holm_adjust = holm_adjust


def _safe_wilcoxon(values: np.ndarray, *, alternative: str = "two-sided") -> Tuple[float, float, str]:
    """Two-sided paired Wilcoxon signed-rank vs 0 with defensive checks.

    Kept for the per-disease degree diagnostics in ``degree_support``, which
    compare rank correlations rather than ranking performance and therefore sit
    outside the k=4 paired-t-test family of the method comparison.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x != 0]
    if len(x) == 0:
        return float("nan"), float("nan"), "all paired differences were zero or missing"
    if len(x) < 2:
        return float("nan"), float("nan"), "fewer than two non-zero paired differences"
    try:
        from scipy.stats import wilcoxon  # type: ignore
    except Exception:  # pragma: no cover
        return float("nan"), float("nan"), "scipy unavailable"
    try:
        res = wilcoxon(x, zero_method="wilcox", alternative=alternative, method="auto")
    except TypeError:  # pragma: no cover - older SciPy
        res = wilcoxon(x, zero_method="wilcox", alternative=alternative)
    except ValueError as exc:
        return float("nan"), float("nan"), str(exc)
    return float(res.statistic), float(res.pvalue), "ok"


def _delta_effect_size(vals: np.ndarray) -> Dict[str, object]:
    finite = np.asarray(vals, dtype=float)
    finite = finite[np.isfinite(finite)]
    n = len(finite)
    if n == 0:
        return {
            "n_diseases": 0,
            "n_nonzero": 0,
            "mean_delta": float("nan"),
            "median_delta": float("nan"),
            "q25_delta": float("nan"),
            "q75_delta": float("nan"),
            "win_rate": float("nan"),
            "frac_tie": float("nan"),
            "frac_negative": float("nan"),
        }
    return {
        "n_diseases": int(n),
        "n_nonzero": int(np.sum(finite != 0)),
        "mean_delta": float(np.mean(finite)),
        "median_delta": float(np.median(finite)),
        "q25_delta": float(np.percentile(finite, 25)),
        "q75_delta": float(np.percentile(finite, 75)),
        "win_rate": float(np.mean(finite > 0)),
        "frac_tie": float(np.mean(finite == 0)),
        "frac_negative": float(np.mean(finite < 0)),
    }


def compute_effect_sizes(
    deltas_df: pd.DataFrame,
    *,
    baselines: List[str] = BASELINES,
    metrics: List[str] | None = None,
    caliper_methods: List[str] | None = None,
    k_list: List[int] | None = None,
) -> pd.DataFrame:
    """Per (baseline, metric) effect size: median ΔAP / median ΔLift@K と win-rate。"""
    k_list = k_list or lift_columns_in_df(deltas_df)
    if metrics is None:
        metrics = default_stats_metrics(k_list or [10])
    col_map = build_delta_col_map(k_list or [10])
    caliper_methods = _resolve_caliper_methods(deltas_df=deltas_df, caliper_methods=caliper_methods)
    rows: List[Dict[str, object]] = []
    for caliper_method, group_df in _delta_groups(deltas_df, caliper_methods):
        for baseline in baselines:
            sub = group_df[group_df["baseline"].astype(str) == str(baseline)]
            for metric in metrics:
                sub_m = _filter_deltas_for_metric(sub, metric)
                vals = sub_m[col_map[metric]].to_numpy(dtype=float)
                row: Dict[str, object] = {"baseline": baseline, "metric": metric}
                if caliper_method is not None:
                    row["caliper_method"] = caliper_method
                row.update(_delta_effect_size(vals))
                rows.append(row)
    return pd.DataFrame(rows)


def _tested_delta_groups(
    deltas_df: pd.DataFrame, caliper_methods: List[str]
) -> List[tuple[str | None, pd.DataFrame]]:
    """Like :func:`_delta_groups`, but always narrows to ``caliper_methods``.

    The significance table is restricted to the primary distance, so a run that
    scored several radii passes a single key here and the other radii's deltas
    must be dropped rather than pooled — which is what ``_delta_groups`` would
    do for a one-element list.
    """
    if "caliper_method" in deltas_df.columns and caliper_methods:
        return [
            (str(cm), deltas_df[deltas_df["caliper_method"].astype(str) == str(cm)].copy())
            for cm in caliper_methods
        ]
    return [(None, deltas_df)]


def resolve_test_caliper_methods(
    deltas_df: pd.DataFrame,
    *,
    caliper_methods: List[str] | None = None,
    k_star: int | None = None,
) -> List[str]:
    """CALIPER series that the significance tests run on (primary distance only)."""
    resolved = _resolve_caliper_methods(deltas_df=deltas_df, caliper_methods=caliper_methods)
    if not resolved:
        return []
    return primary_caliper_methods([m for m in resolved if m is not None], k_star=k_star)


def compute_ttest_table(
    deltas_df: pd.DataFrame,
    *,
    baselines: List[str] = BASELINES,
    metrics: List[str] | None = None,
    caliper_methods: List[str] | None = None,
    k_list: List[int] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Two-sided paired t-test of the per-disease Δ against 0, per metric.

    ``caliper_methods`` must already be restricted to the primary distance
    (see :func:`resolve_test_caliper_methods`); other radii are described by the
    effect-size tables but are not tested. Holm corrects across ``baselines``
    within one (caliper_method, metric) family.
    """
    k_list = k_list or lift_columns_in_df(deltas_df)
    if metrics is None:
        metrics = default_stats_metrics(k_list or [10])
    col_map = build_delta_col_map(k_list or [10])
    caliper_methods = _resolve_caliper_methods(deltas_df=deltas_df, caliper_methods=caliper_methods)
    frames: List[pd.DataFrame] = []
    for caliper_method, group_df in _tested_delta_groups(deltas_df, caliper_methods):
        for metric in metrics:
            rows: List[Dict[str, object]] = []
            for baseline in baselines:
                sub = _filter_deltas_for_metric(
                    group_df[group_df["baseline"].astype(str) == str(baseline)],
                    metric,
                )
                vals = sub[col_map[metric]].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                stat, pval, status = paired_ttest(vals)
                eff = _delta_effect_size(vals)
                row = {
                    "baseline": baseline,
                    "metric": metric,
                    "t_stat": stat,
                    "p_value": pval,
                    "ttest_status": status,
                    "mean_delta": eff["mean_delta"],
                    "median_delta": eff["median_delta"],
                    "win_rate": eff["win_rate"],
                    "n_diseases": int(len(vals)),
                }
                if caliper_method is not None:
                    row["caliper_method"] = caliper_method
                rows.append(row)
            out = pd.DataFrame(rows)
            if out.empty:
                continue
            out["p_holm"] = _holm_adjust(out["p_value"].to_numpy(dtype=float))
            out["correction"] = CORRECTION
            out["alpha"] = float(alpha)
            out["significant"] = out["p_holm"] < float(alpha)
            frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_tie_diagnostics(metrics_df: pd.DataFrame, *, k_list: List[int] | None = None) -> pd.DataFrame:
    k_list = k_list or lift_columns_in_df(metrics_df) or [10]
    cols = [
        "disease_id",
        "method",
        "n_candidate",
        "n_score_groups",
        "n_tie_groups",
        "largest_tie_group",
    ] + [boundary_tie_col(k) for k in k_list]
    if "top10_boundary_tie_size" in metrics_df.columns and boundary_tie_col(10) not in cols:
        cols.append("top10_boundary_tie_size")
    present = [c for c in cols if c in metrics_df.columns]
    return metrics_df[present].copy().sort_values(["disease_id", "method"]).reset_index(drop=True)
