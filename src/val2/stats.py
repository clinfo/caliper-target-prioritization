"""Disease-level significance analysis for Val2 ablation outputs.

This module post-processes already generated ablation metrics
(`ablation/val2_metrics_k*.csv`) without any re-scoring, mirroring the
read-only contract of `viz.make_all_figures`. Significance testing is fixed to
the primary distance `k = PRIMARY_K` (k* selected label-free in Val1): the
other k in `K_valid` are described by the metric curves, not tested. For each
metric (AP, Lift@10) it computes disease-level paired statistics comparing
CALIPER (`proposed`) against each ablation baseline:

- per-disease delta tables (pairwise and best-baseline),
- effect size (median Δ, win-rate across diseases, quartiles),
- paired bootstrap 95% CI (disease resampling),
- two-sided paired t-test of the per-disease Δ against 0,
- Holm-adjusted p-values within each metric; `significant` is `p_holm < alpha`.

`paired_ttest` and `holm_adjust` are the single implementation of the test and
the correction shared with `val2_baseline.evaluate`, so both validations report
the same statistic.

Outputs are written under `<run_root>/stat_tests/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .summary import PRIMARY_K, load_ablation_metrics, read_json_or_empty, resolve_k_valid

DEFAULT_PROPOSED_METHOD = "proposed"
DEFAULT_BASELINES = ["kg_nosign", "kg_sign_noalign", "noKG_sizematch"]
DEFAULT_METRICS = ["AP", "Lift@10"]

# The only multiple-comparison correction used across val2 / val2_baseline.
CORRECTION = "holm"

DEFAULT_STATS: Dict[str, Any] = {
    "enabled": True,
    "n_boot": 10000,
    "bootstrap_seed": 0,
    "alpha": 0.05,
    "metrics": DEFAULT_METRICS,
    "comparisons": ["pairwise", "best_baseline"],
    "baselines": DEFAULT_BASELINES,
    "proposed_method": DEFAULT_PROPOSED_METHOD,
}


def _load_stats_cfg(run_root: Path) -> Dict[str, Any]:
    resolved = read_json_or_empty(run_root / "config" / "resolved_config.json")
    raw = resolved.get("stats") if isinstance(resolved, dict) else None
    cfg = dict(DEFAULT_STATS)
    if isinstance(raw, dict):
        for key in cfg:
            if key in raw and raw[key] is not None:
                cfg[key] = raw[key]
    cfg["metrics"] = [str(m) for m in (cfg.get("metrics") or DEFAULT_METRICS)]
    cfg["baselines"] = [str(b) for b in (cfg.get("baselines") or DEFAULT_BASELINES)]
    cfg["comparisons"] = [str(c).strip().lower() for c in (cfg.get("comparisons") or ["pairwise", "best_baseline"])]
    return cfg


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values, preserving input order."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    out = np.full(m, np.nan, dtype=float)
    valid_mask = np.isfinite(pvals)
    if not valid_mask.any():
        return out
    valid_indices = np.where(valid_mask)[0]
    valid_p = pvals[valid_mask]
    order_local = np.argsort(valid_p)
    ordered_indices = valid_indices[order_local]
    ordered_p = valid_p[order_local]
    m_valid = len(ordered_p)
    running_max = 0.0
    for rank, (idx, p) in enumerate(zip(ordered_indices, ordered_p), start=1):
        candidate = min(1.0, float((m_valid - rank + 1) * p))
        running_max = max(running_max, candidate)
        out[idx] = running_max
    return out


def paired_ttest(values: np.ndarray) -> Tuple[float, float, str]:
    """Two-sided one-sample t-test of the paired differences against 0.

    ``values`` are the per-disease Δ = proposed - baseline. Returns
    ``(t_statistic, p_value, status)``; a Δ vector that is identically zero has
    no effect to detect and is reported as ``p = 1`` rather than as a failure.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), "no finite paired differences"
    if np.all(x == 0.0):
        return float("nan"), 1.0, "all paired differences were zero"
    if len(x) < 2:
        return float("nan"), float("nan"), "fewer than two finite paired differences"
    try:
        from scipy.stats import ttest_1samp  # type: ignore
    except Exception:  # pragma: no cover
        return float("nan"), float("nan"), "scipy unavailable"
    try:
        res = ttest_1samp(x, 0.0)
    except ValueError as exc:  # pragma: no cover - defensive
        return float("nan"), float("nan"), str(exc)
    return float(res.statistic), float(res.pvalue), "ok"


def _paired_bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> Dict[str, float]:
    """Bootstrap CIs (via disease resampling) for both the mean and the median."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        nan = float("nan")
        return {
            "mean": nan, "median": nan,
            "ci_low": nan, "ci_high": nan,
            "median_ci_low": nan, "median_ci_high": nan,
        }
    rng = np.random.default_rng(int(seed))
    boot_means = np.empty(int(n_boot), dtype=float)
    boot_medians = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        sample = values[idx]
        boot_means[b] = sample.mean()
        boot_medians[b] = np.median(sample)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    med_low, med_high = np.percentile(boot_medians, [2.5, 97.5])
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "median_ci_low": float(med_low),
        "median_ci_high": float(med_high),
    }


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


def _prepare_metrics(metrics: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    df = metrics.copy()
    df["method"] = df["method"].astype(str)
    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_pairwise_delta(
    metrics: pd.DataFrame,
    *,
    proposed: str,
    baselines: List[str],
    k_valid: List[int],
    metric_cols: List[str],
) -> pd.DataFrame:
    """Per-(disease, k, baseline) delta = proposed - baseline for each metric."""
    if metrics.empty or not {"disease_id", "k", "method"}.issubset(metrics.columns):
        return pd.DataFrame()
    df = _prepare_metrics(metrics, metric_cols)
    df = df[df["k"].isin([int(k) for k in k_valid])]
    keep_methods = [proposed] + list(baselines)
    df = df[df["method"].isin(keep_methods)]
    if df.empty:
        return pd.DataFrame()
    agg = {col: (col, "mean") for col in metric_cols if col in df.columns}
    df = df.groupby(["disease_id", "k", "method"], as_index=False, dropna=False).agg(**agg)

    rows: List[pd.DataFrame] = []
    for metric in metric_cols:
        if metric not in df.columns:
            continue
        wide = df.pivot(index=["disease_id", "k"], columns="method", values=metric).reset_index()
        wide.columns.name = None
        if proposed not in wide.columns:
            continue
        for baseline in baselines:
            if baseline not in wide.columns:
                continue
            sub = wide[["disease_id", "k", proposed, baseline]].copy()
            sub = sub.rename(columns={proposed: f"{metric}__proposed", baseline: f"{metric}__baseline"})
            sub = sub.dropna(subset=[f"{metric}__proposed", f"{metric}__baseline"])
            sub["baseline"] = baseline
            sub["metric"] = metric
            sub["delta"] = sub[f"{metric}__proposed"] - sub[f"{metric}__baseline"]
            rows.append(sub)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["k"] = out["k"].astype(int)
    return out.sort_values(["metric", "baseline", "k", "disease_id"]).reset_index(drop=True)


def build_best_baseline_delta(
    metrics: pd.DataFrame,
    *,
    proposed: str,
    baselines: List[str],
    k_valid: List[int],
    metric_cols: List[str],
) -> pd.DataFrame:
    """Per-(disease, k) delta vs the best baseline.

    The best baseline is selected by AP (max across baselines) per (disease, k);
    other metrics use that same selected baseline's value (spec §3.1).
    """
    if metrics.empty or not {"disease_id", "k", "method", "AP"}.issubset(metrics.columns):
        return pd.DataFrame()
    cols = list(dict.fromkeys(["AP"] + metric_cols))
    df = _prepare_metrics(metrics, cols)
    df = df[df["k"].isin([int(k) for k in k_valid])]
    keep_methods = [proposed] + list(baselines)
    df = df[df["method"].isin(keep_methods)]
    if df.empty:
        return pd.DataFrame()
    agg = {col: (col, "mean") for col in cols if col in df.columns}
    df = df.groupby(["disease_id", "k", "method"], as_index=False, dropna=False).agg(**agg)

    pivots = {col: df.pivot(index=["disease_id", "k"], columns="method", values=col) for col in cols if col in df.columns}
    ap_pivot = pivots.get("AP")
    if ap_pivot is None:
        return pd.DataFrame()
    available_baselines = [b for b in baselines if b in ap_pivot.columns]
    if proposed not in ap_pivot.columns or not available_baselines:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for idx in ap_pivot.index:
        disease_id, k = idx
        ap_props = ap_pivot.loc[idx, proposed]
        if not np.isfinite(ap_props):
            continue
        baseline_aps = {b: ap_pivot.loc[idx, b] for b in available_baselines if np.isfinite(ap_pivot.loc[idx, b])}
        if not baseline_aps:
            continue
        best_baseline = max(baseline_aps, key=baseline_aps.get)
        for metric in metric_cols:
            pivot = pivots.get(metric)
            if pivot is None or proposed not in pivot.columns or best_baseline not in pivot.columns:
                continue
            prop_val = pivot.loc[idx, proposed]
            base_val = pivot.loc[idx, best_baseline]
            if not (np.isfinite(prop_val) and np.isfinite(base_val)):
                continue
            rows.append(
                {
                    "disease_id": str(disease_id),
                    "k": int(k),
                    "metric": metric,
                    "best_baseline_method": best_baseline,
                    f"{metric}__proposed": float(prop_val),
                    f"{metric}__best_baseline": float(base_val),
                    "delta": float(prop_val) - float(base_val),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(["metric", "k", "disease_id"]).reset_index(drop=True)


def _significance_rows(
    delta_long: pd.DataFrame,
    *,
    group_keys: List[str],
    n_boot: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Per group (e.g. metric × baseline at k=PRIMARY_K) effect size + bootstrap + t-test."""
    rows: List[Dict[str, Any]] = []
    for keys, grp in delta_long.groupby(group_keys, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        vals = grp["delta"].to_numpy(dtype=float)
        row: Dict[str, Any] = dict(zip(group_keys, keys))
        row.update(_delta_effect_size(vals))
        boot = _paired_bootstrap_ci(vals, n_boot, bootstrap_seed)
        row["ci_low"] = boot["ci_low"]
        row["ci_high"] = boot["ci_high"]
        row["median_ci_low"] = boot["median_ci_low"]
        row["median_ci_high"] = boot["median_ci_high"]
        row["n_boot"] = int(n_boot)
        row["bootstrap_seed"] = int(bootstrap_seed)
        stat, pval, status = paired_ttest(vals)
        row["t_stat"] = stat
        row["p_value"] = pval
        row["ttest_status"] = status
        rows.append(row)
    return pd.DataFrame(rows)


def _apply_correction(sig_df: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    """Holm-adjust p_value within each metric (family = all rows of the metric)."""
    if sig_df.empty:
        return sig_df
    out_frames: List[pd.DataFrame] = []
    for _, grp in sig_df.groupby("metric", sort=True):
        grp = grp.copy()
        grp["p_holm"] = holm_adjust(grp["p_value"].to_numpy(dtype=float))
        grp["correction"] = CORRECTION
        grp["alpha"] = float(alpha)
        grp["significant"] = grp["p_holm"] < float(alpha)
        out_frames.append(grp)
    return pd.concat(out_frames, ignore_index=True)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def make_all_stats(run_root: str | Path) -> Dict[str, str]:
    """Compute and write disease-level significance tables from existing metrics.

    Mirrors viz.make_all_figures: reads only `ablation/val2_metrics_k*.csv` and
    the resolved config; performs no scoring. Every table is restricted to the
    primary distance `k = PRIMARY_K`. Returns a mapping of logical name to
    run-root-relative output path.
    """
    run_root = Path(run_root)
    cfg = _load_stats_cfg(run_root)
    outputs: Dict[str, str] = {}
    if not bool(cfg.get("enabled", True)):
        return outputs

    stat_dir = run_root / "stat_tests"
    metrics = load_ablation_metrics(run_root)
    metric_cols = [m for m in cfg["metrics"]]
    proposed = str(cfg.get("proposed_method", DEFAULT_PROPOSED_METHOD))
    baselines = list(cfg["baselines"])
    comparisons = list(cfg["comparisons"])
    n_boot = int(cfg["n_boot"])
    boot_seed = int(cfg["bootstrap_seed"])
    alpha = float(cfg["alpha"])

    k_valid = resolve_k_valid(run_root)
    if not k_valid and not metrics.empty and "k" in metrics.columns:
        k_valid = sorted(int(k) for k in pd.to_numeric(metrics["k"], errors="coerce").dropna().unique())
    # Testing is fixed to the primary distance; K_valid only decides whether
    # PRIMARY_K was in scope for this run at all.
    k_tested = [PRIMARY_K]

    manifest: Dict[str, Any] = {
        "input_metrics": None,
        "k_valid": [int(k) for k in k_valid],
        "k_tested": list(k_tested),
        "test": "paired_t_test_two_sided",
        "correction": CORRECTION,
        "proposed_method": proposed,
        "baselines": baselines,
        "metrics": metric_cols,
        "comparisons": comparisons,
        "n_boot": n_boot,
        "bootstrap_seed": boot_seed,
        "alpha": alpha,
        "outputs": {},
        "diagnostics": {},
    }
    metrics_path = run_root / "ablation"
    hits = sorted(metrics_path.glob("val2_metrics_k*.csv")) if metrics_path.exists() else []
    if hits:
        manifest["input_metrics"] = str(hits[0].relative_to(run_root))

    if metrics.empty or not k_valid:
        manifest["diagnostics"]["error"] = "ablation metrics or k_valid unavailable; nothing computed."
        _write_manifest(manifest, stat_dir, run_root, outputs)
        return outputs
    if PRIMARY_K not in [int(k) for k in k_valid]:
        manifest["diagnostics"]["error"] = (
            f"primary distance k={PRIMARY_K} is not in K_valid={sorted(int(k) for k in k_valid)}; nothing computed."
        )
        _write_manifest(manifest, stat_dir, run_root, outputs)
        return outputs

    if "pairwise" in comparisons:
        pairwise_delta = build_pairwise_delta(
            metrics, proposed=proposed, baselines=baselines, k_valid=k_tested, metric_cols=metric_cols
        )
        if not pairwise_delta.empty:
            delta_path = stat_dir / "val2_pairwise_disease_level_delta.csv"
            _write_csv(_wide_delta_table(pairwise_delta, metric_cols, keys=["disease_id", "k", "baseline"]), delta_path)
            outputs["pairwise_disease_level_delta"] = str(delta_path.relative_to(run_root))

            sig = _significance_rows(
                pairwise_delta, group_keys=["metric", "baseline", "k"], n_boot=n_boot, bootstrap_seed=boot_seed
            )
            sig = _apply_correction(sig, alpha=alpha)
            sig_path = stat_dir / "val2_pairwise_significance.csv"
            _write_csv(sig, sig_path)
            outputs["pairwise_significance"] = str(sig_path.relative_to(run_root))
            manifest["diagnostics"]["n_pairwise_rows"] = int(len(pairwise_delta))

    if "best_baseline" in comparisons:
        best_delta = build_best_baseline_delta(
            metrics, proposed=proposed, baselines=baselines, k_valid=k_tested, metric_cols=metric_cols
        )
        if not best_delta.empty:
            delta_path = stat_dir / "val2_best_baseline_disease_level_delta.csv"
            _write_csv(_wide_best_delta_table(best_delta, metric_cols), delta_path)
            outputs["best_baseline_disease_level_delta"] = str(delta_path.relative_to(run_root))

            sig = _significance_rows(
                best_delta, group_keys=["metric", "k"], n_boot=n_boot, bootstrap_seed=boot_seed
            )
            sig = _apply_correction(sig, alpha=alpha)
            sig_path = stat_dir / "val2_best_baseline_significance.csv"
            _write_csv(sig, sig_path)
            outputs["best_baseline_significance"] = str(sig_path.relative_to(run_root))
            manifest["diagnostics"]["n_best_baseline_rows"] = int(len(best_delta))

    manifest["outputs"] = dict(outputs)
    _write_manifest(manifest, stat_dir, run_root, outputs)
    return outputs


def _wide_delta_table(delta_long: pd.DataFrame, metric_cols: List[str], *, keys: List[str]) -> pd.DataFrame:
    """Pivot long per-metric deltas back to one row per key with per-metric columns."""
    frames: List[pd.DataFrame] = []
    for metric in metric_cols:
        sub = delta_long[delta_long["metric"] == metric].copy()
        if sub.empty:
            continue
        sub = sub.rename(
            columns={
                f"{metric}__proposed": f"{metric}__proposed",
                f"{metric}__baseline": f"{metric}__baseline",
                "delta": f"delta_{metric}",
            }
        )
        cols = keys + [f"{metric}__proposed", f"{metric}__baseline", f"delta_{metric}"]
        frames.append(sub[[c for c in cols if c in sub.columns]])
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on=keys, how="outer")
    return out.sort_values(keys).reset_index(drop=True)


def _wide_best_delta_table(delta_long: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    keys = ["disease_id", "k"]
    frames: List[pd.DataFrame] = []
    for metric in metric_cols:
        sub = delta_long[delta_long["metric"] == metric].copy()
        if sub.empty:
            continue
        sub = sub.rename(columns={"delta": f"delta_{metric}"})
        cols = keys + ["best_baseline_method", f"{metric}__proposed", f"{metric}__best_baseline", f"delta_{metric}"]
        frames.append(sub[[c for c in cols if c in sub.columns]])
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for frame in frames[1:]:
        merge_keys = keys + (["best_baseline_method"] if "best_baseline_method" in frame.columns and "best_baseline_method" in out.columns else [])
        out = out.merge(frame, on=merge_keys, how="outer")
    return out.sort_values(keys).reset_index(drop=True)


def _write_manifest(manifest: Dict[str, Any], stat_dir: Path, run_root: Path, outputs: Dict[str, str]) -> None:
    stat_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = stat_dir / "val2_significance_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    outputs["significance_manifest"] = str(manifest_path.relative_to(run_root))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Compute Val2 disease-level significance tables from existing outputs.")
    p.add_argument("--run-root", required=True)
    args = p.parse_args()
    print(json.dumps(make_all_stats(args.run_root), indent=2, ensure_ascii=False))
