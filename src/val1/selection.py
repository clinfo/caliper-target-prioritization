from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


COVERAGE_AGGREGATION = {"cov_nz": "mean", "cov_ratio": "median"}


def _progress(progress, iterable, **kwargs):
    if progress is None:
        return iterable
    if progress is False:
        return iterable
    return progress(iterable, **kwargs)


def _pair_keys_from_records(records) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    pair_keys: list[tuple[str, str]] = []
    for pair in records:
        key = (str(pair["target_gene"]), str(pair["disease_id"]))
        if key not in seen:
            seen.add(key)
            pair_keys.append(key)
    return pair_keys


def _pair_keys_from_frame(df: pd.DataFrame) -> list[tuple[str, str]]:
    if df.empty:
        return []
    pairs = df[["target_gene", "disease_id"]].drop_duplicates()
    return [(str(r.target_gene), str(r.disease_id)) for r in pairs.itertuples(index=False)]


def _coverage_col(metric: str, td: bool = False) -> str:
    if metric == "cov_nz":
        return "C_nz_td" if td else "C_nz"
    if metric == "cov_ratio":
        return "cov_ratio_td_median" if td else "cov_ratio_median"
    raise ValueError(f"Unsupported coverage metric: {metric}")


def _harmonic(x: float, y: float, zero_division: float = 0.0) -> float:
    if not (np.isfinite(x) and np.isfinite(y)):
        return float("nan")
    denom = x + y
    if denom == 0:
        return float(zero_division)
    return float(2.0 * x * y / denom)


def _range_min_max(values: pd.Series) -> tuple[float, float]:
    arr = values.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.min(arr)), float(np.max(arr))


def _normalize_series(values: pd.Series, *, larger_is_better: bool = True) -> tuple[pd.Series, dict[str, Any]]:
    out = pd.Series(np.nan, index=values.index, dtype=float)
    vmin, vmax = _range_min_max(values)
    finite = values.astype(float).apply(np.isfinite)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return out, {"min": vmin, "max": vmax, "constant": False, "not_identifiable": True}
    if vmax > vmin:
        if larger_is_better:
            out.loc[finite] = (values.loc[finite].astype(float) - vmin) / (vmax - vmin)
        else:
            out.loc[finite] = (vmax - values.loc[finite].astype(float)) / (vmax - vmin)
        constant = False
    else:
        out.loc[finite] = 1.0
        constant = True
    return out, {"min": vmin, "max": vmax, "constant": constant, "not_identifiable": False}


def _json_number(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def build_k_metrics(
    *,
    coverage_summary: pd.DataFrame,
    ambiguity_summary: pd.DataFrame,
    coverage_metric: str,
    ambiguity_metric: str,
    normalize: bool,
    zero_division: float = 0.0,
) -> pd.DataFrame:
    if ambiguity_metric != "amb_median":
        raise ValueError("Only ambiguity_metric='amb_median' is supported.")

    c_col = _coverage_col(coverage_metric, td=False)
    if c_col not in coverage_summary.columns:
        raise KeyError(f"coverage_summary is missing required column: {c_col}")
    if "B_amb_median" not in ambiguity_summary.columns:
        raise KeyError("ambiguity_summary is missing required column: B_amb_median")

    df = pd.merge(coverage_summary, ambiguity_summary, on="k", how="inner", validate="one_to_one")
    df = df.sort_values("k").reset_index(drop=True)
    df["coverage_metric"] = coverage_metric
    df["coverage_aggregation"] = COVERAGE_AGGREGATION[coverage_metric]
    df["ambiguity_metric"] = ambiguity_metric
    df["C"] = df[c_col].astype(float)
    df["B"] = df["B_amb_median"].astype(float)
    df["A"] = 1.0 - df["B"]

    c_min, c_max = _range_min_max(df["C"])
    b_min, b_max = _range_min_max(df["B"])

    if normalize:
        df["C_tilde"], c_diag = _normalize_series(df["C"], larger_is_better=True)
        df["A_tilde"], b_diag = _normalize_series(df["B"], larger_is_better=False)
        x_col, y_col = "C_tilde", "A_tilde"
    else:
        df["C_tilde"] = np.nan
        df["A_tilde"] = np.nan
        c_diag = {"constant": bool(np.isfinite(c_min) and np.isfinite(c_max) and c_min == c_max), "not_identifiable": not np.isfinite(c_min)}
        b_diag = {"constant": bool(np.isfinite(b_min) and np.isfinite(b_max) and b_min == b_max), "not_identifiable": not np.isfinite(b_min)}
        x_col, y_col = "C", "A"

    df["H"] = [_harmonic(x, y, zero_division) for x, y in zip(df[x_col], df[y_col])]
    df["integration_method"] = "harmonic_mean"
    df["normalized"] = bool(normalize)
    df["cov_ratio_empty_base_policy"] = "nan"
    df["C_min"] = c_min
    df["C_max"] = c_max
    df["B_min"] = b_min
    df["B_max"] = b_max
    df["degenerate_constant_coverage"] = bool(c_diag.get("constant", False))
    df["degenerate_constant_ambiguity"] = bool(b_diag.get("constant", False))
    df["degenerate_both_constant"] = bool(c_diag.get("constant", False) and b_diag.get("constant", False))
    df["degenerate_not_identifiable"] = bool(c_diag.get("not_identifiable", False) or b_diag.get("not_identifiable", False))
    return df


def rank_k_values(
    k_metrics: pd.DataFrame,
    *,
    top_n: int,
    tie_breaker: str,
) -> pd.DataFrame:
    if tie_breaker != "smaller_k":
        raise ValueError("Only tie_breaker='smaller_k' is supported.")
    valid = k_metrics[np.isfinite(k_metrics["H"].to_numpy(dtype=float))].copy()
    valid = valid.sort_values(["H", "k"], ascending=[False, True]).reset_index(drop=True)
    valid["rank"] = np.arange(1, len(valid) + 1, dtype=int)
    cutoff = min(int(top_n), len(valid))
    valid["in_K_valid"] = valid["rank"] <= cutoff
    valid["selected"] = valid["rank"] == 1
    valid["ranking_rule"] = "H_desc_then_k_asc"
    return valid


def select_k(
    ranking: pd.DataFrame,
    *,
    top_n: int,
) -> Dict[str, Any]:
    if ranking.empty:
        return {
            "k_star": None,
            "K_valid": [],
            "top_n": int(top_n),
            "ranking_rule": "H_desc_then_k_asc",
            "n_valid": 0,
        }
    first = ranking.iloc[0]
    k_valid = ranking.loc[ranking["in_K_valid"], "k"].astype(int).tolist()
    return {
        "k_star": int(first["k"]),
        "K_valid": [int(k) for k in k_valid],
        "top_n": int(top_n),
        "ranking_rule": "H_desc_then_k_asc",
        "coverage_metric": str(first.get("coverage_metric", "")),
        "coverage_aggregation": str(first.get("coverage_aggregation", "")),
        "ambiguity_metric": str(first.get("ambiguity_metric", "")),
        "score_mode": "normalized_harmonic_mean" if bool(first.get("normalized", False)) else "raw_harmonic_mean",
        "normalized": bool(first.get("normalized", False)),
        "n_valid": int(len(ranking)),
        "C_min": _json_number(first.get("C_min")),
        "C_max": _json_number(first.get("C_max")),
        "B_min": _json_number(first.get("B_min")),
        "B_max": _json_number(first.get("B_max")),
        "degenerate_constant_coverage": bool(first.get("degenerate_constant_coverage", False)),
        "degenerate_constant_ambiguity": bool(first.get("degenerate_constant_ambiguity", False)),
        "degenerate_both_constant": bool(first.get("degenerate_both_constant", False)),
        "degenerate_not_identifiable": bool(first.get("degenerate_not_identifiable", False)),
        "cov_ratio_empty_base_policy": "nan",
        "cov_ratio_primary_aggregation": "median",
    }


def build_td_k_metrics(
    *,
    td_coverage_summary: pd.DataFrame,
    target_ambiguity: pd.DataFrame,
    coverage_metric: str,
    ambiguity_metric: str,
    normalize: bool,
    zero_division: float = 0.0,
    progress=None,
) -> pd.DataFrame:
    if ambiguity_metric != "amb_median":
        raise ValueError("Only ambiguity_metric='amb_median' is supported.")
    c_col = _coverage_col(coverage_metric, td=True)
    if c_col not in td_coverage_summary.columns:
        raise KeyError(f"td_coverage_summary is missing required column: {c_col}")

    amb_cols = ["target_gene", "k", "amb"]
    missing = set(amb_cols) - set(target_ambiguity.columns)
    if missing:
        raise KeyError(f"target_ambiguity is missing required columns: {sorted(missing)}")

    df = pd.merge(
        td_coverage_summary,
        target_ambiguity[amb_cols],
        on=["target_gene", "k"],
        how="left",
        validate="many_to_one",
    )
    df = df.sort_values(["target_gene", "disease_id", "k"]).reset_index(drop=True)
    df["coverage_metric"] = coverage_metric
    df["coverage_aggregation"] = COVERAGE_AGGREGATION[coverage_metric]
    df["ambiguity_metric"] = ambiguity_metric
    df["C_td"] = df[c_col].astype(float)
    df["B_td"] = df["amb"].astype(float)
    df["A_td"] = 1.0 - df["B_td"]
    df["C_td_tilde"] = np.nan
    df["A_td_tilde"] = np.nan
    df["H_td"] = np.nan

    groups = list(df.groupby(["target_gene", "disease_id"], sort=False).groups.items())
    for (_td, idx) in _progress(progress, groups, total=len(groups), desc="td-metrics"):
        sub = df.loc[idx]
        valid_mask = np.isfinite(sub["C_td"].to_numpy(dtype=float)) & np.isfinite(sub["B_td"].to_numpy(dtype=float))
        if not valid_mask.any():
            continue
        valid_index = sub.index[valid_mask]
        if normalize:
            c_norm, _ = _normalize_series(df.loc[valid_index, "C_td"], larger_is_better=True)
            b_norm, _ = _normalize_series(df.loc[valid_index, "B_td"], larger_is_better=False)
            df.loc[valid_index, "C_td_tilde"] = c_norm
            df.loc[valid_index, "A_td_tilde"] = b_norm
            h_vals = [_harmonic(x, y, zero_division) for x, y in zip(c_norm, b_norm)]
        else:
            h_vals = [_harmonic(x, y, zero_division) for x, y in zip(df.loc[valid_index, "C_td"], df.loc[valid_index, "A_td"])]
        df.loc[valid_index, "H_td"] = h_vals

    df["integration_method"] = "harmonic_mean"
    df["normalized"] = bool(normalize)
    df["cov_ratio_empty_base_policy"] = "nan"
    df["td_valid_for_ranking"] = np.isfinite(df["C_td"].to_numpy(dtype=float)) & np.isfinite(df["B_td"].to_numpy(dtype=float)) & np.isfinite(df["H_td"].to_numpy(dtype=float))
    return df


def rank_td_k_values(
    td_k_metrics: pd.DataFrame,
    *,
    top_n: int,
    tie_breaker: str,
    progress=None,
) -> pd.DataFrame:
    if tie_breaker != "smaller_k":
        raise ValueError("Only tie_breaker='smaller_k' is supported.")
    valid = td_k_metrics[td_k_metrics["td_valid_for_ranking"].astype(bool)].copy()
    rows = []
    grouped = valid.groupby(["target_gene", "disease_id"], sort=True)
    for (_td, g) in _progress(progress, grouped, total=grouped.ngroups, desc="td-ranking"):
        g = g.sort_values(["H_td", "k"], ascending=[False, True]).reset_index(drop=True)
        g["td_rank"] = np.arange(1, len(g) + 1, dtype=int)
        cutoff = min(int(top_n), len(g))
        g["in_K_valid_td"] = g["td_rank"] <= cutoff
        g["selected_td"] = g["td_rank"] == 1
        g["ranking_rule"] = "H_td_desc_then_k_asc"
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else valid.copy()


def select_td_k(
    td_ranking: pd.DataFrame,
    *,
    top_n: int,
    all_td_pairs=None,
    progress=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Large DataFrame.attrs cause pandas to deepcopy metadata during groupby/copy
    # operations.  TD pair lists are therefore passed explicitly via
    # all_td_pairs, with this attrs fallback kept only for older callers.
    if td_ranking.attrs:
        td_ranking.attrs = {}

    if all_td_pairs is not None:
        if isinstance(all_td_pairs, pd.DataFrame):
            pair_keys = _pair_keys_from_frame(all_td_pairs)
        else:
            pair_keys = _pair_keys_from_records(all_td_pairs)
    elif not td_ranking.empty:
        pair_keys = _pair_keys_from_frame(td_ranking)
    else:
        pair_keys = []

    if td_ranking.empty:
        ranking_groups: dict[tuple[str, str], pd.DataFrame] = {}
    else:
        ranking = td_ranking.copy()
        ranking["target_gene"] = ranking["target_gene"].astype(str)
        ranking["disease_id"] = ranking["disease_id"].astype(str)
        ranking_groups = {
            (str(t), str(d)): g
            for (t, d), g in ranking.groupby(["target_gene", "disease_id"], sort=False)
        }

    sel_rows = []
    valid_rows = []
    for t, d in _progress(progress, pair_keys, total=len(pair_keys), desc="td-selection"):
        g = ranking_groups.get((t, d), pd.DataFrame()).copy()
        if g.empty:
            sel_rows.append(
                {
                    "target_gene": t,
                    "disease_id": d,
                    "k_star_td": np.nan,
                    "K_valid_td": "",
                    "n_valid_k_td": 0,
                    "top_n": int(top_n),
                    "ranking_rule": "H_td_desc_then_k_asc",
                    "coverage_metric": "",
                    "coverage_aggregation": "",
                    "normalized": np.nan,
                }
            )
            continue
        g = g.sort_values("td_rank")
        top = g.iloc[0]
        k_valid = g.loc[g["in_K_valid_td"], "k"].astype(int).tolist()
        sel_rows.append(
            {
                "target_gene": t,
                "disease_id": d,
                "k_star_td": int(top["k"]),
                "K_valid_td": "|".join(str(int(k)) for k in k_valid),
                "n_valid_k_td": int(len(g)),
                "top_n": int(top_n),
                "ranking_rule": "H_td_desc_then_k_asc",
                "coverage_metric": str(top.get("coverage_metric", "")),
                "coverage_aggregation": str(top.get("coverage_aggregation", "")),
                "normalized": bool(top.get("normalized", False)),
            }
        )
        for _, r in g.loc[g["in_K_valid_td"]].iterrows():
            valid_rows.append(
                {
                    "target_gene": t,
                    "disease_id": d,
                    "k": int(r["k"]),
                    "td_rank": int(r["td_rank"]),
                    "H_td": float(r["H_td"]),
                    "selected_td": bool(r["selected_td"]),
                }
            )
    return pd.DataFrame(sel_rows), pd.DataFrame(valid_rows)
