"""Label-free selection-rule sensitivity for the effective distance k*.

The primary selection rule (coverage-ambiguity harmonic score) lives in
``selection.py`` and is intentionally left untouched: this module only reads the
outputs of a completed Val1 run and re-selects k* under alternative label-free
rules, so that the primary k* is never affected by the sensitivity analysis.

All rules here are label-free: none of them uses therapeutic labels or ranking
performance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd

# Reused so that the sensitivity path normalizes components exactly the way the
# primary harmonic path does.
from .selection import _harmonic, _normalize_series


RULE_HARMONIC_MEAN = "harmonic_mean"
RULE_GEOMETRIC_MEAN = "geometric_mean"
RULE_PARETO_KNEE = "pareto_knee"
RULE_COVERAGE_THRESHOLD = "coverage_threshold"
RULE_AMBIGUITY_CEILING = "ambiguity_ceiling"

SENSITIVITY_RULES = (
    RULE_GEOMETRIC_MEAN,
    RULE_PARETO_KNEE,
    RULE_COVERAGE_THRESHOLD,
    RULE_AMBIGUITY_CEILING,
)

KNEE_METHODS = ("max_chord_distance", "menger_curvature")
THRESHOLD_SCALES = ("raw", "normalized")

DEFAULT_TAUS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]
DEFAULT_BETAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
DEFAULT_PRIMARY_TAU = 0.20
DEFAULT_PRIMARY_BETA = 0.30

# Pooled C(k)/B(k) recomputed from td_k_metrics must reproduce k_metrics; the
# tolerance only absorbs float64 median noise.
POOLED_CONSISTENCY_TOL = 1e-9

# Knee scores are computed on normalized [0, 1] components, so anything below
# this is float noise.  Snapping it to zero keeps collinear points a genuine tie
# that the smaller-k tie-breaker resolves, instead of letting 1e-17 decide k*.
KNEE_NUMERICAL_EPS = 1e-12


def _progress(progress, iterable, **kwargs):
    if progress is None:
        return iterable
    if progress is False:
        return iterable
    return progress(iterable, **kwargs)


def _json_number(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def _json_int(x: Any) -> int | None:
    v = _json_number(x)
    return None if v is None else int(v)


def _nanmedian(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.full(matrix.shape[1] if matrix.ndim == 2 else 0, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(matrix, axis=0)


# ---------------------------------------------------------------------------
# score construction
# ---------------------------------------------------------------------------


def compute_components(
    C: Sequence[float],
    B: Sequence[float],
    *,
    normalize: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (C_tilde, A_tilde) as used by the primary harmonic score."""
    c = pd.Series(np.asarray(C, dtype=float))
    b = pd.Series(np.asarray(B, dtype=float))
    if normalize:
        c_tilde, _ = _normalize_series(c, larger_is_better=True)
        a_tilde, _ = _normalize_series(b, larger_is_better=False)
        return c_tilde.to_numpy(dtype=float), a_tilde.to_numpy(dtype=float)
    return c.to_numpy(dtype=float), (1.0 - b).to_numpy(dtype=float)


def _geometric(x: float, y: float) -> float:
    if not (np.isfinite(x) and np.isfinite(y)):
        return float("nan")
    if x < 0.0 or y < 0.0:
        return float("nan")
    return float(np.sqrt(x * y))


def _sorted_front_indices(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return idx
    order = np.lexsort((y[idx], x[idx]))
    return idx[order]


def pareto_front_mask(x: Sequence[float], y: Sequence[float]) -> np.ndarray:
    """Non-dominated points when maximizing both coordinates."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.zeros(xa.size, dtype=bool)
    finite = np.flatnonzero(np.isfinite(xa) & np.isfinite(ya))
    for i in finite:
        dominated = False
        for j in finite:
            if i == j:
                continue
            if (xa[j] >= xa[i]) and (ya[j] >= ya[i]) and ((xa[j] > xa[i]) or (ya[j] > ya[i])):
                dominated = True
                break
        mask[i] = not dominated
    return mask


def knee_chord_distance(x: Sequence[float], y: Sequence[float], mask: np.ndarray) -> np.ndarray:
    """Perpendicular distance from the chord joining the two front endpoints.

    The endpoints themselves are left undefined: they always sit on the chord,
    and they are the degenerate corners of the trade-off (no coverage, or maximal
    ambiguity), so they must not be selectable as a knee.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    out = np.full(xa.size, np.nan)
    idx = _sorted_front_indices(xa, ya, mask)
    if idx.size < 3:
        return out
    xs = xa[idx]
    ys = ya[idx]
    dx = float(xs[-1] - xs[0])
    dy = float(ys[-1] - ys[0])
    norm = float(np.hypot(dx, dy))
    if norm <= 0.0:
        return out
    ux, uy = dx / norm, dy / norm
    out[idx[1:-1]] = np.abs((xs[1:-1] - xs[0]) * uy - (ys[1:-1] - ys[0]) * ux)
    out[np.abs(out) < KNEE_NUMERICAL_EPS] = 0.0
    return out


def menger_curvature(x: Sequence[float], y: Sequence[float], mask: np.ndarray) -> np.ndarray:
    """Discrete curvature of consecutive Pareto-front points."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    out = np.full(xa.size, np.nan)
    idx = _sorted_front_indices(xa, ya, mask)
    if idx.size < 3:
        return out
    xs = xa[idx]
    ys = ya[idx]
    for i in range(1, idx.size - 1):
        ax, ay = xs[i - 1], ys[i - 1]
        bx, by = xs[i], ys[i]
        cx, cy = xs[i + 1], ys[i + 1]
        area2 = abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
        denom = np.hypot(bx - ax, by - ay) * np.hypot(cx - bx, cy - by) * np.hypot(cx - ax, cy - ay)
        out[idx[i]] = 0.0 if denom <= 0.0 else float(2.0 * area2 / denom)
    out[np.abs(out) < KNEE_NUMERICAL_EPS] = 0.0
    return out


def build_rule_metrics(
    ks: Sequence[int],
    C: Sequence[float],
    B: Sequence[float],
    *,
    normalize: bool,
    zero_division: float = 0.0,
    knee_method: str = "max_chord_distance",
) -> pd.DataFrame:
    """Per-k score table shared by every selection rule."""
    if knee_method not in KNEE_METHODS:
        raise ValueError(f"knee_method must be one of {list(KNEE_METHODS)}. got={knee_method!r}")

    k_arr = np.asarray(ks, dtype=int)
    c_arr = np.asarray(C, dtype=float)
    b_arr = np.asarray(B, dtype=float)
    c_tilde, a_tilde = compute_components(c_arr, b_arr, normalize=normalize)

    df = pd.DataFrame(
        {
            "k": k_arr,
            "C": c_arr,
            "B": b_arr,
            "A": 1.0 - b_arr,
            "C_tilde": c_tilde,
            "A_tilde": a_tilde,
        }
    )
    df[f"score_{RULE_HARMONIC_MEAN}"] = [_harmonic(x, y, zero_division) for x, y in zip(c_tilde, a_tilde)]
    df[f"score_{RULE_GEOMETRIC_MEAN}"] = [_geometric(x, y) for x, y in zip(c_tilde, a_tilde)]

    front = pareto_front_mask(c_tilde, a_tilde)
    df["on_pareto_front"] = front
    df["knee_dist"] = knee_chord_distance(c_tilde, a_tilde, front)
    df["menger_curvature"] = menger_curvature(c_tilde, a_tilde, front)
    df[f"score_{RULE_PARETO_KNEE}"] = df["knee_dist"] if knee_method == "max_chord_distance" else df["menger_curvature"]

    df["knee_method"] = knee_method
    df["normalized"] = bool(normalize)
    return df.sort_values("k").reset_index(drop=True)


# ---------------------------------------------------------------------------
# selection primitives
# ---------------------------------------------------------------------------


def _check_tie_breaker(tie_breaker: str) -> None:
    if tie_breaker != "smaller_k":
        raise ValueError("Only tie_breaker='smaller_k' is supported.")


def rank_by_score(
    ks: Sequence[int],
    scores: Sequence[float],
    *,
    rule: str,
    top_n: int,
    tie_breaker: str,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    _check_tie_breaker(tie_breaker)
    frame = pd.DataFrame({"k": np.asarray(ks, dtype=int), "score": np.asarray(scores, dtype=float)})
    valid = frame[np.isfinite(frame["score"].to_numpy(dtype=float))].copy()
    valid = valid.sort_values(["score", "k"], ascending=[not higher_is_better, True]).reset_index(drop=True)
    valid["rank"] = np.arange(1, len(valid) + 1, dtype=int)
    cutoff = min(int(top_n), len(valid))
    valid["in_K_valid"] = valid["rank"] <= cutoff
    valid["selected"] = valid["rank"] == 1
    valid["rule"] = rule
    valid["ranking_rule"] = f"{'score_desc' if higher_is_better else 'score_asc'}_then_k_asc"
    return valid[["rule", "k", "score", "rank", "in_K_valid", "selected", "ranking_rule"]]


def selection_from_ranking(ranking: pd.DataFrame, *, rule: str, top_n: int, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "rule": rule,
        "k_star": None,
        "K_valid": [],
        "top_n": int(top_n),
        "ranking_rule": "",
        "n_valid": 0,
        "score_at_k_star": None,
        "label_dependent": False,
    }
    if not ranking.empty:
        first = ranking.iloc[0]
        out.update(
            {
                "k_star": int(first["k"]),
                "K_valid": [int(k) for k in ranking.loc[ranking["in_K_valid"], "k"].astype(int).tolist()],
                "ranking_rule": str(first.get("ranking_rule", "")),
                "n_valid": int(len(ranking)),
                "score_at_k_star": _json_number(first.get("score")),
            }
        )
    if extra:
        out.update(extra)
    return out


def rank_by_constrained_objective(
    ks: Sequence[int],
    feasible: Sequence[bool],
    objective: Sequence[float],
    *,
    rule: str,
    top_n: int,
    tie_breaker: str,
    higher_is_better: bool,
) -> pd.DataFrame:
    """Rank the feasible k values by an objective (threshold-style rules).

    The general argmin/argmax form is kept even though C(k) and B(k) are
    monotone on SIGNOR, so the rule stays correct on non-monotone k grids.
    """
    _check_tie_breaker(tie_breaker)
    mask = np.asarray(feasible, dtype=bool)
    obj = np.asarray(objective, dtype=float)
    keep = mask & np.isfinite(obj)
    return rank_by_score(
        np.asarray(ks, dtype=int)[keep],
        obj[keep],
        rule=rule,
        top_n=top_n,
        tie_breaker=tie_breaker,
        higher_is_better=higher_is_better,
    )


def _threshold_vectors(rule_metrics: pd.DataFrame, scale: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (coverage, ambiguity) vectors on the requested scale."""
    if scale not in THRESHOLD_SCALES:
        raise ValueError(f"threshold scale must be one of {list(THRESHOLD_SCALES)}. got={scale!r}")
    if scale == "raw":
        return (
            rule_metrics["C"].to_numpy(dtype=float),
            rule_metrics["B"].to_numpy(dtype=float),
        )
    return (
        rule_metrics["C_tilde"].to_numpy(dtype=float),
        1.0 - rule_metrics["A_tilde"].to_numpy(dtype=float),
    )


def select_by_coverage_threshold(
    rule_metrics: pd.DataFrame,
    *,
    tau: float,
    scale: str,
    top_n: int,
    tie_breaker: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Coverage threshold + ambiguity minimum: argmin B(k) over {k : C(k) >= tau}."""
    cov, amb = _threshold_vectors(rule_metrics, scale)
    ks = rule_metrics["k"].to_numpy(dtype=int)
    feasible = np.isfinite(cov) & (cov >= float(tau))
    ranking = rank_by_constrained_objective(
        ks,
        feasible,
        amb,
        rule=RULE_COVERAGE_THRESHOLD,
        top_n=top_n,
        tie_breaker=tie_breaker,
        higher_is_better=False,
    )
    selection = selection_from_ranking(
        ranking,
        rule=RULE_COVERAGE_THRESHOLD,
        top_n=top_n,
        extra={"tau": float(tau), "scale": str(scale), "n_feasible": int(feasible.sum())},
    )
    return ranking, selection


def select_by_ambiguity_ceiling(
    rule_metrics: pd.DataFrame,
    *,
    beta: float,
    scale: str,
    top_n: int,
    tie_breaker: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Ambiguity ceiling + coverage maximum: argmax C(k) over {k : B(k) <= beta}."""
    cov, amb = _threshold_vectors(rule_metrics, scale)
    ks = rule_metrics["k"].to_numpy(dtype=int)
    feasible = np.isfinite(amb) & (amb <= float(beta))
    ranking = rank_by_constrained_objective(
        ks,
        feasible,
        cov,
        rule=RULE_AMBIGUITY_CEILING,
        top_n=top_n,
        tie_breaker=tie_breaker,
        higher_is_better=True,
    )
    selection = selection_from_ranking(
        ranking,
        rule=RULE_AMBIGUITY_CEILING,
        top_n=top_n,
        extra={"beta": float(beta), "scale": str(scale), "n_feasible": int(feasible.sum())},
    )
    return ranking, selection


def sweep_coverage_threshold(
    rule_metrics: pd.DataFrame,
    *,
    taus: Sequence[float],
    scale: str,
    top_n: int,
    tie_breaker: str,
) -> pd.DataFrame:
    rows = []
    for tau in taus:
        _, sel = select_by_coverage_threshold(
            rule_metrics, tau=float(tau), scale=scale, top_n=top_n, tie_breaker=tie_breaker
        )
        k_star = sel.get("k_star")
        row = rule_metrics.loc[rule_metrics["k"] == k_star] if k_star is not None else rule_metrics.iloc[0:0]
        rows.append(
            {
                "tau": float(tau),
                "scale": str(scale),
                "n_feasible": int(sel.get("n_feasible", 0)),
                "k_star": _json_int(k_star),
                "K_valid": "|".join(str(k) for k in sel.get("K_valid", [])),
                "C_at_k_star": _json_number(row["C"].iloc[0]) if len(row) else None,
                "B_at_k_star": _json_number(row["B"].iloc[0]) if len(row) else None,
            }
        )
    return pd.DataFrame(rows)


def sweep_ambiguity_ceiling(
    rule_metrics: pd.DataFrame,
    *,
    betas: Sequence[float],
    scale: str,
    top_n: int,
    tie_breaker: str,
) -> pd.DataFrame:
    rows = []
    for beta in betas:
        _, sel = select_by_ambiguity_ceiling(
            rule_metrics, beta=float(beta), scale=scale, top_n=top_n, tie_breaker=tie_breaker
        )
        k_star = sel.get("k_star")
        row = rule_metrics.loc[rule_metrics["k"] == k_star] if k_star is not None else rule_metrics.iloc[0:0]
        rows.append(
            {
                "beta": float(beta),
                "scale": str(scale),
                "n_feasible": int(sel.get("n_feasible", 0)),
                "k_star": _json_int(k_star),
                "K_valid": "|".join(str(k) for k in sel.get("K_valid", [])),
                "C_at_k_star": _json_number(row["C"].iloc[0]) if len(row) else None,
                "B_at_k_star": _json_number(row["B"].iloc[0]) if len(row) else None,
            }
        )
    return pd.DataFrame(rows)


def threshold_stability_interval(
    sweep: pd.DataFrame,
    *,
    param_col: str,
    k_star_primary: int | None,
) -> Dict[str, Any]:
    """Longest contiguous run of threshold values that reproduce the primary k*."""
    out: Dict[str, Any] = {
        "param": param_col,
        "k_star_primary": _json_int(k_star_primary),
        "n_values": int(len(sweep)),
        "n_matching": 0,
        "interval_min": None,
        "interval_max": None,
        "interval_width": None,
    }
    if sweep.empty or k_star_primary is None:
        return out
    work = sweep.sort_values(param_col).reset_index(drop=True)
    match = work["k_star"].apply(lambda v: _json_int(v) == int(k_star_primary)).to_numpy(dtype=bool)
    out["n_matching"] = int(match.sum())
    best_start = best_len = cur_start = cur_len = -1
    for i, flag in enumerate(match):
        if flag:
            if cur_len <= 0:
                cur_start, cur_len = i, 1
            else:
                cur_len += 1
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
        else:
            cur_len = 0
    if best_len > 0:
        lo = float(work[param_col].iloc[best_start])
        hi = float(work[param_col].iloc[best_start + best_len - 1])
        out.update({"interval_min": lo, "interval_max": hi, "interval_width": float(hi - lo)})
    return out


# ---------------------------------------------------------------------------
# pooled recomputation from target-disease level metrics
# ---------------------------------------------------------------------------


def prepare_pooled_inputs(td_k_metrics: pd.DataFrame, *, coverage_metric: str) -> Dict[str, Any]:
    """Prepare matrices that reproduce pooled C(k)/B(k) from td_k_metrics.

    Only ``cov_ratio`` is supported: pooled C(k) is then the across-pair median
    of the per-pair medians, which is exactly ``C_td``.  Pooled ``cov_nz`` is a
    mean over (disease, target, cell) rows and cannot be recovered from
    pair-level means without the per-pair row counts.
    """
    if coverage_metric != "cov_ratio":
        raise NotImplementedError(
            "Selection-rule resampling currently supports metrics.coverage.primary='cov_ratio' only. "
            f"got={coverage_metric!r}"
        )
    required = {"target_gene", "disease_id", "k", "C_td", "B_td"}
    missing = required - set(td_k_metrics.columns)
    if missing:
        raise KeyError(f"td_k_metrics is missing required columns: {sorted(missing)}")

    work = td_k_metrics[["target_gene", "disease_id", "k", "C_td", "B_td"]].copy()
    work["target_gene"] = work["target_gene"].astype(str)
    work["disease_id"] = work["disease_id"].astype(str)
    work["k"] = work["k"].astype(int)

    c_pivot = work.pivot_table(index=["target_gene", "disease_id"], columns="k", values="C_td", dropna=False)
    b_pivot = (
        work[["target_gene", "k", "B_td"]]
        .drop_duplicates(["target_gene", "k"])
        .pivot(index="target_gene", columns="k", values="B_td")
    )
    ks = np.asarray(sorted(int(k) for k in c_pivot.columns), dtype=int)
    b_pivot = b_pivot.reindex(columns=ks)
    c_pivot = c_pivot.reindex(columns=ks)

    targets = np.asarray(b_pivot.index.astype(str))
    target_pos = {t: i for i, t in enumerate(targets)}
    pair_targets = np.asarray(c_pivot.index.get_level_values(0).astype(str))
    pair_diseases = np.asarray(c_pivot.index.get_level_values(1).astype(str))
    diseases = np.asarray(sorted(set(pair_diseases.tolist())))
    disease_pos = {d: i for i, d in enumerate(diseases)}

    pair_target_code = np.asarray([target_pos[t] for t in pair_targets], dtype=int)
    pair_disease_code = np.asarray([disease_pos[d] for d in pair_diseases], dtype=int)
    pair_rows_by_target = [np.flatnonzero(pair_target_code == i) for i in range(targets.size)]

    return {
        "ks": ks,
        "C_matrix": c_pivot.to_numpy(dtype=float),
        "B_matrix": b_pivot.to_numpy(dtype=float),
        "targets": targets,
        "diseases": diseases,
        "pair_target_code": pair_target_code,
        "pair_disease_code": pair_disease_code,
        "pair_rows_by_target": pair_rows_by_target,
        "coverage_metric": coverage_metric,
    }


def pooled_with_dropped(
    prepared: Dict[str, Any],
    *,
    drop_target: int | None = None,
    drop_disease: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pooled (ks, C, B) after dropping one target and/or one disease."""
    c_rows = np.ones(prepared["C_matrix"].shape[0], dtype=bool)
    b_rows = np.ones(prepared["B_matrix"].shape[0], dtype=bool)
    if drop_target is not None:
        c_rows &= prepared["pair_target_code"] != int(drop_target)
        b_rows[int(drop_target)] = False
    if drop_disease is not None:
        c_rows &= prepared["pair_disease_code"] != int(drop_disease)
    return (
        prepared["ks"],
        _nanmedian(prepared["C_matrix"][c_rows]),
        _nanmedian(prepared["B_matrix"][b_rows]),
    )


def pooled_with_target_sample(prepared: Dict[str, Any], target_sample: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pooled (ks, C, B) for a target resample drawn with replacement."""
    sample = np.asarray(target_sample, dtype=int)
    if sample.size == 0:
        n_k = prepared["ks"].size
        return prepared["ks"], np.full(n_k, np.nan), np.full(n_k, np.nan)
    rows_by_target = prepared["pair_rows_by_target"]
    c_rows = np.concatenate([rows_by_target[i] for i in sample]) if sample.size else np.asarray([], dtype=int)
    return (
        prepared["ks"],
        _nanmedian(prepared["C_matrix"][c_rows]),
        _nanmedian(prepared["B_matrix"][sample]),
    )


def check_pooled_consistency(
    prepared: Dict[str, Any],
    k_metrics: pd.DataFrame,
    *,
    tol: float = POOLED_CONSISTENCY_TOL,
) -> Dict[str, Any]:
    """Guard: pooled C/B rebuilt from td_k_metrics must match k_metrics."""
    ks, c_vec, b_vec = pooled_with_dropped(prepared)
    ref = k_metrics.set_index("k")
    missing = [int(k) for k in ks if int(k) not in ref.index]
    if missing:
        raise ValueError(f"k_metrics is missing distances present in td_k_metrics: {missing}")
    c_ref = ref.loc[[int(k) for k in ks], "C"].to_numpy(dtype=float)
    b_ref = ref.loc[[int(k) for k in ks], "B"].to_numpy(dtype=float)
    d_c = float(np.nanmax(np.abs(c_vec - c_ref))) if ks.size else 0.0
    d_b = float(np.nanmax(np.abs(b_vec - b_ref))) if ks.size else 0.0
    passed = bool(d_c <= tol and d_b <= tol)
    if not passed:
        raise ValueError(
            "Pooled C(k)/B(k) recomputed from td_k_metrics do not match k_metrics "
            f"(max|dC|={d_c:.3g}, max|dB|={d_b:.3g}, tol={tol:.3g}). "
            "Selection-rule resampling assumes the two are equivalent."
        )
    return {"max_abs_diff_C": d_c, "max_abs_diff_B": d_b, "tol": float(tol), "passed": passed, "n_k": int(ks.size)}


# ---------------------------------------------------------------------------
# rule evaluation over one pooled (C, B) grid
# ---------------------------------------------------------------------------


def select_all_rules(
    ks: Sequence[int],
    C: Sequence[float],
    B: Sequence[float],
    *,
    rules: Sequence[str],
    normalize: bool,
    zero_division: float,
    top_n: int,
    tie_breaker: str,
    knee_method: str,
    tau: float,
    beta: float,
    threshold_scale: str,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]], pd.DataFrame]:
    """Apply the primary rule and every configured sensitivity rule."""
    rule_metrics = build_rule_metrics(
        ks, C, B, normalize=normalize, zero_division=zero_division, knee_method=knee_method
    )
    rankings: List[pd.DataFrame] = []
    selections: Dict[str, Dict[str, Any]] = {}

    for rule in (RULE_HARMONIC_MEAN, *[r for r in rules if r in (RULE_GEOMETRIC_MEAN, RULE_PARETO_KNEE)]):
        ranking = rank_by_score(
            rule_metrics["k"].to_numpy(dtype=int),
            rule_metrics[f"score_{rule}"].to_numpy(dtype=float),
            rule=rule,
            top_n=top_n,
            tie_breaker=tie_breaker,
        )
        if rule == RULE_PARETO_KNEE:
            scores = rule_metrics[f"score_{rule}"].to_numpy(dtype=float)
            finite = scores[np.isfinite(scores)]
            extra = {
                "knee_method": knee_method,
                "n_pareto_front": int(rule_metrics["on_pareto_front"].sum()),
                "degenerate_no_knee": bool(finite.size == 0 or float(np.nanmax(finite)) <= 0.0),
            }
        else:
            extra = None
        rankings.append(ranking)
        selections[rule] = selection_from_ranking(ranking, rule=rule, top_n=top_n, extra=extra)

    if RULE_COVERAGE_THRESHOLD in rules:
        ranking, selection = select_by_coverage_threshold(
            rule_metrics, tau=tau, scale=threshold_scale, top_n=top_n, tie_breaker=tie_breaker
        )
        rankings.append(ranking)
        selections[RULE_COVERAGE_THRESHOLD] = selection

    if RULE_AMBIGUITY_CEILING in rules:
        ranking, selection = select_by_ambiguity_ceiling(
            rule_metrics, beta=beta, scale=threshold_scale, top_n=top_n, tie_breaker=tie_breaker
        )
        rankings.append(ranking)
        selections[RULE_AMBIGUITY_CEILING] = selection

    ranking_frame = pd.concat(rankings, ignore_index=True) if rankings else pd.DataFrame()
    return rule_metrics, selections, ranking_frame


# ---------------------------------------------------------------------------
# resampling
# ---------------------------------------------------------------------------


def run_resampling_selection(
    prepared: Dict[str, Any],
    *,
    rules: Sequence[str],
    normalize: bool,
    zero_division: float,
    top_n: int,
    tie_breaker: str,
    knee_method: str,
    tau: float,
    beta: float,
    threshold_scale: str,
    leave_one_target_out: bool,
    leave_one_disease_out: bool,
    n_bootstrap: int,
    seed: int,
    progress=None,
) -> pd.DataFrame:
    """Re-select k* on leave-one-out folds and optional target bootstrap replicates."""
    kwargs = dict(
        rules=rules,
        normalize=normalize,
        zero_division=zero_division,
        top_n=top_n,
        tie_breaker=tie_breaker,
        knee_method=knee_method,
        tau=tau,
        beta=beta,
        threshold_scale=threshold_scale,
    )
    rows: List[Dict[str, Any]] = []

    def _record(fold_type: str, held_out: str, replicate: int | None, ks, c_vec, b_vec) -> None:
        _, selections, _ = select_all_rules(ks, c_vec, b_vec, **kwargs)
        for rule, sel in selections.items():
            rows.append(
                {
                    "fold_type": fold_type,
                    "held_out_id": held_out,
                    "replicate": replicate,
                    "rule": rule,
                    "k_star": _json_int(sel.get("k_star")),
                    "score_at_k_star": _json_number(sel.get("score_at_k_star")),
                }
            )

    if leave_one_target_out:
        targets = prepared["targets"]
        for i in _progress(progress, range(targets.size), total=int(targets.size), desc="loto"):
            ks, c_vec, b_vec = pooled_with_dropped(prepared, drop_target=int(i))
            _record("leave_one_target_out", str(targets[i]), None, ks, c_vec, b_vec)

    if leave_one_disease_out:
        diseases = prepared["diseases"]
        for i in _progress(progress, range(diseases.size), total=int(diseases.size), desc="lodo"):
            ks, c_vec, b_vec = pooled_with_dropped(prepared, drop_disease=int(i))
            _record("leave_one_disease_out", str(diseases[i]), None, ks, c_vec, b_vec)

    if int(n_bootstrap) > 0:
        rng = np.random.default_rng(int(seed))
        n_targets = int(prepared["targets"].size)
        for b in _progress(progress, range(int(n_bootstrap)), total=int(n_bootstrap), desc="bootstrap"):
            sample = rng.integers(0, n_targets, size=n_targets)
            ks, c_vec, b_vec = pooled_with_target_sample(prepared, sample)
            _record("bootstrap_targets", "", int(b), ks, c_vec, b_vec)

    return pd.DataFrame(
        rows,
        columns=["fold_type", "held_out_id", "replicate", "rule", "k_star", "score_at_k_star"],
    )


def summarize_resampling(resampling: pd.DataFrame, *, k_star_primary: int | None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Per (fold_type, rule) selection frequency of each k."""
    if resampling.empty:
        return pd.DataFrame(columns=["fold_type", "rule", "k", "n", "n_folds", "frequency"]), {"n_records": 0}

    rows = []
    for (fold_type, rule), g in resampling.groupby(["fold_type", "rule"], sort=True):
        n_folds = int(len(g))
        counts = g["k_star"].dropna().astype(int).value_counts().sort_index()
        for k, n in counts.items():
            rows.append(
                {
                    "fold_type": str(fold_type),
                    "rule": str(rule),
                    "k": int(k),
                    "n": int(n),
                    "n_folds": n_folds,
                    "frequency": float(n / n_folds) if n_folds else np.nan,
                }
            )
    frequency = pd.DataFrame(rows)

    summary: Dict[str, Any] = {"n_records": int(len(resampling)), "k_star_primary": _json_int(k_star_primary), "folds": []}
    for (fold_type, rule), g in resampling.groupby(["fold_type", "rule"], sort=True):
        ks = g["k_star"].dropna().astype(int)
        n_folds = int(len(g))
        matched = int((ks == int(k_star_primary)).sum()) if k_star_primary is not None else 0
        summary["folds"].append(
            {
                "fold_type": str(fold_type),
                "rule": str(rule),
                "n_folds": n_folds,
                "n_undefined": int(n_folds - len(ks)),
                "k_star_modal": _json_int(ks.mode().iloc[0]) if len(ks) else None,
                "k_star_primary_frequency": float(matched / n_folds) if n_folds else None,
                "k_star_min": _json_int(ks.min()) if len(ks) else None,
                "k_star_max": _json_int(ks.max()) if len(ks) else None,
            }
        )
    return frequency, summary


# ---------------------------------------------------------------------------
# agreement table and orchestration
# ---------------------------------------------------------------------------


def build_rule_agreement(
    selections: Dict[str, Dict[str, Any]],
    *,
    primary_selection: Dict[str, Any],
    resampling_summary: Dict[str, Any] | None = None,
    stability: Dict[str, Dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """One row per label-free selection rule, compared against the primary k*."""
    k_star_primary = _json_int(primary_selection.get("k_star"))
    rows: List[Dict[str, Any]] = []

    ordered = [RULE_HARMONIC_MEAN] + [r for r in SENSITIVITY_RULES if r in selections]
    for rule in ordered:
        sel = selections.get(rule)
        if sel is None:
            continue
        note_parts = []
        if rule == RULE_PARETO_KNEE and sel.get("knee_method"):
            note_parts.append(f"knee_method={sel['knee_method']}")
        if rule == RULE_COVERAGE_THRESHOLD:
            note_parts.append(f"tau={sel.get('tau')}")
            note_parts.append("objective=min B")
            st = (stability or {}).get("coverage_threshold")
            if st and st.get("interval_min") is not None:
                note_parts.append(f"stable tau in [{st['interval_min']:.3g}, {st['interval_max']:.3g}]")
        if rule == RULE_AMBIGUITY_CEILING:
            note_parts.append(f"beta={sel.get('beta')}")
            note_parts.append("objective=max C")
            st = (stability or {}).get("ambiguity_ceiling")
            if st and st.get("interval_min") is not None:
                note_parts.append(f"stable beta in [{st['interval_min']:.3g}, {st['interval_max']:.3g}]")
        rows.append(
            {
                "rule": f"{rule}_primary" if rule == RULE_HARMONIC_MEAN else rule,
                "k_star": _json_int(sel.get("k_star")),
                "K_valid": "|".join(str(k) for k in sel.get("K_valid", [])),
                "score_at_k_star": _json_number(sel.get("score_at_k_star")),
                "agrees_with_primary": bool(_json_int(sel.get("k_star")) == k_star_primary) if k_star_primary is not None else False,
                "label_dependent": False,
                "notes": "; ".join(note_parts),
            }
        )

    for fold in (resampling_summary or {}).get("folds", []):
        if fold.get("rule") != RULE_HARMONIC_MEAN:
            continue
        freq = fold.get("k_star_primary_frequency")
        rows.append(
            {
                "rule": f"{fold['fold_type']}_modal",
                "k_star": fold.get("k_star_modal"),
                "K_valid": "",
                "score_at_k_star": None,
                "agrees_with_primary": bool(fold.get("k_star_modal") == k_star_primary) if k_star_primary is not None else False,
                "label_dependent": False,
                "notes": f"n_folds={fold.get('n_folds')}; primary-k frequency={freq:.4g}" if freq is not None else f"n_folds={fold.get('n_folds')}",
            }
        )
    return pd.DataFrame(rows)


def run_selection_sensitivity(
    *,
    k_metrics: pd.DataFrame,
    td_k_metrics: pd.DataFrame,
    primary_selection: Dict[str, Any],
    settings: Dict[str, Any],
    coverage_metric: str,
    normalize: bool,
    zero_division: float,
    top_n: int,
    tie_breaker: str,
    seed: int = 0,
    progress=None,
) -> Dict[str, Any]:
    """Run every configured label-free selection rule on a completed Val1 run."""
    required = {"k", "C", "B"}
    missing = required - set(k_metrics.columns)
    if missing:
        raise KeyError(f"k_metrics is missing required columns: {sorted(missing)}")

    rules = list(settings.get("rules", list(SENSITIVITY_RULES)))
    cov_cfg = dict(settings.get("coverage_threshold") or {})
    amb_cfg = dict(settings.get("ambiguity_ceiling") or {})
    pareto_cfg = dict(settings.get("pareto") or {})
    resample_cfg = dict(settings.get("resampling") or {})

    threshold_scale = str(cov_cfg.get("scale", "raw"))
    taus = [float(t) for t in cov_cfg.get("taus", DEFAULT_TAUS)]
    tau = float(cov_cfg.get("primary_tau", DEFAULT_PRIMARY_TAU))
    betas = [float(b) for b in amb_cfg.get("betas", DEFAULT_BETAS)]
    beta = float(amb_cfg.get("primary_beta", DEFAULT_PRIMARY_BETA))
    knee_method = str(pareto_cfg.get("knee_method", "max_chord_distance"))

    base = k_metrics.sort_values("k").reset_index(drop=True)
    rule_metrics, selections, ranking = select_all_rules(
        base["k"].to_numpy(dtype=int),
        base["C"].to_numpy(dtype=float),
        base["B"].to_numpy(dtype=float),
        rules=rules,
        normalize=normalize,
        zero_division=zero_division,
        top_n=top_n,
        tie_breaker=tie_breaker,
        knee_method=knee_method,
        tau=tau,
        beta=beta,
        threshold_scale=threshold_scale,
    )

    # The recomputed harmonic score must reproduce the primary k_metrics column.
    harmonic_check: Dict[str, Any] = {"checked": False}
    if "H" in base.columns:
        diff = np.abs(
            rule_metrics[f"score_{RULE_HARMONIC_MEAN}"].to_numpy(dtype=float) - base["H"].to_numpy(dtype=float)
        )
        max_diff = float(np.nanmax(diff)) if diff.size else 0.0
        harmonic_check = {"checked": True, "max_abs_diff_H": max_diff, "tol": POOLED_CONSISTENCY_TOL}
        if max_diff > POOLED_CONSISTENCY_TOL:
            raise ValueError(
                f"Recomputed harmonic score does not match k_metrics.H (max|dH|={max_diff:.3g}). "
                "The sensitivity path must reproduce the primary score exactly."
            )

    k_star_primary = _json_int(primary_selection.get("k_star"))
    if k_star_primary is None:
        k_star_primary = _json_int(selections[RULE_HARMONIC_MEAN].get("k_star"))

    tau_sweep = (
        sweep_coverage_threshold(rule_metrics, taus=taus, scale=threshold_scale, top_n=top_n, tie_breaker=tie_breaker)
        if RULE_COVERAGE_THRESHOLD in rules
        else pd.DataFrame()
    )
    beta_sweep = (
        sweep_ambiguity_ceiling(rule_metrics, betas=betas, scale=threshold_scale, top_n=top_n, tie_breaker=tie_breaker)
        if RULE_AMBIGUITY_CEILING in rules
        else pd.DataFrame()
    )
    stability = {
        "coverage_threshold": threshold_stability_interval(tau_sweep, param_col="tau", k_star_primary=k_star_primary),
        "ambiguity_ceiling": threshold_stability_interval(beta_sweep, param_col="beta", k_star_primary=k_star_primary),
    }

    pooled_consistency: Dict[str, Any] = {"checked": False}
    resampling = pd.DataFrame()
    frequency = pd.DataFrame()
    resampling_summary: Dict[str, Any] = {"n_records": 0}

    want_resampling = bool(
        resample_cfg.get("leave_one_target_out", False)
        or resample_cfg.get("leave_one_disease_out", False)
        or int(resample_cfg.get("bootstrap_targets", 0)) > 0
    )
    if want_resampling and not td_k_metrics.empty:
        prepared = prepare_pooled_inputs(td_k_metrics, coverage_metric=coverage_metric)
        pooled_consistency = check_pooled_consistency(prepared, base)
        pooled_consistency["checked"] = True
        resampling = run_resampling_selection(
            prepared,
            rules=rules,
            normalize=normalize,
            zero_division=zero_division,
            top_n=top_n,
            tie_breaker=tie_breaker,
            knee_method=knee_method,
            tau=tau,
            beta=beta,
            threshold_scale=threshold_scale,
            leave_one_target_out=bool(resample_cfg.get("leave_one_target_out", False)),
            leave_one_disease_out=bool(resample_cfg.get("leave_one_disease_out", False)),
            n_bootstrap=int(resample_cfg.get("bootstrap_targets", 0)),
            seed=int(seed),
            progress=progress,
        )
        frequency, resampling_summary = summarize_resampling(resampling, k_star_primary=k_star_primary)
        resampling_summary["n_targets"] = int(prepared["targets"].size)
        resampling_summary["n_diseases"] = int(prepared["diseases"].size)

    agreement = build_rule_agreement(
        selections,
        primary_selection={"k_star": k_star_primary},
        resampling_summary=resampling_summary,
        stability=stability,
    )

    rule_selection = {
        "k_star_primary": k_star_primary,
        "rules": rules,
        "selections": selections,
        "threshold_stability": stability,
        "harmonic_consistency": harmonic_check,
        "pooled_consistency": pooled_consistency,
        "settings": {
            "coverage_metric": coverage_metric,
            "normalize": bool(normalize),
            "zero_division": float(zero_division),
            "top_n": int(top_n),
            "tie_breaker": str(tie_breaker),
            "knee_method": knee_method,
            "threshold_scale": threshold_scale,
            "primary_tau": tau,
            "primary_beta": beta,
            "taus": taus,
            "betas": betas,
            "resampling": {
                "leave_one_target_out": bool(resample_cfg.get("leave_one_target_out", False)),
                "leave_one_disease_out": bool(resample_cfg.get("leave_one_disease_out", False)),
                "bootstrap_targets": int(resample_cfg.get("bootstrap_targets", 0)),
                "seed": int(seed),
            },
        },
        "label_free": True,
    }

    tables = {
        "rule_k_metrics": rule_metrics,
        "rule_ranking": ranking,
        "rule_agreement": agreement,
    }
    if not tau_sweep.empty:
        tables["coverage_threshold_sweep"] = tau_sweep
    if not beta_sweep.empty:
        tables["ambiguity_ceiling_sweep"] = beta_sweep
    if not resampling.empty:
        tables["resampling_k_star"] = resampling
        tables["resampling_frequency"] = frequency

    return {
        "tables": tables,
        "json": {"rule_selection": rule_selection, "resampling_summary": resampling_summary},
        "selections": selections,
    }
