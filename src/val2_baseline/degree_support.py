"""Per-disease degree dependence and method-exclusive top-K support profiles.

Pure-computation companions of the CALIPER-vs-Classical-RWR target-degree
figures in ``viz.py``:

* ``compute_per_disease_degree_spearman`` — the pooled Spearman rho(degree,
  score) in ``degree_diagnostics_spearman.tsv`` mixes between-disease
  differences in degree and prevalence into a single correlation. This
  recomputes rho *within* each disease, i.e. on exactly the candidate pool each
  disease is ranked and evaluated over, and aggregates it with the disease as
  the unit (median + disease-bootstrap CI + paired Wilcoxon), matching the
  aggregation the rest of the val2_baseline pipeline uses. The primary variant
  runs over all common candidates of a disease (n ~ 4e2 per disease); the
  positives-only variant reproduces the pooled table's definition and is gated
  by ``min_positives`` because the per-disease positive count is small
  (median ~1e1).
* ``compute_topk_overlap`` / ``compute_topk_group_profile`` — CALIPER's and
  Classical RWR's top-K candidates barely overlap, so the two methods are
  described by how their *exclusive* candidates differ in target degree,
  transcriptomic coverage and cell-line support.

Group definitions here are label-free by construction: top-K membership,
overlap and the support profile never read the label column, so these tables
cannot be misread as precision comparisons.

Matplotlib is intentionally not imported — everything in this module is
table-level and unit-testable, and ``viz.py`` owns the plotting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

# The Holm step-down and the defensive paired Wilcoxon are reused from the main
# evaluation module so every table in a run shares one implementation of the
# correction and the test.
from .evaluate import _holm_adjust, _safe_wilcoxon


DEGREE_COL = "degree_H"

# Per-disease Spearman defaults.
CANDIDATE_SETS = ("all", "positive")
SPEARMAN_MIN_PAIRS = 3
SPEARMAN_MIN_POSITIVES = 5
# "Strongly degree-dependent" disease threshold reported alongside the median.
RHO_REPORT_THRESHOLD = 0.3
N_BOOT = 1000
BOOTSTRAP_SEED = 20260727

SPEARMAN_BY_DISEASE_COLUMNS = [
    "disease_id",
    "method",
    "candidate_set",
    "n_pairs",
    "n_positive",
    "n_candidate",
    "spearman_rho",
    "spearman_p",
    "included",
]
SPEARMAN_SUMMARY_COLUMNS = [
    "method",
    "candidate_set",
    "n_diseases",
    "n_diseases_undefined",
    "median_rho",
    "ci_lo",
    "ci_hi",
    "mean_rho",
    "iqr_lo",
    "iqr_hi",
    "rho_threshold",
    "frac_rho_ge_threshold",
    "pooled_rho",
    "delta_vs_caliper",
    "delta_lo",
    "delta_hi",
    "wilcoxon_p",
    "p_holm",
    "significant",
    "n_diseases_paired",
]

# Top-K overlap / support profile.
GROUP_A_ONLY = "caliper_only"
GROUP_B_ONLY = "rwr_only"
GROUP_BOTH = "both"
GROUP_NEITHER = "neither"
TOPK_GROUPS = (GROUP_A_ONLY, GROUP_BOTH, GROUP_B_ONLY, GROUP_NEITHER)
EXCLUSIVE_GROUPS = (GROUP_A_ONLY, GROUP_BOTH, GROUP_B_ONLY)

TOPK_OVERLAP_BY_DISEASE_COLUMNS = [
    "disease_id",
    "K",
    "n_candidate",
    "n_top_a",
    "n_top_b",
    "n_both",
    "n_caliper_only",
    "n_rwr_only",
    "jaccard",
    "boundary_tie_size_a",
    "boundary_tie_size_b",
]
TOPK_OVERLAP_SUMMARY_COLUMNS = [
    "K",
    "n_diseases",
    "median_jaccard",
    "jaccard_ci_lo",
    "jaccard_ci_hi",
    "mean_jaccard",
    "median_n_both",
    "pooled_n_both",
    "pooled_n_caliper_only",
    "pooled_n_rwr_only",
    "n_unique_targets_both",
    "n_unique_targets_caliper_only",
    "n_unique_targets_rwr_only",
    "n_diseases_boundary_tie_a",
    "n_diseases_boundary_tie_b",
]

# Support metrics profiled per top-K group. The first four are also the
# Mann-Whitney family; the remaining two are descriptive context.
SUPPORT_METRICS: Tuple[str, ...] = (
    DEGREE_COL,
    "n_common_cells",
    "mean_covAlign",
    "mean_support",
    "mean_n_meas",
    "ncs_n_common_cells",
    "s_d_h_size",
)
SUPPORT_TEST_METRICS: Tuple[str, ...] = (
    DEGREE_COL,
    "n_common_cells",
    "mean_covAlign",
    "mean_support",
)
SUPPORT_PAIR_COLUMNS = ["disease_id", "target_gene", "K", "group", *SUPPORT_METRICS]
TOPK_GROUP_TESTS_COLUMNS = [
    "K",
    "metric",
    "group_a",
    "group_b",
    "n_a",
    "n_b",
    "median_a",
    "median_b",
    "mannwhitney_p",
    "p_holm",
    "significant",
    "n_tests_in_family",
]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def paired_median_ci_p(
    deltas: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT
) -> Tuple[float, float, float, float]:
    """Median paired difference, bootstrap 95% CI of that median, Wilcoxon p.

    The disease-bootstrap CI and the two-sided paired signed-rank test against 0
    are the same pair of estimates ``effect_size_forest_ap`` reports, so a delta
    computed here can be read next to the main effect-size tables.
    """
    dd = np.asarray(deltas, dtype=float)
    dd = dd[np.isfinite(dd)]
    if dd.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    boot = np.array([np.median(dd[rng.integers(0, dd.size, dd.size)]) for _ in range(n_boot)])
    _stat, p, _note = _safe_wilcoxon(dd)
    return (
        float(np.median(dd)),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
        float(p),
    )


def median_ci_bootstrap(
    vals: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT
) -> Tuple[float, float, float]:
    """Median and its bootstrap 95% CI (resampling the unit of analysis)."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    boot = np.array([np.median(v[rng.integers(0, v.size, v.size)]) for _ in range(n_boot)])
    return float(np.median(v)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _spearman(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    """Spearman rho/p, returning NaN for constant input instead of warning."""
    xv = pd.to_numeric(x, errors="coerce")
    yv = pd.to_numeric(y, errors="coerce")
    mask = xv.notna() & yv.notna()
    xv, yv = xv[mask], yv[mask]
    if len(xv) < SPEARMAN_MIN_PAIRS or xv.nunique() < 2 or yv.nunique() < 2:
        return float("nan"), float("nan")
    from scipy.stats import spearmanr  # type: ignore

    res = spearmanr(xv.to_numpy(), yv.to_numpy())
    return float(res.statistic), float(res.pvalue)


def compute_per_disease_degree_spearman(
    feat: pd.DataFrame,
    methods: Sequence[str] = ("CALIPER", "RWR"),
    *,
    min_positives: int = SPEARMAN_MIN_POSITIVES,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Within-disease Spearman rho(target degree, score) per method.

    ``candidate_set='all'`` (primary) correlates degree and score over every
    common candidate of a disease; ``candidate_set='positive'`` restricts to
    known positives, reproducing the pooled table's definition, and marks
    diseases with fewer than ``min_positives`` positives as ``included=False``.

    Returns ``(by_disease_df, summary_df)``. The summary aggregates across
    diseases (median + bootstrap CI) and, for each baseline, adds the paired
    rho difference vs CALIPER with a signed-rank p corrected across baselines
    within each ``candidate_set``.
    """
    methods = [m for m in methods if m in feat.columns]
    required = {"disease_id", DEGREE_COL, "positive"}
    if feat.empty or not methods or not required.issubset(feat.columns):
        return (
            pd.DataFrame(columns=SPEARMAN_BY_DISEASE_COLUMNS),
            pd.DataFrame(columns=SPEARMAN_SUMMARY_COLUMNS),
        )

    df = feat.copy()
    df["disease_id"] = df["disease_id"].astype(str)
    by_disease = df.groupby("disease_id")
    n_candidate = by_disease["positive"].size()
    n_positive = by_disease["positive"].sum()

    rows: List[Dict[str, object]] = []
    for candidate_set in CANDIDATE_SETS:
        sub = df[df["positive"]] if candidate_set == "positive" else df
        for disease_id, grp in sub.groupby("disease_id"):
            n_pos = int(n_positive.get(disease_id, 0))
            n_cand = int(n_candidate.get(disease_id, 0))
            included = len(grp) >= SPEARMAN_MIN_PAIRS
            if candidate_set == "positive":
                included = included and n_pos >= int(min_positives)
            for method in methods:
                rho, p = _spearman(grp[DEGREE_COL], grp[method])
                rows.append(
                    {
                        "disease_id": disease_id,
                        "method": method,
                        "candidate_set": candidate_set,
                        "n_pairs": int(len(grp)),
                        "n_positive": n_pos,
                        "n_candidate": n_cand,
                        "spearman_rho": rho,
                        "spearman_p": p,
                        "included": bool(included),
                    }
                )
    by_disease_df = pd.DataFrame(rows, columns=SPEARMAN_BY_DISEASE_COLUMNS)
    summary_df = _summarize_per_disease_spearman(
        by_disease_df,
        df,
        methods,
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
    )
    return by_disease_df, summary_df


def _summarize_per_disease_spearman(
    by_disease_df: pd.DataFrame,
    feat: pd.DataFrame,
    methods: Sequence[str],
    *,
    n_boot: int,
    seed: int,
    alpha: float,
) -> pd.DataFrame:
    rng = _rng(seed)
    rows: List[Dict[str, object]] = []
    for candidate_set in CANDIDATE_SETS:
        pool = feat[feat["positive"]] if candidate_set == "positive" else feat
        rho_by_method: Dict[str, pd.Series] = {}
        for method in methods:
            sel = by_disease_df[
                (by_disease_df["candidate_set"] == candidate_set)
                & (by_disease_df["method"] == method)
                & (by_disease_df["included"])
            ]
            rho_by_method[method] = pd.Series(
                sel["spearman_rho"].to_numpy(dtype=float), index=sel["disease_id"].to_numpy()
            )
        for method in methods:
            rho = rho_by_method[method]
            finite = rho[np.isfinite(rho.to_numpy(dtype=float))]
            median, lo, hi = median_ci_bootstrap(finite.to_numpy(dtype=float), rng, n_boot)
            pooled_rho, _pooled_p = _spearman(pool[DEGREE_COL], pool[method])
            row: Dict[str, object] = {
                "method": method,
                "candidate_set": candidate_set,
                "n_diseases": int(finite.size),
                "n_diseases_undefined": int(rho.size - finite.size),
                "median_rho": median,
                "ci_lo": lo,
                "ci_hi": hi,
                "mean_rho": float(finite.mean()) if finite.size else float("nan"),
                "iqr_lo": float(np.percentile(finite, 25)) if finite.size else float("nan"),
                "iqr_hi": float(np.percentile(finite, 75)) if finite.size else float("nan"),
                "rho_threshold": RHO_REPORT_THRESHOLD,
                "frac_rho_ge_threshold": float(np.mean(finite >= RHO_REPORT_THRESHOLD))
                if finite.size
                else float("nan"),
                "pooled_rho": pooled_rho,
                "delta_vs_caliper": float("nan"),
                "delta_lo": float("nan"),
                "delta_hi": float("nan"),
                "wilcoxon_p": float("nan"),
                "p_holm": float("nan"),
                "significant": False,
                "n_diseases_paired": 0,
            }
            if method != "CALIPER" and "CALIPER" in rho_by_method:
                paired = pd.concat(
                    {"CALIPER": rho_by_method["CALIPER"], "baseline": rho}, axis=1
                ).dropna()
                deltas = (paired["CALIPER"] - paired["baseline"]).to_numpy(dtype=float)
                med, dlo, dhi, p = paired_median_ci_p(deltas, rng, n_boot)
                row.update(
                    {
                        "delta_vs_caliper": med,
                        "delta_lo": dlo,
                        "delta_hi": dhi,
                        "wilcoxon_p": p,
                        "n_diseases_paired": int(len(paired)),
                    }
                )
            rows.append(row)

    summary_df = pd.DataFrame(rows, columns=SPEARMAN_SUMMARY_COLUMNS)
    if summary_df.empty:
        return summary_df
    # Correction family: the baselines compared with CALIPER inside one candidate_set.
    for candidate_set in CANDIDATE_SETS:
        mask = (summary_df["candidate_set"] == candidate_set) & summary_df["method"].ne("CALIPER")
        if not mask.any():
            continue
        pvals = summary_df.loc[mask, "wilcoxon_p"].to_numpy(dtype=float)
        adjusted = _holm_adjust(pvals)
        summary_df.loc[mask, "p_holm"] = adjusted
        summary_df.loc[mask, "significant"] = np.isfinite(adjusted) & (adjusted < float(alpha))
    return summary_df


def _boundary_tie_size(scores: np.ndarray, k: int) -> int:
    """Size of the equal-score block the top-K cutoff splits (0 if it does not).

    Same semantics as ``expected_precision_at_k_with_ties``' second return value,
    so a non-zero value here means the same thing as ``topK_boundary_tie_size``
    in ``metrics_by_disease_method.tsv``.
    """
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    k_eff = min(int(k), s.size)
    if k_eff <= 0:
        return 0
    s = np.sort(s)[::-1]
    rows_seen = 0
    start = 0
    while start < s.size and rows_seen < k_eff:
        end = start + 1
        while end < s.size and s[end] == s[start]:
            end += 1
        block = end - start
        if rows_seen + block > k_eff:
            return int(block)
        rows_seen += block
        start = end
    return 0


def _topk_targets(grp: pd.DataFrame, method: str, k: int) -> set:
    rank_col = f"{method}_rank_min"
    sel = grp[grp[rank_col].le(k)]
    return set(sel["target_gene"].astype(str))


def compute_topk_overlap(
    feat: pd.DataFrame,
    method_a: str = "CALIPER",
    method_b: str = "RWR",
    *,
    k_list: Iterable[int] = (10, 20),
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-disease overlap of the two methods' top-K candidate sets.

    Membership is ``rank_min <= K`` on the shared common candidate set, the same
    rule the top-10 degree panels use, so a tie block straddling K can push the
    selected count above K; ``n_top_*`` and ``boundary_tie_size_*`` record that
    explicitly. Labels are not read.
    """
    required = {"disease_id", "target_gene", f"{method_a}_rank_min", f"{method_b}_rank_min"}
    if feat.empty or not required.issubset(feat.columns):
        return (
            pd.DataFrame(columns=TOPK_OVERLAP_BY_DISEASE_COLUMNS),
            pd.DataFrame(columns=TOPK_OVERLAP_SUMMARY_COLUMNS),
        )

    rows: List[Dict[str, object]] = []
    unique_targets: Dict[Tuple[int, str], set] = {}
    for k in [int(k) for k in k_list]:
        for group in EXCLUSIVE_GROUPS:
            unique_targets[(k, group)] = set()
        for disease_id, grp in feat.groupby(feat["disease_id"].astype(str)):
            set_a = _topk_targets(grp, method_a, k)
            set_b = _topk_targets(grp, method_b, k)
            union = set_a | set_b
            both = set_a & set_b
            a_only = set_a - set_b
            b_only = set_b - set_a
            unique_targets[(k, GROUP_BOTH)] |= both
            unique_targets[(k, GROUP_A_ONLY)] |= a_only
            unique_targets[(k, GROUP_B_ONLY)] |= b_only
            rows.append(
                {
                    "disease_id": disease_id,
                    "K": k,
                    "n_candidate": int(len(grp)),
                    "n_top_a": int(len(set_a)),
                    "n_top_b": int(len(set_b)),
                    "n_both": int(len(both)),
                    "n_caliper_only": int(len(a_only)),
                    "n_rwr_only": int(len(b_only)),
                    "jaccard": float(len(both) / len(union)) if union else float("nan"),
                    "boundary_tie_size_a": _boundary_tie_size(grp[method_a].to_numpy(), k)
                    if method_a in grp.columns
                    else 0,
                    "boundary_tie_size_b": _boundary_tie_size(grp[method_b].to_numpy(), k)
                    if method_b in grp.columns
                    else 0,
                }
            )
    by_disease_df = pd.DataFrame(rows, columns=TOPK_OVERLAP_BY_DISEASE_COLUMNS)

    rng = _rng(seed)
    summary_rows: List[Dict[str, object]] = []
    for k, grp in by_disease_df.groupby("K"):
        median, lo, hi = median_ci_bootstrap(grp["jaccard"].to_numpy(dtype=float), rng, n_boot)
        summary_rows.append(
            {
                "K": int(k),
                "n_diseases": int(len(grp)),
                "median_jaccard": median,
                "jaccard_ci_lo": lo,
                "jaccard_ci_hi": hi,
                "mean_jaccard": float(grp["jaccard"].mean(skipna=True)),
                "median_n_both": float(grp["n_both"].median()),
                "pooled_n_both": int(grp["n_both"].sum()),
                "pooled_n_caliper_only": int(grp["n_caliper_only"].sum()),
                "pooled_n_rwr_only": int(grp["n_rwr_only"].sum()),
                "n_unique_targets_both": int(len(unique_targets[(int(k), GROUP_BOTH)])),
                "n_unique_targets_caliper_only": int(len(unique_targets[(int(k), GROUP_A_ONLY)])),
                "n_unique_targets_rwr_only": int(len(unique_targets[(int(k), GROUP_B_ONLY)])),
                "n_diseases_boundary_tie_a": int((grp["boundary_tie_size_a"] > 0).sum()),
                "n_diseases_boundary_tie_b": int((grp["boundary_tie_size_b"] > 0).sum()),
            }
        )
    summary_df = pd.DataFrame(summary_rows, columns=TOPK_OVERLAP_SUMMARY_COLUMNS)
    return by_disease_df, summary_df


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def _k_star(run_root: Path) -> int | None:
    path = run_root / "metadata.json"
    if not path.exists():
        return None
    try:
        meta = json.loads(path.read_text())
    except Exception:
        return None
    k_star = meta.get("k_star")
    return int(k_star) if k_star is not None else None


def load_support_features(run_root: str | Path) -> pd.DataFrame:
    """Pair-level coverage / cell-line-support features written by the run.

    CALIPER's own diagnostics (``scores/caliper_target_level.tsv``, restricted to
    ``k_star`` when the run scored several radii) supply the number of evaluable
    cell lines, the mean measured-gene count, the coverage of the aligned
    neighborhood and the evaluable-gene support; ``common_sets/
    availability_by_pair.tsv`` adds the NCS-side evaluable cell count and the
    disease seed-set size in H. Missing files simply drop their columns.
    """
    run_root = Path(run_root)
    k_star = _k_star(run_root)

    target_level = _read_tsv(run_root / "scores" / "caliper_target_level.tsv")
    frames: List[pd.DataFrame] = []
    if not target_level.empty:
        if "k" in target_level.columns and k_star is not None:
            at_k = target_level[pd.to_numeric(target_level["k"], errors="coerce").eq(k_star)]
            if not at_k.empty:
                target_level = at_k
        keep = [
            c
            for c in ("n_common_cells", "mean_n_meas", "mean_covAlign", "mean_support")
            if c in target_level.columns
        ]
        if keep:
            frames.append(
                target_level.drop_duplicates(subset=["disease_id", "target_gene"])[
                    ["disease_id", "target_gene", *keep]
                ]
            )

    availability = _read_tsv(run_root / "common_sets" / "availability_by_pair.tsv")
    if not availability.empty:
        keep = [c for c in ("ncs_n_common_cells", "s_d_h_size") if c in availability.columns]
        if keep:
            frames.append(
                availability.drop_duplicates(subset=["disease_id", "target_gene"])[
                    ["disease_id", "target_gene", *keep]
                ]
            )

    if not frames:
        return pd.DataFrame(columns=["disease_id", "target_gene"])
    support = frames[0]
    for extra in frames[1:]:
        support = support.merge(extra, on=["disease_id", "target_gene"], how="outer")
    support["disease_id"] = support["disease_id"].astype(str)
    support["target_gene"] = support["target_gene"].astype(str)
    return support


def assign_topk_groups(
    feat: pd.DataFrame, k: int, method_a: str = "CALIPER", method_b: str = "RWR"
) -> pd.Series:
    """Label every candidate pair caliper_only / both / rwr_only / neither at K."""
    in_a = feat[f"{method_a}_rank_min"].le(k)
    in_b = feat[f"{method_b}_rank_min"].le(k)
    return pd.Series(
        np.select(
            [in_a & in_b, in_a & ~in_b, ~in_a & in_b],
            [GROUP_BOTH, GROUP_A_ONLY, GROUP_B_ONLY],
            default=GROUP_NEITHER,
        ),
        index=feat.index,
        dtype=object,
    )


def compute_topk_group_profile(
    feat: pd.DataFrame,
    support: pd.DataFrame,
    *,
    method_a: str = "CALIPER",
    method_b: str = "RWR",
    k_list: Iterable[int] = (10, 20),
    alpha: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Degree / coverage / cell-line-support profile of the method-exclusive top-K.

    Returns ``(pair_df, profile_df, tests_df)``. ``pair_df`` carries one row per
    (disease, target, K) for the three top-K groups (the ``neither`` background
    is summarized in ``profile_df`` only, to keep the table small);
    ``profile_df`` has the per-group median/IQR of every support metric;
    ``tests_df`` has Mann-Whitney comparisons between groups, corrected across
    all tests within one K.
    """
    required = {"disease_id", "target_gene", f"{method_a}_rank_min", f"{method_b}_rank_min"}
    if feat.empty or not required.issubset(feat.columns):
        empty_pairs = pd.DataFrame(columns=SUPPORT_PAIR_COLUMNS)
        return empty_pairs, pd.DataFrame(), pd.DataFrame(columns=TOPK_GROUP_TESTS_COLUMNS)

    base = feat[["disease_id", "target_gene", *[c for c in (DEGREE_COL,) if c in feat.columns]]].copy()
    base["disease_id"] = base["disease_id"].astype(str)
    base["target_gene"] = base["target_gene"].astype(str)
    if support is not None and not support.empty:
        base = base.merge(support, on=["disease_id", "target_gene"], how="left")
    metrics = [m for m in SUPPORT_METRICS if m in base.columns]

    pair_frames: List[pd.DataFrame] = []
    profile_rows: List[Dict[str, object]] = []
    test_rows: List[Dict[str, object]] = []
    for k in [int(k) for k in k_list]:
        groups = assign_topk_groups(feat, k, method_a, method_b)
        at_k = base.assign(K=k, group=groups.to_numpy())
        pair_frames.append(at_k[at_k["group"].isin(EXCLUSIVE_GROUPS)])

        for group in TOPK_GROUPS:
            sub = at_k[at_k["group"] == group]
            if sub.empty:
                continue
            row: Dict[str, object] = {
                "K": k,
                "group": group,
                "n_pairs": int(len(sub)),
                "n_unique_targets": int(sub["target_gene"].nunique()),
                "n_unique_diseases": int(sub["disease_id"].nunique()),
            }
            for metric in metrics:
                vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
                row[f"{metric}_median"] = float(vals.median()) if len(vals) else float("nan")
                row[f"{metric}_q25"] = float(np.percentile(vals, 25)) if len(vals) else float("nan")
                row[f"{metric}_q75"] = float(np.percentile(vals, 75)) if len(vals) else float("nan")
                row[f"{metric}_n"] = int(len(vals))
            profile_rows.append(row)

        test_metrics = [m for m in SUPPORT_TEST_METRICS if m in metrics]
        comparisons = [
            (GROUP_A_ONLY, GROUP_B_ONLY),
            (GROUP_A_ONLY, GROUP_NEITHER),
            (GROUP_B_ONLY, GROUP_NEITHER),
        ]
        k_rows: List[Dict[str, object]] = []
        for metric in test_metrics:
            for group_a, group_b in comparisons:
                x = pd.to_numeric(
                    at_k.loc[at_k["group"] == group_a, metric], errors="coerce"
                ).dropna()
                y = pd.to_numeric(
                    at_k.loc[at_k["group"] == group_b, metric], errors="coerce"
                ).dropna()
                if x.empty or y.empty:
                    continue
                from scipy.stats import mannwhitneyu  # type: ignore

                try:
                    p = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
                except ValueError:
                    p = float("nan")
                k_rows.append(
                    {
                        "K": k,
                        "metric": metric,
                        "group_a": group_a,
                        "group_b": group_b,
                        "n_a": int(len(x)),
                        "n_b": int(len(y)),
                        "median_a": float(x.median()),
                        "median_b": float(y.median()),
                        "mannwhitney_p": p,
                    }
                )
        if k_rows:
            pvals = np.array([r["mannwhitney_p"] for r in k_rows], dtype=float)
            adjusted = _holm_adjust(pvals)
            for row, adj in zip(k_rows, adjusted):
                row["p_holm"] = float(adj)
                row["significant"] = bool(np.isfinite(adj) and adj < float(alpha))
                row["n_tests_in_family"] = len(k_rows)
            test_rows.extend(k_rows)

    pair_df = (
        pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame(columns=SUPPORT_PAIR_COLUMNS)
    )
    if not pair_df.empty:
        ordered = ["disease_id", "target_gene", "K", "group", *metrics]
        pair_df = pair_df[[c for c in ordered if c in pair_df.columns]]
    profile_df = pd.DataFrame(profile_rows)
    tests_df = pd.DataFrame(test_rows, columns=TOPK_GROUP_TESTS_COLUMNS)
    return pair_df, profile_df, tests_df
