from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


GetNodesFn = Callable[[str, int], Set[str]]
GetVotesFn = Callable[[str, int, int], Dict[str, Dict[int, float]]]

ALLOWED_COVERAGE_METRICS = {"cov_nz", "cov_ratio"}
ALLOWED_AMBIGUITY_METRICS = {"amb_median"}


def _finite_array(x: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(x), dtype=float)
    return arr[np.isfinite(arr)]


def median_iqr(x: np.ndarray) -> Tuple[float, float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    q25 = float(np.quantile(x, 0.25))
    q50 = float(np.quantile(x, 0.50))
    q75 = float(np.quantile(x, 0.75))
    return q50, q25, q75


def mad(x: np.ndarray, center: float | None = None) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if center is None or not np.isfinite(center):
        center = float(np.median(x))
    return float(np.median(np.abs(x - center)))


def _progress(progress, iterable, **kwargs):
    if progress is None:
        return tqdm(iterable, **kwargs)
    if progress is False:
        return iterable
    return progress(iterable, **kwargs)


def compute_pair_coverage(
    *,
    pairs: pd.DataFrame,
    ks: List[int],
    disease_gene_sets: Dict[str, Set[str]],
    perturbation_gene_sets: Dict[Tuple[str, str], Set[str]],
    kg_nodes: Set[str],
    get_nodes_k: GetNodesFn,
    progress=None,
) -> pd.DataFrame:
    """Compute pair-level coverage for every (d,t,c,k).

    The perturbation gene set is a gene universe and therefore includes zero-valued
    perturbation genes as long as the corresponding profile column exists.
    Disease gene sets are non-zero disease genes.  When Base is empty, the
    pair-level ratio is undefined and retained as NaN.
    """
    rows: list[dict[str, object]] = []
    kg_nodes = set(map(str, kg_nodes))
    ks_int = [int(k) for k in ks]
    iter_pairs = pairs[["disease_id", "target_gene", "cell_id"]].drop_duplicates().to_numpy(dtype=object)
    for d, t, c in _progress(progress, iter_pairs, total=len(iter_pairs), desc="coverage"):
        d = str(d)
        t = str(t)
        c = str(c)
        D = disease_gene_sets.get(d, set())
        P = perturbation_gene_sets.get((t, c), set())
        if not isinstance(D, set):
            D = set(D)
        if not isinstance(P, set):
            P = set(P)
        base = D & P & kg_nodes
        base_count = int(len(base))
        if base_count == 0:
            for k in ks_int:
                rows.append(
                    {
                        "disease_id": d,
                        "target_gene": t,
                        "cell_id": c,
                        "k": k,
                        "base_count": 0,
                        "covered_count": 0,
                        "cov_nz_pair": 0,
                        "cov_ratio_pair": float("nan"),
                    }
                )
            continue

        for k in ks_int:
            Nk = get_nodes_k(t, k)
            covered_count = int(len(Nk & base))
            rows.append(
                {
                    "disease_id": d,
                    "target_gene": t,
                    "cell_id": c,
                    "k": k,
                    "base_count": base_count,
                    "covered_count": covered_count,
                    "cov_nz_pair": int(covered_count > 0),
                    "cov_ratio_pair": float(covered_count / base_count),
                }
            )
    return pd.DataFrame(rows)


def summarize_coverage(pair_coverage_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, g in pair_coverage_df.groupby("k", sort=True):
        ratio = g["cov_ratio_pair"].to_numpy(dtype=float)
        td_ratio = (
            g.groupby(["target_gene", "disease_id"], sort=False)["cov_ratio_pair"]
            .median()
            .to_numpy(dtype=float)
        )
        rows.append(
            {
                "k": int(k),
                "C_nz": float(g["cov_nz_pair"].mean()) if len(g) else float("nan"),
                "cov_ratio_median": float(np.nanmedian(td_ratio)) if np.isfinite(td_ratio).any() else float("nan"),
                "cov_ratio_mean": float(np.nanmean(ratio)) if np.isfinite(ratio).any() else float("nan"),
                "n_pairs": int(len(g)),
                "n_empty_base_pairs": int((g["base_count"].astype(int) == 0).sum()),
                "n_cov_ratio_non_na": int(np.isfinite(ratio).sum()),
                "base_count_median": float(np.median(g["base_count"].to_numpy(dtype=float))) if len(g) else float("nan"),
                "covered_count_median": float(np.median(g["covered_count"].to_numpy(dtype=float))) if len(g) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def summarize_coverage_by_td(pair_coverage_df: pd.DataFrame, progress=None) -> pd.DataFrame:
    rows = []
    group_cols = ["target_gene", "disease_id", "k"]
    grouped = pair_coverage_df.groupby(group_cols, sort=True)
    for (t, d, k), g in _progress(progress, grouped, total=grouped.ngroups, desc="td-coverage"):
        ratio = g["cov_ratio_pair"].to_numpy(dtype=float)
        rows.append(
            {
                "target_gene": str(t),
                "disease_id": str(d),
                "k": int(k),
                "C_nz_td": float(g["cov_nz_pair"].mean()) if len(g) else float("nan"),
                "cov_ratio_td_median": float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else float("nan"),
                "cov_ratio_td_mean": float(np.nanmean(ratio)) if np.isfinite(ratio).any() else float("nan"),
                "n_pairs_td": int(len(g)),
                "n_cells_td": int(g["cell_id"].astype(str).nunique()),
                "n_empty_base_pairs_td": int((g["base_count"].astype(int) == 0).sum()),
                "n_cov_ratio_non_na_td": int(np.isfinite(ratio).sum()),
                "base_count_td_median": float(np.median(g["base_count"].to_numpy(dtype=float))) if len(g) else float("nan"),
                "covered_count_td_median": float(np.median(g["covered_count"].to_numpy(dtype=float))) if len(g) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def compute_target_ambiguity(
    *,
    targets: List[str],
    ks: List[int],
    start_sign: int,
    get_nodes_k: GetNodesFn,
    get_votes: GetVotesFn,
    progress=None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ks_int = [int(k) for k in ks]
    for t in _progress(progress, [str(x) for x in targets], total=len(targets), desc="ambiguity"):
        for k in ks_int:
            Nk = get_nodes_k(t, k)
            n_nodes = int(len(Nk))
            if n_nodes == 0:
                rows.append(
                    {
                        "target_gene": t,
                        "k": k,
                        "n_nodes": 0,
                        "n_voted": 0,
                        "n_multi": 0,
                        "n_single": 0,
                        "amb": float("nan"),
                    }
                )
                continue

            votes = get_votes(t, k, int(start_sign))
            n_voted = 0
            n_multi = 0
            n_single = 0
            for g in Nk:
                vv = votes.get(g)
                if not vv:
                    continue
                pos = float(vv.get(1, 0.0))
                neg = float(vv.get(-1, 0.0))
                if (pos <= 0.0) and (neg <= 0.0):
                    continue
                n_voted += 1
                if (pos > 0.0) and (neg > 0.0):
                    n_multi += 1
                else:
                    n_single += 1
            amb = float(n_multi / n_nodes) if n_nodes > 0 else float("nan")
            rows.append(
                {
                    "target_gene": t,
                    "k": k,
                    "n_nodes": n_nodes,
                    "n_voted": int(n_voted),
                    "n_multi": int(n_multi),
                    "n_single": int(n_single),
                    "amb": amb,
                }
            )
    return pd.DataFrame(rows)


def summarize_ambiguity(target_ambiguity_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, g in target_ambiguity_df.groupby("k", sort=True):
        amb = g["amb"].to_numpy(dtype=float)
        rows.append(
            {
                "k": int(k),
                "B_amb_median": float(np.nanmedian(amb)) if np.isfinite(amb).any() else float("nan"),
                "amb_mean": float(np.nanmean(amb)) if np.isfinite(amb).any() else float("nan"),
                "n_targets": int(len(g)),
                "n_amb_non_na": int(np.isfinite(amb).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


# Legacy wrappers retained for downstream imports.  They are not used by the
# revised Val1 main pipeline.
def run_val1_1(
    *,
    pairs: pd.DataFrame,
    ks: List[int],
    dis_keys: Dict[str, Set[str]],
    pert_all_keys: Dict[Tuple[str, str], Set[str]],
    kg_nodes: Set[str],
    get_nodes_k: GetNodesFn,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pair_df = compute_pair_coverage(
        pairs=pairs,
        ks=ks,
        disease_gene_sets=dis_keys,
        perturbation_gene_sets=pert_all_keys,
        kg_nodes=kg_nodes,
        get_nodes_k=get_nodes_k,
    )
    legacy_pair = pair_df.rename(
        columns={"covered_count": "cov_count", "cov_ratio_pair": "cov_ratio"}
    ).copy()
    legacy_summary = summarize_coverage(pair_df).rename(
        columns={"C_nz": "cov_nz", "covered_count_median": "cov_count_median"}
    )
    return legacy_pair, legacy_summary


def run_val1_2(
    *,
    targets: List[str],
    ks: List[int],
    start_sign: int,
    get_nodes_k: GetNodesFn,
    get_votes: GetVotesFn,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target_df = compute_target_ambiguity(
        targets=targets,
        ks=ks,
        start_sign=start_sign,
        get_nodes_k=get_nodes_k,
        get_votes=get_votes,
    )
    legacy_target = target_df.copy()
    legacy_target["sign_agree"] = 1.0 - legacy_target["amb"]
    summary = summarize_ambiguity(target_df).rename(columns={"B_amb_median": "amb_median"})
    summary["sign_agree_median"] = 1.0 - summary["amb_median"]
    return legacy_target, summary


def run_val1_3(*args, **kwargs):  # pragma: no cover - intentionally deprecated in v2.5
    raise RuntimeError(
        "Alignment diagnostics are not part of the revised Val1 main pipeline. "
        "Use legacy code or a separate diagnostics module if alignment is needed."
    )
