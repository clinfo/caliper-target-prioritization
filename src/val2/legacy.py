"""Target ranking performance evaluation (v2 — optimized).

Optimizations over val2.py (logic-preserving, output-identical):
  1. Vectorised size-matched scoring: reversal flags are pre-computed once
     per (pair, k), then reused across all reps via index-based sampling.
  2. Single-pass sizematch: mean scores *and* per-rep scores are computed
     in one pass instead of two, eliminating redundant score_pair calls.
  3. Batched SEM merge: per-rep DataFrames are merged with labels once per k
     instead of once per (k, rep), reducing pandas overhead.
  4. SHA-256 prefix caching: the hash prefix shared across reps is computed
     once, avoiding repeated string concatenation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import argparse
import json
import random
import hashlib

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

from inference.signature_reversal_nodes import (
    khop_nodes,
    propagate_sign_votes_shortest_paths,
    predicted_sign_sets_from_votes,
    confidence_weight_from_votes,
    _score_proposed,
)

# pert signature is keyed by (pert_id, cell_id)
PertKey = Tuple[str, str]  # (pert_id, cell_id)

Gene = str
Disease = str
Target = str
Edge = Tuple[str, str]  # (src, dst)
VALID_TIE_POLICIES = {"legacy", "expected"}

def stable_uint32_seed(*parts) -> int:
    """Create a stable 32-bit seed from arbitrary parts (no Python hash())."""
    s = "|".join(str(p) for p in parts)
    d = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(d[:4], "little", signed=False)


# -----------------------------
# Loading (KG)
# -----------------------------
def load_signed_multidigraph_from_tsv(
    path: str,
    src_col: str = "source",
    dst_col: str = "target",
    sign_col: str = "polarity",
    sep: str = "\t",
) -> nx.MultiDiGraph:
    """
    kg_openbiolink.tsv:
      - source, target, polarity (-1 or 1 only; int or string "1"/"-1")
    """
    df = pd.read_csv(path, sep=sep)
    required = {src_col, dst_col, sign_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[KG] Missing columns: {missing}. existing={list(df.columns)[:20]}")
    if len(df) == 0:
        raise ValueError(f"[KG] File is empty or has no data rows: {path}")

    G = nx.MultiDiGraph()
    src_i = df.columns.get_loc(src_col)
    dst_i = df.columns.get_loc(dst_col)
    sign_i = df.columns.get_loc(sign_col)

    for idx, row in enumerate(df.itertuples(index=False, name=None)):
        s = str(row[src_i]).strip()
        t = str(row[dst_i]).strip()
        pol_raw = row[sign_i]
        if not s or not t:
            raise ValueError(
                f"[KG] Row {idx}: source and target must be non-empty. got source={repr(s)}, target={repr(t)}"
            )
        try:
            pol = int(pol_raw) if not isinstance(pol_raw, str) else int(pol_raw.strip())
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"[KG] Row {idx}: polarity must be -1 or 1 (int or string). got {repr(pol_raw)}"
            ) from e
        if pol not in (-1, 1):
            raise ValueError(
                f"[KG] Row {idx}: polarity must be exactly -1 or 1. got {pol}"
            )
        G.add_edge(s, t, polarity=pol)

    return G


# -----------------------------
# Graph utilities
# -----------------------------


# -----------------------------
# Ranking metrics (no sklearn)
# -----------------------------
def normalize_tie_policy(tie_policy: str = "expected") -> str:
    policy = str(tie_policy or "expected").strip().lower()
    aliases = {
        "default": "expected",
        "tie_aware": "expected",
        "tie-aware": "expected",
        "average": "expected",
        "average_ties": "expected",
        "expected_ties": "expected",
    }
    policy = aliases.get(policy, policy)
    if policy not in VALID_TIE_POLICIES:
        raise ValueError(f"Unknown tie_policy={tie_policy!r}. valid={sorted(VALID_TIE_POLICIES)}")
    return policy


def precision_at_k(y_true: List[int], y_score: List[float], k: int) -> float:
    idx = np.argsort(-np.array(y_score))[:k]
    yt = np.array(y_true)[idx]
    return float(yt.mean()) if len(yt) > 0 else float("nan")


def average_precision(y_true: List[int], y_score: List[float]) -> float:
    y_true = np.array(y_true, dtype=int)
    y_score = np.array(y_score, dtype=float)
    order = np.argsort(-y_score)
    y_true = y_true[order]

    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")

    cum_tp = 0
    ap = 0.0
    for i, yt in enumerate(y_true, start=1):
        if yt == 1:
            cum_tp += 1
            ap += cum_tp / i
    return float(ap / n_pos)


def expected_precision_at_k_with_ties(y_true: List[int], y_score: List[float], k: int) -> Tuple[float, int]:
    """Expected P@K when tied score blocks are internally uniformly randomized.

    Returns (P@K, boundary_tie_size). boundary_tie_size is non-zero only when
    the top-K cutoff splits an equal-score block.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    finite = np.isfinite(y_score_arr)
    y_true_arr = y_true_arr[finite]
    y_score_arr = y_score_arr[finite]
    if len(y_true_arr) == 0:
        return float("nan"), 0

    k_eff = min(int(k), len(y_true_arr))
    if k_eff <= 0:
        return float("nan"), 0

    order = np.argsort(-y_score_arr, kind="mergesort")
    y_true_arr = y_true_arr[order]
    y_score_arr = y_score_arr[order]

    expected_pos = 0.0
    rows_seen = 0
    boundary_tie_size = 0
    start = 0
    n = len(y_true_arr)
    while start < n and rows_seen < k_eff:
        end = start + 1
        while end < n and y_score_arr[end] == y_score_arr[start]:
            end += 1

        block_size = int(end - start)
        block_pos = int(y_true_arr[start:end].sum())
        remaining = k_eff - rows_seen
        if block_size <= remaining:
            expected_pos += block_pos
            rows_seen += block_size
        else:
            expected_pos += remaining * (block_pos / block_size)
            boundary_tie_size = block_size
            rows_seen = k_eff
        start = end

    return float(expected_pos / k_eff), int(boundary_tie_size)


def expected_average_precision_with_ties(y_true: List[int], y_score: List[float]) -> Tuple[float, int, int, int]:
    """Expected AP when tied score blocks are internally uniformly randomized.

    Returns (AP, n_score_groups, n_tie_groups, largest_tie_group).
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    finite = np.isfinite(y_score_arr)
    y_true_arr = y_true_arr[finite]
    y_score_arr = y_score_arr[finite]

    n_pos = int(y_true_arr.sum())
    if n_pos == 0:
        return float("nan"), 0, 0, 0

    order = np.argsort(-y_score_arr, kind="mergesort")
    y_true_arr = y_true_arr[order]
    y_score_arr = y_score_arr[order]

    ap = 0.0
    positives_before = 0
    rows_before = 0
    n_score_groups = 0
    n_tie_groups = 0
    largest_tie_group = 0

    start = 0
    n = len(y_true_arr)
    while start < n:
        end = start + 1
        while end < n and y_score_arr[end] == y_score_arr[start]:
            end += 1

        block_size = int(end - start)
        block_pos = int(y_true_arr[start:end].sum())
        n_score_groups += 1
        if block_size > 1:
            n_tie_groups += 1
            largest_tie_group = max(largest_tie_group, block_size)

        if block_pos > 0:
            if block_size == 1:
                ap += (positives_before + 1) / (rows_before + 1)
            else:
                for j in range(1, block_size + 1):
                    expected_prev_in_block = (
                        ((j - 1) * (block_pos - 1) / (block_size - 1)) if block_pos > 1 else 0.0
                    )
                    expected_positive_at_j = block_pos / block_size
                    expected_precision_numerator = positives_before + 1.0 + expected_prev_in_block
                    ap += expected_positive_at_j * expected_precision_numerator / (rows_before + j)

        positives_before += block_pos
        rows_before += block_size
        start = end

    return float(ap / n_pos), int(n_score_groups), int(n_tie_groups), int(largest_tie_group)


def _ranking_metrics_for_scores(
    y_true: List[int],
    y_score: List[float],
    K_list: List[int],
    tie_policy: str = "expected",
) -> Dict[str, object]:
    policy = normalize_tie_policy(tie_policy)
    n_total = int(len(y_true))
    n_pos = int(sum(y_true))
    chance = float(np.mean(y_true)) if n_total > 0 else float("nan")
    out: Dict[str, object] = {
        "chance": chance,
        "n_pos": n_pos,
        "n_neg": int(n_total - n_pos),
        "tie_policy": policy,
    }
    if policy == "legacy":
        out["AP"] = average_precision(y_true, y_score)
        out["n_score_groups"] = np.nan
        out["n_tie_groups"] = np.nan
        out["largest_tie_group"] = np.nan
        for K in K_list:
            p_at_k = precision_at_k(y_true, y_score, int(K))
            out[f"P@{K}"] = p_at_k
            out[f"Lift@{K}"] = (p_at_k / chance) if chance > 0 and not np.isnan(p_at_k) else np.nan
            out[f"top{K}_boundary_tie_size"] = np.nan
        return out

    ap, n_score_groups, n_tie_groups, largest_tie_group = expected_average_precision_with_ties(y_true, y_score)
    out["AP"] = ap
    out["n_score_groups"] = n_score_groups
    out["n_tie_groups"] = n_tie_groups
    out["largest_tie_group"] = largest_tie_group
    for K in K_list:
        p_at_k, boundary_tie_size = expected_precision_at_k_with_ties(y_true, y_score, int(K))
        out[f"P@{K}"] = p_at_k
        out[f"Lift@{K}"] = (p_at_k / chance) if chance > 0 and not np.isnan(p_at_k) else np.nan
        out[f"top{K}_boundary_tie_size"] = boundary_tie_size
    return out


def _detect_profile_columns(
    df: pd.DataFrame,
    pert_suffix: str = "_gene",
    disease_suffix: str = "_disease",
) -> Tuple[List[str], List[str]]:
    pert_cols = [c for c in df.columns if c.endswith(pert_suffix)]
    dis_cols = [c for c in df.columns if c.endswith(disease_suffix)]

    if len(pert_cols) == 0 or len(dis_cols) == 0:
        raise ValueError(
            "Gene-profile columns not found.\n"
            f"  detected pert_cols={len(pert_cols)}, disease_cols={len(dis_cols)}\n"
            f"  expected suffixes: pert_suffix='{pert_suffix}', disease_suffix='{disease_suffix}'\n"
            "If your parquet uses different suffixes, change the args."
        )
    return pert_cols, dis_cols


def _row_to_sig_dict_disease(row: pd.Series, cols: List[str], suffix: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c in cols:
        v = int(row[c])
        if v == 0:
            continue
        gene = str(c[: -len(suffix)])
        out[gene] = v
    return out


def _row_to_sig_dict_pert(row: pd.Series, cols: List[str], suffix: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c in cols:
        v = int(row[c])
        gene = str(c[: -len(suffix)])
        out[gene] = v
    return out


def load_knockdown_parquet_as_dicts(
    parquet_path: str,
    pert_suffix: str = "_gene",
    disease_suffix: str = "_disease",
    kg_nodes: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, Dict[PertKey, Dict[str, int]], Dict[str, Dict[str, int]], List[str]]:
    """
    Input parquet columns (expected):
      - pert_id, do_id, cell_id, label
      - <gene><pert_suffix>    : perturbation sign (-1/0/1)
      - <gene><disease_suffix> : disease sign (-1/0/1)

    Returns:
      labels_df: columns [disease_id, target_gene, cell_id, label]
      pert_sigs: dict[(pert_id, cell_id)] -> dict[gene]->sign
      dis_sigs : dict[do_id] -> dict[gene]->sign
      common_genes: intersection gene list (string). If kg_nodes is given, additionally filtered by KG nodes.
    """
    import pyarrow.parquet as pq

    # Phase 1: inspect schema to determine needed columns (no data loaded)
    all_columns = pq.read_schema(parquet_path).names

    _meta_cols = ["pert_id", "do_id", "cell_id", "label"]
    missing = set(_meta_cols) - set(all_columns)
    if missing:
        raise ValueError(f"[parquet] Missing columns: {missing}. existing={all_columns[:20]}")

    pert_cols = [c for c in all_columns if c.endswith(pert_suffix)]
    dis_cols = [c for c in all_columns if c.endswith(disease_suffix)]
    if len(pert_cols) == 0 or len(dis_cols) == 0:
        raise ValueError(
            "Gene-profile columns not found.\n"
            f"  detected pert_cols={len(pert_cols)}, disease_cols={len(dis_cols)}\n"
            f"  expected suffixes: pert_suffix='{pert_suffix}', disease_suffix='{disease_suffix}'\n"
            "If your parquet uses different suffixes, change the args."
        )

    pert_genes = [str(c[: -len(pert_suffix)]) for c in pert_cols]
    dis_genes = [str(c[: -len(disease_suffix)]) for c in dis_cols]
    common_genes = sorted(set(pert_genes) & set(dis_genes))

    if kg_nodes is not None:
        kg_nodes = set(map(str, kg_nodes))
        common_genes = [g for g in common_genes if g in kg_nodes]

    if len(common_genes) == 0:
        raise ValueError("No common genes between perturbation columns and disease columns (after KG filtering).")

    pert_cols_common = [g + pert_suffix for g in common_genes]
    dis_cols_common = [g + disease_suffix for g in common_genes]

    # Phase 2: pre-compute row mask from meta columns only (low memory)
    import gc
    try:
        df_meta = pd.read_parquet(parquet_path, columns=_meta_cols, engine='pyarrow')
    except (OSError, Exception):
        df_meta = pd.read_parquet(parquet_path, columns=_meta_cols, engine='fastparquet')

    if len(df_meta) == 0:
        raise ValueError(f"[parquet] File is empty or has no data rows: {parquet_path}")

    label_vals = df_meta["label"].dropna().astype(int)
    if not label_vals.isin([0, 1]).all():
        bad = label_vals[~label_vals.isin([0, 1])].unique().tolist()
        raise ValueError(f"[parquet] Column 'label' must contain only 0 or 1. got values: {bad}")
    if df_meta["label"].isna().any():
        raise ValueError("[parquet] Column 'label' must not contain NaN.")

    if kg_nodes is not None:
        keep_mask = df_meta["pert_id"].astype(str).isin(kg_nodes).values
    else:
        keep_mask = np.ones(len(df_meta), dtype=bool)
    del df_meta, label_vals
    gc.collect()

    if not keep_mask.any():
        raise ValueError("No rows left after filtering: pert_id not found in KG nodes.")

    # Phase 3: read profile columns in chunks, applying row mask
    col_chunk_size = 2000
    profile_cols = pert_cols_common + dis_cols_common
    chunks = [pd.read_parquet(parquet_path, columns=_meta_cols, engine='fastparquet')[keep_mask].reset_index(drop=True)]
    for i in range(0, len(profile_cols), col_chunk_size):
        batch = profile_cols[i:i + col_chunk_size]
        try:
            chunk = pd.read_parquet(parquet_path, columns=batch, engine='pyarrow')
        except (OSError, Exception):
            chunk = pd.read_parquet(parquet_path, columns=batch, engine='fastparquet')
        chunks.append(chunk[keep_mask].reset_index(drop=True))
        del chunk
        gc.collect()
    del keep_mask
    gc.collect()
    df = pd.concat(chunks, axis=1)
    del chunks
    gc.collect()

    _VALID_SIG = {-1, 0, 1}
    for col in pert_cols_common + dis_cols_common:
        if df[col].isna().any():
            raise ValueError(f"[parquet] Column '{col}' must not contain NaN.")
        try:
            vals = df[col].astype(int)
        except (TypeError, ValueError) as e:
            raise ValueError(f"[parquet] Column '{col}' must be integer (-1/0/1). {e}") from e
        if not vals.isin(_VALID_SIG).all():
            bad = vals[~vals.isin(_VALID_SIG)].dropna().unique().tolist()
            raise ValueError(f"[parquet] Column '{col}' must contain only -1, 0, 1. got values: {bad}")

    # label table
    labels_df = (
        df[["do_id", "pert_id", "cell_id", "label"]]
        .drop_duplicates()
        .rename(columns={"do_id": "disease_id", "pert_id": "target_gene"})
    )
    labels_df["disease_id"] = labels_df["disease_id"].astype(str)
    labels_df["target_gene"] = labels_df["target_gene"].astype(str)
    labels_df["cell_id"] = labels_df["cell_id"].astype(str)
    labels_df["label"] = labels_df["label"].astype(int)

    # perturb signatures: unique (pert_id, cell_id)
    pert_sigs: Dict[PertKey, Dict[str, int]] = {}
    sub_pert = df[["pert_id", "cell_id"] + pert_cols_common].drop_duplicates(subset=["pert_id", "cell_id"])
    for _, r in sub_pert.iterrows():
        key = (str(r["pert_id"]), str(r["cell_id"]))
        pert_sigs[key] = _row_to_sig_dict_pert(r, pert_cols_common, pert_suffix)

    # disease signatures: unique do_id
    dis_sigs: Dict[str, Dict[str, int]] = {}
    sub_dis = df[["do_id"] + dis_cols_common].drop_duplicates(subset=["do_id"])
    for _, r in sub_dis.iterrows():
        did = str(r["do_id"])
        dis_sigs[did] = _row_to_sig_dict_disease(r, dis_cols_common, disease_suffix)

    return labels_df, pert_sigs, dis_sigs, common_genes


# -----------------------------
# Core scoring for Val2
# -----------------------------
def _score_noKG(B, pert, dis):
    n_meas = len(B)
    if n_meas == 0:
        return 0.0, 0, 0, 0.0, 0.0, 0
    n_rev = sum(1 for g in B if pert.get(g, 0) * dis.get(g, 0) == -1)
    return (n_rev / n_meas), n_rev, n_meas, 0.0, 0.0, 0


def _sizematch_reversal_flags(
    B: Set[str],
    pert: Dict[str, int],
    dis: Dict[str, int],
) -> Tuple[List[str], np.ndarray]:
    """Build a sorted gene list and a boolean reversal-flag array for B.

    A gene g is a "reversal" when pert[g] * dis[g] == -1.
    The returned array has shape (len(B),) with dtype int32 (0 or 1).
    Sorting B ensures deterministic ordering for reproducible sampling.
    """
    genes_sorted = sorted(B)
    if len(genes_sorted) == 0:
        return genes_sorted, np.empty(0, dtype=np.int32)
    is_reversal = np.array(
        [1 if pert.get(g, 0) * dis.get(g, 0) == -1 else 0 for g in genes_sorted],
        dtype=np.int32,
    )
    return genes_sorted, is_reversal


def _score_noKG_sizematch(
    *,
    B: Set[str],
    U: Set[str],
    pert: Dict[str, int],
    dis: Dict[str, int],
    target_gene: str,
    cell_id: str,
    disease_id: str,
    k: int,
    sizematch_reps: int,
    sizematch_seed: int,
    sizematch_rep: Optional[int] = None,
):
    """Score a pair using size-matched random subsets of B (noKG baseline).

    Randomly samples |U| genes from B without replacement and counts reversals.
    If sizematch_rep is given, only that single rep is computed; otherwise all
    reps are averaged.

    Returns the standard 6-tuple: (score, n_rev, n_meas, covK, covAlign, support).
    """
    sample_size = len(U)
    if sample_size == 0:
        return 0.0, 0, 0, 0.0, 0.0, 0

    genes_sorted, is_reversal = _sizematch_reversal_flags(B, pert, dis)
    pool_size = len(genes_sorted)
    if pool_size == 0:
        return 0.0, 0, sample_size, 0.0, 0.0, 0

    def _one_rep(r: int) -> Tuple[float, int]:
        seed = stable_uint32_seed(target_gene, cell_id, disease_id, int(k), int(sizematch_seed), int(r))
        rng = np.random.default_rng(seed)
        sampled_idx = rng.choice(pool_size, size=sample_size, replace=False)
        n_rev_r = int(is_reversal[sampled_idx].sum())
        return float(n_rev_r / sample_size), n_rev_r

    # Single rep mode (used by SEM computation in the original code path)
    if sizematch_rep is not None:
        score, n_rev = _one_rep(int(sizematch_rep))
        return score, n_rev, sample_size, 0.0, 0.0, 0

    # All-reps mode: average across reps
    scores = []
    n_revs = []
    for r in range(int(sizematch_reps)):
        score, n_rev = _one_rep(r)
        scores.append(score)
        n_revs.append(n_rev)

    return float(np.mean(scores)), float(np.mean(n_revs)), sample_size, 0.0, 0.0, 0


def _score_noKG_sizematch_all_reps(
    *,
    B: Set[str],
    U: Set[str],
    pert: Dict[str, int],
    dis: Dict[str, int],
    target_gene: str,
    cell_id: str,
    disease_id: str,
    k: int,
    sizematch_reps: int,
    sizematch_seed: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Compute size-matched scores for *all* reps in a single pass.

    This is the main optimisation entry point used by run_val2.  It avoids
    calling score_pair per rep and instead:
      - builds the reversal-flag array once,
      - caches the SHA-256 hash prefix shared across reps, and
      - samples via integer indices rather than string arrays.

    Returns:
        scores_per_rep : ndarray of shape (sizematch_reps,) — score per rep
        n_revs_per_rep : ndarray of shape (sizematch_reps,) — reversal count per rep
        sample_size    : int — |U|, the number of genes sampled in each rep
    """
    sample_size = len(U)
    n_reps = int(sizematch_reps)
    if sample_size == 0:
        return np.zeros(n_reps), np.zeros(n_reps, dtype=int), 0

    genes_sorted, is_reversal = _sizematch_reversal_flags(B, pert, dis)
    pool_size = len(genes_sorted)
    if pool_size == 0:
        return np.zeros(n_reps), np.zeros(n_reps, dtype=int), sample_size

    scores_per_rep = np.empty(n_reps)
    n_revs_per_rep = np.empty(n_reps, dtype=int)

    # Pre-compute the SHA-256 state for the fixed prefix (everything except rep index).
    # sha256("A|B|C|D|E|" + str(r)) == sha256("A|B|C|D|E|r") == stable_uint32_seed(A,B,C,D,E,r)
    hash_prefix = hashlib.sha256(
        ("|".join(str(p) for p in (target_gene, cell_id, disease_id, int(k), int(sizematch_seed))) + "|").encode("utf-8")
    )
    for r in range(n_reps):
        h = hash_prefix.copy()
        h.update(str(r).encode("utf-8"))
        seed = int.from_bytes(h.digest()[:4], "little", signed=False)
        rng = np.random.default_rng(seed)
        sampled_idx = rng.choice(pool_size, size=sample_size, replace=False)
        n_rev = int(is_reversal[sampled_idx].sum())
        scores_per_rep[r] = n_rev / sample_size
        n_revs_per_rep[r] = n_rev

    return scores_per_rep, n_revs_per_rep, sample_size


def _score_kg_nosign(U, pert, dis):
    n_meas = len(U)
    if n_meas == 0:
        return 0.0, 0, 0, 0.0, 0.0, 0
    n_rev = sum(1 for g in U if pert.get(g, 0) * dis.get(g, 0) == -1)
    return (n_rev / n_meas), n_rev, n_meas, 0.0, 0.0, 0


def _score_kg_sign_noalign(U, votes, dis):
    n_meas = len(U)
    if n_meas == 0:
        return 0.0, 0, 0, 0.0, 0.0, 0

    n_rev = 0
    n_cov = 0
    for g in U:
        ds = dis.get(g, 0)
        if ds == 0:
            continue
        vv = votes.get(g)
        has_pred = (vv is not None) and ((vv.get(1, 0.0) + vv.get(-1, 0.0)) > 0.0)
        if has_pred:
            n_cov += 1

        w = confidence_weight_from_votes(votes, g, int(-ds))
        n_rev += w

    score = n_rev / n_meas
    covK = n_cov / n_meas
    return score, n_rev, n_meas, covK, 0.0, 0


def score_pair(
    G,
    pert_sigs,
    dis_sigs,
    target_gene,
    cell_id,
    disease_id,
    k,
    method,
    start_sign=-1,
    # For step (1b)
    sizematch_reps=1,
    sizematch_seed=0,
    sizematch_rep=None,
    cache=None,
):
    if not isinstance(G, nx.MultiDiGraph):
        raise TypeError("score_pair: G must be nx.MultiDiGraph.")
    if start_sign not in (-1, 1):
        raise ValueError(f"score_pair: start_sign must be -1 or 1. got {start_sign}")
    pert = pert_sigs.get((target_gene, cell_id), {})
    dis = dis_sigs.get(disease_id, {})
    if len(dis) == 0:
        return 0.0, 0, 0, 0.0, 0.0, 0  # score, n_rev, n_meas, covK, covAlign, support

    # Look up (or build) gene-name sets for fast intersection
    if cache is None:
        dis_keys = set(dis.keys())
        pert_keys = set(pert.keys())
        kg_nodes = set(G.nodes())
    else:
        dis_keys = cache.get("dis_keys", {}).get(disease_id, set(dis.keys()))
        pert_keys = cache.get("pert_keys", {}).get((target_gene, cell_id), set(pert.keys()))
        kg_nodes = cache.get("kg_nodes", set(G.nodes()))

    # B(t,d) = D_d ∩ P_t ∩ V(KG)  — genes in both disease & pert profiles that are KG nodes
    B = dis_keys & pert_keys & kg_nodes

    # U(t,d,k) = N_k(t) ∩ D_d ∩ P_t  — B restricted to the k-hop neighbourhood of t
    U = set()
    if method in ("kg_nosign", "kg_sign_noalign", "proposed", "noKG_sizematch"):
        if target_gene in G:
            if cache is not None:
                nk_store = cache.setdefault("nodes_k", {})
                nodes_k = nk_store.get((target_gene, int(k)))
                if nodes_k is None:
                    nodes_k = khop_nodes(G, target_gene, int(k))
                    nk_store[(target_gene, int(k))] = nodes_k
            else:
                nodes_k = khop_nodes(G, target_gene, int(k))
            U = set(nodes_k) & dis_keys & pert_keys

    # ---- Method dispatch ----
    if method == "noKG":
        return _score_noKG(B, pert, dis)

    if method == "noKG_sizematch":
        return _score_noKG_sizematch(
            B=B,
            U=U,
            pert=pert,
            dis=dis,
            target_gene=target_gene,
            cell_id=cell_id,
            disease_id=disease_id,
            k=k,
            sizematch_reps=int(sizematch_reps),
            sizematch_seed=int(sizematch_seed),
            sizematch_rep=sizematch_rep,
        )

    if method == "kg_nosign":
        return _score_kg_nosign(U, pert, dis)

    # Sign-aware methods require KG-propagated sign votes
    if target_gene not in G:
        return 0.0, 0, 0, 0.0, 0.0, 0

    # votes / pred depend only on (target_gene, k, start_sign) — cache them
    cache_key = (target_gene, int(k), int(start_sign))
    if cache is not None:
        pred_cache = cache.setdefault("pred", {})
        votes_cache = cache.setdefault("votes", {})
        pred = pred_cache.get(cache_key)
        votes = votes_cache.get(cache_key)
        if pred is None or votes is None:
            votes = propagate_sign_votes_shortest_paths(
                G,
                target_gene,
                int(k),
                start_sign=start_sign,
                sign_attr="polarity",
                max_neighbors=None,
                seed=0,
            )
            pred = predicted_sign_sets_from_votes(votes, eps=0.0)
            pred_cache[cache_key] = pred
            votes_cache[cache_key] = votes
    else:
        votes = propagate_sign_votes_shortest_paths(
            G,
            target_gene,
            int(k),
            start_sign=start_sign,
            sign_attr="polarity",
            max_neighbors=None,
            seed=0,
        )
        pred = predicted_sign_sets_from_votes(votes, eps=0.0)

    if method == "kg_sign_noalign":
        return _score_kg_sign_noalign(U, votes, dis)

    if method == "proposed":
        return _score_proposed(U, votes, pert, dis)

    raise ValueError(f"Unknown method: {method}")


def evaluate_ranking_by_query(
    labels: pd.DataFrame,
    scores_df: pd.DataFrame,
    k: int,
    method: str,
    K_list: List[int] = [10, 20, 50],
    tie_policy: str = "expected",
) -> pd.DataFrame:
    """
    query = disease_id
    Note: aggregate over cell_id for evaluation.
    """
    merged = labels.merge(
        scores_df.query("k == @k and method == @method"),
        on=["disease_id", "target_gene", "cell_id"],
        how="inner",
    )

    merged = merged.dropna(subset=["score"])
    merged = (
        merged.groupby(["disease_id", "target_gene"], as_index=False)
        .agg(label=("label", "max"), score=("score", "mean"))
    )

    rows = []
    for disease_id, g in merged.groupby("disease_id"):
        y_true = g["label"].astype(int).tolist()
        y_score = g["score"].astype(float).tolist()

        n_pos = sum(y_true)
        n_neg = len(y_true) - n_pos

        if sum(y_true) == 0:
            continue

        metric_values = _ranking_metrics_for_scores(y_true, y_score, K_list, tie_policy=tie_policy)
        out = {
            "disease_id": disease_id,
            "k": int(k),
            "method": method,
            **metric_values,
        }

        rows.append(out)

    return pd.DataFrame(rows)


def filter_by_min_n_meas_all_methods(
    scores_df: pd.DataFrame,
    methods: List[str],
    min_n_meas: int,
) -> pd.DataFrame:
    sub = scores_df[scores_df["method"].isin(methods)].copy()
    sub["ok"] = sub["n_meas"].fillna(0).astype(int) >= int(min_n_meas)

    piv = sub.pivot_table(
        index=["disease_id", "target_gene", "cell_id", "k"],
        columns="method",
        values="ok",
        aggfunc="any",
        fill_value=False,
    )
    valid = piv[piv.all(axis=1)].index
    keep = sub.set_index(["disease_id", "target_gene", "cell_id", "k"]).index.isin(valid)
    return sub.loc[keep].drop(columns=["ok"])


def run_val2(
    G: nx.MultiDiGraph,
    pert_sigs: Dict[PertKey, Dict[str, int]],
    dis_sigs: Dict[str, Dict[str, int]],
    labels: pd.DataFrame,
    ks: List[int],
    methods: List[str],
    start_sign: int = -1,
    sizematch_reps: int = 100,
    sizematch_seed: int = 0,
    K_list: List[int] = [10, 20, 50],
    tie_policy: str = "expected",
    ncs_scores: Optional[pd.DataFrame] = None,
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not isinstance(G, nx.MultiDiGraph):
        raise TypeError("run_val2: G must be nx.MultiDiGraph.")
    if not isinstance(ks, list) or len(ks) == 0:
        raise ValueError("run_val2: ks must be a non-empty list.")
    if not isinstance(methods, list) or len(methods) == 0:
        raise ValueError("run_val2: methods must be a non-empty list.")
    if start_sign not in (-1, 1):
        raise ValueError(f"run_val2: start_sign must be -1 or 1. got {start_sign}")
    tie_policy = normalize_tie_policy(tie_policy)
    req_labels = {"disease_id", "target_gene", "cell_id", "label"}
    if not req_labels.issubset(labels.columns):
        missing = req_labels - set(labels.columns)
        raise ValueError(f"run_val2: labels DataFrame must have columns {req_labels}. missing: {missing}")

    pairs = labels[["disease_id", "target_gene", "cell_id", "label"]].drop_duplicates()

    if "ncs" in methods and ncs_scores is None:
        raise ValueError(
            "run_val2: methods contains 'ncs' but ncs_scores was not provided."
        )

    # NCS and noKG_sizematch are special methods scored outside the score_pair
    # loop; they are excluded from the deterministic filter so a missing score
    # for one of them cannot drop a pair for the other methods.
    det_methods = [m for m in methods if m not in ("noKG_sizematch", "ncs")]

    # Pre-build caches to avoid redundant computation across pairs / methods / ks.
    # - nodes_k, votes, pred: populated lazily by score_pair
    # - pert_keys, dis_keys: gene-name sets per signature, used for set intersections
    cache = {"nodes_k": {}, "pred": {}, "votes": {}, "kg_nodes": set(G.nodes())}
    cache["pert_keys"] = {k: set(v.keys()) for k, v in pert_sigs.items()}
    cache["dis_keys"] = {k: set(v.keys()) for k, v in dis_sigs.items()}

    # ---- Parallel precompute of khop / votes / pred (n_jobs > 1) ----
    # When n_jobs > 1, all (target, k) entries are pre-filled in cache so that
    # the scoring loop below is pure cache-hit (no recomputation). This is
    # numerically identical to the lazy path because each computation is
    # deterministic (seed=0, no state shared between targets).
    if int(n_jobs) > 1:
        import multiprocessing as mp
        import pickle as _pickle

        unique_targets = sorted(
            set(str(t) for t in pairs[["target_gene"]].drop_duplicates()["target_gene"])
        )
        work_items = [
            (t, int(k), int(start_sign))
            for t in unique_targets
            for k in sorted(set(int(x) for x in ks))
        ]
        if work_items:
            G_pkl = _pickle.dumps(G)
            n_workers = min(int(n_jobs), len(work_items))
            print(f"[PAR] precompute votes/pred/khop: {len(work_items)} tasks, workers={n_workers}")
            with mp.Pool(
                processes=n_workers,
                initializer=_mp_precompute_init,
                initargs=(G_pkl,),
            ) as pool:
                results = pool.map(_mp_precompute_task, work_items)
            for (t, k_r, s0, nodes_k, votes, pred) in results:
                cache["nodes_k"][(t, int(k_r))] = nodes_k
                cache["votes"][(t, int(k_r), int(s0))] = votes
                cache["pred"][(t, int(k_r), int(s0))] = pred
            del G_pkl

    score_rows = []
    pairs_arr = pairs[["disease_id", "target_gene", "cell_id", "label"]].to_numpy(dtype=object)

    for k in tqdm(ks, desc="Scoring pairs", total=len(ks)):
        for method in det_methods:
            for d, t, c, y in pairs_arr:
                d = str(d)
                t = str(t)
                c = str(c)
                y = int(y)

                score, n_rev, n_meas, covK, covAlign, support = score_pair(
                    G,
                    pert_sigs,
                    dis_sigs,
                    target_gene=t,
                    cell_id=c,
                    disease_id=d,
                    k=int(k),
                    method=method,
                    start_sign=start_sign,
                    cache=cache,
                )

                score_rows.append(
                    {
                        "disease_id": d,
                        "target_gene": t,
                        "cell_id": c,
                        "k": int(k),
                        "method": method,
                        "score": score,
                        "n_rev": n_rev,
                        "n_meas": n_meas,
                        "covK": covK,
                        "covAlign": covAlign,
                        "support": support,
                        "pair_label": y,
                    }
                )

    scores_df = pd.DataFrame(score_rows)

    # For the proposed method, n_rev and support carry weighted sums (not counts).
    # Expose them as w_rev / w_sum for clarity; other methods get NaN.
    if len(scores_df) > 0:
        scores_df["w_rev"] = np.where(scores_df["method"] == "proposed", scores_df["n_rev"].astype(float), np.nan)
        scores_df["w_sum"] = np.where(scores_df["method"] == "proposed", scores_df["support"].astype(float), np.nan)

    # Drop pairs that have zero n_meas in *any* deterministic method (min_n_meas=0
    # means only structurally absent pairs are removed).
    if len(scores_df) > 0 and len(det_methods) > 0:
        scores_df = filter_by_min_n_meas_all_methods(scores_df, det_methods, min_n_meas=0)

    # ---- Step (1b): size-matched scoring (single-pass optimisation) ----
    # In val2.py this was two separate passes:
    #   Pass 1 — compute mean score per pair (score_pair with all reps)
    #   Pass 2 — re-compute each rep individually for SEM
    # Here we compute all per-rep scores once via _score_noKG_sizematch_all_reps
    # (or _mp_sizematch_task for the parallel path) and derive both mean scores
    # and per-rep data from the same result.
    #
    # `rep_data_cache` stores per-rep arrays keyed by (disease, target, cell, k)
    # so the SEM section can look them up without re-scoring.

    if "noKG_sizematch" in methods and len(scores_df) > 0:
        sizematch_mean_rows = []
        rep_data_cache: Dict[Tuple[str, str, str, int], Tuple[np.ndarray, np.ndarray, int]] = {}

        kg_nodes_set = cache["kg_nodes"]

        if int(n_jobs) > 1:
            # ---- Parallel size-match path ----
            # Enumerate all (k, pair) work items on the main process:
            #   - zero-dis pairs are handled inline (no worker needed)
            #   - remaining pairs: precompute is_reversal here, send arrays to workers
            # Results are collected in the original k-outer / pair-inner order.

            import multiprocessing as mp

            # Ordered list of records to fill sizematch_mean_rows.
            # Each entry is either a "ready" dict (zero-dis pair) or a sentinel
            # that is replaced by a worker result after pool.map.
            sizematch_order: List[dict] = []
            # Parallel work items: (task_args, index_in_sizematch_order)
            parallel_tasks: List[tuple] = []

            for k in sorted(set(int(x) for x in ks)):
                eligible = scores_df.query("k == @k")[["disease_id", "target_gene", "cell_id", "pair_label"]].drop_duplicates()
                eligible_arr_k = eligible[["disease_id", "target_gene", "cell_id", "pair_label"]].to_numpy(dtype=object)

                for disease_id, target, cell, label in eligible_arr_k:
                    disease_id = str(disease_id)
                    target = str(target)
                    cell = str(cell)
                    label = int(label)

                    pert = pert_sigs.get((target, cell), {})
                    dis = dis_sigs.get(disease_id, {})

                    if len(dis) == 0:
                        sizematch_order.append({
                            "disease_id": disease_id, "target_gene": target, "cell_id": cell,
                            "k": k, "pair_label": label, "_zero": True,
                        })
                        if int(sizematch_reps) >= 2:
                            rep_data_cache[(disease_id, target, cell, k)] = (
                                np.zeros(int(sizematch_reps)),
                                np.zeros(int(sizematch_reps), dtype=int),
                                0,
                            )
                        continue

                    dis_keys = cache["dis_keys"].get(disease_id, set(dis.keys()))
                    pert_keys = cache["pert_keys"].get((target, cell), set(pert.keys()))
                    B = dis_keys & pert_keys & kg_nodes_set
                    U = set()
                    if target in G:
                        nk_store = cache.setdefault("nodes_k", {})
                        nodes_k = nk_store.get((target, k))
                        if nodes_k is None:
                            nodes_k = khop_nodes(G, target, k)
                            nk_store[(target, k)] = nodes_k
                        U = set(nodes_k) & dis_keys & pert_keys

                    genes_sorted, is_reversal = _sizematch_reversal_flags(B, pert, dis)
                    pool_size = len(genes_sorted)
                    sample_size = len(U)

                    idx = len(sizematch_order)
                    sizematch_order.append({
                        "disease_id": disease_id, "target_gene": target, "cell_id": cell,
                        "k": k, "pair_label": label, "_zero": False,
                    })
                    parallel_tasks.append((
                        target, cell, disease_id, k,
                        is_reversal, pool_size, sample_size,
                        int(sizematch_reps), int(sizematch_seed),
                        idx,
                    ))

            # Run workers (returns in input order via pool.map)
            task_args = [t[:9] for t in parallel_tasks]
            task_indices = [t[9] for t in parallel_tasks]
            if task_args:
                n_workers = min(int(n_jobs), len(task_args))
                print(f"[PAR] size-match: {len(task_args)} tasks, workers={n_workers}")
                with mp.Pool(processes=n_workers) as pool:
                    par_results = pool.map(_mp_sizematch_task, task_args)
            else:
                par_results = []

            result_by_idx: Dict[int, tuple] = {
                task_indices[i]: par_results[i] for i in range(len(par_results))
            }

            for slot_idx, slot in enumerate(sizematch_order):
                d_id = slot["disease_id"]
                t_gene = slot["target_gene"]
                cell_id = slot["cell_id"]
                k_slot = slot["k"]
                pair_label = slot["pair_label"]
                if slot["_zero"]:
                    sizematch_mean_rows.append({
                        "disease_id": d_id, "target_gene": t_gene, "cell_id": cell_id,
                        "k": k_slot, "method": "noKG_sizematch",
                        "score": 0.0, "n_rev": 0, "n_meas": 0,
                        "covK": 0.0, "covAlign": 0.0, "support": 0, "pair_label": pair_label,
                    })
                else:
                    res = result_by_idx[slot_idx]
                    _, _, _, k_r, scores_per_rep, n_revs_per_rep, n_meas = res
                    sizematch_mean_rows.append({
                        "disease_id": d_id, "target_gene": t_gene, "cell_id": cell_id,
                        "k": int(k_r), "method": "noKG_sizematch",
                        "score": float(np.mean(scores_per_rep)),
                        "n_rev": float(np.mean(n_revs_per_rep)),
                        "n_meas": n_meas,
                        "covK": 0.0, "covAlign": 0.0, "support": 0, "pair_label": pair_label,
                    })
                    if int(sizematch_reps) >= 2:
                        rep_data_cache[(d_id, t_gene, cell_id, int(k_r))] = (
                            scores_per_rep, n_revs_per_rep, n_meas,
                        )

        else:
            # ---- Sequential size-match path (n_jobs == 1, identical to original) ----
            for k in tqdm(ks, desc="Scoring size-matched (all reps)", total=len(ks)):
                eligible = scores_df.query("k == @k")[["disease_id", "target_gene", "cell_id", "pair_label"]].drop_duplicates()
                eligible_arr = eligible[["disease_id", "target_gene", "cell_id", "pair_label"]].to_numpy(dtype=object)

                for disease_id, target, cell, label in eligible_arr:
                    disease_id = str(disease_id)
                    target = str(target)
                    cell = str(cell)
                    label = int(label)

                    pert = pert_sigs.get((target, cell), {})
                    dis = dis_sigs.get(disease_id, {})

                    if len(dis) == 0:
                        sizematch_mean_rows.append(
                            {
                                "disease_id": disease_id, "target_gene": target, "cell_id": cell,
                                "k": int(k), "method": "noKG_sizematch",
                                "score": 0.0, "n_rev": 0, "n_meas": 0,
                                "covK": 0.0, "covAlign": 0.0, "support": 0, "pair_label": label,
                            }
                        )
                        if int(sizematch_reps) >= 2:
                            rep_data_cache[(disease_id, target, cell, int(k))] = (
                                np.zeros(int(sizematch_reps)),
                                np.zeros(int(sizematch_reps), dtype=int),
                                0,
                            )
                        continue

                    # Reconstruct B and U the same way score_pair does
                    dis_keys = cache["dis_keys"].get(disease_id, set(dis.keys()))
                    pert_keys = cache["pert_keys"].get((target, cell), set(pert.keys()))
                    B = dis_keys & pert_keys & kg_nodes_set
                    U = set()
                    if target in G:
                        nk_store = cache.setdefault("nodes_k", {})
                        nodes_k = nk_store.get((target, int(k)))
                        if nodes_k is None:
                            nodes_k = khop_nodes(G, target, int(k))
                            nk_store[(target, int(k))] = nodes_k
                        U = set(nodes_k) & dis_keys & pert_keys

                    scores_per_rep, n_revs_per_rep, n_meas = _score_noKG_sizematch_all_reps(
                        B=B, U=U, pert=pert, dis=dis,
                        target_gene=target, cell_id=cell, disease_id=disease_id, k=int(k),
                        sizematch_reps=int(sizematch_reps),
                        sizematch_seed=int(sizematch_seed),
                    )

                    sizematch_mean_rows.append(
                        {
                            "disease_id": disease_id, "target_gene": target, "cell_id": cell,
                            "k": int(k), "method": "noKG_sizematch",
                            "score": float(np.mean(scores_per_rep)),
                            "n_rev": float(np.mean(n_revs_per_rep)),
                            "n_meas": n_meas,
                            "covK": 0.0, "covAlign": 0.0, "support": 0, "pair_label": label,
                        }
                    )
                    if int(sizematch_reps) >= 2:
                        rep_data_cache[(disease_id, target, cell, int(k))] = (
                            scores_per_rep, n_revs_per_rep, n_meas,
                        )

        scores_df = pd.concat([scores_df, pd.DataFrame(sizematch_mean_rows)], ignore_index=True)

    # ---- NCS (L1000-NCS reference line): inject precomputed target-level scores ----
    # ncs_scores is a target-level table [disease_id, target_gene, score] computed
    # in val2.ncs (following val2_baseline's WTCS + NCS-normalization algorithm) on
    # val2's own label pairs. NCS is k-independent, so the same target score is
    # replicated across every k and every label cell of the pair; the downstream
    # mean-over-cells in evaluate_ranking_by_query reproduces -mean_ncs exactly and
    # renders NCS as a flat reference line. NCS never enters the deterministic
    # filter above, so it cannot drop pairs for the other methods.
    if "ncs" in methods and ncs_scores is not None and len(ncs_scores) > 0:
        ncs_tgt = ncs_scores[["disease_id", "target_gene", "score"]].copy()
        ncs_tgt["disease_id"] = ncs_tgt["disease_id"].astype(str)
        ncs_tgt["target_gene"] = ncs_tgt["target_gene"].astype(str)

        pairs_str = pairs.copy()
        for col in ("disease_id", "target_gene", "cell_id"):
            pairs_str[col] = pairs_str[col].astype(str)
        merged_pairs = pairs_str.merge(ncs_tgt, on=["disease_id", "target_gene"], how="inner")

        if len(merged_pairs) > 0:
            ncs_frames = []
            for k in ks:
                f = merged_pairs.rename(columns={"label": "pair_label"}).copy()
                f["k"] = int(k)
                f["method"] = "ncs"
                f["score"] = f["score"].astype(float)
                f["n_rev"] = np.nan
                f["n_meas"] = 0
                f["covK"] = 0.0
                f["covAlign"] = 0.0
                f["support"] = 0
                ncs_frames.append(f)
            scores_df = pd.concat([scores_df] + ncs_frames, ignore_index=True)

    # ---- Ranking metrics (AP, P@K, Lift@K per disease) ----
    metric_rows = []
    sem_cols = ["AP"] + [f"P@{K}" for K in K_list] + [f"Lift@{K}" for K in K_list]

    # Deterministic methods produce a single score per pair, so SEM is undefined.
    for k in ks:
        for method in det_methods:
            md = evaluate_ranking_by_query(
                labels, scores_df, k=int(k), method=method, K_list=K_list, tie_policy=tie_policy
            )
            for col in sem_cols:
                md[f"{col}_SEM"] = np.nan
            metric_rows.append(md)

    # NCS is a single deterministic score per pair (SEM undefined), evaluated like
    # the deterministic methods above.
    if "ncs" in methods and ncs_scores is not None:
        for k in ks:
            md = evaluate_ranking_by_query(
                labels, scores_df, k=int(k), method="ncs", K_list=K_list, tie_policy=tie_policy
            )
            for col in sem_cols:
                md[f"{col}_SEM"] = np.nan
            metric_rows.append(md)

    # ---- Step (1b) SEM: evaluate per-rep ranking metrics from cached rep data ----
    # Instead of calling evaluate_ranking_by_query once per (k, rep) — which
    # creates and merges a new DataFrame each time — we build a single long-form
    # DataFrame for all reps at each k and merge with labels just once.

    if "noKG_sizematch" in methods and len(scores_df) > 0 and int(sizematch_reps) >= 2:
        n_reps = int(sizematch_reps)
        rep_metric_rows = []

        for k in ks:
            eligible = scores_df.query("k == @k and method in @det_methods")[["disease_id", "target_gene", "cell_id"]].drop_duplicates()
            eligible_arr = eligible[["disease_id", "target_gene", "cell_id"]].to_numpy(dtype=object)
            n_eligible = len(eligible_arr)
            total_rows = n_eligible * n_reps

            # Pre-allocate flat arrays for the long-form DataFrame
            disease_ids = np.empty(total_rows, dtype=object)
            target_genes = np.empty(total_rows, dtype=object)
            cell_ids = np.empty(total_rows, dtype=object)
            rep_scores = np.empty(total_rows, dtype=np.float64)
            rep_indices = np.empty(total_rows, dtype=np.int32)
            rep_range = np.arange(n_reps, dtype=np.int32)

            for pair_idx, (disease_id, target, cell) in enumerate(eligible_arr):
                disease_id, target, cell = str(disease_id), str(target), str(cell)
                cached = rep_data_cache.get((disease_id, target, cell, int(k)))
                scores_arr = cached[0] if cached is not None else np.zeros(n_reps)
                start = pair_idx * n_reps
                end = start + n_reps
                disease_ids[start:end] = disease_id
                target_genes[start:end] = target
                cell_ids[start:end] = cell
                rep_scores[start:end] = scores_arr
                rep_indices[start:end] = rep_range

            all_reps_df = pd.DataFrame({
                "disease_id": disease_ids, "target_gene": target_genes, "cell_id": cell_ids,
                "score": rep_scores, "rep": rep_indices,
            })

            # Single merge for all reps at this k
            merged = labels.merge(all_reps_df, on=["disease_id", "target_gene", "cell_id"], how="inner")
            merged = merged.dropna(subset=["score"])
            merged = merged.groupby(["disease_id", "target_gene", "rep"], as_index=False).agg(
                label=("label", "max"), score=("score", "mean"),
            )

            # Compute ranking metrics per (rep, disease)
            for rep, rep_group in merged.groupby("rep"):
                metric_entries = []
                for disease_id, grp in rep_group.groupby("disease_id"):
                    y_true = grp["label"].astype(int).tolist()
                    y_score = grp["score"].astype(float).tolist()
                    n_pos = sum(y_true)
                    if n_pos == 0:
                        continue
                    metric_values = _ranking_metrics_for_scores(y_true, y_score, K_list, tie_policy=tie_policy)
                    entry = {
                        "disease_id": disease_id, "k": int(k), "method": "noKG_sizematch",
                        **metric_values,
                    }
                    metric_entries.append(entry)
                md = pd.DataFrame(metric_entries)
                md["rep"] = int(rep)
                rep_metric_rows.append(md)

        rep_metrics = pd.concat(rep_metric_rows, ignore_index=True) if rep_metric_rows else pd.DataFrame()
        if len(rep_metrics) > 0:

            def _sem(x: pd.Series) -> float:
                n = int(x.count())
                if n <= 1:
                    return 0.0
                return float(x.std(ddof=1) / np.sqrt(n))

            group_keys = ["disease_id", "k", "method"]
            mean_df = rep_metrics.groupby(group_keys, as_index=False).mean(numeric_only=True)
            sem_df = rep_metrics.groupby(group_keys)[sem_cols].agg(_sem).reset_index()
            sem_df = sem_df.rename(columns={c: f"{c}_SEM" for c in sem_cols})

            base_df = mean_df.merge(sem_df, on=group_keys, how="left").drop(columns=["rep"], errors="ignore")
            base_df["tie_policy"] = tie_policy
            metric_rows.append(base_df)

    metrics_df = pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()

    # ---- Predicted-sign ambiguity summary (per target_gene, k) ----
    amb_rows = []
    _pred_items = cache.get("pred", {})
    for (t, kk, s0), pred_map in tqdm(_pred_items.items(), desc="Predicted-sign ambiguity", total=len(_pred_items)):
        if int(s0) != int(start_sign):
            continue
        n_pred = int(len(pred_map))
        n_single = int(sum(1 for ss in pred_map.values() if len(ss) == 1))
        n_multi = int(sum(1 for ss in pred_map.values() if len(ss) >= 2))
        amb_rows.append(
            {
                "target_gene": str(t),
                "k": int(kk),
                "start_sign": int(s0),
                "n_predicted_nodes": n_pred,
                "n_single_sign": n_single,
                "n_multi_sign": n_multi,
                "frac_multi": (n_multi / n_pred) if n_pred > 0 else np.nan,
            }
        )
    pred_ambiguity_df = pd.DataFrame(amb_rows).sort_values(["target_gene", "k"]) if amb_rows else pd.DataFrame()

    return scores_df, metrics_df, pred_ambiguity_df


def export_proposed_w_values(
    *,
    G: nx.MultiDiGraph,
    pert_sigs: Dict[PertKey, Dict[str, int]],
    dis_sigs: Dict[str, Dict[str, int]],
    labels_df: pd.DataFrame,
    ks: List[int],
    out_path: Path,
    start_sign: int = -1,
) -> None:
    """Export per-gene confidence weights w(g) used by the proposed method.

    Output columns:
      disease_id, target_gene, cell_id, k, gene, w, pert_sign, dis_sign, is_reversal
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache = {"nodes_k": {}, "votes": {}, "kg_nodes": set(G.nodes())}
    cache["pert_keys"] = {k: set(v.keys()) for k, v in pert_sigs.items()}
    cache["dis_keys"] = {k: set(v.keys()) for k, v in dis_sigs.items()}

    pairs = labels_df[["disease_id", "target_gene", "cell_id"]].drop_duplicates()
    pairs_arr = pairs[["disease_id", "target_gene", "cell_id"]].to_numpy(dtype=object)

    rows = []
    for k in ks:
        k = int(k)
        for d, t, c in pairs_arr:
            d = str(d)
            t = str(t)
            c = str(c)

            pert = pert_sigs.get((t, c), {})
            dis = dis_sigs.get(d, {})
            if len(dis) == 0 or t not in G:
                continue

            disK = cache["dis_keys"].get(d, set(dis.keys()))
            pertK = cache["pert_keys"].get((t, c), set(pert.keys()))

            # U(t,d,k)=N_k(t)∩D_d∩P_t
            nk_cache = cache.setdefault("nodes_k", {})
            nodes_k = nk_cache.get((t, k))
            if nodes_k is None:
                nodes_k = khop_nodes(G, t, k)
                nk_cache[(t, k)] = nodes_k
            U = set(nodes_k) & disK & pertK
            if len(U) == 0:
                continue

            # votes cache (target_gene, k, start_sign)
            vkey = (t, k, int(start_sign))
            votes_cache = cache.setdefault("votes", {})
            votes = votes_cache.get(vkey)
            if votes is None:
                votes = propagate_sign_votes_shortest_paths(
                    G,
                    t,
                    k,
                    start_sign=int(start_sign),
                    sign_attr="polarity",
                    max_neighbors=None,
                    seed=0,
                )
                votes_cache[vkey] = votes

            for g in U:
                ps = int(pert.get(g, 0))
                ds = int(dis.get(g, 0))
                w = float(confidence_weight_from_votes(votes, g, ps))
                rows.append(
                    {
                        "disease_id": d,
                        "target_gene": t,
                        "cell_id": c,
                        "k": k,
                        "gene": str(g),
                        "w": w,
                        "pert_sign": ps,
                        "dis_sign": ds,
                        "is_reversal": int(ps * ds == -1),
                    }
                )

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[write] {out_path}")


# -----------------------------
# val2_2 permutation sanity checks (optional)
# -----------------------------

def permute_signs_preserving_counts(sig: Dict[str, int], rng: np.random.Generator) -> Dict[str, int]:
    """Shuffle the sign assignments across genes while preserving the multiset of values in `sig`."""
    _VALID = {-1, 0, 1}
    for g, v in sig.items():
        if v not in _VALID:
            raise ValueError(
                f"permute_signs_preserving_counts: signature values must be -1, 0, or 1. got gene={g!r} value={v}"
            )
    genes = sorted(sig.keys())
    if not genes:
        return {}
    signs = np.array([sig[g] for g in genes], dtype=int)
    rng.shuffle(signs)
    return {g: int(s) for g, s in zip(genes, signs)}


def shuffle_kg_polarity_across_edges(G: nx.MultiDiGraph, rng: np.random.Generator) -> nx.MultiDiGraph:
    """Shuffle edge polarity values across edges while keeping graph structure fixed."""
    edges_pols = []
    for u, v, key, data in G.edges(keys=True, data=True):
        raw = data.get("polarity")
        if raw is None:
            raise ValueError(
                f"shuffle_kg_polarity_across_edges: edge ({u!r}, {v!r}, key={key}) missing 'polarity'."
            )
        pol = int(raw)
        if pol not in (-1, 1):
            raise ValueError(
                f"shuffle_kg_polarity_across_edges: edge ({u!r}, {v!r}) polarity must be -1 or 1. got {pol}"
            )
        edges_pols.append((str(u), str(v), int(key), pol))
    # Stable ordering so the shuffle is fully reproducible given the same RNG seed.
    edges_pols.sort(key=lambda x: (x[0], x[1], x[2]))

    edges = [(u, v, k) for (u, v, k, _) in edges_pols]
    pols = np.array([p for (_, _, _, p) in edges_pols], dtype=int)
    rng.shuffle(pols)

    G_shuf = G.copy()
    for (u, v, key), pol in zip(edges, pols):
        G_shuf[u][v][key]["polarity"] = int(1 if int(pol) > 0 else -1)
    return G_shuf


def _build_score_cache(
    G: nx.MultiDiGraph,
    pert_sigs: Dict[PertKey, Dict[str, int]],
    dis_sigs: Dict[str, Dict[str, int]],
) -> dict:
    cache = {"nodes_k": {}, "pred": {}, "votes": {}, "kg_nodes": set(G.nodes())}
    cache["pert_keys"] = {k: set(v.keys()) for k, v in pert_sigs.items()}
    cache["dis_keys"] = {k: set(v.keys()) for k, v in dis_sigs.items()}
    return cache


# ---- Parallel precompute workers (n_jobs > 1) ----
# Module-level slot populated by Pool initializer via pickle.
_mp_worker_G: Optional[nx.MultiDiGraph] = None


def _mp_precompute_init(G_pkl: bytes) -> None:
    """Pool initializer: unpickle G once per worker process."""
    import pickle
    global _mp_worker_G
    _mp_worker_G = pickle.loads(G_pkl)


def _mp_precompute_task(args: Tuple) -> Tuple:
    """Worker: compute khop_nodes + votes + pred for one (target, k, start_sign).

    Returns (target, k, start_sign, nodes_k, votes, pred).
    votes is returned as a plain dict (not defaultdict) so it is picklable.
    """
    target, k, start_sign = args
    G = _mp_worker_G
    if G is None or target not in G:
        return (target, k, start_sign, set(), {}, {})
    nodes_k = khop_nodes(G, target, int(k))
    votes_raw = propagate_sign_votes_shortest_paths(
        G, target, int(k),
        start_sign=int(start_sign),
        sign_attr="polarity",
        max_neighbors=None,
        seed=0,
    )
    # defaultdict with lambda factory is not picklable; convert to plain dict.
    votes = {node: dict(v) for node, v in votes_raw.items()}
    pred = predicted_sign_sets_from_votes(votes, eps=0.0)
    return (target, k, start_sign, nodes_k, votes, pred)


def _mp_sizematch_task(args: Tuple) -> Tuple:
    """Worker: all-reps size-match sampling for one (disease, target, cell, k).

    args = (target_gene, cell_id, disease_id, k,
             is_reversal, pool_size, sample_size,
             n_reps, sizematch_seed)
    Returns (target_gene, cell_id, disease_id, k,
              scores_per_rep, n_revs_per_rep, sample_size).
    """
    (target_gene, cell_id, disease_id, k,
     is_reversal, pool_size, sample_size,
     n_reps, sizematch_seed) = args
    n_reps = int(n_reps)
    sample_size_i = int(sample_size)
    pool_size_i = int(pool_size)

    scores_per_rep = np.empty(n_reps)
    n_revs_per_rep = np.empty(n_reps, dtype=int)

    if sample_size_i == 0 or pool_size_i == 0:
        scores_per_rep[:] = 0.0
        n_revs_per_rep[:] = 0
        return (target_gene, cell_id, disease_id, int(k),
                scores_per_rep, n_revs_per_rep, sample_size_i)

    hash_prefix = hashlib.sha256(
        ("|".join(str(p) for p in (
            target_gene, cell_id, disease_id, int(k), int(sizematch_seed)
        )) + "|").encode("utf-8")
    )
    for r in range(n_reps):
        h = hash_prefix.copy()
        h.update(str(r).encode("utf-8"))
        seed_val = int.from_bytes(h.digest()[:4], "little", signed=False)
        rng = np.random.default_rng(seed_val)
        sampled_idx = rng.choice(pool_size_i, size=sample_size_i, replace=False)
        n_rev = int(is_reversal[sampled_idx].sum())
        scores_per_rep[r] = n_rev / sample_size_i
        n_revs_per_rep[r] = n_rev

    return (target_gene, cell_id, disease_id, int(k),
            scores_per_rep, n_revs_per_rep, sample_size_i)


def _compute_scores_for_pairs_single_k(
    *,
    G: nx.MultiDiGraph,
    pert_sigs: Dict[PertKey, Dict[str, int]],
    dis_sigs: Dict[str, Dict[str, int]],
    labels_df: pd.DataFrame,
    k: int,
    method: str,
    start_sign: int,
    cache: Optional[dict] = None,
) -> pd.DataFrame:
    """Compute pair-level scores for a single k.

    Returned columns:
      disease_id, target_gene, cell_id, score
    """
    k = int(k)

    pairs = labels_df[["disease_id", "target_gene", "cell_id"]].drop_duplicates()
    pairs_arr = pairs[["disease_id", "target_gene", "cell_id"]].to_numpy(dtype=object)

    rows = []
    for d, t, c in pairs_arr:
        d = str(d)
        t = str(t)
        c = str(c)

        score, *_ = score_pair(
            G,
            pert_sigs,
            dis_sigs,
            target_gene=t,
            cell_id=c,
            disease_id=d,
            k=int(k),
            method=method,
            start_sign=int(start_sign),
            cache=cache,
        )
        rows.append({"disease_id": d, "target_gene": t, "cell_id": c, "score": float(score)})

    if len(rows) == 0:
        return pd.DataFrame(columns=["disease_id", "target_gene", "cell_id", "score"])
    return pd.DataFrame(rows)


def _compute_mean_ap_and_lift10_from_pair_scores(
    labels_df: pd.DataFrame,
    pair_scores_df: pd.DataFrame,
    tie_policy: str = "expected",
) -> Tuple[float, float]:
    """Return (mean AP, mean Lift@10) over diseases for a pair-level score table."""
    tie_policy = normalize_tie_policy(tie_policy)
    merged = labels_df.merge(pair_scores_df, on=["disease_id", "target_gene", "cell_id"], how="inner")
    merged = (
        merged.groupby(["disease_id", "target_gene"], as_index=False)
        .agg(label=("label", "max"), score=("score", "mean"))
    )

    ap_list = []
    lift_list = []
    for disease_id, g in merged.groupby("disease_id"):
        y_true = g["label"].astype(int).tolist()
        y_score = g["score"].astype(float).tolist()
        if sum(y_true) == 0:
            continue

        metric_values = _ranking_metrics_for_scores(y_true, y_score, [10], tie_policy=tie_policy)
        ap = float(metric_values["AP"])
        lift10 = float(metric_values["Lift@10"])

        ap_list.append(float(ap))
        lift_list.append(float(lift10))

    if len(ap_list) == 0:
        return float("nan"), float("nan")

    return float(np.mean(ap_list)), float(np.mean(lift_list))


def _compute_per_disease_ap_lift10_from_pair_scores(
    labels_df: pd.DataFrame,
    pair_scores_df: pd.DataFrame,
    tie_policy: str = "expected",
) -> pd.DataFrame:
    """Return per-disease AP and Lift@10 for a pair-level score table.

    Returns DataFrame with columns: disease_id, AP, Lift@10
    """
    tie_policy = normalize_tie_policy(tie_policy)
    merged = labels_df.merge(pair_scores_df, on=["disease_id", "target_gene", "cell_id"], how="inner")
    merged = (
        merged.groupby(["disease_id", "target_gene"], as_index=False)
        .agg(label=("label", "max"), score=("score", "mean"))
    )

    rows = []
    for disease_id, g in merged.groupby("disease_id"):
        y_true = g["label"].astype(int).tolist()
        y_score = g["score"].astype(float).tolist()
        if sum(y_true) == 0:
            continue

        metric_values = _ranking_metrics_for_scores(y_true, y_score, [10], tie_policy=tie_policy)
        ap = float(metric_values["AP"])
        lift10 = float(metric_values["Lift@10"])

        rows.append({
            "disease_id": str(disease_id),
            "AP": float(ap),
            "Lift@10": float(lift10),
            "tie_policy": tie_policy,
        })

    if len(rows) == 0:
        return pd.DataFrame(columns=["disease_id", "AP", "Lift@10", "tie_policy"])
    return pd.DataFrame(rows)


def run_val2_2_permutation_shuffles(
    *,
    G: nx.MultiDiGraph,
    pert_sigs: Dict[PertKey, Dict[str, int]],
    dis_sigs: Dict[str, Dict[str, int]],
    labels_df: pd.DataFrame,
    out_dir: Path,
    k_band: List[int],
    start_sign: int = -1,
    n_perm: int = 30,
    seed: int = 0,
    tie_policy: str = "expected",
) -> pd.DataFrame:
    """Run 3 permutation sanity checks (disease sign, pert sign, KG sign) and save results.

    Writes:
      - val2_2_permutation_summary.csv: one row per (perm_type, perm_idx, k_star).
        Columns: perm_type, perm_idx, k_star, k_band, mAP, mean_Lift@10, mAP_base, mean_Lift@10_base
      - val2_2_permutation_summary_by_disease.csv: one row per (perm_type, perm_idx, k_star, disease_id).
        Columns: perm_type, perm_idx, k_star, k_band, disease_id, AP, Lift@10, AP_base, Lift@10_base
    """
    if not isinstance(G, nx.MultiDiGraph):
        raise TypeError("run_val2_2_permutation_shuffles: G must be nx.MultiDiGraph.")
    if start_sign not in (-1, 1):
        raise ValueError(f"run_val2_2_permutation_shuffles: start_sign must be -1 or 1. got {start_sign}")
    tie_policy = normalize_tie_policy(tie_policy)
    req_labels = {"disease_id", "target_gene", "cell_id"}
    if not req_labels.issubset(labels_df.columns):
        missing = req_labels - set(labels_df.columns)
        raise ValueError(f"run_val2_2_permutation_shuffles: labels_df must have columns {req_labels}. missing: {missing}")

    n_perm = int(n_perm)
    if n_perm <= 0:
        print("[perm] skipped (n_perm_val2_2 <= 0)")
        return pd.DataFrame()

    k_band = [int(k) for k in k_band]
    if len(k_band) == 0:
        raise ValueError("k_band must be non-empty")
    # Stable representation in outputs.
    k_band_sorted = sorted(dict.fromkeys(k_band))
    k_band_str = ",".join(str(k) for k in k_band_sorted)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Cache KG-dependent quantities once (for disease/pert shuffles).
    base_cache = _build_score_cache(G, pert_sigs, dis_sigs)
    targets = sorted(labels_df["target_gene"].dropna().astype(str).unique().tolist())
    nk_cache = base_cache.setdefault("nodes_k", {})
    for t in targets:
        if t in G:
            for k in k_band_sorted:
                nk_cache[(t, int(k))] = khop_nodes(G, t, int(k))

    # Base (unpermuted) scores/metrics for reference (computed per k).
    base_metrics: Dict[int, Tuple[float, float]] = {}
    base_per_disease: Dict[int, pd.DataFrame] = {}
    for k in tqdm(k_band_sorted, desc="Base (unpermuted)", total=len(k_band_sorted)):
        base_pair_scores = _compute_scores_for_pairs_single_k(
            G=G,
            pert_sigs=pert_sigs,
            dis_sigs=dis_sigs,
            labels_df=labels_df,
            k=int(k),
            method="proposed",
            start_sign=int(start_sign),
            cache=base_cache,
        )
        base_map, base_lift = _compute_mean_ap_and_lift10_from_pair_scores(
            labels_df, base_pair_scores, tie_policy=tie_policy
        )
        base_metrics[int(k)] = (float(base_map), float(base_lift))
        base_per_disease[int(k)] = _compute_per_disease_ap_lift10_from_pair_scores(
            labels_df, base_pair_scores, tie_policy=tie_policy
        )

    perm_rows = []
    perm_by_disease_rows: List[dict] = []

    # --- disease sign shuffle ---
    for i in tqdm(range(n_perm), desc="disease_sign_shuffle", total=n_perm):
        rng_i = np.random.default_rng(stable_uint32_seed(int(seed), "disease_sign_shuffle", int(i)))
        dis_perm = {d: permute_signs_preserving_counts(dis_sigs[d], rng_i) for d in sorted(dis_sigs.keys())}

        for k in k_band_sorted:
            perm_pair_scores = _compute_scores_for_pairs_single_k(
                G=G,
                pert_sigs=pert_sigs,
                dis_sigs=dis_perm,
                labels_df=labels_df,
                k=int(k),
                method="proposed",
                start_sign=int(start_sign),
                cache=base_cache,
            )
            mAP, mLift = _compute_mean_ap_and_lift10_from_pair_scores(
                labels_df, perm_pair_scores, tie_policy=tie_policy
            )
            base_map, base_lift = base_metrics[int(k)]
            perm_rows.append(
                {
                    "perm_type": "disease_sign_shuffle",
                    "perm_idx": int(i),
                    "k_star": int(k),
                    "k_band": k_band_str,
                    "mAP": float(mAP),
                    "mean_Lift@10": float(mLift),
                    "mAP_base": float(base_map),
                    "mean_Lift@10_base": float(base_lift),
                }
            )
            per_disease_df = _compute_per_disease_ap_lift10_from_pair_scores(
                labels_df, perm_pair_scores, tie_policy=tie_policy
            )
            base_dis = base_per_disease[int(k)].rename(columns={"AP": "AP_base", "Lift@10": "Lift@10_base"})
            for _, row in per_disease_df.merge(base_dis, on="disease_id", how="inner").iterrows():
                perm_by_disease_rows.append(
                    {
                        "perm_type": "disease_sign_shuffle",
                        "perm_idx": int(i),
                        "k_star": int(k),
                        "k_band": k_band_str,
                        "disease_id": row["disease_id"],
                        "AP": float(row["AP"]),
                        "Lift@10": float(row["Lift@10"]),
                        "AP_base": float(row["AP_base"]),
                        "Lift@10_base": float(row["Lift@10_base"]),
                    }
                )

# --- perturbation sign shuffle ---
    for i in tqdm(range(n_perm), desc="pert_sign_shuffle", total=n_perm):
        rng_i = np.random.default_rng(stable_uint32_seed(int(seed), "pert_sign_shuffle", int(i)))
        pert_perm = {k: permute_signs_preserving_counts(pert_sigs[k], rng_i) for k in sorted(pert_sigs.keys())}

        for k in k_band_sorted:
            perm_pair_scores = _compute_scores_for_pairs_single_k(
                G=G,
                pert_sigs=pert_perm,
                dis_sigs=dis_sigs,
                labels_df=labels_df,
                k=int(k),
                method="proposed",
                start_sign=int(start_sign),
                cache=base_cache,
            )
            mAP, mLift = _compute_mean_ap_and_lift10_from_pair_scores(
                labels_df, perm_pair_scores, tie_policy=tie_policy
            )
            base_map, base_lift = base_metrics[int(k)]
            perm_rows.append(
                {
                    "perm_type": "pert_sign_shuffle",
                    "perm_idx": int(i),
                    "k_star": int(k),
                    "k_band": k_band_str,
                    "mAP": float(mAP),
                    "mean_Lift@10": float(mLift),
                    "mAP_base": float(base_map),
                    "mean_Lift@10_base": float(base_lift),
                }
            )
            per_disease_df = _compute_per_disease_ap_lift10_from_pair_scores(
                labels_df, perm_pair_scores, tie_policy=tie_policy
            )
            base_dis = base_per_disease[int(k)].rename(columns={"AP": "AP_base", "Lift@10": "Lift@10_base"})
            for _, row in per_disease_df.merge(base_dis, on="disease_id", how="inner").iterrows():
                perm_by_disease_rows.append(
                    {
                        "perm_type": "pert_sign_shuffle",
                        "perm_idx": int(i),
                        "k_star": int(k),
                        "k_band": k_band_str,
                        "disease_id": row["disease_id"],
                        "AP": float(row["AP"]),
                        "Lift@10": float(row["Lift@10"]),
                        "AP_base": float(row["AP_base"]),
                        "Lift@10_base": float(row["Lift@10_base"]),
                    }
                )

# --- KG sign shuffle ---
    for i in tqdm(range(n_perm), desc="kg_sign_shuffle", total=n_perm):
        rng_i = np.random.default_rng(stable_uint32_seed(int(seed), "kg_sign_shuffle", int(i)))
        kg_shuf = shuffle_kg_polarity_across_edges(G, rng_i)

        cache_shuf = _build_score_cache(kg_shuf, pert_sigs, dis_sigs)
        # Reuse k-hop neighborhoods (structure fixed).
        cache_shuf["nodes_k"] = dict(nk_cache)

        for k in k_band_sorted:
            perm_pair_scores = _compute_scores_for_pairs_single_k(
                G=kg_shuf,
                pert_sigs=pert_sigs,
                dis_sigs=dis_sigs,
                labels_df=labels_df,
                k=int(k),
                method="proposed",
                start_sign=int(start_sign),
                cache=cache_shuf,
            )
            mAP, mLift = _compute_mean_ap_and_lift10_from_pair_scores(
                labels_df, perm_pair_scores, tie_policy=tie_policy
            )
            base_map, base_lift = base_metrics[int(k)]
            perm_rows.append(
                {
                    "perm_type": "kg_sign_shuffle",
                    "perm_idx": int(i),
                    "k_star": int(k),
                    "k_band": k_band_str,
                    "mAP": float(mAP),
                    "mean_Lift@10": float(mLift),
                    "mAP_base": float(base_map),
                    "mean_Lift@10_base": float(base_lift),
                }
            )
            per_disease_df = _compute_per_disease_ap_lift10_from_pair_scores(
                labels_df, perm_pair_scores, tie_policy=tie_policy
            )
            base_dis = base_per_disease[int(k)].rename(columns={"AP": "AP_base", "Lift@10": "Lift@10_base"})
            for _, row in per_disease_df.merge(base_dis, on="disease_id", how="inner").iterrows():
                perm_by_disease_rows.append(
                    {
                        "perm_type": "kg_sign_shuffle",
                        "perm_idx": int(i),
                        "k_star": int(k),
                        "k_band": k_band_str,
                        "disease_id": row["disease_id"],
                        "AP": float(row["AP"]),
                        "Lift@10": float(row["Lift@10"]),
                        "AP_base": float(row["AP_base"]),
                        "Lift@10_base": float(row["Lift@10_base"]),
                    }
                )

    perm_df = pd.DataFrame(perm_rows)
    out_path = out_dir / "val2_2_permutation_summary.csv"
    perm_df.to_csv(out_path, index=False)
    print(f"[write] {out_path}")

    if perm_by_disease_rows:
        by_disease_df = pd.DataFrame(perm_by_disease_rows)
        by_disease_path = out_dir / "val2_2_permutation_summary_by_disease.csv"
        by_disease_df.to_csv(by_disease_path, index=False)
        print(f"[write] {by_disease_path}")

    return perm_df


def load_config(path: str) -> dict:
    """Load a config file (YAML/JSON) and return a dict.

    Supported extensions:
      - .json
      - .yml / .yaml (requires PyYAML)

    All parameters are read from the config file.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    ext = p.suffix.lower()
    if ext == ".json":
        with p.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg is None:
            raise ValueError("Config file must contain a single mapping (dict). Got null.")
        if not isinstance(cfg, dict):
            raise ValueError(f"Config file must contain a single mapping (dict). Got {type(cfg).__name__}.")
        return cfg

    if ext in (".yml", ".yaml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise ImportError(
                "YAML config requires PyYAML. Install with: pip install pyyaml"
            ) from e
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if cfg is None:
            raise ValueError("Config file must contain a single mapping (dict). Got null/empty.")
        if not isinstance(cfg, dict):
            raise ValueError(f"Config file must contain a single mapping (dict). Got {type(cfg).__name__}.")
        return cfg

    raise ValueError(f"Unsupported config extension: {ext} (use .json/.yml/.yaml)")

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Run Val2 scoring/evaluation using a config file (.yml/.yaml/.json). "
            "All parameters are loaded from the config file (no CLI overrides)."
        )
    )
    p.add_argument("--config", type=str, required=True, help="config file path (.yml/.yaml/.json)")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config)

    def req(key: str):
        """Fetch a required key from config."""
        if key not in cfg:
            raise ValueError(f"Missing required key in config: '{key}'")
        return cfg[key]

    # Required inputs
    parquet_path = str(req("parquet"))
    ppi_path = str(req("ppi"))
    if not Path(parquet_path).exists():
        raise FileNotFoundError(f"Config 'parquet' file not found: {parquet_path}")
    if not Path(ppi_path).exists():
        raise FileNotFoundError(f"Config 'ppi' file not found: {ppi_path}")

    out_dir = Path(str(req("out_dir")))
    out_dir.mkdir(parents=True, exist_ok=True)

    pert_suffix = str(req("pert_suffix"))
    disease_suffix = str(req("disease_suffix"))

    methods = req("methods")
    ks = req("ks")
    start_sign_raw = req("start_sign")
    try:
        start_sign = int(start_sign_raw)
    except (TypeError, ValueError) as e:
        raise TypeError("Config key 'start_sign' must be an int (-1 or 1).") from e
    if start_sign not in (-1, 1):
        raise ValueError(f"Config key 'start_sign' must be -1 or 1. got {start_sign}")
    K_list = req("K_list")
    analysis_cfg = cfg.get("analysis", {}) if isinstance(cfg.get("analysis"), dict) else {}
    tie_policy = normalize_tie_policy(cfg.get("tie_policy", analysis_cfg.get("tie_policy", "expected")))

    if not isinstance(methods, list):
        raise TypeError("Config key 'methods' must be a list.")
    if not isinstance(ks, list):
        raise TypeError("Config key 'ks' must be a list.")
    if not isinstance(K_list, list):
        raise TypeError("Config key 'K_list' must be a list.")
    if len(methods) == 0:
        raise ValueError("Config key 'methods' must be a non-empty list.")
    if len(ks) == 0:
        raise ValueError("Config key 'ks' must be a non-empty list.")
    if len(K_list) == 0:
        raise ValueError("Config key 'K_list' must be a non-empty list.")

    methods = [str(m) for m in methods]
    ks = [int(k) for k in ks]
    K_list = [int(k) for k in K_list]

    # Size-matched params (required if method enabled)
    sizematch_reps = int(req("sizematch_reps")) if "sizematch_reps" in cfg else 100

    # Reproducibility seed (used for all stochastic components)
    seed_base = None
    if isinstance(cfg.get("reproducibility"), dict) and "seed" in cfg.get("reproducibility", {}):
        seed_base = int(cfg["reproducibility"]["seed"])
    elif "seed" in cfg:
        seed_base = int(cfg["seed"])
    elif "sizematch_seed" in cfg:
        seed_base = int(cfg["sizematch_seed"])
    else:
        raise ValueError("Missing reproducibility seed: set 'seed' or 'reproducibility.seed' in config")

    random.seed(int(seed_base))
    np.random.seed(int(seed_base))

    sizematch_seed = int(cfg.get("sizematch_seed", seed_base))

    # 1) load KG
    G = load_signed_multidigraph_from_tsv(ppi_path, src_col="source", dst_col="target", sign_col="polarity", sep="\t")
    print(f"[KG] nodes={G.number_of_nodes():,}, edges={G.number_of_edges():,}")

    # 2) load parquet -> labels & signatures
    labels_df, pert_sigs, dis_sigs, common_genes = load_knockdown_parquet_as_dicts(
        parquet_path,
        pert_suffix=pert_suffix,
        disease_suffix=disease_suffix,
        kg_nodes=set(G.nodes()),
    )
    print(
        f"[DATA] pairs={len(labels_df):,}  pert_sigs={len(pert_sigs):,}  dis_sigs={len(dis_sigs):,}  common_genes={len(common_genes):,}"
    )

    # 3) run Val2
    scores_df, metrics_df, pred_ambiguity_df = run_val2(
        G,
        pert_sigs=pert_sigs,
        dis_sigs=dis_sigs,
        labels=labels_df,
        ks=ks,
        methods=methods,
        start_sign=start_sign,
        sizematch_reps=int(sizematch_reps),
        sizematch_seed=int(sizematch_seed),
        K_list=K_list,
        tie_policy=tie_policy,
    )

    k_min, k_max = min(ks), max(ks)
    k_tag = f"k{k_min}-{k_max}"
    scores_path = out_dir / f"val2_scores_{k_tag}.csv"
    metrics_path = out_dir / f"val2_metrics_{k_tag}.csv"
    pred_amb_path = out_dir / f"val2_pred_sign_ambiguity_{k_tag}.csv"
    scores_df.to_csv(scores_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    pred_ambiguity_df.to_csv(pred_amb_path, index=False)

    # 4) optional: val2_2 permutation sanity checks (3 shuffles)
    n_perm_val2_2 = 0
    if "n_perm_val2_2" in cfg:
        n_perm_val2_2 = int(cfg.get("n_perm_val2_2", 0))
    elif "n_perm_fig3" in cfg:
        n_perm_val2_2 = int(cfg.get("n_perm_fig3", 0))
    elif isinstance(cfg.get("analysis"), dict):
        n_perm_val2_2 = int(cfg.get("analysis", {}).get("n_perm_val2_2", cfg.get("analysis", {}).get("n_perm_fig3", 0)))

    if n_perm_val2_2 > 0:
        k_band_cfg = (
            cfg.get("k_band_val2_2")
            or cfg.get("k_band_fig3")
            or cfg.get("k_perm_val2_2")
            or cfg.get("k_perm_fig3")
            or cfg.get("k_star")
        )
        if k_band_cfg is None and isinstance(cfg.get("analysis"), dict):
            k_band_cfg = (
                cfg.get("analysis", {}).get("k_band_val2_2")
                or cfg.get("analysis", {}).get("k_band_fig3")
                or cfg.get("analysis", {}).get("k_star")
            )

        if k_band_cfg is None:
            if len(ks) == 1:
                k_band_cfg = [int(ks[0])]
            else:
                raise ValueError(
                    "Config must provide 'k_band_val2_2' (or 'k_band_fig3' / 'k_perm_val2_2' / 'k_star') when n_perm_val2_2 > 0 and multiple ks are provided."
                )

        if isinstance(k_band_cfg, list):
            k_band_cfg = [int(k) for k in k_band_cfg]
        else:
            k_band_cfg = [int(k_band_cfg)]

        run_val2_2_permutation_shuffles(
            G=G,
            pert_sigs=pert_sigs,
            dis_sigs=dis_sigs,
            labels_df=labels_df,
            out_dir=out_dir,
            k_band=list(k_band_cfg),
            start_sign=int(start_sign),
            n_perm=int(n_perm_val2_2),
            seed=int(seed_base),
            tie_policy=tie_policy,
        )

    # 5) optional: record per-gene weights used by proposed method
    record_w = bool(
        cfg.get("record_proposed_w", False)
        or (isinstance(cfg.get("analysis"), dict) and cfg.get("analysis", {}).get("record_proposed_w", False))
    )
    if record_w:
        w_ks = cfg.get("w_record_ks", ks)
        if not isinstance(w_ks, list):
            w_ks = ks
        w_ks = [int(k) for k in w_ks]
        w_out = out_dir / f"val2_proposed_w_values_k{min(w_ks)}-{max(w_ks)}.csv"
        export_proposed_w_values(
            G=G,
            pert_sigs=pert_sigs,
            dis_sigs=dis_sigs,
            labels_df=labels_df,
            ks=w_ks,
            out_path=w_out,
            start_sign=int(start_sign),
        )

    # 6) quick summary
    if len(metrics_df) > 0:
        cols = [c for c in ["AP", "P@10", "P@20", "Lift@10", "Lift@20"] if c in metrics_df.columns]
        print("\n[SUMMARY] mean over disease_id:")
        print(metrics_df.groupby(["method", "k"])[cols].mean(numeric_only=True).sort_index())
    else:
        print("[WARN] metrics_df is empty (e.g., no query had positive labels).")

    print(f"\n[DONE] saved:\n  {scores_path}\n  {metrics_path}\n  {pred_amb_path}\n")


if __name__ == "__main__":
    main()
