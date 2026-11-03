"""L1000-NCS signature-reversal baseline (methodology §6.2).

Weighted enrichment score (WTCS) equivalent to cmapM
``+mortar/+compute/@Connectivity/fastESCore.m`` evaluated per reference
profile and query gene set; the combined WTCS follows ``getCombinedES``
without rescale. On top of WTCS this module adds the cell-line NCS
normalization of methodology §6.2.6: within each cell line the reference
population's WTCS values are sign-split, averaged to mu+/mu-, and each
profile's WTCS is divided by its sign's coefficient to yield NCS.

Profile unit deviation: per user direction the reference profiles are the
target-cell averaged continuous KD profiles (``BING_knockdown_signature``),
NOT the per-sig_id Level 5 signatures required by §6.2.1 / validation #4.
The resulting score is therefore a "target-cell-averaged adapted NCS", which
is neither the official Touchstone NCS nor the sig_id-level adapted NCS. The
normalization population is still fixed independently of val2 candidates / OT
labels / SIGNOR coverage (§6.2.2, validation #13): all target-cell profiles of
a cell line are used. ``BING_knockdown_signature`` is built from
``pert_type == trt_sh.cgs`` rows only, so validation #14 holds by construction.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def gene_ids_to_int(feature_genes: List[str]) -> np.ndarray:
    """Secondary sort key: integer part of 'NCBIGENE:<n>' (ascending).

    Falls back to a stable hash-free ordering for non-conforming IDs by using a
    large sentinel so they sort after numeric IDs deterministically.
    """
    out = np.empty(len(feature_genes), dtype=np.int64)
    for i, g in enumerate(feature_genes):
        s = str(g)
        digits = s.split(":")[-1]
        try:
            out[i] = int(digits)
        except ValueError:
            out[i] = np.iinfo(np.int64).max
    return out


def rank_descending(ref_values: np.ndarray, gene_id_int: np.ndarray) -> np.ndarray:
    """1-indexed ranks by descending reference value, ties broken by ascending
    NCBI gene ID (methodology §6.2.2). Deterministic."""
    neg = -ref_values.astype(np.float64)
    order = np.lexsort((gene_id_int, neg))  # primary: neg (asc => value desc), secondary: gene_id asc
    ranks = np.empty(len(ref_values), dtype=np.int64)
    ranks[order] = np.arange(1, len(ref_values) + 1, dtype=np.int64)
    return ranks


def _es_core(srt_rank: np.ndarray, weight_sorted: np.ndarray, N: int, G: int) -> float:
    """Signed weighted enrichment score (fastESCore.m, single column)."""
    abs_w = np.abs(weight_sorted)
    total = float(abs_w.sum())
    hit_cost = abs_w / total  # weighted hit increments
    miss_cost = -1.0 / (N - G)

    prev = np.concatenate(([0], srt_rank[:-1]))
    delta = (srt_rank - prev - 1).astype(np.float64) * miss_cost
    es = np.cumsum(delta + hit_cost)
    es_pre = es - hit_cost
    use_pre = np.abs(es) < np.abs(es_pre)
    es_used = np.where(use_pre, es_pre, es)
    idx = int(np.argmax(np.abs(es_used)))
    return float(es_used[idx])


def es_for_query(
    ranks: np.ndarray,
    ref_values: np.ndarray,
    positions: np.ndarray,
) -> Tuple[Optional[float], str]:
    """Enrichment score for a query whose members occupy ``positions`` in F.

    Returns (es, reason). es is None when undefined; reason is "" on success.
    Undefined cases (methodology §6.2.3): empty query, query == F, zero total
    weight.
    """
    N = int(len(ranks))
    G = int(len(positions))
    if G == 0:
        return None, "empty_query"
    if G == N:
        return None, "query_equals_F"
    weights = ref_values[positions]
    if float(np.abs(weights).sum()) == 0.0:
        return None, "zero_weight"
    member_ranks = ranks[positions]
    order = np.argsort(member_ranks, kind="mergesort")
    srt_rank = member_ranks[order]
    weight_sorted = weights[order]
    es = _es_core(srt_rank, weight_sorted, N, G)
    return es, ""


def combined_wtcs(es_up: float, es_down: float) -> float:
    """getCombinedES without rescale: 0.5*(up-down), nullized when signs match."""
    if np.sign(es_up) == np.sign(es_down):
        return 0.0
    return 0.5 * (es_up - es_down)


def compute_wtcs_cell(
    ref_values: np.ndarray,
    gene_id_int: np.ndarray,
    up_positions: np.ndarray,
    down_positions: np.ndarray,
) -> Dict[str, object]:
    """Compute ES_up, ES_down, WTCS for one reference profile."""
    ranks = rank_descending(ref_values, gene_id_int)
    es_up, reason_up = es_for_query(ranks, ref_values, up_positions)
    es_dn, reason_dn = es_for_query(ranks, ref_values, down_positions)
    if es_up is None or es_dn is None:
        reasons = [r for r in (reason_up, reason_dn) if r]
        return {
            "es_up": np.nan,
            "es_down": np.nan,
            "wtcs": np.nan,
            "valid": False,
            "reason": ";".join(reasons) if reasons else "undefined",
        }
    wtcs = combined_wtcs(float(es_up), float(es_dn))
    return {
        "es_up": float(es_up),
        "es_down": float(es_dn),
        "wtcs": float(wtcs),
        "valid": True,
        "reason": "",
    }


def ranks_for_matrix(ref_matrix: np.ndarray, gene_id_int: np.ndarray) -> np.ndarray:
    """Descending ranks for every row of a cell's reference matrix (§6.2.3).

    Computed once per cell line and reused across all disease queries, because
    a profile's gene ranking does not depend on the disease query.

    Vectorized equivalent of a per-row :func:`rank_descending`: columns are
    pre-ordered by ascending gene ID so a per-row stable sort on the negated
    values reproduces the ``lexsort((gene_id, neg))`` tie-breaking exactly.
    """
    ref = np.asarray(ref_matrix)
    if ref.ndim != 2:
        raise ValueError("ranks_for_matrix expects a 2D reference matrix.")
    gid = np.asarray(gene_id_int, dtype=np.int64)
    n_rows, n_cols = ref.shape
    if n_cols == 0:
        return np.empty((n_rows, 0), dtype=np.int32)
    gorder = np.argsort(gid, kind="mergesort")  # columns by ascending gene ID (stable)
    neg_g = (-ref[:, gorder]).astype(np.float64)
    ord_in_g = np.argsort(neg_g, axis=1, kind="stable")  # value desc, ties gene-ID asc
    ranks_g = np.empty((n_rows, n_cols), dtype=np.int64)
    seq = np.broadcast_to(np.arange(1, n_cols + 1, dtype=np.int64), (n_rows, n_cols))
    np.put_along_axis(ranks_g, ord_in_g, seq, axis=1)
    out = np.empty((n_rows, n_cols), dtype=np.int64)
    out[:, gorder] = ranks_g
    return out.astype(np.int32)


def _es_for_query_matrix(
    ranks_matrix: np.ndarray,
    ref_matrix: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Vectorized :func:`es_for_query` over every reference row for one query.

    Returns a ``(n_rows,)`` float64 array of enrichment scores; ``nan`` marks
    rows where the score is undefined (zero total query weight). Empty queries
    or ``G == N`` yield an all-``nan`` array, mirroring the scalar path
    (``empty_query`` / ``query_equals_F``). Dtypes follow the scalar
    :func:`_es_core` so results are bit-identical.
    """
    n_rows = int(ref_matrix.shape[0])
    N = int(ranks_matrix.shape[1])
    G = int(positions.shape[0])
    es = np.full(n_rows, np.nan, dtype=np.float64)
    if G == 0 or G == N:
        return es
    weights = ref_matrix[:, positions]
    member_ranks = ranks_matrix[:, positions]
    order = np.argsort(member_ranks, axis=1, kind="stable")  # ranks distinct -> deterministic
    srt_rank = np.take_along_axis(member_ranks, order, axis=1)
    weight_sorted = np.take_along_axis(weights, order, axis=1)
    abs_w = np.abs(weight_sorted)
    total = abs_w.sum(axis=1)  # keep query-weight dtype to match scalar float() upcast round-trip
    miss_cost = -1.0 / (N - G)
    with np.errstate(invalid="ignore", divide="ignore"):
        hit_cost = abs_w / total[:, None]
    prev = np.concatenate([np.zeros((n_rows, 1), dtype=srt_rank.dtype), srt_rank[:, :-1]], axis=1)
    delta = (srt_rank - prev - 1).astype(np.float64) * miss_cost
    es_cum = np.cumsum(delta + hit_cost, axis=1)
    es_pre = es_cum - hit_cost
    use_pre = np.abs(es_cum) < np.abs(es_pre)
    es_used = np.where(use_pre, es_pre, es_cum)
    idx = np.argmax(np.abs(es_used), axis=1)
    es_val = es_used[np.arange(n_rows), idx]
    es_val[total == 0.0] = np.nan
    return es_val


def compute_cell_ncs(
    ref_matrix: np.ndarray,
    ranks_matrix: np.ndarray,
    up_positions: np.ndarray,
    down_positions: np.ndarray,
) -> Dict[str, object]:
    """NCS normalization for one (disease, cell) over the cell's full reference
    population (methodology §6.2.5-6.2.6).

    ``ref_matrix`` / ``ranks_matrix`` cover every reference profile of the cell
    line (the normalization population R_{c,sh}), not just candidate targets.

    Returns per-row signed WTCS and NCS arrays plus the normalization
    diagnostics (mu+/mu-, sign counts). When either sign class is empty or its
    coefficient is 0 the NCS is left undefined for the whole (d,c) so that no
    finite NCS is returned for a missing normalizer (validation #15).
    """
    n_rows = int(ref_matrix.shape[0])
    up_positions = np.asarray(up_positions, dtype=np.int64)
    down_positions = np.asarray(down_positions, dtype=np.int64)
    es_up = _es_for_query_matrix(ranks_matrix, ref_matrix, up_positions)
    es_dn = _es_for_query_matrix(ranks_matrix, ref_matrix, down_positions)
    both_defined = np.isfinite(es_up) & np.isfinite(es_dn)
    # combined_wtcs: 0 when the two enrichment signs match, else 0.5*(up-down).
    combined = np.where(np.sign(es_up) == np.sign(es_dn), 0.0, 0.5 * (es_up - es_dn))
    wtcs = np.full(n_rows, np.nan, dtype=np.float64)
    wtcs[both_defined] = combined[both_defined]

    defined = np.isfinite(wtcs)
    pos_mask = defined & (wtcs > 0.0)
    neg_mask = defined & (wtcs < 0.0)
    zero_mask = defined & (wtcs == 0.0)
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    n_zero = int(zero_mask.sum())
    mu_pos = float(wtcs[pos_mask].mean()) if n_pos else float("nan")
    mu_neg = float(np.abs(wtcs[neg_mask]).mean()) if n_neg else float("nan")

    valid = bool(n_pos > 0 and n_neg > 0 and mu_pos > 0.0 and mu_neg > 0.0)
    ncs = np.full(n_rows, np.nan, dtype=np.float64)
    if valid:
        ncs[pos_mask] = wtcs[pos_mask] / mu_pos
        ncs[neg_mask] = wtcs[neg_mask] / mu_neg
        ncs[zero_mask] = 0.0

    return {
        "wtcs": wtcs,
        "ncs": ncs,
        "mu_pos": mu_pos,
        "mu_neg": mu_neg,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_zero": n_zero,
        "n_ref_defined": int(defined.sum()),
        "n_ref_total": n_rows,
        "valid": valid,
    }


def compute_cell_ncs_local(
    ref_matrix: np.ndarray,
    local_cols: np.ndarray,
    gene_id_int_local: np.ndarray,
    up_local_positions: np.ndarray,
    down_local_positions: np.ndarray,
) -> Dict[str, object]:
    """NCS normalization on a target-specific local feature universe.

    ``ref_matrix`` covers all reference profiles for one cell line in the
    global feature space. ``local_cols`` selects ``F_h(t)`` columns from that
    matrix, and query positions are relative to this local matrix.
    """
    local_cols = np.asarray(local_cols, dtype=np.int64)
    if local_cols.size == 0:
        n_rows = int(ref_matrix.shape[0])
        return {
            "wtcs": np.full(n_rows, np.nan, dtype=np.float64),
            "ncs": np.full(n_rows, np.nan, dtype=np.float64),
            "mu_pos": float("nan"),
            "mu_neg": float("nan"),
            "n_pos": 0,
            "n_neg": 0,
            "n_zero": 0,
            "n_ref_defined": 0,
            "n_ref_total": n_rows,
            "valid": False,
            "reason": "empty_local_feature_universe",
        }
    local_matrix = ref_matrix[:, local_cols]
    ranks_matrix = ranks_for_matrix(local_matrix, np.asarray(gene_id_int_local, dtype=np.int64))
    out = compute_cell_ncs(
        local_matrix,
        ranks_matrix,
        np.asarray(up_local_positions, dtype=np.int64),
        np.asarray(down_local_positions, dtype=np.int64),
    )
    out["reason"] = "" if out["valid"] else "norm_undefined"
    return out
