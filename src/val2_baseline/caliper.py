"""CALIPER scoring wrapper (methodology §6.1).

Reuses the existing val2 ``score_pair`` (method='proposed') at the fixed
``k_star`` from val1. The val2 label is never used to choose k, and the full
K_valid set is not re-scanned. Cell-level scores are emitted with their
``n_meas``, ``covAlign`` and ``support`` diagnostics; alignment-zero pairs keep
score 0 and are not treated as missing.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

import pandas as pd

_CALIPER_METHOD_RE = re.compile(r"^CALIPER@k(\d+)$")

from val2.legacy import score_pair
from val2.summary import PRIMARY_K


def caliper_method_key(k: int, *, multi_k: bool) -> str:
    """Internal method key for a CALIPER hop radius ``k``."""
    if multi_k:
        return f"CALIPER@k{int(k)}"
    return "CALIPER"


def caliper_method_keys(caliper_k_list: List[int]) -> List[str]:
    """Method keys for each hop radius in ``caliper_k_list``."""
    multi = len(caliper_k_list) > 1
    return [caliper_method_key(k, multi_k=multi) for k in caliper_k_list]


def is_caliper_method(method: str) -> bool:
    m = str(method)
    return m == "CALIPER" or bool(_CALIPER_METHOD_RE.match(m))


def parse_caliper_method_k(method: str) -> int | None:
    m = str(method)
    if m == "CALIPER":
        return None
    match = _CALIPER_METHOD_RE.match(m)
    return int(match.group(1)) if match else None


def caliper_method_display(method: str) -> str:
    k = parse_caliper_method_k(method)
    if k is not None:
        return f"CALIPER (k={k})"
    return "CALIPER"


def primary_caliper_methods(methods: Iterable[str], *, k_star: int | None = None) -> List[str]:
    """CALIPER method keys scored at the primary distance ``PRIMARY_K``.

    Significance testing is restricted to this radius (see
    :mod:`val2.stats`), so a multi-k run contributes only its ``CALIPER@k4``
    series. A single-k run emits the bare ``CALIPER`` key, whose radius is
    ``k_star`` and is therefore resolved from the caller.
    """
    out: List[str] = []
    for method in methods:
        m = str(method)
        k = parse_caliper_method_k(m)
        if k is None:
            k = int(k_star) if k_star is not None else None
        if k == PRIMARY_K:
            out.append(m)
    return list(dict.fromkeys(out))


def detect_caliper_methods(methods: Iterable[str]) -> List[str]:
    """Return CALIPER method keys present in ``methods``, sorted by hop k."""
    out = [str(m) for m in methods if is_caliper_method(str(m))]

    def _sort_key(method: str) -> tuple[int, str]:
        k = parse_caliper_method_k(method)
        return (k if k is not None else 10**9, method)

    return sorted(dict.fromkeys(out), key=_sort_key)


def compute_caliper_cell_scores(
    *,
    graph,
    pert_sigs: Dict[Tuple[str, str], Dict[str, int]],
    dis_sigs: Dict[str, Dict[str, int]],
    cell_triples: Iterable[Tuple[str, str, str]],
    k_star: int,
    start_sign: int,
    progress=None,
) -> pd.DataFrame:
    """Compute CALIPER cell-level scores for (disease_id, target_gene, cell_id).

    ``cell_triples`` yields (disease_id, target_gene, cell_id). Returns a long
    DataFrame with one row per triple.
    """
    cache: Dict[str, object] = {}
    rows: List[Dict[str, object]] = []
    triples = list(cell_triples)
    iterator = progress(triples, desc="caliper", total=len(triples)) if progress else triples
    for disease_id, target_gene, cell_id in iterator:
        score, n_rev, n_meas, _covK, cov_align, support = score_pair(
            graph,
            pert_sigs,
            dis_sigs,
            target_gene,
            cell_id,
            disease_id,
            int(k_star),
            "proposed",
            start_sign=int(start_sign),
            cache=cache,
        )
        rows.append(
            {
                "disease_id": str(disease_id),
                "target_gene": str(target_gene),
                "cell_id": str(cell_id),
                "score": float(score),
                "n_rev": float(n_rev),
                "n_meas": int(n_meas),
                "covAlign": float(cov_align),
                "support": float(support),
            }
        )
    cols = ["disease_id", "target_gene", "cell_id", "score", "n_rev", "n_meas", "covAlign", "support"]
    return pd.DataFrame(rows, columns=cols)


def aggregate_target_level(
    cell_scores: pd.DataFrame,
    common_cells: Dict[Tuple[str, str], List[str]],
) -> pd.DataFrame:
    """Arithmetic mean over the common cell set L_{t,d}^common (methodology §6.1)."""
    rows: List[Dict[str, object]] = []
    indexed = cell_scores.set_index(["disease_id", "target_gene", "cell_id"]).sort_index()
    for (disease_id, target_gene), cells in common_cells.items():
        vals = []
        n_meas_vals = []
        cov_vals = []
        support_vals = []
        for c in cells:
            key = (str(disease_id), str(target_gene), str(c))
            if key in indexed.index:
                r = indexed.loc[key]
                vals.append(float(r["score"]))
                n_meas_vals.append(float(r["n_meas"]))
                cov_vals.append(float(r["covAlign"]))
                support_vals.append(float(r["support"]))
        if not vals:
            continue
        rows.append(
            {
                "disease_id": str(disease_id),
                "target_gene": str(target_gene),
                "score": float(sum(vals) / len(vals)),
                "n_common_cells": int(len(vals)),
                "mean_n_meas": float(sum(n_meas_vals) / len(n_meas_vals)),
                "mean_covAlign": float(sum(cov_vals) / len(cov_vals)),
                "mean_support": float(sum(support_vals) / len(support_vals)),
            }
        )
    cols = ["disease_id", "target_gene", "score", "n_common_cells", "mean_n_meas", "mean_covAlign", "mean_support"]
    return pd.DataFrame(rows, columns=cols)
