"""L1000-NCS baseline scoring on the Val2 ablation evaluation space.

This adapts the val2_baseline NCS baseline (methodology §6.2) so it can be used
as an ablation reference line in ``src/val2/`` *in place of* ``noKG``
("Global no-KG"). The NCS scoring algorithm is unchanged from val2_baseline
(WTCS + per-cell-line NCS normalization); only the evaluation space differs:

- Candidates / diseases / cells come from the Val2 label table (``labels_df``),
  NOT val2_baseline's separate "common universe" / disease gates.
- Scores are aggregated to the target level (``-mean_ncs`` over the label cells
  where NCS is defined), matching :func:`val2.legacy.evaluate_ranking_by_query`
  which averages per-cell scores over cells before ranking.
- Pairs where NCS is undefined are handled per ``undefined_policy`` (``floor``
  fills them with a per-disease floor so NCS ranks the same candidate set as the
  other methods; ``drop`` omits them).

The heavy NCS math is reused verbatim from :mod:`val2_baseline.wtcs`
(numpy-only, no back-import of ``val2``); the parquet loaders from
:mod:`val2_baseline.io` are imported lazily inside the function to keep the
``val2`` package import light and defensive against import-order changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from val2_baseline import wtcs as wtcs_mod

_NCS_COLUMNS = ["disease_id", "target_gene", "score", "n_valid_cells", "is_floor", "reason"]


def _build_disease_queries(
    disease_ids: Set[str],
    disease_signatures: Dict[str, Dict[str, int]],
    feature_index: Dict[str, int],
) -> Dict[str, Dict[str, object]]:
    """Per-disease up/down query positions in feature space F.

    Mirrors ``val2_baseline.runner._build_disease_inputs`` (only the NCS-query
    part): q_up/q_down are the sign +1 / -1 genes, intersected with F.
    ``ncs_query_valid`` requires both up and down to be non-empty in F.
    """
    feature_set = set(feature_index.keys())
    info: Dict[str, Dict[str, object]] = {}
    for d in sorted(disease_ids):
        sig = disease_signatures.get(d, {})
        q_up = {g for g, s in sig.items() if int(s) == 1}
        q_down = {g for g, s in sig.items() if int(s) == -1}
        up_in_f = sorted(q_up & feature_set)
        down_in_f = sorted(q_down & feature_set)
        info[d] = {
            "up_positions": np.array([feature_index[g] for g in up_in_f], dtype=np.int64),
            "down_positions": np.array([feature_index[g] for g in down_in_f], dtype=np.int64),
            "ncs_query_valid": bool(up_in_f) and bool(down_in_f),
        }
    return info


def compute_ncs_target_scores(
    *,
    labels_df: pd.DataFrame,
    disease_signatures: Dict[str, Dict[str, int]],
    continuous_kd_parquet: str | Path,
    id_col: str = "ncbigene_id",
    cell_col: str = "cell_id",
    disease_signature_source: str = "full_creeds",
    creeds_parquet: Optional[str | Path] = None,
    creeds_id_col: str = "doid",
    undefined_policy: str = "floor",
) -> pd.DataFrame:
    """Compute target-level NCS scores on the Val2 ablation evaluation space.

    Returns a DataFrame with columns
    ``[disease_id, target_gene, score, n_valid_cells, is_floor, reason]`` where
    ``score = -mean_ncs`` over the label cells with a defined NCS (higher =
    stronger reversal). Only ``(disease, target)`` pairs present in ``labels_df``
    are considered, and only diseases whose NCS up/down query is valid.

    ``undefined_policy``:
      - ``"floor"``: pairs with no defined cell get ``min(valid score in
        disease) - 1.0`` (rank strictly last); diseases with no valid pair at
        all are skipped (no floor is definable).
      - ``"drop"``: pairs with no defined cell are omitted.
    """
    policy = str(undefined_policy).strip().lower()
    if policy not in {"floor", "drop"}:
        raise ValueError("undefined_policy must be 'floor' or 'drop'.")
    source = str(disease_signature_source).strip().lower()
    if source not in {"full_creeds", "val2"}:
        raise ValueError("disease_signature_source must be 'full_creeds' or 'val2'.")

    # Lazy import to keep the val2 package import light and avoid pulling
    # pyarrow-heavy val2_baseline.io at import time.
    from val2_baseline.io import load_continuous_kd, load_creeds_disease_signatures

    labels = labels_df[["disease_id", "target_gene", "cell_id"]].copy()
    labels["disease_id"] = labels["disease_id"].astype(str)
    labels["target_gene"] = labels["target_gene"].astype(str)
    labels["cell_id"] = labels["cell_id"].astype(str)

    # Full reference population (needed_genes=None) so per-cell normalization is
    # candidate-independent (methodology §6.2.2).
    continuous_kd = load_continuous_kd(
        continuous_kd_parquet, id_col=id_col, cell_col=cell_col, needed_genes=None
    )

    # Discreteness guard (validation #4): reject a discretized (-1/0/1) profile.
    sample = continuous_kd.matrix[: min(5, continuous_kd.matrix.shape[0])]
    if sample.size and np.all(
        np.isin(sample, np.array([-1.0, 0.0, 1.0], dtype=continuous_kd.matrix.dtype))
    ):
        raise ValueError(
            "[NCS] continuous KD profile appears to be discretized (only -1/0/1 values). "
            "The discretized profile must not be used for WTCS/NCS (validation #4)."
        )

    # Resolve the disease-signature source used for the up/down query.
    if source == "full_creeds":
        if not creeds_parquet:
            raise ValueError("creeds_parquet is required when disease_signature_source='full_creeds'.")
        query_signatures = load_creeds_disease_signatures(creeds_parquet, id_col=creeds_id_col)
    else:
        query_signatures = disease_signatures

    disease_ids = set(labels["disease_id"].unique())
    disease_info = _build_disease_queries(disease_ids, query_signatures, continuous_kd.feature_index)

    # Candidate (disease, target) pairs and their label cells (val2's space).
    pair_cells: Dict[Tuple[str, str], List[str]] = {}
    for r in labels.itertuples(index=False):
        pair_cells.setdefault((r.disease_id, r.target_gene), []).append(r.cell_id)

    continuous_cells: Set[Tuple[str, str]] = set(continuous_kd.keys)

    # (disease, cell) -> candidate targets, and cell -> diseases, restricted to
    # candidate pairs whose (target, cell) profile exists and whose disease has a
    # valid NCS query.
    cand_targets_by_dc: Dict[Tuple[str, str], List[str]] = {}
    cell_to_diseases: Dict[str, Set[str]] = {}
    for (d, t), cells in pair_cells.items():
        if not disease_info.get(d, {}).get("ncs_query_valid", False):
            continue
        for c in cells:
            if (t, c) in continuous_cells:
                cand_targets_by_dc.setdefault((d, c), []).append(t)
                cell_to_diseases.setdefault(c, set()).add(d)

    gene_id_int = wtcs_mod.gene_ids_to_int(continuous_kd.feature_genes)
    cell_to_rows = continuous_kd.rows_by_cell()

    # ncs_cell_value[(disease, target, cell)] = finite NCS (per-cell readout).
    ncs_cell_value: Dict[Tuple[str, str, str], float] = {}
    for c in sorted(cell_to_diseases.keys()):
        rows = cell_to_rows.get(c, [])
        if not rows:
            continue
        ref_matrix = continuous_kd.matrix[rows]
        ranks_2d = wtcs_mod.ranks_for_matrix(ref_matrix, gene_id_int)
        local_of_global = {grow: i for i, grow in enumerate(rows)}
        for d in sorted(cell_to_diseases[c]):
            di = disease_info[d]
            res = wtcs_mod.compute_cell_ncs(
                ref_matrix, ranks_2d, di["up_positions"], di["down_positions"]
            )
            ncs_arr = res["ncs"]
            for t in cand_targets_by_dc.get((d, c), []):
                grow = continuous_kd.key_to_row.get((t, c))
                if grow is None or grow not in local_of_global:
                    continue
                ncs_v = float(ncs_arr[local_of_global[grow]])
                if np.isfinite(ncs_v):
                    ncs_cell_value[(d, t, c)] = ncs_v

    # Target-level aggregation: score = -mean(ncs over defined label cells).
    valid_scores: Dict[Tuple[str, str], Tuple[float, int]] = {}
    for (d, t), cells in pair_cells.items():
        vals = [ncs_cell_value[(d, t, c)] for c in cells if (d, t, c) in ncs_cell_value]
        if vals:
            mean_ncs = float(sum(vals) / len(vals))
            valid_scores[(d, t)] = (float(-mean_ncs), len(vals))

    # Emit rows per undefined policy, restricted to diseases with a valid query.
    valid_diseases = {d for d, di in disease_info.items() if di["ncs_query_valid"]}
    out_rows: List[Dict[str, object]] = []
    for d in sorted(valid_diseases):
        disease_pairs = [t for (dd, t) in pair_cells.keys() if dd == d]
        disease_valid_scores = [
            valid_scores[(d, t)][0] for t in disease_pairs if (d, t) in valid_scores
        ]
        floor_value = (min(disease_valid_scores) - 1.0) if disease_valid_scores else None
        for t in disease_pairs:
            if (d, t) in valid_scores:
                score, n_valid = valid_scores[(d, t)]
                out_rows.append(
                    {
                        "disease_id": d, "target_gene": t, "score": score,
                        "n_valid_cells": int(n_valid), "is_floor": False, "reason": "",
                    }
                )
            elif policy == "floor" and floor_value is not None:
                out_rows.append(
                    {
                        "disease_id": d, "target_gene": t, "score": float(floor_value),
                        "n_valid_cells": 0, "is_floor": True, "reason": "ncs_undefined",
                    }
                )
            # policy == "drop", or floor undefinable (no valid pair): emit nothing.

    return pd.DataFrame(out_rows, columns=_NCS_COLUMNS)
