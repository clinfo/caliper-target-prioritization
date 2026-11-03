"""Availability table and common-set determination (methodology §5).

Builds the per-pair availability table, then the single shared evaluation
universe used by all four methods: C_d^common (common candidates),
L_{t,d}^common (common cells) and D^common (common diseases). Labels are used
only here for disease selection (>=1 positive) and never for score-defining
parameters (methodology §3.2). Once fixed, exclusion rules must not change
based on observed scores.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import pandas as pd

PairKey = Tuple[str, str]  # (disease_id, target_gene)
DEFAULT_BASELINES = ["NCS", "proximity", "RWR"]


def _selected_set(selected_baselines: List[str] | None) -> Set[str]:
    return set(selected_baselines or DEFAULT_BASELINES)


def _pair_cells_from_labels(labels_df: pd.DataFrame) -> Dict[PairKey, Set[str]]:
    out: Dict[PairKey, Set[str]] = {}
    for r in labels_df.itertuples(index=False):
        key = (str(r.disease_id), str(r.target_gene))
        out.setdefault(key, set()).add(str(r.cell_id))
    return out


def build_common_sets(
    *,
    labels_df: pd.DataFrame,
    H_nodes: Set[str],
    disease_info: Dict[str, Dict[str, object]],
    continuous_cells: Set[Tuple[str, str]],
    ncs_valid_cells: Set[Tuple[str, str, str]],
    ncs_ref_counts: Dict[Tuple[str, str], Tuple[int, int]],
    proximity_valid: Dict[PairKey, bool],
    rwr_valid: Dict[PairKey, bool],
    min_disease_seeds: int,
    selected_baselines: List[str] | None = None,
    dada_valid: Dict[PairKey, bool] | None = None,
) -> Dict[str, object]:
    selected = _selected_set(selected_baselines)
    require_ncs = "NCS" in selected
    require_proximity = "proximity" in selected
    require_rwr = "RWR" in selected
    require_dada = "DADA" in selected
    # DADA is a shared-network baseline: it activates the disease-level
    # min_disease_seeds gate exactly like proximity/RWR.
    require_network = require_proximity or require_rwr or require_dada
    dada_valid = dada_valid or {}

    pair_cells = _pair_cells_from_labels(labels_df)
    pair_label = (
        labels_df.groupby(["disease_id", "target_gene"])["label"].max().astype(int).to_dict()
    )

    avail_rows: List[Dict[str, object]] = []
    common_candidate_by_disease: Dict[str, List[str]] = {}
    common_cells: Dict[PairKey, List[str]] = {}

    for (disease_id, target_gene), cells in pair_cells.items():
        d_info = disease_info.get(disease_id, {})
        ncs_query_valid = bool(d_info.get("ncs_query_valid", False))
        s_d_h_size = int(d_info.get("s_d_h_size", 0))

        target_in_H = target_gene in H_nodes
        cont_cells = {c for c in cells if (target_gene, c) in continuous_cells}
        # L_{t,d}^{NCS valid}: cells where the candidate profile has a finite
        # target-cell NCS (both normalization coefficients defined; §5.4).
        ncs_cells = {c for c in cells if (disease_id, target_gene, c) in ncs_valid_cells}
        ncs_common_cells = sorted(cells & cont_cells & ncs_cells)
        l_common = ncs_common_cells if require_ncs else sorted(cells)
        n_common_cells = len(l_common)

        # ncs_reference_valid: at least one common cell has both pos/neg
        # normalization coefficients defined. ncs_ref_*_count_min: min over
        # common cells of the pos/neg WTCS reference counts.
        pos_counts = [ncs_ref_counts.get((disease_id, c), (0, 0))[0] for c in ncs_common_cells]
        neg_counts = [ncs_ref_counts.get((disease_id, c), (0, 0))[1] for c in ncs_common_cells]
        ncs_reference_valid = len(ncs_common_cells) >= 1
        ncs_ref_pos_count_min = int(min(pos_counts)) if pos_counts else 0
        ncs_ref_neg_count_min = int(min(neg_counts)) if neg_counts else 0

        caliper_defined = n_common_cells >= 1
        ncs_defined = len(ncs_common_cells) >= 1
        prox_defined = bool(proximity_valid.get((disease_id, target_gene), False))
        rwr_defined = bool(rwr_valid.get((disease_id, target_gene), False))
        dada_defined = bool(dada_valid.get((disease_id, target_gene), False))

        reasons: List[str] = []
        if not target_in_H:
            reasons.append("target_not_in_H")
        if require_ncs and not ncs_query_valid:
            reasons.append("ncs_query_invalid")
        if n_common_cells == 0:
            reasons.append("no_common_cell")
        if require_proximity and not prox_defined:
            reasons.append("proximity_undefined")
        if require_rwr and not rwr_defined:
            reasons.append("rwr_undefined")
        if require_dada and not dada_defined:
            reasons.append("dada_undefined")

        in_common = (
            target_in_H
            and caliper_defined
            and (ncs_defined if require_ncs else True)
            and (prox_defined if require_proximity else True)
            and (rwr_defined if require_rwr else True)
            and (dada_defined if require_dada else True)
            and (ncs_query_valid if require_ncs else True)
        )

        avail_rows.append(
            {
                "disease_id": disease_id,
                "target_gene": target_gene,
                "in_val2_universe": True,
                "target_in_H": target_in_H,
                "ncs_query_valid": ncs_query_valid,
                "ncs_reference_valid": ncs_reference_valid,
                "s_d_h_size": s_d_h_size,
                "ncs_n_common_cells": int(len(ncs_common_cells)),
                "caliper_score_defined": caliper_defined,
                "ncs_score_defined": ncs_defined,
                "ncs_ref_pos_count_min": ncs_ref_pos_count_min,
                "ncs_ref_neg_count_min": ncs_ref_neg_count_min,
                "proximity_score_defined": prox_defined,
                "rwr_score_defined": rwr_defined,
                # Only emitted when DADA is selected, so runs without it keep
                # their existing availability columns unchanged.
                **({"dada_score_defined": dada_defined} if require_dada else {}),
                "in_common_candidate": in_common,
                "label": int(pair_label.get((disease_id, target_gene), 0)),
                "exclusion_reason": ";".join(reasons),
            }
        )

        if in_common:
            common_candidate_by_disease.setdefault(disease_id, []).append(target_gene)
            common_cells[(disease_id, target_gene)] = l_common

    availability_df = pd.DataFrame(avail_rows).sort_values(["disease_id", "target_gene"]).reset_index(drop=True)

    # ---- D^common (methodology §5.3) ----
    common_diseases: List[Dict[str, object]] = []
    disease_exclusion: Dict[str, List[str]] = {}
    for disease_id, d_info in disease_info.items():
        ncs_query_valid = bool(d_info.get("ncs_query_valid", False))
        s_d_h_size = int(d_info.get("s_d_h_size", 0))
        cands = common_candidate_by_disease.get(disease_id, [])
        n_cand = len(cands)
        n_pos = sum(int(pair_label.get((disease_id, t), 0)) for t in cands)

        reasons: List[str] = []
        if require_ncs and not ncs_query_valid:
            reasons.append("ncs_query_invalid")
        if require_network and s_d_h_size < int(min_disease_seeds):
            reasons.append(f"seeds_lt_{int(min_disease_seeds)}")
        if n_cand == 0:
            reasons.append("empty_common_candidates")
        if n_pos < 1:
            reasons.append("no_positive_in_common")

        selected = (
            (ncs_query_valid if require_ncs else True)
            and (s_d_h_size >= int(min_disease_seeds) if require_network else True)
            and n_cand >= 1
            and n_pos >= 1
        )
        if selected:
            common_diseases.append(
                {
                    "disease_id": disease_id,
                    "s_d_h_size": s_d_h_size,
                    "n_common_candidates": int(n_cand),
                    "n_positive": int(n_pos),
                }
            )
        else:
            disease_exclusion[disease_id] = reasons

    common_diseases_df = pd.DataFrame(
        common_diseases, columns=["disease_id", "s_d_h_size", "n_common_candidates", "n_positive"]
    ).sort_values("disease_id").reset_index(drop=True)
    selected_disease_ids = set(common_diseases_df["disease_id"].astype(str))

    # ---- Final common candidate table (only diseases in D^common) ----
    cand_rows: List[Dict[str, object]] = []
    final_common_cells: Dict[PairKey, List[str]] = {}
    for disease_id in selected_disease_ids:
        for target_gene in sorted(common_candidate_by_disease.get(disease_id, [])):
            label = int(pair_label.get((disease_id, target_gene), 0))
            cells = common_cells[(disease_id, target_gene)]
            final_common_cells[(disease_id, target_gene)] = cells
            cand_rows.append(
                {
                    "disease_id": disease_id,
                    "target_gene": target_gene,
                    "label": label,
                    "is_positive": bool(label == 1),
                    "n_common_cells": int(len(cells)),
                }
            )
    common_candidates_df = (
        pd.DataFrame(cand_rows, columns=["disease_id", "target_gene", "label", "is_positive", "n_common_cells"])
        .sort_values(["disease_id", "target_gene"])
        .reset_index(drop=True)
    )

    # ---- common cells long table ----
    cells_rows: List[Dict[str, object]] = []
    for (disease_id, target_gene), cells in final_common_cells.items():
        for c in cells:
            cells_rows.append({"disease_id": disease_id, "target_gene": target_gene, "cell_id": c})
    common_cells_df = (
        pd.DataFrame(cells_rows, columns=["disease_id", "target_gene", "cell_id"])
        .sort_values(["disease_id", "target_gene", "cell_id"])
        .reset_index(drop=True)
    )

    # ---- exclusion summary (counts by reason) ----
    pair_reason_counts: Dict[str, int] = {}
    for r in avail_rows:
        if not r["in_common_candidate"]:
            for reason in str(r["exclusion_reason"]).split(";"):
                if reason:
                    pair_reason_counts[f"pair:{reason}"] = pair_reason_counts.get(f"pair:{reason}", 0) + 1
    for _disease_id, reasons in disease_exclusion.items():
        for reason in reasons:
            pair_reason_counts[f"disease:{reason}"] = pair_reason_counts.get(f"disease:{reason}", 0) + 1
    exclusion_summary_df = pd.DataFrame(
        sorted(({"scope_reason": k, "count": v} for k, v in pair_reason_counts.items()), key=lambda x: x["scope_reason"]),
        columns=["scope_reason", "count"],
    )

    # ---- flow (methodology §12-1) ----
    orig_diseases = set(labels_df["disease_id"].astype(str))
    orig_pairs = set((str(r.disease_id), str(r.target_gene)) for r in labels_df.itertuples(index=False))
    orig_pos = int((labels_df.groupby(["disease_id", "target_gene"])["label"].max() == 1).sum())
    flow = {
        "original_n_diseases": int(len(orig_diseases)),
        "original_n_pairs": int(len(orig_pairs)),
        "original_n_positive_pairs": int(orig_pos),
        "common_n_diseases": int(len(common_diseases_df)),
        "common_n_pairs": int(len(common_candidates_df)),
        "common_n_positive_pairs": int(common_candidates_df["is_positive"].sum()) if len(common_candidates_df) else 0,
    }

    return {
        "availability_df": availability_df,
        "common_diseases_df": common_diseases_df,
        "common_candidates_df": common_candidates_df,
        "common_cells_df": common_cells_df,
        "exclusion_summary_df": exclusion_summary_df,
        "common_cells": final_common_cells,
        "selected_disease_ids": selected_disease_ids,
        "pair_label": pair_label,
        "flow": flow,
    }


def build_common_sets_directed_target_reach(
    *,
    labels_df: pd.DataFrame,
    G_nodes: Set[str],
    disease_info: Dict[str, Dict[str, object]],
    continuous_cells: Set[Tuple[str, str]],
    reach_info: Dict[str, Dict[str, object]],
    ncs_fallback_by_pair: Dict[PairKey, int],
    proximity_info: Dict[PairKey, Dict[str, object]],
    rwr_info: Dict[PairKey, Dict[str, object]],
    selected_baselines: List[str] | None = None,
    min_disease_seeds: int = 20,
) -> Dict[str, object]:
    """Common-set determination for directed target-reach matched analysis.

    Local evidence absence and local network fallback are diagnostics, not
    exclusion criteria. The disease-level ``min_disease_seeds`` gate (applied
    only when proximity/RWR are selected) does apply here: it guards against
    diseases whose network baselines cannot function anywhere in the graph
    (methodology parity with the native shared-network gate).
    """
    selected = _selected_set(selected_baselines)
    require_ncs = "NCS" in selected
    require_proximity = "proximity" in selected
    require_rwr = "RWR" in selected
    require_network = require_proximity or require_rwr

    pair_cells = _pair_cells_from_labels(labels_df)
    pair_label = (
        labels_df.groupby(["disease_id", "target_gene"])["label"].max().astype(int).to_dict()
    )

    avail_rows: List[Dict[str, object]] = []
    common_candidate_by_disease: Dict[str, List[str]] = {}
    common_cells: Dict[PairKey, List[str]] = {}

    for (disease_id, target_gene), cells in pair_cells.items():
        d_info = disease_info.get(disease_id, {})
        ncs_query_valid = bool(d_info.get("ncs_query_valid", False))
        target_in_G = target_gene in G_nodes
        cont_cells = {c for c in cells if (target_gene, c) in continuous_cells}
        ncs_common_cell_count = len(cells & cont_cells)
        l_common = sorted(cells & cont_cells) if require_ncs else sorted(cells)
        n_common_cells = len(l_common)
        caliper_defined = n_common_cells >= 1
        ncs_defined = (ncs_common_cell_count >= 1 and ncs_query_valid) if require_ncs else False
        reach = reach_info.get(target_gene, {})
        prox = proximity_info.get((disease_id, target_gene), {})
        rwr = rwr_info.get((disease_id, target_gene), {})

        reasons: List[str] = []
        if not target_in_G:
            reasons.append("target_not_in_G")
        if require_ncs and not ncs_query_valid:
            reasons.append("ncs_query_invalid_global")
        if n_common_cells == 0:
            reasons.append("no_common_cell")

        in_common = target_in_G and (ncs_query_valid if require_ncs else True) and n_common_cells >= 1
        avail_rows.append(
            {
                "disease_id": disease_id,
                "target_gene": target_gene,
                "primary_analysis": "directed_target_reach_matched",
                "in_val2_universe": True,
                "target_in_G": target_in_G,
                "ncs_query_valid": ncs_query_valid,
                "ncs_query_valid_global": ncs_query_valid,
                "ncs_reference_valid": ncs_common_cell_count >= 1,
                "ncs_n_common_cells": int(ncs_common_cell_count),
                "caliper_score_defined": caliper_defined,
                "ncs_score_defined": ncs_defined,
                "n_reach_nodes": int(reach.get("n_reach_nodes", 0)),
                "n_anchor_nodes": int(reach.get("n_anchor_nodes", 0)),
                "n_local_edges": int(reach.get("n_local_edges", 0)),
                "n_local_seed": int(prox.get("n_seed_in_local", rwr.get("n_seed_in_local", 0)) or 0),
                "ncs_local_fallback_cells": int(ncs_fallback_by_pair.get((disease_id, target_gene), 0)),
                "proximity_score_defined": bool(require_proximity),
                "proximity_fallback": bool(prox.get("fallback", False)),
                "proximity_fallback_reason": str(prox.get("reason", "")),
                "rwr_score_defined": bool(require_rwr),
                "rwr_fallback": bool(rwr.get("fallback", False)),
                "rwr_fallback_reason": str(rwr.get("reason", "")),
                "in_common_candidate": in_common,
                "label": int(pair_label.get((disease_id, target_gene), 0)),
                "exclusion_reason": ";".join(reasons),
            }
        )

        if in_common:
            common_candidate_by_disease.setdefault(disease_id, []).append(target_gene)
            common_cells[(disease_id, target_gene)] = l_common

    availability_df = pd.DataFrame(avail_rows).sort_values(["disease_id", "target_gene"]).reset_index(drop=True)

    common_diseases: List[Dict[str, object]] = []
    disease_exclusion: Dict[str, List[str]] = {}
    for disease_id, d_info in disease_info.items():
        cands = common_candidate_by_disease.get(disease_id, [])
        n_cand = len(cands)
        n_pos = sum(int(pair_label.get((disease_id, t), 0)) for t in cands)
        s_d_h_size = int(d_info.get("s_d_h_size", 0))
        seeds_ok = (s_d_h_size >= int(min_disease_seeds)) if require_network else True
        reasons: List[str] = []
        if not seeds_ok:
            reasons.append(f"seeds_lt_{int(min_disease_seeds)}")
        if n_cand == 0:
            reasons.append("empty_common_candidates")
        if n_pos < 1:
            reasons.append("no_positive_in_common")
        if seeds_ok and n_cand >= 1 and n_pos >= 1:
            common_diseases.append(
                {
                    "disease_id": disease_id,
                    "s_d_h_size": s_d_h_size,
                    "n_common_candidates": int(n_cand),
                    "n_positive": int(n_pos),
                    "ncs_query_valid": bool(d_info.get("ncs_query_valid", False)),
                }
            )
        else:
            disease_exclusion[disease_id] = reasons

    common_diseases_df = pd.DataFrame(
        common_diseases,
        columns=["disease_id", "s_d_h_size", "n_common_candidates", "n_positive", "ncs_query_valid"],
    ).sort_values("disease_id").reset_index(drop=True)
    selected_disease_ids = set(common_diseases_df["disease_id"].astype(str))

    cand_rows: List[Dict[str, object]] = []
    final_common_cells: Dict[PairKey, List[str]] = {}
    for disease_id in selected_disease_ids:
        for target_gene in sorted(common_candidate_by_disease.get(disease_id, [])):
            label = int(pair_label.get((disease_id, target_gene), 0))
            cells = common_cells[(disease_id, target_gene)]
            final_common_cells[(disease_id, target_gene)] = cells
            cand_rows.append(
                {
                    "disease_id": disease_id,
                    "target_gene": target_gene,
                    "label": label,
                    "is_positive": bool(label == 1),
                    "n_common_cells": int(len(cells)),
                }
            )
    common_candidates_df = (
        pd.DataFrame(cand_rows, columns=["disease_id", "target_gene", "label", "is_positive", "n_common_cells"])
        .sort_values(["disease_id", "target_gene"])
        .reset_index(drop=True)
    )

    cells_rows: List[Dict[str, object]] = []
    for (disease_id, target_gene), cells in final_common_cells.items():
        for c in cells:
            cells_rows.append({"disease_id": disease_id, "target_gene": target_gene, "cell_id": c})
    common_cells_df = (
        pd.DataFrame(cells_rows, columns=["disease_id", "target_gene", "cell_id"])
        .sort_values(["disease_id", "target_gene", "cell_id"])
        .reset_index(drop=True)
    )

    reason_counts: Dict[str, int] = {}
    for r in avail_rows:
        if not r["in_common_candidate"]:
            for reason in str(r["exclusion_reason"]).split(";"):
                if reason:
                    reason_counts[f"pair:{reason}"] = reason_counts.get(f"pair:{reason}", 0) + 1
    for _disease_id, reasons in disease_exclusion.items():
        for reason in reasons:
            reason_counts[f"disease:{reason}"] = reason_counts.get(f"disease:{reason}", 0) + 1
    exclusion_summary_df = pd.DataFrame(
        sorted(({"scope_reason": k, "count": v} for k, v in reason_counts.items()), key=lambda x: x["scope_reason"]),
        columns=["scope_reason", "count"],
    )

    orig_diseases = set(labels_df["disease_id"].astype(str))
    orig_pairs = set((str(r.disease_id), str(r.target_gene)) for r in labels_df.itertuples(index=False))
    orig_pos = int((labels_df.groupby(["disease_id", "target_gene"])["label"].max() == 1).sum())
    flow = {
        "original_n_diseases": int(len(orig_diseases)),
        "original_n_pairs": int(len(orig_pairs)),
        "original_n_positive_pairs": int(orig_pos),
        "common_n_diseases": int(len(common_diseases_df)),
        "common_n_pairs": int(len(common_candidates_df)),
        "common_n_positive_pairs": int(common_candidates_df["is_positive"].sum()) if len(common_candidates_df) else 0,
    }

    return {
        "availability_df": availability_df,
        "common_diseases_df": common_diseases_df,
        "common_candidates_df": common_candidates_df,
        "common_cells_df": common_cells_df,
        "exclusion_summary_df": exclusion_summary_df,
        "common_cells": final_common_cells,
        "selected_disease_ids": selected_disease_ids,
        "pair_label": pair_label,
        "flow": flow,
    }
