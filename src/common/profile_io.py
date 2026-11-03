from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


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
    # Keep zero-valued perturbation genes in the gene universe.
    out: Dict[str, int] = {}
    for c in cols:
        v = int(row[c])
        gene = str(c[: -len(suffix)])
        out[gene] = v
    return out


def load_caliper_feature_matrix_as_dicts(
    parquet_path: str,
    pert_suffix: str = "_gene",
    disease_suffix: str = "_disease",
    kg_nodes: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], Dict[str, int]], Dict[str, Dict[str, int]], List[str]]:
    try:
        df = pd.read_parquet(parquet_path, engine="pyarrow")
    except Exception:
        df = pd.read_parquet(parquet_path, engine="fastparquet")

    required = {"pert_id", "do_id", "cell_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[parquet] Missing columns: {missing}. existing={list(df.columns)[:20]}")

    pert_cols, dis_cols = _detect_profile_columns(df, pert_suffix=pert_suffix, disease_suffix=disease_suffix)
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

    df = df[["do_id", "pert_id", "cell_id", "label"] + pert_cols_common + dis_cols_common]

    if kg_nodes is not None:
        df = df[df["pert_id"].astype(str).isin(kg_nodes)]
        if len(df) == 0:
            raise ValueError("No rows left after filtering: pert_id not found in KG nodes.")

    labels_df = (
        df[["do_id", "pert_id", "cell_id", "label"]]
        .drop_duplicates()
        .rename(columns={"do_id": "disease_id", "pert_id": "target_gene"})
    )
    labels_df["disease_id"] = labels_df["disease_id"].astype(str)
    labels_df["target_gene"] = labels_df["target_gene"].astype(str)
    labels_df["cell_id"] = labels_df["cell_id"].astype(str)
    labels_df["label"] = labels_df["label"].astype(int)

    pert_sigs: Dict[Tuple[str, str], Dict[str, int]] = {}
    sub_pert = df[["pert_id", "cell_id"] + pert_cols_common].drop_duplicates(subset=["pert_id", "cell_id"])
    for _, r in sub_pert.iterrows():
        key = (str(r["pert_id"]), str(r["cell_id"]))
        pert_sigs[key] = _row_to_sig_dict_pert(r, pert_cols_common, pert_suffix)

    dis_sigs: Dict[str, Dict[str, int]] = {}
    sub_dis = df[["do_id"] + dis_cols_common].drop_duplicates(subset=["do_id"])
    for _, r in sub_dis.iterrows():
        did = str(r["do_id"])
        dis_sigs[did] = _row_to_sig_dict_disease(r, dis_cols_common, disease_suffix)

    return labels_df, pert_sigs, dis_sigs, common_genes


def load_knockdown_parquet_as_dicts(*args, **kwargs):
    return load_caliper_feature_matrix_as_dicts(*args, **kwargs)
