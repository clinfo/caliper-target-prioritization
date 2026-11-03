from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import networkx as nx
import pandas as pd

from common.io import ensure_dir, read_yaml_or_json, write_json, write_table
from common.profile_io import (
    _detect_profile_columns,
    _row_to_sig_dict_disease,
    _row_to_sig_dict_pert,
    load_caliper_feature_matrix_as_dicts,
)



def load_signed_multidigraph_from_tsv(
    path: str,
    src_col: str = "source",
    dst_col: str = "target",
    sign_col: str = "polarity",
    sep: str = "\t",
) -> nx.MultiDiGraph:
    df = pd.read_csv(path, sep=sep)
    required = {src_col, dst_col, sign_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[KG] Missing columns: {missing}. existing={list(df.columns)[:20]}")

    G = nx.MultiDiGraph()
    src_i = df.columns.get_loc(src_col)
    dst_i = df.columns.get_loc(dst_col)
    sign_i = df.columns.get_loc(sign_col)

    for row in df.itertuples(index=False, name=None):
        s = str(row[src_i])
        t = str(row[dst_i])
        pol_raw = row[sign_i]
        if isinstance(pol_raw, str):
            pol_raw = pol_raw.strip()
            pol = 1 if pol_raw in ["+1", "+", "1", "pos", "activate", "activation"] else -1
        else:
            pol = 1 if int(pol_raw) > 0 else -1
        G.add_edge(s, t, polarity=int(pol))

    return G



def load_knockdown_parquet_as_dicts(
    parquet_path: str,
    pert_suffix: str = "_gene",
    disease_suffix: str = "_disease",
    kg_nodes: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], Dict[str, int]], Dict[str, Dict[str, int]], list[str]]:
    return load_caliper_feature_matrix_as_dicts(
        parquet_path,
        pert_suffix=pert_suffix,
        disease_suffix=disease_suffix,
        kg_nodes=kg_nodes,
    )
