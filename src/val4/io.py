from __future__ import annotations

from typing import Optional, Set, Tuple, Dict

import networkx as nx
import pandas as pd

try:
    from val2.legacy import load_knockdown_parquet_as_dicts as _load_val2_profiles
except Exception:  # pragma: no cover - fallback for isolated imports
    from common.profile_io import load_caliper_feature_matrix_as_dicts as _load_val2_profiles


def load_mechanism_multidigraph_from_tsv(
    path: str,
    src_col: str = "source",
    dst_col: str = "target",
    sign_col: str = "polarity",
    mechanism_col: str = "mechanism",
    sep: str = "\t",
) -> nx.MultiDiGraph:
    df = pd.read_csv(path, sep=sep)
    required = {src_col, dst_col, sign_col, mechanism_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[KG] Missing columns: {missing}. existing={list(df.columns)[:20]}")
    if len(df) == 0:
        raise ValueError(f"[KG] File is empty or has no data rows: {path}")

    G = nx.MultiDiGraph()
    src_i = df.columns.get_loc(src_col)
    dst_i = df.columns.get_loc(dst_col)
    sign_i = df.columns.get_loc(sign_col)
    mech_i = df.columns.get_loc(mechanism_col)

    for edge_id, row in enumerate(df.itertuples(index=False, name=None)):
        s = str(row[src_i]).strip()
        t = str(row[dst_i]).strip()
        if not s or not t:
            raise ValueError(f"[KG] Row {edge_id}: source and target must be non-empty.")
        pol_raw = row[sign_i]
        if isinstance(pol_raw, str):
            pol_token = pol_raw.strip().lower()
            if pol_token in {"+1", "+", "1", "pos", "activate", "activation"}:
                pol = 1
            elif pol_token in {"-1", "-", "neg", "inhibit", "inhibition"}:
                pol = -1
            else:
                try:
                    pol = int(pol_token)
                except ValueError as e:
                    raise ValueError(f"[KG] Row {edge_id}: invalid polarity={pol_raw!r}") from e
        else:
            pol = int(pol_raw)
        if pol not in {-1, 1}:
            raise ValueError(f"[KG] Row {edge_id}: polarity must be exactly -1 or +1. got={pol!r}")
        mechanism = "" if pd.isna(row[mech_i]) else str(row[mech_i])
        G.add_edge(s, t, polarity=int(pol), mechanism=mechanism, edge_id=int(edge_id))
    return G


def load_caliper_profiles_for_val4(
    parquet_path: str,
    pert_suffix: str = "_gene",
    disease_suffix: str = "_disease",
    kg_nodes: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], Dict[str, int]], Dict[str, Dict[str, int]], list[str]]:
    return _load_val2_profiles(
        parquet_path,
        pert_suffix=pert_suffix,
        disease_suffix=disease_suffix,
        kg_nodes=kg_nodes,
    )


# Backward-compatible H1 function name.
def load_knockdown_parquet_as_dicts(*args, **kwargs):
    return load_caliper_profiles_for_val4(*args, **kwargs)
