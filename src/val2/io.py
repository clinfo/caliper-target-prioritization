from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import importlib.util
import sys

import networkx as nx
import pandas as pd

from .registry import DatasetSpec


PertKey = Tuple[str, str]


@dataclass
class DataBundle:
    graph: nx.MultiDiGraph
    labels_df: pd.DataFrame
    perturbation_signatures: Dict[PertKey, Dict[str, int]]
    disease_signatures: Dict[str, Dict[str, int]]
    common_genes: list[str]


def _load_legacy_val2_module():
    module_name = "val2_legacy_module"
    if module_name in sys.modules:
        return sys.modules[module_name]

    candidate = Path(__file__).resolve().parent / "legacy.py"
    if not candidate.exists():
        raise FileNotFoundError(
            "Legacy val2 module not found inside the unified package. "
            f"Expected: {candidate}"
        )
    spec = importlib.util.spec_from_file_location(module_name, candidate)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {candidate}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_legacy_val2():
    return _load_legacy_val2_module()


def load_signed_multidigraph_with_optional_mechanism(
    path: str | Path,
    *,
    src_col: str,
    dst_col: str,
    sign_col: str,
    mechanism_col: Optional[str],
    sep: str,
) -> nx.MultiDiGraph:
    """Load a signed MultiDiGraph while optionally preserving mechanism annotations.

    When mechanism_col is present, the row index is used as a stable edge key.
    This does not change the graph structure or edge signs used by legacy scoring.
    """
    df = pd.read_csv(path, sep=sep)
    required = {src_col, dst_col, sign_col}
    if mechanism_col is not None:
        required.add(mechanism_col)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[KG] Missing columns: {missing}. existing={list(df.columns)[:20]}")
    if len(df) == 0:
        raise ValueError(f"[KG] File is empty or has no data rows: {path}")

    graph = nx.MultiDiGraph()
    src_idx = df.columns.get_loc(src_col)
    dst_idx = df.columns.get_loc(dst_col)
    sign_idx = df.columns.get_loc(sign_col)
    mechanism_idx = None if mechanism_col is None else df.columns.get_loc(mechanism_col)

    for row_index, row in enumerate(df.itertuples(index=False, name=None)):
        source = str(row[src_idx]).strip()
        target = str(row[dst_idx]).strip()
        if not source or not target:
            raise ValueError(
                f"[KG] Row {row_index}: source and target must be non-empty. "
                f"got source={source!r}, target={target!r}"
            )
        raw_sign = row[sign_idx]
        try:
            polarity = int(raw_sign) if not isinstance(raw_sign, str) else int(raw_sign.strip())
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"[KG] Row {row_index}: polarity must be -1 or 1. got {raw_sign!r}"
            ) from e
        if polarity not in (-1, 1):
            raise ValueError(
                f"[KG] Row {row_index}: polarity must be exactly -1 or 1. got {polarity}"
            )
        attrs = {"polarity": int(polarity)}
        if mechanism_idx is not None:
            raw_mechanism = row[mechanism_idx]
            attrs["mechanism"] = "" if pd.isna(raw_mechanism) else str(raw_mechanism).strip()
            attrs["edge_id"] = int(row_index)
            graph.add_edge(source, target, key=int(row_index), **attrs)
        else:
            graph.add_edge(source, target, **attrs)
    return graph


def load_data_bundle(dataset_spec: DatasetSpec) -> DataBundle:
    legacy_val2 = load_legacy_val2()
    graph = load_signed_multidigraph_with_optional_mechanism(
        dataset_spec.ppi_path,
        src_col=dataset_spec.source_col,
        dst_col=dataset_spec.target_col,
        sign_col=dataset_spec.sign_col,
        mechanism_col=dataset_spec.mechanism_col,
        sep=dataset_spec.sep,
    )
    labels_df, pert_sigs, dis_sigs, common_genes = legacy_val2.load_knockdown_parquet_as_dicts(
        str(dataset_spec.parquet_path),
        pert_suffix=dataset_spec.pert_suffix,
        disease_suffix=dataset_spec.disease_suffix,
        kg_nodes=set(graph.nodes()),
    )
    return DataBundle(
        graph=graph,
        labels_df=labels_df,
        perturbation_signatures=pert_sigs,
        disease_signatures=dis_sigs,
        common_genes=common_genes,
    )
