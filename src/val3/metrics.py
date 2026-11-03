from __future__ import annotations

import ast
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from inference.signature_reversal_paths import load_permutation_specs

from .decoy import (
    DECOY_NULL_TYPE,
    DECOY_NULL_TYPES,
    TARGET_DECOY_NULL_TYPE,
    TARGET_DECOY_SELECTION_MATCHED,
    build_target_decoy_evaluability,
    build_target_decoy_matching_acceptance,
    build_target_decoy_matching_balance,
    build_target_decoy_triplet_evaluability,
    canonical_decoy_null_type,
    load_directed_kg_graph,
    null_representative_kind,
    summarize_target_decoy_delta_by_n_evaluable,
    summarize_target_decoy_evaluability,
    target_decoy_acceptance_metadata,
)
from .io import ensure_dir, read_parquet_any, signed_edge_set_to_json_str, write_json
from .null1_engine import (
    TrieCache,
    null1_signed_edge_df,
    prepare_infer_input,
)
from .null1_parallel import Null1ParallelEvaluator
from .streaming import CellpairDebugWriter, prune_non_v23_null_types, write_parquet_zstd


# Edge scopes for the first-edge exclusion sensitivity analysis.
# ``all_edges`` is the primary analysis and its outputs are unchanged; the
# ``first_edge_excluded`` scope only replaces the edge set used by the overlap
# statistic, never the evaluable triplet set, the contributing cell pairs, or the
# cell-pair weights.
EDGE_SCOPE_ALL = "all_edges"
EDGE_SCOPE_FIRST_EDGE_EXCLUDED = "first_edge_excluded"
EDGE_SCOPES: tuple[str, ...] = (EDGE_SCOPE_ALL, EDGE_SCOPE_FIRST_EDGE_EXCLUDED)
EDGE_COLUMN_SUFFIX_BY_SCOPE: dict[str, str] = {
    EDGE_SCOPE_ALL: "",
    EDGE_SCOPE_FIRST_EDGE_EXCLUDED: "_first_edge_excluded",
}
FIRST_EDGE_EXCLUDED_SUFFIX = EDGE_COLUMN_SUFFIX_BY_SCOPE[EDGE_SCOPE_FIRST_EDGE_EXCLUDED]


def edge_column_suffix_for_scope(edge_scope: str) -> str:
    try:
        return EDGE_COLUMN_SUFFIX_BY_SCOPE[str(edge_scope)]
    except KeyError as e:
        raise ValueError(f"Unsupported edge scope: {edge_scope!r}") from e


def first_edge_excluded_edges(
    signed_edges: Any,
    *,
    root: Any,
) -> set[tuple[str, str, int]]:
    """Drop every signed edge emanating from the path root.

    Accepted paths are simple, so the root only ever appears as a path start and
    ``u == root`` selects exactly the first hop of each accepted path.
    """
    root_s = str(root)
    return {edge for edge in _edge_set(signed_edges) if str(edge[0]) != root_s}


def _to_python_obj(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, (list, tuple, dict)):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            pass
        try:
            return ast.literal_eval(s)
        except Exception:
            return None
    return x


def infer_path_count(row: Mapping[str, Any]) -> int:
    paths_obj = _to_python_obj(row.get("signature_reversal_paths")) or []
    if not isinstance(paths_obj, (list, tuple)):
        return 0
    return int(len(paths_obj))


def _edge_sign_from_polarities(prev_pol: int, curr_pol: int) -> int:
    return 1 if int(prev_pol) == int(curr_pol) else -1


def _initial_polarity_for_mode(mode: str) -> int:
    if mode == "knockdown":
        return -1
    if mode == "overexpression":
        return 1
    raise ValueError(f"Unsupported mode: {mode}")


def _signed_edge_payload_from_infer_row(
    row: Mapping[str, Any],
    *,
    mode: str,
) -> tuple[set[tuple[str, str, int]], list[set[tuple[str, str, int]]], int]:
    """Build signed-edge outputs from one infer row with a single path/polarity parse.

    This preserves the legacy semantics of ``infer_path_count``: ``n_paths`` is
    the raw number of entries in ``signature_reversal_paths`` whenever that
    column is list-like, irrespective of whether the row later yields any valid
    signed edges.
    """
    source = str(row.get("pert_id"))
    paths_obj = _to_python_obj(row.get("signature_reversal_paths")) or []
    n_paths = int(len(paths_obj)) if isinstance(paths_obj, (list, tuple)) else 0

    if source in {"", "None", "nan"}:
        return set(), [], n_paths
    initial_polarity = _initial_polarity_for_mode(mode)

    pols_obj = _to_python_obj(row.get("signature_reversal_polarities")) or []
    if not isinstance(paths_obj, (list, tuple)) or not isinstance(pols_obj, (list, tuple)):
        return set(), [], n_paths

    signed_edges: set[tuple[str, str, int]] = set()
    path_edges: list[set[tuple[str, str, int]]] = []
    for path_nodes, node_pols in zip(paths_obj, pols_obj):
        if not isinstance(path_nodes, (list, tuple)) or not path_nodes:
            continue
        if not isinstance(node_pols, (list, tuple)) or len(node_pols) != len(path_nodes):
            continue
        prev_node = source
        prev_pol = int(initial_polarity)
        edges: set[tuple[str, str, int]] = set()
        for node, curr_pol in zip(path_nodes, node_pols):
            curr_node = str(node)
            curr_pol_i = int(curr_pol)
            edge = (str(prev_node), curr_node, _edge_sign_from_polarities(prev_pol, curr_pol_i))
            signed_edges.add(edge)
            edges.add(edge)
            prev_node = curr_node
            prev_pol = curr_pol_i
        path_edges.append(edges)
    return signed_edges, path_edges, n_paths


def signed_edges_from_infer_row(row: Mapping[str, Any], *, mode: str) -> set[tuple[str, str, int]]:
    signed_edges, _, _ = _signed_edge_payload_from_infer_row(row, mode=mode)
    return signed_edges


def signed_edge_paths_from_infer_row(row: Mapping[str, Any], *, mode: str) -> list[set[tuple[str, str, int]]]:
    _, signed_edge_paths, _ = _signed_edge_payload_from_infer_row(row, mode=mode)
    return signed_edge_paths


def _validate_infer_parquet_path_column_readable(df: pd.DataFrame, *, path: str | Path) -> None:
    col = "signature_reversal_paths"
    if col not in df.columns or "num_nodes_in_paths" not in df.columns:
        return
    non_null = int(df[col].notna().sum())
    if non_null == 0 and bool((df["num_nodes_in_paths"] > 0).any()):
        raise ValueError(
            f"Infer parquet column {col!r} is entirely null/NA while num_nodes_in_paths>0. "
            f"Re-run infer_signature_reversal_paths.py. File: {path}"
        )


def _attach_signed_edge_columns(
    out: pd.DataFrame,
    *,
    mode: str,
    include_signed_edge_paths: bool,
    include_first_edge_excluded: bool,
) -> pd.DataFrame:
    """Single place where the Val3 side-level signed-edge columns are derived.

    ``signed_edges_first_edge_excluded`` is defined here and nowhere else; the
    Null 1 traversal engine constructs the same set directly and is pinned to this
    definition by the equivalence tests.
    """
    records = out.to_dict(orient="records")
    payloads = [_signed_edge_payload_from_infer_row(rec, mode=mode) for rec in records]
    out["signed_edges"] = [signed_edges for signed_edges, _, _ in payloads]
    if include_signed_edge_paths:
        out["signed_edge_paths"] = [signed_edge_paths for _, signed_edge_paths, _ in payloads]
    out["n_paths"] = [n_paths for _, _, n_paths in payloads]
    if include_first_edge_excluded:
        out["signed_edges" + FIRST_EDGE_EXCLUDED_SUFFIX] = [
            first_edge_excluded_edges(signed_edges, root=rec.get("pert_id"))
            for (signed_edges, _, _), rec in zip(payloads, records)
        ]
    if "cell_id" not in out.columns and "pert_cell" in out.columns:
        out["cell_id"] = out["pert_cell"]
    return out


def load_infer_signed_edge_df(
    path: str | Path,
    *,
    mode: str,
    include_signed_edge_paths: bool = True,
    include_first_edge_excluded: bool = False,
) -> pd.DataFrame:
    df = read_parquet_any(path).copy()
    _validate_infer_parquet_path_column_readable(df, path=path)
    return _attach_signed_edge_columns(
        df,
        mode=mode,
        include_signed_edge_paths=include_signed_edge_paths,
        include_first_edge_excluded=include_first_edge_excluded,
    )


def signed_edge_df_from_infer_df(
    df: pd.DataFrame,
    *,
    mode: str,
    path: str | Path | None = None,
    include_signed_edge_paths: bool = True,
    include_first_edge_excluded: bool = False,
) -> pd.DataFrame:
    """Attach Val3 signed-edge columns to an in-memory infer DataFrame.

    This is the in-process equivalent of ``load_infer_signed_edge_df`` and uses
    the same parser/edge reconstruction path, so downstream metric values are
    unchanged relative to reading an infer parquet from disk.
    """
    out = df.copy()
    _validate_infer_parquet_path_column_readable(out, path=path or "<in-memory>")
    return _attach_signed_edge_columns(
        out,
        mode=mode,
        include_signed_edge_paths=include_signed_edge_paths,
        include_first_edge_excluded=include_first_edge_excluded,
    )


def infer_signed_edge_df_inprocess(
    *,
    input_path: str | Path,
    graph: nx.MultiDiGraph,
    mode: str,
    cutoff: int,
    gap: int,
    permutation_specs_path: str | Path | None = None,
    path_cache_dir: str | Path | None = None,
    path_strategy: str = "simple",
    path_pool_cache: dict[tuple[str, str, int, str], list[dict[str, int]]] | None = None,
    include_signed_edge_paths: bool = True,
    include_first_edge_excluded: bool = False,
) -> pd.DataFrame:
    """Run signature-reversal inference in-process and return signed-edge rows.

    The accepted-path logic is delegated to ``signature_reversal_paths_from_graph``.
    Only the execution boundary changes: no subprocess, temporary config, or infer
    parquet is materialized. ``path_pool_cache`` shares the exact path-polarity
    skeleton across observed and Null 1 iterations within the same Python process.

    This is the reference engine kept for equivalence testing; the production Null 1
    path uses ``val3.null1_engine``.
    """
    from inference.signature_reversal_paths import load_permutation_specs, signature_reversal_paths_from_graph

    input_df = read_parquet_any(input_path)
    permutation_specs = load_permutation_specs(permutation_specs_path) if permutation_specs_path is not None else None
    infer_df = signature_reversal_paths_from_graph(
        df=input_df,
        graph=graph,
        mode=mode,
        cutoff=int(cutoff),
        gap=int(gap),
        permutation_specs=permutation_specs,
        path_cache_dir=path_cache_dir,
        path_strategy=path_strategy,
        path_pool_cache=path_pool_cache,
    )
    return signed_edge_df_from_infer_df(
        infer_df,
        mode=mode,
        path="<in-process>",
        include_signed_edge_paths=include_signed_edge_paths,
        include_first_edge_excluded=include_first_edge_excluded,
    )


def _infer_part_paths(path: str | Path) -> list[Path]:
    p = Path(path)
    if p.is_dir():
        parts = sorted(p.glob("part-*.parquet"))
        return parts if parts else sorted(p.glob("*.parquet"))
    return [p]


def _paired_infer_part_paths(drug_path: str | Path, pert_path: str | Path) -> list[tuple[Path, Path]]:
    d = Path(drug_path)
    p = Path(pert_path)
    if d.is_dir() and p.is_dir():
        d_parts = {x.name: x for x in _infer_part_paths(d)}
        p_parts = {x.name: x for x in _infer_part_paths(p)}
        return [(d_parts[name], p_parts[name]) for name in sorted(set(d_parts) & set(p_parts))]
    return [(d, p)]


def metric_value_from_counts(metric: str, *, n_intersection: int, n_drug_edges: int, n_pert_edges: int) -> float:
    """Metric value from cardinalities only.

    ``len(a | b) == len(a) + len(b) - len(a & b)`` by inclusion-exclusion, so this
    returns exactly the same float as the set-based metric functions while letting
    callers skip building the union set.
    """
    n_inter = int(n_intersection)
    n_a = int(n_drug_edges)
    n_b = int(n_pert_edges)
    if metric == "jaccard":
        if n_a == 0 and n_b == 0:
            return float("nan")
        if n_a == 0 or n_b == 0:
            return 0.0
        return n_inter / (n_a + n_b - n_inter)
    raise ValueError(f"Unsupported metric for global evaluation: {metric}")


def _pair_edge_counts(ea: set[Any], eb: set[Any]) -> tuple[int, int, int, int]:
    n_a = len(ea)
    n_b = len(eb)
    # Intersect the smaller set into the larger one; set.intersection already does
    # this internally, but choosing explicitly keeps the union count derivable.
    n_inter = len(ea & eb) if n_a <= n_b else len(eb & ea)
    return n_inter, n_a + n_b - n_inter, n_a, n_b


def metric_jaccard(a: set[tuple[str, str, int]], b: set[tuple[str, str, int]]) -> float:
    if not a and not b:
        return float("nan")
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_cell_pairs(drug_df: pd.DataFrame, pert_df: pd.DataFrame, *, cell_pair_mode: str = "cartesian") -> pd.DataFrame:
    common_cols = [c for c in ["triplet_id", "drug_id", "target_id", "direction"] if c in drug_df.columns and c in pert_df.columns]
    left = drug_df.copy()
    right = pert_df.copy()
    if cell_pair_mode == "cartesian":
        merged = left.merge(right, on=common_cols, how="inner", suffixes=("_drug", "_pert"))
    elif cell_pair_mode == "shared_only":
        if "cell_id" not in left.columns or "cell_id" not in right.columns:
            raise ValueError("shared_only requires 'cell_id' on both sides")
        merged = left.merge(right, on=common_cols + ["cell_id"], how="inner", suffixes=("_drug", "_pert"))
        merged["cell_id_drug"] = merged["cell_id"]
        merged["cell_id_pert"] = merged["cell_id"]
    else:
        raise ValueError(f"Unsupported cell_pair_mode: {cell_pair_mode}")
    # Pandas suffixing creates these automatically in cartesian mode.
    if "n_paths_drug" in merged.columns:
        merged = merged.rename(columns={"n_paths_drug": "n_drug_paths", "n_paths_pert": "n_pert_paths"})
    if "cell_id_drug" not in merged.columns and "cell_id_x" in merged.columns:
        merged = merged.rename(columns={"cell_id_x": "cell_id_drug", "cell_id_y": "cell_id_pert"})
    if "n_drug_paths" not in merged.columns:
        if "n_paths_x" in merged.columns:
            merged = merged.rename(columns={"n_paths_x": "n_drug_paths", "n_paths_y": "n_pert_paths"})
    return merged


def cellpair_filter_diagnostics(pair_df: pd.DataFrame) -> pd.DataFrame:
    if pair_df.empty:
        cols = [
            "triplet_id", "drug_id", "target_id", "direction", "n_cell_pairs_raw", "n_cell_pairs_evaluable",
            "n_cell_pairs_excluded_no_drug_path", "n_cell_pairs_excluded_no_pert_path", "n_cell_pairs_excluded_no_both_path",
        ]
        return pd.DataFrame(columns=cols)
    group_cols = [c for c in ["triplet_id", "drug_id", "target_id", "direction"] if c in pair_df.columns]
    tmp = pair_df.copy()
    nd = pd.to_numeric(tmp.get("n_drug_paths", 0), errors="coerce").fillna(0).astype(int)
    npth = pd.to_numeric(tmp.get("n_pert_paths", 0), errors="coerce").fillna(0).astype(int)
    tmp["_no_drug"] = nd <= 0
    tmp["_no_pert"] = npth <= 0
    tmp["_no_both"] = tmp["_no_drug"] & tmp["_no_pert"]
    tmp["_evaluable"] = ~(tmp["_no_drug"] | tmp["_no_pert"])
    out = tmp.groupby(group_cols, dropna=False).agg(
        n_cell_pairs_raw=("triplet_id", "size"),
        n_cell_pairs_evaluable=("_evaluable", "sum"),
        n_cell_pairs_excluded_no_drug_path=("_no_drug", "sum"),
        n_cell_pairs_excluded_no_pert_path=("_no_pert", "sum"),
        n_cell_pairs_excluded_no_both_path=("_no_both", "sum"),
    ).reset_index()
    for c in out.columns:
        if c.startswith("n_cell_pairs"):
            out[c] = out[c].astype(int)
    return out


def filter_evaluable_cell_pairs(
    pair_df: pd.DataFrame,
    *,
    require_both_nonempty_inferred_paths: bool = True,
    require_both_nonempty_signed_edges: bool = False,
) -> pd.DataFrame:
    if pair_df.empty:
        return pair_df.copy()
    mask = pd.Series(True, index=pair_df.index)
    if require_both_nonempty_inferred_paths:
        mask &= pd.to_numeric(pair_df["n_drug_paths"], errors="coerce").fillna(0).gt(0)
        mask &= pd.to_numeric(pair_df["n_pert_paths"], errors="coerce").fillna(0).gt(0)
    if require_both_nonempty_signed_edges:
        mask &= pair_df["signed_edges_drug"].map(lambda x: len(x if isinstance(x, set) else set(x or []))).gt(0)
        mask &= pair_df["signed_edges_pert"].map(lambda x: len(x if isinstance(x, set) else set(x or []))).gt(0)
    return pair_df.loc[mask].copy()


def compute_cellpair_metrics(
    drug_df: pd.DataFrame,
    pert_df: pd.DataFrame,
    *,
    cell_pair_mode: str = "cartesian",
    filters: Mapping[str, Any] | None = None,
    iteration: int | None = None,
    progress: bool = False,
    tqdm_kwargs: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_raw = build_cell_pairs(drug_df, pert_df, cell_pair_mode=cell_pair_mode)
    diag = cellpair_filter_diagnostics(pair_raw)
    if pair_raw.empty:
        return pair_raw, diag
    filters = dict(filters or {})
    pair_df = filter_evaluable_cell_pairs(
        pair_raw,
        require_both_nonempty_inferred_paths=bool(filters.get("require_both_nonempty_inferred_paths", True)),
        require_both_nonempty_signed_edges=bool(filters.get("require_both_nonempty_signed_edges", False)),
    )
    if pair_df.empty:
        if iteration is not None:
            pair_df = pair_df.copy()
            pair_df["iteration"] = int(iteration)
            diag = diag.copy()
            diag["iteration"] = int(iteration)
        return pair_df, diag

    rows = pair_df[["signed_edges_drug", "signed_edges_pert"]].itertuples(index=False, name=None)
    if progress:
        rows = tqdm(list(rows), desc="cell-pair metrics", unit="pair", **(tqdm_kwargs or {}))
    else:
        rows = list(rows)

    jaccard_vals: list[float] = []
    n_intersection: list[int] = []
    n_union: list[int] = []
    n_drug_edges: list[int] = []
    n_pert_edges: list[int] = []
    # Within a triplet many cells share the same edge set, so the set algebra is
    # memoized per unique (drug edge set, pert edge set) pair.
    pair_cache: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    group_ids: dict[tuple[tuple[str, str, int], ...], int] = {}

    def _group_id(edges: set[Any]) -> int:
        key = tuple(sorted(edges))
        gid = group_ids.get(key)
        if gid is None:
            gid = len(group_ids)
            group_ids[key] = gid
        return gid

    for a, b in rows:
        ea = a if isinstance(a, set) else set(a or [])
        eb = b if isinstance(b, set) else set(b or [])
        cache_key = (_group_id(ea), _group_id(eb))
        counts = pair_cache.get(cache_key)
        if counts is None:
            counts = _pair_edge_counts(ea, eb)
            pair_cache[cache_key] = counts
        n_inter, n_uni, n_a, n_b = counts
        n_intersection.append(n_inter)
        n_union.append(n_uni)
        n_drug_edges.append(n_a)
        n_pert_edges.append(n_b)
        jaccard_vals.append(metric_value_from_counts("jaccard", n_intersection=n_inter, n_drug_edges=n_a, n_pert_edges=n_b))

    pair_df = pair_df.copy()
    pair_df["jaccard"] = jaccard_vals
    pair_df["n_intersection"] = n_intersection
    pair_df["n_union"] = n_union
    pair_df["n_drug_edges"] = n_drug_edges
    pair_df["n_pert_edges"] = n_pert_edges
    if iteration is not None:
        pair_df["iteration"] = int(iteration)
        diag = diag.copy()
        diag["iteration"] = int(iteration)
    return pair_df, diag


def cellpair_metrics_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("signed_edges_drug", "signed_edges_pert"):
        if col in out.columns:
            out[col] = out[col].map(signed_edge_set_to_json_str)
    for col in ("signed_edge_paths_drug", "signed_edge_paths_pert"):
        if col in out.columns:
            out[col] = out[col].map(signed_edge_paths_to_json_str)
    return out


def signed_edge_paths_to_json_str(paths: Any) -> str:
    if paths is None or (isinstance(paths, float) and pd.isna(paths)):
        return "[]"
    if isinstance(paths, str):
        return paths if paths.strip() else "[]"
    normalized = []
    for edge_set in paths or []:
        edges = edge_set if isinstance(edge_set, set) else set(edge_set or [])
        normalized.append(sorted((str(u), str(v), int(sig)) for (u, v, sig) in edges))
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def _nanmean(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce")
    return float(x.mean()) if x.notna().any() else float("nan")


def _nanmedian(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce")
    return float(x.median()) if x.notna().any() else float("nan")


def aggregate_triplet_metrics(cellpair_df: pd.DataFrame, diag_df: pd.DataFrame | None = None, *, include_iteration: bool = False) -> pd.DataFrame:
    if cellpair_df.empty:
        base = pd.DataFrame()
    else:
        group_cols = ["triplet_id"]
        for c in ["drug_id", "target_id", "direction"]:
            if c in cellpair_df.columns:
                group_cols.append(c)
        if include_iteration and "iteration" in cellpair_df.columns:
            group_cols.append("iteration")
        base = cellpair_df.groupby(group_cols, dropna=False).agg(
            n_cell_pairs=("triplet_id", "size"),
            jaccard=("jaccard", _nanmean),
            n_valid_jaccard=("jaccard", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            mean_n_drug_paths=("n_drug_paths", "mean"),
            mean_n_pert_paths=("n_pert_paths", "mean"),
            mean_n_drug_edges=("n_drug_edges", "mean"),
            mean_n_pert_edges=("n_pert_edges", "mean"),
        ).reset_index()
    if diag_df is not None and not diag_df.empty:
        keys = [c for c in ["triplet_id", "drug_id", "target_id", "direction", "iteration"] if c in diag_df.columns and (base.empty or c in base.columns)]
        if base.empty:
            base = diag_df.copy()
            base["n_cell_pairs"] = base.get("n_cell_pairs_evaluable", 0)
            base["jaccard"] = float("nan")
            base["n_valid_jaccard"] = 0
            for c in ["mean_n_drug_paths", "mean_n_pert_paths", "mean_n_drug_edges", "mean_n_pert_edges"]:
                base[c] = float("nan")
        elif keys:
            base = base.merge(diag_df, on=keys, how="left")
    return base


def empirical_p_value(true_value: float, baseline_values: Sequence[float]) -> float:
    if pd.isna(true_value):
        return float("nan")
    xs = pd.Series(list(baseline_values), dtype=float)
    xs = xs[xs.notna()]
    if xs.empty:
        return float("nan")
    return float((1 + int((xs >= float(true_value)).sum())) / (len(xs) + 1))


def compare_true_vs_baseline(true_triplet_df: pd.DataFrame, baseline_triplet_df: pd.DataFrame) -> pd.DataFrame:
    if true_triplet_df.empty:
        return pd.DataFrame()
    if baseline_triplet_df.empty:
        out = true_triplet_df.copy()
        out["baseline_median_jaccard"] = float("nan")
        out["delta_jaccard"] = float("nan")
        out["p_jaccard"] = float("nan")
        return out

    if "null_type" not in baseline_triplet_df.columns:
        baseline_triplet_df = baseline_triplet_df.copy()
        baseline_triplet_df["null_type"] = "global_paired_gene_label_permutation"

    outputs: list[pd.DataFrame] = []
    for null_type, base_sub in baseline_triplet_df.groupby("null_type", dropna=False):
        base_medians = base_sub.groupby("triplet_id", dropna=False).agg(
            baseline_median_jaccard=("jaccard", _nanmedian),
        ).reset_index()
        out = true_triplet_df.merge(base_medians, on="triplet_id", how="left")
        out["null_type"] = str(null_type)
        out["delta_jaccard"] = out["jaccard"] - out["baseline_median_jaccard"]
        base_map = {tid: sub for tid, sub in base_sub.groupby("triplet_id", dropna=False)}
        pvals = []
        for tid, val in out[["triplet_id", "jaccard"]].itertuples(index=False, name=None):
            base_tid = base_map.get(tid)
            pvals.append(
                empirical_p_value(
                    float(val) if not pd.isna(val) else float("nan"),
                    [] if base_tid is None else base_tid["jaccard"].tolist(),
                )
            )
        out["p_jaccard"] = pvals
        outputs.append(out)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()



def _required_global_pair_columns(pair_df: pd.DataFrame) -> None:
    required = {"num_nodes_in_paths_drug", "num_nodes_in_paths_pert", "n_drug_paths", "n_pert_paths"}
    missing = required - set(pair_df.columns)
    if missing:
        raise ValueError(
            "Global evaluation requires infer outputs with coverage/path columns. "
            f"Missing columns after cell-pair merge: {sorted(missing)}"
        )


def _triplet_group_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in ["triplet_id", "drug_id", "target_id", "direction"] if c in df.columns]


def build_global_evaluable_triplets(
    true_drug_df: pd.DataFrame,
    true_pert_df: pd.DataFrame,
    *,
    cell_pair_mode: str,
    min_candidate_nodes: int,
) -> pd.DataFrame:
    """Build the fixed coverage-based triplet set from observed raw cell-pairs.

    This table is intentionally based on candidate-node coverage before applying
    path-existence filters, so the fixed set is not conditioned on accepted paths.
    """
    pair_raw = build_cell_pairs(true_drug_df, true_pert_df, cell_pair_mode=cell_pair_mode)
    if pair_raw.empty:
        raise ValueError("Fixed-coverage evaluation cannot build an evaluable set: observed raw cell-pair table is empty.")
    _required_global_pair_columns(pair_raw)

    group_cols = _triplet_group_cols(pair_raw)
    tmp = pair_raw.copy()
    tmp["candidate_nodes_drug"] = pd.to_numeric(tmp["num_nodes_in_paths_drug"], errors="coerce").fillna(0).astype(int)
    tmp["candidate_nodes_pert"] = pd.to_numeric(tmp["num_nodes_in_paths_pert"], errors="coerce").fillna(0).astype(int)
    tmp["has_drug_path"] = pd.to_numeric(tmp["n_drug_paths"], errors="coerce").fillna(0).gt(0)
    tmp["has_pert_path"] = pd.to_numeric(tmp["n_pert_paths"], errors="coerce").fillna(0).gt(0)
    tmp["both_paths"] = tmp["has_drug_path"] & tmp["has_pert_path"]
    out = tmp.groupby(group_cols, dropna=False).agg(
        n_raw_cell_pairs=("triplet_id", "size"),
        n_path_existing_cell_pairs=("both_paths", "sum"),
        candidate_nodes_drug_min=("candidate_nodes_drug", "min"),
        candidate_nodes_pert_min=("candidate_nodes_pert", "min"),
    ).reset_index()
    out["candidate_nodes_min"] = out[["candidate_nodes_drug_min", "candidate_nodes_pert_min"]].min(axis=1)
    out["is_global_evaluable"] = (
        pd.to_numeric(out["candidate_nodes_min"], errors="coerce").fillna(0).ge(int(min_candidate_nodes))
        & pd.to_numeric(out["n_raw_cell_pairs"], errors="coerce").fillna(0).ge(1)
    )
    out["evaluable_set_type"] = "fixed_coverage"
    for c in ["n_raw_cell_pairs", "n_path_existing_cell_pairs", "candidate_nodes_drug_min", "candidate_nodes_pert_min", "candidate_nodes_min"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    if int(out["is_global_evaluable"].sum()) < 1:
        raise ValueError(
            "Fixed-coverage evaluation found no evaluable triplets. "
            f"Relax evaluation.evaluable_sets.fixed_coverage.min_candidate_nodes={min_candidate_nodes} or inspect coverage."
        )
    return out


def build_path_existing_evaluable_triplets(
    true_drug_df: pd.DataFrame,
    true_pert_df: pd.DataFrame,
    *,
    cell_pair_mode: str,
) -> pd.DataFrame:
    """Build triplets with at least one observed drug/pert cell-pair where both sides have paths."""
    pair_raw = build_cell_pairs(true_drug_df, true_pert_df, cell_pair_mode=cell_pair_mode)
    if pair_raw.empty:
        raise ValueError("Path-existing evaluation cannot build an evaluable set: observed raw cell-pair table is empty.")
    _required_global_pair_columns(pair_raw)
    group_cols = _triplet_group_cols(pair_raw)
    tmp = pair_raw.copy()
    tmp["candidate_nodes_drug"] = pd.to_numeric(tmp["num_nodes_in_paths_drug"], errors="coerce").fillna(0).astype(int)
    tmp["candidate_nodes_pert"] = pd.to_numeric(tmp["num_nodes_in_paths_pert"], errors="coerce").fillna(0).astype(int)
    tmp["has_drug_path"] = pd.to_numeric(tmp["n_drug_paths"], errors="coerce").fillna(0).gt(0)
    tmp["has_pert_path"] = pd.to_numeric(tmp["n_pert_paths"], errors="coerce").fillna(0).gt(0)
    tmp["both_paths"] = tmp["has_drug_path"] & tmp["has_pert_path"]
    out = tmp.groupby(group_cols, dropna=False).agg(
        n_raw_cell_pairs=("triplet_id", "size"),
        n_path_existing_cell_pairs=("both_paths", "sum"),
        candidate_nodes_drug_min=("candidate_nodes_drug", "min"),
        candidate_nodes_pert_min=("candidate_nodes_pert", "min"),
    ).reset_index()
    out["candidate_nodes_min"] = out[["candidate_nodes_drug_min", "candidate_nodes_pert_min"]].min(axis=1)
    out["is_global_evaluable"] = pd.to_numeric(out["n_path_existing_cell_pairs"], errors="coerce").fillna(0).gt(0)
    out["evaluable_set_type"] = "path_existing_pairs"
    for c in ["n_raw_cell_pairs", "n_path_existing_cell_pairs", "candidate_nodes_drug_min", "candidate_nodes_pert_min", "candidate_nodes_min"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    if int(out["is_global_evaluable"].sum()) < 1:
        raise ValueError("Path-existing evaluation found no observed triplets with both-side paths.")
    return out


def _edge_set(x: Any) -> set[tuple[str, str, int]]:
    return x if isinstance(x, set) else set(x or [])


def _path_list(x: Any) -> list[set[tuple[str, str, int]]]:
    if isinstance(x, list):
        return [e if isinstance(e, set) else set(e or []) for e in x]
    if isinstance(x, tuple):
        return [e if isinstance(e, set) else set(e or []) for e in x]
    return []


def _initial_polarity_for_mode(mode: str) -> int:
    if mode == "knockdown":
        return -1
    if mode == "overexpression":
        return 1
    raise ValueError(f"Unsupported mode: {mode}")


def _load_kg_graph(kg_tsv: str | Path) -> nx.MultiDiGraph:
    # Shared with Target-Decoy matching so out-degree and reach are computed on the
    # same graph representation as path inference.
    return load_directed_kg_graph(kg_tsv)


def _profile_valid_genes(path: str | Path | None) -> set[str] | None:
    if path is None:
        return None
    df = read_parquet_any(path)
    gene_cols = {str(c)[:-5] for c in df.columns if str(c).endswith("_gene")}
    disease_cols = {str(c)[:-8] for c in df.columns if str(c).endswith("_disease")}
    valid = gene_cols & disease_cols
    return valid or None


def _path_signed_edge_set(
    graph: nx.MultiDiGraph,
    path: Sequence[tuple[str, str, int]],
    *,
    initial_polarity: int,
) -> set[tuple[str, str, int]]:
    prev_pol = int(initial_polarity)
    signed_edges: set[tuple[str, str, int]] = set()
    for u, v, key in path:
        curr_pol = prev_pol * int(graph[u][v][key]["polarity"])
        signed_edges.add((str(u), str(v), _edge_sign_from_polarities(prev_pol, curr_pol)))
        prev_pol = curr_pol
    return signed_edges


def _valid_gene_digest(valid_genes: set[str] | None) -> str:
    if valid_genes is None:
        return "kg_only"
    import hashlib

    h = hashlib.sha256()
    for gene in sorted(map(str, valid_genes)):
        h.update(gene.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def build_shared_target_khop_path_pool(
    *,
    graph: nx.MultiDiGraph,
    targets: set[str],
    mode: str,
    cutoff: int,
    valid_genes: set[str] | None,
    path_strategy: str,
    side: str,
    pool_cache: dict[tuple[Any, ...], tuple[dict[str, dict[int, list[set[tuple[str, str, int]]]]], pd.DataFrame]] | None = None,
) -> tuple[dict[str, dict[int, list[set[tuple[str, str, int]]]]], pd.DataFrame]:
    """Build (or reuse) the target k-hop structural pool for one side.

    ``_path_signed_edge_set`` yields ``sign == KG edge polarity`` regardless of the
    initial polarity, so two sides that agree on graph, cutoff, path strategy and
    valid-gene set share the identical candidate pool and only differ in the
    ``side`` label of the diagnostics rows. The candidate order is preserved so the
    sampling RNG draws exactly the same indices as the per-side build.
    """
    if pool_cache is None:
        return _build_target_khop_path_pool(
            graph=graph,
            targets=targets,
            mode=mode,
            cutoff=int(cutoff),
            valid_genes=valid_genes,
            path_strategy=str(path_strategy),
            side=side,
        )
    key = (
        id(graph),
        int(cutoff),
        str(path_strategy),
        _valid_gene_digest(valid_genes),
        tuple(sorted(map(str, targets))),
    )
    cached = pool_cache.get(key)
    if cached is None:
        cached = _build_target_khop_path_pool(
            graph=graph,
            targets=targets,
            mode=mode,
            cutoff=int(cutoff),
            valid_genes=valid_genes,
            path_strategy=str(path_strategy),
            side=side,
        )
        pool_cache[key] = cached
    pool, diag = cached
    diag = diag.copy()
    if not diag.empty:
        diag["side"] = side
    return pool, diag


def _build_target_khop_path_pool(
    *,
    graph: nx.MultiDiGraph,
    targets: set[str],
    mode: str,
    cutoff: int,
    valid_genes: set[str] | None,
    path_strategy: str,
    side: str,
) -> tuple[dict[str, dict[int, list[set[tuple[str, str, int]]]]], pd.DataFrame]:
    if path_strategy not in {"simple", "shortest"}:
        raise ValueError("path_strategy must be 'simple' or 'shortest'")
    pool: dict[str, dict[int, list[set[tuple[str, str, int]]]]] = {}
    rows: list[dict[str, Any]] = []
    initial_polarity = _initial_polarity_for_mode(mode)
    kg_nodes = set(map(str, graph.nodes))
    valid = set(map(str, valid_genes)) & kg_nodes if valid_genes is not None else None

    for target in sorted(map(str, targets)):
        by_length: dict[int, list[set[tuple[str, str, int]]]] = {}
        if target in graph:
            lengths = nx.single_source_shortest_path_length(graph, target, int(cutoff))
            reachable = set(lengths)
            reachable.discard(target)
            if path_strategy == "simple":
                path_iter = nx.all_simple_edge_paths(graph, source=target, target=reachable, cutoff=int(cutoff))
                for path in path_iter:
                    if valid is not None and any(str(v) not in valid for _, v, _ in path):
                        continue
                    edge_set = _path_signed_edge_set(graph, path, initial_polarity=initial_polarity)
                    if not edge_set:
                        continue
                    by_length.setdefault(int(len(edge_set)), []).append(edge_set)
            else:
                for end in sorted(reachable):
                    shortest_len = int(lengths[end])
                    for path in nx.all_simple_edge_paths(graph, source=target, target=end, cutoff=shortest_len):
                        if len(path) != shortest_len:
                            continue
                        if valid is not None and any(str(v) not in valid for _, v, _ in path):
                            continue
                        edge_set = _path_signed_edge_set(graph, path, initial_polarity=initial_polarity)
                        if not edge_set:
                            continue
                        by_length.setdefault(int(len(edge_set)), []).append(edge_set)
        pool[target] = by_length
        if by_length:
            for path_length, paths in sorted(by_length.items()):
                rows.append({
                    "side": side,
                    "target_id": target,
                    "path_length": int(path_length),
                    "n_structural_paths": int(len(paths)),
                    "pool_scope": "target_khop",
                    "pool_filter": "valid_profile_genes" if valid is not None else "kg_only",
                })
        else:
            rows.append({
                "side": side,
                "target_id": target,
                "path_length": 0,
                "n_structural_paths": 0,
                "pool_scope": "target_khop",
                "pool_filter": "valid_profile_genes" if valid is not None else "kg_only",
            })
    return pool, pd.DataFrame(rows)


def _add_observed_pool_diagnostics(
    diag: pd.DataFrame,
    df: pd.DataFrame,
    pool: dict[str, dict[int, list[set[tuple[str, str, int]]]]],
    *,
    side: str,
) -> pd.DataFrame:
    observed_rows: list[dict[str, Any]] = []
    for target, path_list in df[["target_id", "signed_edge_paths"]].itertuples(index=False, name=None):
        target_s = str(target)
        for edge_set in _path_list(path_list):
            if not edge_set:
                continue
            length = int(len(edge_set))
            candidates = {frozenset(x) for x in pool.get(target_s, {}).get(length, [])}
            observed_rows.append({
                "side": side,
                "target_id": target_s,
                "path_length": length,
                "observed_path_present_in_pool": frozenset(edge_set) in candidates,
            })
    if observed_rows:
        obs = pd.DataFrame(observed_rows)
        obs_agg = obs.groupby(["side", "target_id", "path_length"], dropna=False).agg(
            n_observed_paths=("observed_path_present_in_pool", "size"),
            n_observed_paths_missing_from_pool=("observed_path_present_in_pool", lambda s: int((~s.astype(bool)).sum())),
        ).reset_index()
    else:
        obs_agg = pd.DataFrame(columns=["side", "target_id", "path_length", "n_observed_paths", "n_observed_paths_missing_from_pool"])
    out = diag.merge(obs_agg, on=["side", "target_id", "path_length"], how="outer")
    out["n_structural_paths"] = pd.to_numeric(out.get("n_structural_paths"), errors="coerce").fillna(0).astype(int)
    out["n_observed_paths"] = pd.to_numeric(out.get("n_observed_paths"), errors="coerce").fillna(0).astype(int)
    out["n_observed_paths_missing_from_pool"] = pd.to_numeric(out.get("n_observed_paths_missing_from_pool"), errors="coerce").fillna(0).astype(int)
    if "pool_scope" in out.columns:
        out["pool_scope"] = out["pool_scope"].fillna("target_khop")
    if "pool_filter" in out.columns:
        out["pool_filter"] = out["pool_filter"].fillna("unknown")
    return out


def _median_from_counter(counter: Mapping[int, int]) -> float:
    """Median of a value multiset held as a frequency table.

    The removed-edge counts are small non-negative integers, so keeping a
    frequency table instead of one array entry per cell pair bounds the memory of
    the first-edge diagnostics by the value range rather than by the pair count.
    """
    total = int(sum(int(v) for v in counter.values()))
    if total <= 0:
        return float("nan")
    values = sorted(int(k) for k in counter)
    lower_rank = (total - 1) // 2
    upper_rank = total // 2
    lower_value: float | None = None
    upper_value: float | None = None
    seen = 0
    for value in values:
        seen += int(counter[value])
        if lower_value is None and seen > lower_rank:
            lower_value = float(value)
        if upper_value is None and seen > upper_rank:
            upper_value = float(value)
            break
    if lower_value is None or upper_value is None:
        return float("nan")
    return float((lower_value + upper_value) / 2.0)


class FirstEdgeExclusionDiagnostics:
    """Streaming counters for the first-edge exclusion diagnostic columns.

    Only used by the ``first_edge_excluded`` scope; the ``all_edges`` scope never
    passes an accumulator, so the primary path keeps its original cost and output.
    """

    def __init__(self) -> None:
        self._empty_pairs: Counter[tuple[str, str]] = Counter()
        self._removed_drug: dict[tuple[str, str], Counter[int]] = {}
        self._removed_pert: dict[tuple[str, str], Counter[int]] = {}
        self._key: tuple[str, str] | None = None

    def scoped(self, *, null_type: str, evaluable_set_type: str) -> "FirstEdgeExclusionDiagnostics":
        self._key = (str(null_type), str(evaluable_set_type))
        self._removed_drug.setdefault(self._key, Counter())
        self._removed_pert.setdefault(self._key, Counter())
        self._empty_pairs.setdefault(self._key, 0)
        return self

    def add_pair(self, *, n_removed_drug: int, n_removed_pert: int, is_empty_after_exclusion: bool, weight: int = 1) -> None:
        if self._key is None:
            raise ValueError("FirstEdgeExclusionDiagnostics.scoped() must be called before add_pair().")
        w = int(weight)
        if w <= 0:
            return
        self._removed_drug[self._key][int(n_removed_drug)] += w
        self._removed_pert[self._key][int(n_removed_pert)] += w
        if is_empty_after_exclusion:
            self._empty_pairs[self._key] += w

    def to_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for key in sorted(set(self._removed_drug) | set(self._removed_pert) | set(self._empty_pairs)):
            null_type, eval_type = key
            rows.append({
                "null_type": null_type,
                "evaluable_set_type": eval_type,
                "n_pairs_empty_after_exclusion": int(self._empty_pairs.get(key, 0)),
                "n_edges_removed_median_drug": _median_from_counter(self._removed_drug.get(key, Counter())),
                "n_edges_removed_median_pert": _median_from_counter(self._removed_pert.get(key, Counter())),
            })
        return pd.DataFrame(
            rows,
            columns=[
                "null_type",
                "evaluable_set_type",
                "n_pairs_empty_after_exclusion",
                "n_edges_removed_median_drug",
                "n_edges_removed_median_pert",
            ],
        )


def _joint_pair_values(
    pair: pd.DataFrame,
    metrics: Sequence[str],
    *,
    edge_column_suffix: str = "",
    diagnostics_accumulator: FirstEdgeExclusionDiagnostics | None = None,
) -> pd.DataFrame:
    out = pair.copy()
    drug_col = f"signed_edges{edge_column_suffix}_drug"
    pert_col = f"signed_edges{edge_column_suffix}_pert"
    joint_values: dict[str, list[float]] = {f"joint_{m}": [] for m in metrics}
    jaccard_vals: list[float] = []
    n_intersection: list[int] = []
    n_union: list[int] = []
    n_drug_edges: list[int] = []
    n_pert_edges: list[int] = []

    # Unique edge-set pairs, not expanded cell pairs, drive the set algebra cost.
    pair_cache: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    group_ids: dict[tuple[tuple[str, str, int], ...], int] = {}

    def _group_id(edges: set[Any]) -> int:
        key = _edge_key(edges)
        gid = group_ids.get(key)
        if gid is None:
            gid = len(group_ids)
            group_ids[key] = gid
        return gid

    if diagnostics_accumulator is not None:
        columns = ["both_paths", drug_col, pert_col, "signed_edges_drug", "signed_edges_pert"]
    else:
        columns = ["both_paths", drug_col, pert_col]
    for record in out[columns].itertuples(index=False, name=None):
        both, a, b = record[0], record[1], record[2]
        ea = _edge_set(a)
        eb = _edge_set(b)
        cache_key = (_group_id(ea), _group_id(eb))
        counts = pair_cache.get(cache_key)
        if counts is None:
            counts = _pair_edge_counts(ea, eb)
            pair_cache[cache_key] = counts
        n_inter, n_uni, n_a, n_b = counts
        n_intersection.append(int(n_inter))
        n_union.append(int(n_uni))
        n_drug_edges.append(int(n_a))
        n_pert_edges.append(int(n_b))
        raw_j = metric_value_from_counts("jaccard", n_intersection=n_inter, n_drug_edges=n_a, n_pert_edges=n_b)
        jaccard_vals.append(float(raw_j))
        for m in metrics:
            if not bool(both):
                val = 0.0
            else:
                val = metric_value_from_counts(str(m), n_intersection=n_inter, n_drug_edges=n_a, n_pert_edges=n_b)
                if pd.isna(val):
                    val = 0.0
            joint_values[f"joint_{m}"].append(float(val))
        if diagnostics_accumulator is not None:
            diagnostics_accumulator.add_pair(
                n_removed_drug=len(_edge_set(record[3])) - n_a,
                n_removed_pert=len(_edge_set(record[4])) - n_b,
                is_empty_after_exclusion=bool(both) and (n_a == 0 or n_b == 0),
            )
    out["jaccard"] = jaccard_vals
    out["n_intersection"] = n_intersection
    out["n_union"] = n_union
    out["n_drug_edges"] = n_drug_edges
    out["n_pert_edges"] = n_pert_edges
    for col, values in joint_values.items():
        out[col] = values
    return out


def _aggregate_joint_pairs(
    pair_raw: pd.DataFrame,
    pair_used: pd.DataFrame,
    *,
    fixed_triplet_ids: set[str],
    metrics: Sequence[str],
    evaluable_set_type: str,
    iteration: int | None = None,
    structural: bool = False,
) -> pd.DataFrame:
    group_cols = _triplet_group_cols(pair_raw)
    raw_counts = pair_raw.groupby(group_cols, dropna=False).size().rename("n_cell_pairs_raw").reset_index()
    meta = pair_raw.groupby("triplet_id", dropna=False)[[c for c in ["drug_id", "target_id", "direction"] if c in pair_raw.columns]].first().reset_index()

    if pair_used.empty:
        agg = meta.copy()
        agg["n_cell_pairs_used"] = 0
        agg["path_recoverability"] = float("nan")
        for m in metrics:
            agg[f"joint_{m}"] = float("nan")
    else:
        agg_spec: dict[str, Any] = {"n_cell_pairs_used": ("triplet_id", "size")}
        if structural or evaluable_set_type == "path_existing_pairs":
            agg_spec["path_recoverability"] = ("both_paths", lambda s: float("nan"))
        else:
            agg_spec["path_recoverability"] = ("both_paths", "mean")
        for m in metrics:
            agg_spec[f"joint_{m}"] = (f"joint_{m}", "mean")
        agg = pair_used.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()

    out = meta.merge(raw_counts, on=[c for c in group_cols if c in meta.columns and c in raw_counts.columns], how="left")
    out = out.merge(agg, on=[c for c in group_cols if c in out.columns and c in agg.columns], how="left", suffixes=("", "_agg"))
    if "n_cell_pairs_used" not in out.columns:
        out["n_cell_pairs_used"] = 0
    out["n_cell_pairs_used"] = pd.to_numeric(out["n_cell_pairs_used"], errors="coerce").fillna(0).astype(int)
    out["n_cell_pairs_raw"] = pd.to_numeric(out["n_cell_pairs_raw"], errors="coerce").fillna(0).astype(int)
    for m in metrics:
        col = f"joint_{m}"
        if col not in out.columns:
            out[col] = float("nan")
    if "path_recoverability" not in out.columns:
        out["path_recoverability"] = float("nan")
    out = out.loc[out["triplet_id"].astype(str).isin({str(x) for x in fixed_triplet_ids})].copy()
    missing = sorted({str(x) for x in fixed_triplet_ids} - set(out["triplet_id"].astype(str)))
    if missing:
        label = f"baseline iteration {iteration}" if iteration is not None else "observed"
        raise ValueError(
            f"Global evaluation {evaluable_set_type} triplet set is incomplete in {label}: "
            f"{len(missing)} missing triplets, first few={missing[:10]}"
        )
    out["evaluable_set_type"] = evaluable_set_type
    if iteration is not None:
        out["iteration"] = int(iteration)
    # Stable column order.
    cols = [c for c in ["evaluable_set_type", "iteration", "triplet_id", "drug_id", "target_id", "direction", "n_cell_pairs_raw", "n_cell_pairs_used", "path_recoverability"] if c in out.columns]
    cols += [f"joint_{m}" for m in metrics if f"joint_{m}" in out.columns]
    return out[cols]




def _edge_key(edges: Any) -> tuple[tuple[str, str, int], ...]:
    return tuple(sorted(_edge_set(edges)))


def _side_edge_groups(
    df: pd.DataFrame,
    *,
    edge_column_suffix: str = "",
    track_removed_edges: bool = False,
) -> list[dict[str, Any]]:
    """Collapse rows with identical edge-set/path-count state for exact weighted pairing."""
    if df.empty:
        return []
    edge_col = f"signed_edges{edge_column_suffix}"
    tmp = df.copy()
    tmp["_edge_key"] = tmp[edge_col].map(_edge_key)
    tmp["_n_paths_key"] = pd.to_numeric(tmp.get("n_paths", 0), errors="coerce").fillna(0).astype(int)
    group_keys = ["_edge_key", "_n_paths_key"]
    if track_removed_edges:
        # Splitting groups further never changes the weighted sums (the metric only
        # depends on the scoped edge set and n_paths), it only keeps the per-pair
        # removed-edge count available for the diagnostics.
        tmp["_n_removed_key"] = (
            tmp["signed_edges"].map(lambda x: len(_edge_set(x))) - tmp[edge_col].map(lambda x: len(_edge_set(x)))
        ).astype(int)
        group_keys.append("_n_removed_key")
    grouped = tmp.groupby(group_keys, sort=False, dropna=False).size().reset_index(name="weight")
    rows: list[dict[str, Any]] = []
    for record in grouped[group_keys + ["weight"]].itertuples(index=False, name=None):
        edge_key, n_paths = record[0], record[1]
        edges = set(edge_key)
        rows.append({
            "edges": edges,
            "n_paths": int(n_paths),
            "n_edges": int(len(edges)),
            "weight": int(record[-1]),
            "n_edges_removed": int(record[2]) if track_removed_edges else 0,
        })
    return rows


def _compute_joint_metrics_for_triplet_weighted(
    drug_sub: pd.DataFrame,
    pert_sub: pd.DataFrame,
    *,
    metrics: Sequence[str],
    evaluable_set_type: str,
    edge_column_suffix: str = "",
    diagnostics_accumulator: FirstEdgeExclusionDiagnostics | None = None,
) -> dict[str, Any]:
    track_removed = diagnostics_accumulator is not None
    d_groups = _side_edge_groups(drug_sub, edge_column_suffix=edge_column_suffix, track_removed_edges=track_removed)
    p_groups = _side_edge_groups(pert_sub, edge_column_suffix=edge_column_suffix, track_removed_edges=track_removed)
    raw_count = int(len(drug_sub) * len(pert_sub))
    if raw_count < 1:
        raise ValueError("Cannot compute weighted joint metrics for an empty triplet side.")

    metric_weighted_sums = {f"joint_{m}": 0.0 for m in metrics}
    used_weight = 0
    both_weight = 0

    for dg in d_groups:
        d_has_path = int(dg["n_paths"]) > 0
        for pg in p_groups:
            p_has_path = int(pg["n_paths"]) > 0
            w = int(dg["weight"]) * int(pg["weight"])
            both = bool(d_has_path and p_has_path)
            if both:
                both_weight += w
            if diagnostics_accumulator is not None:
                diagnostics_accumulator.add_pair(
                    n_removed_drug=int(dg["n_edges_removed"]),
                    n_removed_pert=int(pg["n_edges_removed"]),
                    is_empty_after_exclusion=both and (int(dg["n_edges"]) == 0 or int(pg["n_edges"]) == 0),
                    weight=w,
                )
            if evaluable_set_type == "path_existing_pairs" and not both:
                continue
            if evaluable_set_type not in {"fixed_coverage", "path_existing_pairs"}:
                raise ValueError(f"Unsupported evaluable_set_type: {evaluable_set_type}")
            used_weight += w
            if both:
                n_inter, _, n_a, n_b = _pair_edge_counts(dg["edges"], pg["edges"])
            for m in metrics:
                if not both:
                    val = 0.0
                else:
                    val = metric_value_from_counts(str(m), n_intersection=n_inter, n_drug_edges=n_a, n_pert_edges=n_b)
                    if pd.isna(val):
                        val = 0.0
                metric_weighted_sums[f"joint_{m}"] += float(val) * w

    out: dict[str, Any] = {
        "n_cell_pairs_raw": raw_count,
        "n_cell_pairs_used": int(used_weight),
        "path_recoverability": float(both_weight / raw_count) if evaluable_set_type == "fixed_coverage" else float("nan"),
    }
    for m in metrics:
        col = f"joint_{m}"
        out[col] = float(metric_weighted_sums[col] / used_weight) if used_weight > 0 else float("nan")
    return out


def compute_joint_triplet_metrics_weighted(
    drug_df: pd.DataFrame,
    pert_df: pd.DataFrame,
    *,
    cell_pair_mode: str,
    fixed_triplet_ids: set[str],
    metrics: Sequence[str],
    evaluable_set_type: str = "fixed_coverage",
    iteration: int | None = None,
    edge_column_suffix: str = "",
    diagnostics_accumulator: FirstEdgeExclusionDiagnostics | None = None,
) -> pd.DataFrame:
    """Exact joint metric aggregation using unique edge-set groups.

    For cartesian cell-pairing, expanding all drug×pert rows and then taking a
    mean is algebraically identical to computing each unique edge-set pair once
    with ``weight = n_drug_profiles * n_pert_profiles``. The implementation
    first partitions by the same merge keys used by ``build_cell_pairs`` so that
    metadata mismatches are handled identically to the expanded inner merge.
    """
    if str(cell_pair_mode) != "cartesian":
        raise ValueError("weighted joint aggregation currently supports only cell_pair_mode='cartesian'.")
    if not fixed_triplet_ids:
        raise ValueError("fixed_triplet_ids must not be empty.")
    metrics = [str(m) for m in metrics]
    fixed_keys = {str(x) for x in fixed_triplet_ids}
    common_cols = [c for c in ["triplet_id", "drug_id", "target_id", "direction"] if c in drug_df.columns and c in pert_df.columns]
    if "triplet_id" not in common_cols:
        raise ValueError("weighted joint aggregation requires triplet_id on both sides.")

    drug_by_key = {tuple(k if isinstance(k, tuple) else (k,)): v.copy() for k, v in drug_df.groupby(common_cols, sort=False, dropna=False)}
    pert_by_key = {tuple(k if isinstance(k, tuple) else (k,)): v.copy() for k, v in pert_df.groupby(common_cols, sort=False, dropna=False)}
    tid_pos = common_cols.index("triplet_id")
    matching_keys = sorted(set(drug_by_key) & set(pert_by_key), key=lambda x: tuple(str(y) for y in x))
    keys_by_tid: dict[str, list[tuple[Any, ...]]] = {}
    for key in matching_keys:
        keys_by_tid.setdefault(str(key[tid_pos]), []).append(key)

    missing = sorted(fixed_keys - set(keys_by_tid))
    if missing:
        label = f"baseline iteration {iteration}" if iteration is not None else "observed"
        raise ValueError(
            f"Global evaluation {evaluable_set_type} fixed triplet set is incomplete in {label}: "
            f"{len(missing)} missing triplets, first few={missing[:10]}"
        )

    rows: list[dict[str, Any]] = []
    for tid in sorted(fixed_keys):
        for key in keys_by_tid[tid]:
            dsub = drug_by_key[key]
            psub = pert_by_key[key]
            row: dict[str, Any] = {"evaluable_set_type": str(evaluable_set_type)}
            if iteration is not None:
                row["iteration"] = int(iteration)
            for col, value in zip(common_cols, key):
                row[col] = value
            row.update(_compute_joint_metrics_for_triplet_weighted(
                dsub,
                psub,
                metrics=metrics,
                evaluable_set_type=str(evaluable_set_type),
                edge_column_suffix=edge_column_suffix,
                diagnostics_accumulator=diagnostics_accumulator,
            ))
            rows.append(row)

    out = pd.DataFrame(rows)
    cols = [c for c in ["evaluable_set_type", "iteration", "triplet_id", "drug_id", "target_id", "direction", "n_cell_pairs_raw", "n_cell_pairs_used", "path_recoverability"] if c in out.columns]
    cols += [f"joint_{m}" for m in metrics if f"joint_{m}" in out.columns]
    return out[cols]

def compute_joint_triplet_metrics(
    drug_df: pd.DataFrame,
    pert_df: pd.DataFrame,
    *,
    cell_pair_mode: str,
    fixed_triplet_ids: set[str],
    metrics: Sequence[str],
    evaluable_set_type: str = "fixed_coverage",
    iteration: int | None = None,
    debug_writer: CellpairDebugWriter | None = None,
    debug_source: str | None = None,
    debug_null_type: str | None = None,
    debug_direction: str | None = None,
    edge_column_suffix: str = "",
    diagnostics_accumulator: FirstEdgeExclusionDiagnostics | None = None,
) -> pd.DataFrame:
    """Compute joint path-overlap scores for an explicit evaluable set."""
    if str(cell_pair_mode) == "cartesian" and debug_writer is None:
        return compute_joint_triplet_metrics_weighted(
            drug_df,
            pert_df,
            cell_pair_mode=cell_pair_mode,
            fixed_triplet_ids=fixed_triplet_ids,
            metrics=metrics,
            evaluable_set_type=evaluable_set_type,
            iteration=iteration,
            edge_column_suffix=edge_column_suffix,
            diagnostics_accumulator=diagnostics_accumulator,
        )
    if not fixed_triplet_ids:
        raise ValueError("fixed_triplet_ids must not be empty.")
    pair_raw = build_cell_pairs(drug_df, pert_df, cell_pair_mode=cell_pair_mode)
    if pair_raw.empty:
        raise ValueError("Global evaluation raw cell-pair table is empty.")
    _required_global_pair_columns(pair_raw)

    fixed_keys = {str(x) for x in fixed_triplet_ids}
    pair_raw = pair_raw.copy()
    pair_raw["_triplet_key"] = pair_raw["triplet_id"].astype(str)
    present = set(pair_raw["_triplet_key"])
    missing = sorted(fixed_keys - present)
    if missing:
        label = f"baseline iteration {iteration}" if iteration is not None else "observed"
        raise ValueError(
            f"Global evaluation {evaluable_set_type} fixed triplet set is incomplete in {label}: "
            f"{len(missing)} missing triplets, first few={missing[:10]}"
        )

    pair_raw = pair_raw.loc[pair_raw["_triplet_key"].isin(fixed_keys)].copy()
    pair_raw["has_drug_path"] = pd.to_numeric(pair_raw["n_drug_paths"], errors="coerce").fillna(0).gt(0)
    pair_raw["has_pert_path"] = pd.to_numeric(pair_raw["n_pert_paths"], errors="coerce").fillna(0).gt(0)
    pair_raw["both_paths"] = pair_raw["has_drug_path"] & pair_raw["has_pert_path"]

    metrics = [str(m) for m in metrics]
    if evaluable_set_type == "fixed_coverage":
        pair_used = _joint_pair_values(
            pair_raw,
            metrics,
            edge_column_suffix=edge_column_suffix,
            diagnostics_accumulator=diagnostics_accumulator,
        )
    elif evaluable_set_type == "path_existing_pairs":
        pair_used = _joint_pair_values(
            pair_raw.loc[pair_raw["both_paths"]].copy(),
            metrics,
            edge_column_suffix=edge_column_suffix,
            diagnostics_accumulator=diagnostics_accumulator,
        )
    else:
        raise ValueError(f"Unsupported evaluable_set_type: {evaluable_set_type}")
    if debug_writer is not None:
        debug_writer.write(
            pair_used,
            source=str(debug_source or ("null" if iteration is not None else "observed")),
            null_type=str(debug_null_type or ("observed" if iteration is None else "unknown")),
            evaluable_set_type=str(evaluable_set_type),
            iteration=iteration,
            direction=debug_direction,
        )
    return _aggregate_joint_pairs(
        pair_raw,
        pair_used,
        fixed_triplet_ids=fixed_keys,
        metrics=metrics,
        evaluable_set_type=evaluable_set_type,
        iteration=iteration,
        structural=False,
    )


def _group_by_triplet(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if df.empty or "triplet_id" not in df.columns:
        return {}
    return {str(k): v.copy() for k, v in df.groupby(df["triplet_id"].astype(str), sort=False)}


def build_evaluable_triplets_by_triplet(
    drug_df: pd.DataFrame,
    pert_df: pd.DataFrame,
    *,
    cell_pair_mode: str,
    evaluable_set_type: str,
    min_candidate_nodes: int = 10,
) -> pd.DataFrame:
    drug_groups = _group_by_triplet(drug_df)
    pert_groups = _group_by_triplet(pert_df)
    rows: list[pd.DataFrame] = []
    for tid in sorted(set(drug_groups) & set(pert_groups)):
        pair_raw = build_cell_pairs(drug_groups[tid], pert_groups[tid], cell_pair_mode=cell_pair_mode)
        if pair_raw.empty:
            continue
        _required_global_pair_columns(pair_raw)
        group_cols = _triplet_group_cols(pair_raw)
        tmp = pair_raw.copy()
        tmp["candidate_nodes_drug"] = pd.to_numeric(tmp["num_nodes_in_paths_drug"], errors="coerce").fillna(0).astype(int)
        tmp["candidate_nodes_pert"] = pd.to_numeric(tmp["num_nodes_in_paths_pert"], errors="coerce").fillna(0).astype(int)
        tmp["has_drug_path"] = pd.to_numeric(tmp["n_drug_paths"], errors="coerce").fillna(0).gt(0)
        tmp["has_pert_path"] = pd.to_numeric(tmp["n_pert_paths"], errors="coerce").fillna(0).gt(0)
        tmp["both_paths"] = tmp["has_drug_path"] & tmp["has_pert_path"]
        tmp["n_drug_paths_num"] = pd.to_numeric(tmp["n_drug_paths"], errors="coerce").fillna(0)
        tmp["n_pert_paths_num"] = pd.to_numeric(tmp["n_pert_paths"], errors="coerce").fillna(0)
        part = tmp.groupby(group_cols, dropna=False).agg(
            n_raw_cell_pairs=("triplet_id", "size"),
            n_path_existing_cell_pairs=("both_paths", "sum"),
            candidate_nodes_drug_min=("candidate_nodes_drug", "min"),
            candidate_nodes_pert_min=("candidate_nodes_pert", "min"),
            n_drug_paths=("n_drug_paths_num", "sum"),
            n_kd_paths=("n_pert_paths_num", "sum"),
        ).reset_index()
        part["candidate_nodes_min"] = part[["candidate_nodes_drug_min", "candidate_nodes_pert_min"]].min(axis=1)
        if evaluable_set_type == "fixed_coverage":
            part["is_global_evaluable"] = (
                pd.to_numeric(part["candidate_nodes_min"], errors="coerce").fillna(0).ge(int(min_candidate_nodes))
                & pd.to_numeric(part["n_raw_cell_pairs"], errors="coerce").fillna(0).ge(1)
            )
        elif evaluable_set_type == "path_existing_pairs":
            part["is_global_evaluable"] = pd.to_numeric(part["n_path_existing_cell_pairs"], errors="coerce").fillna(0).gt(0)
        else:
            raise ValueError(f"Unsupported evaluable_set_type: {evaluable_set_type}")
        part["evaluable_set_type"] = evaluable_set_type
        rows.append(part)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    for c in [
        "n_raw_cell_pairs", "n_path_existing_cell_pairs", "candidate_nodes_drug_min",
        "candidate_nodes_pert_min", "candidate_nodes_min", "n_drug_paths", "n_kd_paths",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    return out


def compute_joint_triplet_metrics_by_triplet(
    drug_df: pd.DataFrame,
    pert_df: pd.DataFrame,
    *,
    cell_pair_mode: str,
    fixed_triplet_ids: set[str],
    metrics: Sequence[str],
    evaluable_set_type: str,
    iteration: int | None = None,
    debug_writer: CellpairDebugWriter | None = None,
    debug_source: str | None = None,
    debug_null_type: str | None = None,
    debug_direction: str | None = None,
    edge_column_suffix: str = "",
    diagnostics_accumulator: FirstEdgeExclusionDiagnostics | None = None,
    drug_groups: Mapping[str, pd.DataFrame] | None = None,
    pert_groups: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    drug_groups = _group_by_triplet(drug_df) if drug_groups is None else drug_groups
    pert_groups = _group_by_triplet(pert_df) if pert_groups is None else pert_groups
    parts: list[pd.DataFrame] = []
    for tid in sorted({str(x) for x in fixed_triplet_ids}):
        if tid not in drug_groups or tid not in pert_groups:
            continue
        try:
            part = compute_joint_triplet_metrics(
                drug_groups[tid],
                pert_groups[tid],
                cell_pair_mode=cell_pair_mode,
                fixed_triplet_ids={tid},
                metrics=metrics,
                evaluable_set_type=evaluable_set_type,
                iteration=iteration,
                debug_writer=debug_writer,
                debug_source=debug_source,
                debug_null_type=debug_null_type,
                debug_direction=debug_direction,
                edge_column_suffix=edge_column_suffix,
                diagnostics_accumulator=diagnostics_accumulator,
            )
        except ValueError:
            continue
        parts.append(part)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    missing = sorted({str(x) for x in fixed_triplet_ids} - set(out.get("triplet_id", pd.Series(dtype=str)).astype(str)))
    if missing:
        label = f"baseline iteration {iteration}" if iteration is not None else "observed"
        raise ValueError(
            f"Global evaluation {evaluable_set_type} triplet set is incomplete in {label}: "
            f"{len(missing)} missing triplets, first few={missing[:10]}"
        )
    return out


def _sample_target_edges_for_lengths(
    lengths: list[int],
    pool: dict[str, dict[int, list[set[tuple[str, str, int]]]]],
    *,
    target_id: str,
    side: str,
    rng: np.random.Generator,
    with_first_edge_excluded: bool = False,
) -> tuple[set[tuple[str, str, int]], set[tuple[str, str, int]]]:
    """Draw one structural path per observed path length and union their edges.

    Both edge scopes are derived from the *same* draw: the RNG is consumed exactly
    once per observed path regardless of ``with_first_edge_excluded``, so adding the
    first-edge scope cannot change the Null 2 sample.
    """
    sampled: set[tuple[str, str, int]] = set()
    sampled_excluded: set[tuple[str, str, int]] = set()
    root = str(target_id)
    target_pool = pool.get(root, {})
    for length in lengths:
        candidates = target_pool.get(int(length), [])
        if not candidates:
            raise ValueError(
                f"Structural resampling target_khop pool is empty for "
                f"side={side}, target_id={target_id}, path_length={length}."
            )
        idx = int(rng.integers(0, len(candidates)))
        path_edges = candidates[idx]
        sampled.update(path_edges)
        if with_first_edge_excluded:
            sampled_excluded.update(edge for edge in path_edges if str(edge[0]) != root)
    return sampled, sampled_excluded


def compute_structural_resampling_joint_metrics(
    true_drug_df: pd.DataFrame,
    true_pert_df: pd.DataFrame,
    *,
    cell_pair_mode: str,
    fixed_triplet_ids: set[str],
    metrics: Sequence[str],
    evaluable_set_type: str,
    n_shuffle: int,
    rng: np.random.Generator,
    kg_tsv: str | Path | None,
    cutoff: int | None,
    drug_mode: str,
    pert_mode: str,
    path_strategy: str = "simple",
    true_drug_input_path: str | Path | None = None,
    true_pert_input_path: str | Path | None = None,
    debug_writer: CellpairDebugWriter | None = None,
    debug_direction: str | None = None,
    edge_scopes: Sequence[str] = (EDGE_SCOPE_ALL,),
    diagnostics_accumulator: FirstEdgeExclusionDiagnostics | None = None,
    pool_cache: dict[tuple[Any, ...], Any] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Target k-hop path-count and length-matched structural resampling null."""
    if n_shuffle < 1:
        raise ValueError("Structural resampling requires n_shuffle >= 1.")
    if kg_tsv is None:
        raise ValueError("target_khop structural resampling requires kg_tsv.")
    if cutoff is None:
        raise ValueError("target_khop structural resampling requires cutoff.")

    graph = _load_kg_graph(kg_tsv)
    targets = set(true_drug_df.get("target_id", pd.Series(dtype=str)).astype(str)).union(
        set(true_pert_df.get("target_id", pd.Series(dtype=str)).astype(str))
    )
    drug_valid_genes = _profile_valid_genes(true_drug_input_path)
    pert_valid_genes = _profile_valid_genes(true_pert_input_path)
    shared_pool_cache = {} if pool_cache is None else pool_cache
    drug_pool, drug_diag = build_shared_target_khop_path_pool(
        graph=graph,
        targets=targets,
        mode=drug_mode,
        cutoff=int(cutoff),
        valid_genes=drug_valid_genes,
        path_strategy=str(path_strategy),
        side="drug",
        pool_cache=shared_pool_cache,
    )
    pert_pool, pert_diag = build_shared_target_khop_path_pool(
        graph=graph,
        targets=targets,
        mode=pert_mode,
        cutoff=int(cutoff),
        valid_genes=pert_valid_genes,
        path_strategy=str(path_strategy),
        side="pert",
        pool_cache=shared_pool_cache,
    )
    pool_diag = pd.concat([
        _add_observed_pool_diagnostics(drug_diag, true_drug_df, drug_pool, side="drug"),
        _add_observed_pool_diagnostics(pert_diag, true_pert_df, pert_pool, side="pert"),
    ], ignore_index=True)
    missing_obs = int(pd.to_numeric(pool_diag.get("n_observed_paths_missing_from_pool", 0), errors="coerce").fillna(0).sum())
    if missing_obs:
        missing = pool_diag.loc[pd.to_numeric(pool_diag["n_observed_paths_missing_from_pool"], errors="coerce").fillna(0).gt(0)]
        first = missing[["side", "target_id", "path_length", "n_observed_paths_missing_from_pool"]].head(10).to_dict(orient="records")
        raise ValueError(
            "Observed accepted paths are not fully contained in the target_khop structural pool: "
            f"{missing_obs} missing paths, first few={first}"
        )

    pair_raw = build_cell_pairs(true_drug_df, true_pert_df, cell_pair_mode=cell_pair_mode)
    if pair_raw.empty:
        raise ValueError("Structural resampling raw cell-pair table is empty.")
    _required_global_pair_columns(pair_raw)
    fixed_keys = {str(x) for x in fixed_triplet_ids}
    pair_raw = pair_raw.copy()
    pair_raw["_triplet_key"] = pair_raw["triplet_id"].astype(str)
    missing = sorted(fixed_keys - set(pair_raw["_triplet_key"]))
    if missing:
        raise ValueError(
            f"Structural resampling fixed triplet set is incomplete in observed data: "
            f"{len(missing)} missing triplets, first few={missing[:10]}"
        )
    pair_raw = pair_raw.loc[pair_raw["_triplet_key"].isin(fixed_keys)].copy()
    pair_raw["has_drug_path"] = pd.to_numeric(pair_raw["n_drug_paths"], errors="coerce").fillna(0).gt(0)
    pair_raw["has_pert_path"] = pd.to_numeric(pair_raw["n_pert_paths"], errors="coerce").fillna(0).gt(0)
    pair_raw["both_paths"] = pair_raw["has_drug_path"] & pair_raw["has_pert_path"]
    if evaluable_set_type == "path_existing_pairs":
        source_pairs = pair_raw.loc[pair_raw["both_paths"]].copy()
    elif evaluable_set_type == "fixed_coverage":
        source_pairs = pair_raw.copy()
    else:
        raise ValueError(f"Unsupported evaluable_set_type: {evaluable_set_type}")

    scopes = [str(s) for s in edge_scopes]
    unknown_scopes = sorted(set(scopes) - set(EDGE_SCOPES))
    if unknown_scopes:
        raise ValueError(f"Unsupported edge scopes for structural resampling: {unknown_scopes}")
    want_excluded = EDGE_SCOPE_FIRST_EDGE_EXCLUDED in scopes
    parts_by_scope: dict[str, list[pd.DataFrame]] = {scope: [] for scope in scopes}
    metrics = [str(m) for m in metrics]
    if diagnostics_accumulator is not None:
        diagnostics_accumulator.scoped(
            null_type="path_count_length_matched_structural_resampling",
            evaluable_set_type=str(evaluable_set_type),
        )
    for iteration in range(1, int(n_shuffle) + 1):
        pair = source_pairs.copy()
        joint_values: dict[str, dict[str, list[float]]] = {
            scope: {f"joint_{m}": [] for m in metrics} for scope in scopes
        }
        sampled_drug_edges: list[set[tuple[str, str, int]]] = []
        sampled_pert_edges: list[set[tuple[str, str, int]]] = []
        sampled_drug_edges_excl: list[set[tuple[str, str, int]]] = []
        sampled_pert_edges_excl: list[set[tuple[str, str, int]]] = []
        pair_counts: list[tuple[int, int, int, int]] = []
        for both, target_id, d_paths, p_paths in pair[["both_paths", "target_id", "signed_edge_paths_drug", "signed_edge_paths_pert"]].itertuples(index=False, name=None):
            if not bool(both):
                d_edges: set[tuple[str, str, int]] = set()
                p_edges: set[tuple[str, str, int]] = set()
                d_edges_excl: set[tuple[str, str, int]] = set()
                p_edges_excl: set[tuple[str, str, int]] = set()
            else:
                d_lengths = [len(x) for x in _path_list(d_paths) if len(x) > 0]
                p_lengths = [len(x) for x in _path_list(p_paths) if len(x) > 0]
                # The structural pool for this row is rooted at the shared target,
                # which is also the ``pert_id`` of both observed sides in Val3.
                d_edges, d_edges_excl = _sample_target_edges_for_lengths(
                    d_lengths, drug_pool, target_id=str(target_id), side="drug", rng=rng,
                    with_first_edge_excluded=want_excluded,
                )
                p_edges, p_edges_excl = _sample_target_edges_for_lengths(
                    p_lengths, pert_pool, target_id=str(target_id), side="pert", rng=rng,
                    with_first_edge_excluded=want_excluded,
                )
            sampled_drug_edges.append(d_edges)
            sampled_pert_edges.append(p_edges)
            sampled_drug_edges_excl.append(d_edges_excl)
            sampled_pert_edges_excl.append(p_edges_excl)
            counts = _pair_edge_counts(d_edges, p_edges)
            pair_counts.append(counts)
            for scope in scopes:
                if scope == EDGE_SCOPE_ALL:
                    n_inter, _, n_a, n_b = counts
                else:
                    n_inter, _, n_a, n_b = _pair_edge_counts(d_edges_excl, p_edges_excl)
                for m in metrics:
                    if not bool(both):
                        val = 0.0
                    else:
                        val = metric_value_from_counts(m, n_intersection=n_inter, n_drug_edges=n_a, n_pert_edges=n_b)
                        if pd.isna(val):
                            val = 0.0
                    joint_values[scope][f"joint_{m}"].append(float(val))
            if diagnostics_accumulator is not None and want_excluded:
                diagnostics_accumulator.add_pair(
                    n_removed_drug=len(d_edges) - len(d_edges_excl),
                    n_removed_pert=len(p_edges) - len(p_edges_excl),
                    is_empty_after_exclusion=bool(both) and (not d_edges_excl or not p_edges_excl),
                )
        pair["signed_edges_drug"] = sampled_drug_edges
        pair["signed_edges_pert"] = sampled_pert_edges
        pair["n_intersection"] = [int(c[0]) for c in pair_counts]
        pair["n_union"] = [int(c[1]) for c in pair_counts]
        pair["n_drug_edges"] = [int(c[2]) for c in pair_counts]
        pair["n_pert_edges"] = [int(c[3]) for c in pair_counts]
        pair["jaccard"] = [metric_value_from_counts("jaccard", n_intersection=c[0], n_drug_edges=c[2], n_pert_edges=c[3]) for c in pair_counts]
        for col, vals in joint_values[EDGE_SCOPE_ALL if EDGE_SCOPE_ALL in scopes else scopes[0]].items():
            pair[col] = vals
        if debug_writer is not None:
            debug_writer.write(
                pair,
                source="null",
                null_type="path_count_length_matched_structural_resampling",
                evaluable_set_type=str(evaluable_set_type),
                iteration=iteration,
                direction=debug_direction,
            )
        for scope in scopes:
            scoped_pair = pair
            if scope != EDGE_SCOPE_ALL:
                scoped_pair = pair.copy()
                for col, vals in joint_values[scope].items():
                    scoped_pair[col] = vals
            part = _aggregate_joint_pairs(
                pair_raw,
                scoped_pair,
                fixed_triplet_ids=fixed_keys,
                metrics=metrics,
                evaluable_set_type=evaluable_set_type,
                iteration=iteration,
                structural=True,
            )
            part["null_type"] = "path_count_length_matched_structural_resampling"
            part["pool_scope"] = "target_khop"
            part["pool_filter"] = "valid_profile_genes" if drug_valid_genes is not None and pert_valid_genes is not None else "kg_only"
            parts_by_scope[scope].append(part)
    baseline_by_scope = {
        scope: (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame())
        for scope, parts in parts_by_scope.items()
    }
    return baseline_by_scope, pool_diag


def _valid_statistics_for_group(null_type: str, evaluable_set_type: str, metrics: Sequence[str]) -> list[str]:
    return ["joint_jaccard"] if "jaccard" in {str(metric) for metric in metrics} else []


def _per_triplet_null_representative(
    base_sub: pd.DataFrame,
    stat: str,
    *,
    kind: str = "median",
    name: str = "null_median",
) -> pd.DataFrame:
    """Per-triplet null representative score.

    Every null model uses the ``median`` over the triplet's null draws / evaluable
    decoys. ``kind="mean"`` is only used to retain a non-primary diagnostic reference.
    """
    if kind not in {"mean", "median"}:
        raise ValueError("kind must be either 'mean' or 'median'.")
    grouped = (
        base_sub.assign(_stat=pd.to_numeric(base_sub[stat], errors="coerce"))
        .groupby("triplet_id", dropna=False)["_stat"]
    )
    rep = grouped.mean() if kind == "mean" else grouped.median()
    return rep.rename(name).reset_index()


def compute_global_effect_size(
    true_joint_triplet_df: pd.DataFrame,
    baseline_joint_triplet_df: pd.DataFrame,
    *,
    metrics: Sequence[str],
    comparison_triplet_sets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Paired global effect size for every null model.

    The null representative is the median null/control score within each triplet.
    The primary effect is the mean across triplet-level ``observed - null median``
    deltas. Mean-null values are retained only as ``ref_*`` diagnostics.
    """
    if true_joint_triplet_df.empty:
        raise ValueError("Observed global joint triplet table is empty.")
    if baseline_joint_triplet_df.empty:
        raise ValueError("Baseline global joint triplet table is empty.")
    rows: list[dict[str, Any]] = []
    for (null_type, eval_type), base_sub in baseline_joint_triplet_df.groupby(["null_type", "evaluable_set_type"], dropna=False):
        null_type_s = canonical_decoy_null_type(null_type)
        true_sub = true_joint_triplet_df.loc[true_joint_triplet_df["evaluable_set_type"].astype(str) == str(eval_type)].copy()
        ids = _comparison_ids_for_group(comparison_triplet_sets, null_type=null_type_s, eval_type=str(eval_type))
        if ids is None and str(null_type) != null_type_s:
            ids = _comparison_ids_for_group(comparison_triplet_sets, null_type=str(null_type), eval_type=str(eval_type))
        if ids is not None:
            true_sub = true_sub.loc[true_sub["triplet_id"].astype(str).isin(ids)].copy()
            base_sub = base_sub.loc[base_sub["triplet_id"].astype(str).isin(ids)].copy()
        if true_sub.empty or base_sub.empty:
            continue
        for stat in _valid_statistics_for_group(null_type_s, str(eval_type), metrics):
            if stat not in true_sub.columns or stat not in base_sub.columns:
                continue
            null_rep = _per_triplet_null_representative(base_sub, stat)
            null_ref = _per_triplet_null_representative(
                base_sub, stat, kind="mean", name="ref_null_mean",
            )
            obs = true_sub[["triplet_id", stat]].copy()
            obs["observed"] = pd.to_numeric(obs[stat], errors="coerce")
            merged = (
                obs[["triplet_id", "observed"]]
                .merge(null_rep, on="triplet_id", how="inner")
                .merge(null_ref, on="triplet_id", how="inner")
            )
            merged["delta"] = merged["observed"] - merged["null_median"]
            merged["ref_delta"] = merged["observed"] - merged["ref_null_mean"]
            # The reference columns are computed on the same valid rows as the main
            # result, so the two are directly comparable.
            valid = merged.dropna(subset=["observed", "null_median", "delta"])
            if valid.empty:
                continue
            rows.append({
                "null_type": null_type_s,
                "evaluable_set_type": str(eval_type),
                "statistic": stat,
                "estimand": "mean_observed_minus_triplet_null_median",
                "n_triplets": int(len(valid)),
                "observed_mean": float(valid["observed"].mean()),
                "observed_median": float(valid["observed"].median()),
                "baseline_representative": null_representative_kind(null_type_s),
                "null_median_mean": float(valid["null_median"].mean()),
                "null_median_median": float(valid["null_median"].median()),
                "paired_delta_mean": float(valid["delta"].mean()),
                "paired_delta_median": float(valid["delta"].median()),
                "ref_null_mean_mean": float(valid["ref_null_mean"].mean()),
                "ref_null_mean_median": float(valid["ref_null_mean"].median()),
                "ref_paired_delta_mean": float(valid["ref_delta"].mean()),
                "ref_paired_delta_median": float(valid["ref_delta"].median()),
            })
    return pd.DataFrame(rows)


def _spearman_or_nan(x: pd.Series, y: pd.Series) -> float:
    tmp = pd.DataFrame({
        "x": pd.to_numeric(pd.Series(list(x)), errors="coerce"),
        "y": pd.to_numeric(pd.Series(list(y)), errors="coerce"),
    }).dropna()
    if len(tmp) < 2 or tmp["x"].nunique(dropna=True) < 2 or tmp["y"].nunique(dropna=True) < 2:
        return float("nan")
    return float(tmp["x"].corr(tmp["y"], method="spearman"))


def _enabled_global_options(evaluation_cfg: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "global_effect_size": bool((evaluation_cfg or {}).get("global_effect_size", {}).get("enabled", False)),
    }


def _enabled_evaluable_sets(evaluation_cfg: Mapping[str, Any]) -> list[str]:
    ev = (evaluation_cfg or {}).get("evaluable_sets") or {}
    out: list[str] = []
    if bool((ev.get("fixed_coverage") or {}).get("enabled", False)):
        out.append("fixed_coverage")
    if bool((ev.get("path_existing_pairs") or {}).get("enabled", False)):
        out.append("path_existing_pairs")
    return out


def _enabled_null_model(baseline_cfg: Mapping[str, Any] | None, name: str, default: bool = False) -> bool:
    cfg = dict(baseline_cfg or {})
    null_models = cfg.get("null_models") or {}
    if name in null_models and isinstance(null_models[name], Mapping):
        return bool(null_models[name].get("enabled", default))
    return bool(default)


def _inference_null_types_enabled(baseline_cfg: Mapping[str, Any] | None, baseline_loaded: Sequence[Any]) -> list[str]:
    loaded_types = []
    for item in baseline_loaded:
        if len(item) >= 4:
            loaded_types.append(str(item[3]))
    out: list[str] = []
    for name in [
        "target_khop_restricted_paired_gene_label_permutation",
        "global_paired_gene_label_permutation",
    ]:
        if _enabled_null_model(baseline_cfg, name, False) or name in loaded_types:
            out.append(name)
    if not out and bool((baseline_cfg or {}).get("enabled", len(baseline_loaded) > 0)) and len(baseline_loaded) > 0:
        out.append("global_paired_gene_label_permutation")
    return out


def _structural_null_enabled(baseline_cfg: Mapping[str, Any] | None) -> bool:
    cfg = dict(baseline_cfg or {})
    null_models = cfg.get("null_models") or {}
    if "path_count_length_matched_structural_resampling" in null_models:
        return bool(null_models["path_count_length_matched_structural_resampling"].get("enabled", False))
    structural = cfg.get("structural_null") or {}
    return bool(structural.get("enabled", False)) if isinstance(structural, dict) else False


def _decoy_null_enabled(baseline_cfg: Mapping[str, Any] | None) -> bool:
    return any(_enabled_null_model(baseline_cfg, name, False) for name in sorted(DECOY_NULL_TYPES))


def _enabled_decoy_null_type(baseline_cfg: Mapping[str, Any] | None) -> str:
    """Resolve which decoy control this run uses.

    Enabling both the legacy observed-target-pool control and the canonical
    topology- and profile-matched control is rejected in the config loader.
    """
    if _enabled_null_model(baseline_cfg, TARGET_DECOY_NULL_TYPE, False):
        return TARGET_DECOY_NULL_TYPE
    return DECOY_NULL_TYPE


def _normalize_baseline_pair(item: Sequence[Any]) -> tuple[int, str | Path, str | Path, str]:
    if len(item) >= 4:
        return int(item[0]), item[1], item[2], str(item[3])
    return int(item[0]), item[1], item[2], "global_paired_gene_label_permutation"




def _is_evaluable_joint_row(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    eval_type = df.get("evaluable_set_type", pd.Series("", index=df.index)).astype(str)
    n_used = pd.to_numeric(
        df["n_cell_pairs_used"] if "n_cell_pairs_used" in df.columns else pd.Series(0, index=df.index),
        errors="coerce",
    ).fillna(0)
    n_raw = pd.to_numeric(
        df["n_cell_pairs_raw"] if "n_cell_pairs_raw" in df.columns else pd.Series(0, index=df.index),
        errors="coerce",
    ).fillna(0)
    global_eval = (
        df["is_global_evaluable"].fillna(False).astype(bool)
        if "is_global_evaluable" in df.columns else pd.Series(False, index=df.index)
    )
    fixed = eval_type.eq("fixed_coverage") & (n_raw.ge(1) | global_eval)
    path_existing = eval_type.eq("path_existing_pairs") & (n_used.gt(0) | global_eval)
    other = ~(eval_type.isin(["fixed_coverage", "path_existing_pairs"])) & (n_raw.ge(1) | global_eval)
    return fixed | path_existing | other


def build_comparison_triplet_sets(
    true_joint_triplet_df: pd.DataFrame,
    baseline_joint_triplet_df: pd.DataFrame,
    *,
    n_shuffle_by_null_type: dict[str, int],
) -> pd.DataFrame:
    """Build T_compare(null_type, evaluable_set_type) for observed-vs-null summaries.

    A comparison triplet must be observed-evaluable and have at least one
    evaluable null/control score. Expected iteration counts are retained as QC
    only and are not used to require K_i = B.
    """
    if true_joint_triplet_df.empty or baseline_joint_triplet_df.empty:
        return pd.DataFrame(columns=[
            "null_type", "evaluable_set_type", "triplet_id", "in_observed_evaluable",
            "in_null_evaluable", "n_null_iterations_available", "n_decoy_anchors_available",
            "expected_null_iterations", "is_comparison_triplet",
        ])
    true_tmp = true_joint_triplet_df.copy()
    base_tmp = baseline_joint_triplet_df.copy()
    true_tmp["_is_eval"] = _is_evaluable_joint_row(true_tmp)
    base_tmp["_is_eval"] = _is_evaluable_joint_row(base_tmp)
    observed = true_tmp.loc[true_tmp["_is_eval"], ["evaluable_set_type", "triplet_id"]].drop_duplicates()
    observed["triplet_id"] = observed["triplet_id"].astype(str)
    rows: list[dict[str, Any]] = []
    for (null_type, eval_type), base_sub in base_tmp.groupby(["null_type", "evaluable_set_type"], dropna=False):
        null_type_s = canonical_decoy_null_type(null_type)
        is_decoy = null_type_s in DECOY_NULL_TYPES
        obs_ids = set(observed.loc[observed["evaluable_set_type"].astype(str).eq(str(eval_type)), "triplet_id"].astype(str))
        eval_base = base_sub.loc[base_sub["_is_eval"]].copy()
        if is_decoy and "decoy_anchor_id" in eval_base.columns:
            null_counts = (
                eval_base.assign(triplet_id=lambda x: x["triplet_id"].astype(str))
                .groupby("triplet_id", dropna=False)["decoy_anchor_id"]
                .nunique()
                .rename("n_decoy_anchors_available")
                .reset_index()
                if not eval_base.empty else pd.DataFrame(columns=["triplet_id", "n_decoy_anchors_available"])
            )
            null_ids = set(null_counts["triplet_id"].astype(str)) if not null_counts.empty else set()
            count_map = dict(zip(null_counts.get("triplet_id", pd.Series(dtype=str)).astype(str), pd.to_numeric(null_counts.get("n_decoy_anchors_available", pd.Series(dtype=int)), errors="coerce").fillna(0).astype(int)))
            all_ids = sorted(obs_ids | null_ids)
            for tid in all_ids:
                n_decoys = int(count_map.get(str(tid), 0))
                in_obs = str(tid) in obs_ids
                in_null = str(tid) in null_ids
                rows.append({
                    "null_type": null_type_s,
                    "evaluable_set_type": str(eval_type),
                    "triplet_id": str(tid),
                    "in_observed_evaluable": bool(in_obs),
                    "in_null_evaluable": bool(in_null),
                    "n_null_iterations_available": pd.NA,
                    "n_decoy_anchors_available": n_decoys,
                    "expected_null_iterations": pd.NA,
                    "is_comparison_triplet": bool(in_obs and in_null and n_decoys >= 1),
                })
            continue

        null_counts = (
            eval_base.assign(triplet_id=lambda x: x["triplet_id"].astype(str))
            .groupby("triplet_id", dropna=False)["iteration"]
            .nunique()
            .rename("n_null_iterations_available")
            .reset_index()
            if not eval_base.empty else pd.DataFrame(columns=["triplet_id", "n_null_iterations_available"])
        )
        null_ids = set(null_counts["triplet_id"].astype(str)) if not null_counts.empty else set()
        all_ids = sorted(obs_ids | null_ids)
        expected_n = int(n_shuffle_by_null_type.get(null_type_s, base_sub["iteration"].nunique() if "iteration" in base_sub.columns else 0))
        count_map = dict(zip(null_counts.get("triplet_id", pd.Series(dtype=str)).astype(str), pd.to_numeric(null_counts.get("n_null_iterations_available", pd.Series(dtype=int)), errors="coerce").fillna(0).astype(int)))
        for tid in all_ids:
            n_avail = int(count_map.get(str(tid), 0))
            in_obs = str(tid) in obs_ids
            in_null = str(tid) in null_ids
            rows.append({
                "null_type": null_type_s,
                "evaluable_set_type": str(eval_type),
                "triplet_id": str(tid),
                "in_observed_evaluable": bool(in_obs),
                "in_null_evaluable": bool(in_null),
                "n_null_iterations_available": n_avail,
                "n_decoy_anchors_available": pd.NA,
                "expected_null_iterations": expected_n,
                "is_comparison_triplet": bool(in_obs and in_null and n_avail >= 1),
            })
    return pd.DataFrame(rows)

def _comparison_ids_for_group(
    comparison_triplet_sets: pd.DataFrame | None,
    *,
    null_type: str,
    eval_type: str,
) -> set[str] | None:
    if comparison_triplet_sets is None or comparison_triplet_sets.empty:
        return None
    sub = comparison_triplet_sets.loc[
        comparison_triplet_sets["null_type"].astype(str).eq(str(null_type))
        & comparison_triplet_sets["evaluable_set_type"].astype(str).eq(str(eval_type))
        & comparison_triplet_sets["is_comparison_triplet"].astype(bool)
    ]
    return set(sub["triplet_id"].astype(str))


def compute_global_joint_triplet_comparison(
    true_joint_triplet_df: pd.DataFrame,
    baseline_joint_triplet_df: pd.DataFrame,
    *,
    metrics: Sequence[str],
    comparison_triplet_sets: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if true_joint_triplet_df.empty or baseline_joint_triplet_df.empty:
        return pd.DataFrame()
    for (null_type, eval_type), base_sub0 in baseline_joint_triplet_df.groupby(["null_type", "evaluable_set_type"], dropna=False):
        null_type_s = canonical_decoy_null_type(null_type)
        ids = _comparison_ids_for_group(comparison_triplet_sets, null_type=null_type_s, eval_type=str(eval_type))
        if ids is None and null_type_s != str(null_type):
            ids = _comparison_ids_for_group(comparison_triplet_sets, null_type=str(null_type), eval_type=str(eval_type))
        if not ids:
            continue
        true_sub0 = true_joint_triplet_df.loc[
            true_joint_triplet_df["evaluable_set_type"].astype(str).eq(str(eval_type))
            & true_joint_triplet_df["triplet_id"].astype(str).isin(ids)
        ].copy()
        base_sub0 = base_sub0.loc[base_sub0["triplet_id"].astype(str).isin(ids)].copy()
        meta_cols = [c for c in ["triplet_id", "drug_id", "target_id", "direction"] if c in true_sub0.columns]
        is_decoy = null_type_s in DECOY_NULL_TYPES
        representative = null_representative_kind(null_type_s)
        for stat in _valid_statistics_for_group(null_type_s, str(eval_type), metrics):
            if stat not in true_sub0.columns or stat not in base_sub0.columns:
                continue
            base_map = {str(tid): sub for tid, sub in base_sub0.groupby("triplet_id", dropna=False)}
            for rec in true_sub0[meta_cols + [stat]].to_dict(orient="records"):
                tid = str(rec["triplet_id"])
                obs_val = pd.to_numeric(pd.Series([rec.get(stat)]), errors="coerce").iloc[0]
                base_tid = base_map.get(tid, pd.DataFrame())
                bvals = pd.to_numeric(base_tid.get(stat, pd.Series(dtype=float)), errors="coerce").dropna()
                if pd.isna(obs_val) or bvals.empty:
                    continue
                baseline_mean = float(bvals.mean())
                baseline_median = float(bvals.median())
                row = {
                    "null_type": null_type_s,
                    "evaluable_set_type": str(eval_type),
                    "triplet_id": tid,
                    "drug_id": rec.get("drug_id"),
                    "target_id": rec.get("target_id"),
                    "direction": rec.get("direction"),
                    "statistic": stat,
                    "observed_value": float(obs_val),
                    "baseline_representative": representative,
                    "baseline_mean": baseline_mean,
                    "baseline_median": baseline_median,
                    "delta": float(obs_val - baseline_median),
                    "n_null_values": int(len(bvals)),
                    # Mean-null difference is retained only as a non-primary diagnostic.
                    "ref_delta_vs_null_mean": float(obs_val - baseline_mean),
                }
                if is_decoy:
                    row["n_shuffle"] = pd.NA
                    row["n_decoy_anchors"] = int(base_tid["decoy_anchor_id"].nunique()) if "decoy_anchor_id" in base_tid.columns else int(len(bvals))
                else:
                    row["n_shuffle"] = int(len(bvals))
                rows.append(row)
    return pd.DataFrame(rows)

TRUE_JOINT_CANONICAL_SORT_KEYS: tuple[str, ...] = ("evaluable_set_type", "triplet_id")
BASELINE_JOINT_CANONICAL_SORT_KEYS: tuple[str, ...] = (
    "evaluable_set_type",
    "null_type",
    "iteration",
    "triplet_id",
    "decoy_triplet_id",
)
# Scope of the explicit canonical row order (spec §6.1). The §6.1 pre-check on the
# reference run confirmed that a stable sort on the keys above reproduces the
# historical row order inside every null_type block, while the block concatenation
# order itself (Null 1 -> Null 2 -> Null 3) is generation order and is preserved.
CANONICAL_ROW_ORDER_SCOPE = "per_null_type_block_stable_sort_with_generation_block_order"


def canonical_sort_true_joint_triplet(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the observed joint-metric row order so it is reproducible."""
    if df.empty:
        return df
    keys = [k for k in TRUE_JOINT_CANONICAL_SORT_KEYS if k in df.columns]
    if not keys:
        return df.reset_index(drop=True)
    return df.sort_values(keys, kind="stable", na_position="last").reset_index(drop=True)


def canonical_sort_baseline_joint_triplet(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the baseline joint-metric row order within each null_type block.

    The normalization guarantees a deterministic row order in the stored baseline
    table regardless of how the parts were generated, so byte-level regression checks
    stay meaningful. Sorting per block keeps the Target-Decoy block in its generation
    order relative to the other nulls, which is what the current implementation
    produces.
    """
    if df.empty or "null_type" not in df.columns:
        return df.reset_index(drop=True) if not df.empty else df
    keys = [k for k in BASELINE_JOINT_CANONICAL_SORT_KEYS if k in df.columns]
    if not keys:
        return df.reset_index(drop=True)
    out = df.reset_index(drop=True)
    null_types = out["null_type"].astype(str)
    block_order = list(dict.fromkeys(null_types))
    parts = [
        out.loc[null_types.eq(null_type)].sort_values(keys, kind="stable", na_position="last")
        for null_type in block_order
    ]
    return pd.concat(parts, ignore_index=True) if parts else out


def compute_decoy_candidate_joint_metrics_streaming(
    *,
    drug_infer_path: str | Path,
    pert_infer_path: str | Path,
    decoy_candidates: pd.DataFrame,
    drug_mode: str,
    pert_mode: str,
    cell_pair_mode: str,
    eval_set_types: Sequence[str],
    ev_cfg: Mapping[str, Any],
    metrics: Sequence[str],
    edge_scopes: Sequence[str] = (EDGE_SCOPE_ALL,),
    diagnostics_accumulator: FirstEdgeExclusionDiagnostics | None = None,
    debug_writer: CellpairDebugWriter | None = None,
    debug_direction: str | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Joint metrics for evaluable decoy candidates plus per-candidate pair counts.

    The second frame covers every inferred candidate, evaluable or not, so that
    non-evaluable candidates stay auditable without being scored. Every requested
    edge scope is scored from a single read of each inference part.

    With a ``debug_writer`` the all-edges cell-pair records of every decoy candidate
    are also written, which is what lets the same-cell-line subset be rebuilt for the
    Target-Decoy control.  The shards are keyed by the pseudo triplet id, so this
    output is ``n_decoys_per_triplet`` times larger than the observed one (ten times
    in the primary spec).  Like every other ``evaluation/`` artifact it is outside the
    scope of ``storage.intermediate_retention``.
    """
    scopes = [str(s) for s in edge_scopes]
    unknown_scopes = sorted(set(scopes) - set(EDGE_SCOPES))
    if unknown_scopes:
        raise ValueError(f"Unsupported edge scopes for the decoy control: {unknown_scopes}")
    if decoy_candidates.empty:
        return {scope: pd.DataFrame() for scope in scopes}, pd.DataFrame()
    cand_meta = decoy_candidates.rename(columns={"pseudo_triplet_id": "triplet_id"}).copy()[
        ["triplet_id", "original_triplet_id", "original_target_id", "decoy_anchor_id", "disease_id"]
    ]
    parts_by_scope: dict[str, list[pd.DataFrame]] = {scope: [] for scope in scopes}
    eval_parts: list[pd.DataFrame] = []
    want_excluded = EDGE_SCOPE_FIRST_EDGE_EXCLUDED in scopes
    for drug_part_path, pert_part_path in _paired_infer_part_paths(drug_infer_path, pert_infer_path):
        decoy_drug_df = load_infer_signed_edge_df(
            drug_part_path, mode=drug_mode, include_first_edge_excluded=want_excluded,
        )
        decoy_pert_df = load_infer_signed_edge_df(
            pert_part_path, mode=pert_mode, include_first_edge_excluded=want_excluded,
        )
        if decoy_drug_df.empty or decoy_pert_df.empty:
            continue
        decoy_drug_groups = _group_by_triplet(decoy_drug_df)
        decoy_pert_groups = _group_by_triplet(decoy_pert_df)
        for eval_type in eval_set_types:
            try:
                if eval_type == "fixed_coverage":
                    fixed_cfg = dict(ev_cfg.get("fixed_coverage") or {})
                    cand_eval = build_evaluable_triplets_by_triplet(
                        decoy_drug_df,
                        decoy_pert_df,
                        cell_pair_mode=cell_pair_mode,
                        evaluable_set_type=eval_type,
                        min_candidate_nodes=int(fixed_cfg.get("min_candidate_nodes", 10)),
                    )
                elif eval_type == "path_existing_pairs":
                    cand_eval = build_evaluable_triplets_by_triplet(
                        decoy_drug_df,
                        decoy_pert_df,
                        cell_pair_mode=cell_pair_mode,
                        evaluable_set_type=eval_type,
                    )
                else:
                    cand_eval = pd.DataFrame()
            except ValueError:
                cand_eval = pd.DataFrame()
            if cand_eval.empty:
                continue
            eval_parts.append(cand_eval)
            candidate_fixed_ids = set(cand_eval.loc[cand_eval["is_global_evaluable"], "triplet_id"].astype(str))
            if not candidate_fixed_ids:
                continue
            extra_cols = [
                c for c in [
                    "triplet_id", "candidate_nodes_min", "n_raw_cell_pairs",
                    "n_path_existing_cell_pairs", "n_drug_paths", "n_kd_paths",
                ] if c in cand_eval.columns
            ]
            for scope in scopes:
                cand_joint = compute_joint_triplet_metrics_by_triplet(
                    decoy_drug_df,
                    decoy_pert_df,
                    cell_pair_mode=cell_pair_mode,
                    fixed_triplet_ids=candidate_fixed_ids,
                    metrics=metrics,
                    evaluable_set_type=str(eval_type),
                    edge_column_suffix=edge_column_suffix_for_scope(scope),
                    diagnostics_accumulator=(
                        diagnostics_accumulator.scoped(
                            null_type=TARGET_DECOY_NULL_TYPE, evaluable_set_type=str(eval_type),
                        )
                        if diagnostics_accumulator is not None and scope == EDGE_SCOPE_FIRST_EDGE_EXCLUDED
                        else None
                    ),
                    # The cell-pair debug shard stays an all-edges artifact (spec §2.2).
                    debug_writer=debug_writer if scope == EDGE_SCOPE_ALL else None,
                    debug_source="null",
                    debug_null_type=TARGET_DECOY_NULL_TYPE,
                    debug_direction=debug_direction,
                    iteration=None,
                    drug_groups=decoy_drug_groups,
                    pert_groups=decoy_pert_groups,
                )
                cand_joint = cand_joint.merge(cand_meta, on="triplet_id", how="left")
                if extra_cols:
                    cand_joint = cand_joint.merge(cand_eval[extra_cols].drop_duplicates("triplet_id"), on="triplet_id", how="left")
                parts_by_scope[scope].append(cand_joint)
    return (
        {
            scope: (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame())
            for scope, parts in parts_by_scope.items()
        },
        pd.concat(eval_parts, ignore_index=True) if eval_parts else pd.DataFrame(),
    )


def sample_decoy_anchor_joint_baseline(
    decoy_candidate_joint_df: pd.DataFrame,
    true_joint_triplet_df: pd.DataFrame,
    *,
    n_shuffle: int,
    rng: np.random.Generator,
    evaluable_set_types: Sequence[str],
    sample_with_replacement: bool = False,
    null_type: str = DECOY_NULL_TYPE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use all final evaluable unique decoy anchors/targets for the decoy control.

    ``n_shuffle`` and replacement sampling are accepted for backward API
    compatibility but are intentionally not used for the decoy control after v2.6.
    """
    if decoy_candidate_joint_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    required = {"original_triplet_id", "original_target_id", "decoy_anchor_id", "evaluable_set_type"}
    missing = required - set(decoy_candidate_joint_df.columns)
    if missing:
        raise ValueError(f"decoy_candidate_joint_df missing columns: {sorted(missing)}")
    true_ids_by_set = {
        str(eval_type): set(true_joint_triplet_df.loc[true_joint_triplet_df["evaluable_set_type"].astype(str).eq(str(eval_type)), "triplet_id"].astype(str))
        for eval_type in evaluable_set_types
    }
    parts: list[pd.DataFrame] = []
    diag_rows: list[dict[str, Any]] = []
    for eval_type in map(str, evaluable_set_types):
        sub = decoy_candidate_joint_df.loc[decoy_candidate_joint_df["evaluable_set_type"].astype(str).eq(eval_type)].copy()
        sub = sub.loc[_is_evaluable_joint_row(sub)].copy()
        if sub.empty:
            continue
        for original_triplet_id, cand in sub.groupby("original_triplet_id", dropna=False):
            original_triplet_id = str(original_triplet_id)
            if original_triplet_id not in true_ids_by_set.get(eval_type, set()):
                continue
            cand = cand.drop_duplicates(["decoy_anchor_id", "evaluable_set_type"]).copy()
            n = int(len(cand))
            for _, drec in cand.iterrows():
                diag_rows.append({
                    "null_type": str(null_type),
                    "triplet_id": original_triplet_id,
                    "drug_id": drec.get("drug_id"),
                    "target_id": drec.get("original_target_id"),
                    "disease_id": drec.get("disease_id"),
                    "evaluable_set_type": eval_type,
                    "n_decoy_anchors": n,
                    "decoy_anchor_id": drec.get("decoy_anchor_id"),
                    "decoy_anchor_degree_in": drec.get("decoy_anchor_degree_in", np.nan),
                    "decoy_anchor_degree_out": drec.get("decoy_anchor_degree_out", np.nan),
                    "n_reachable_nodes_decoy": drec.get("n_reachable_nodes_decoy", np.nan),
                    "n_decoy_paths": drec.get("n_cell_pairs_used", np.nan),
                    "profile_coverage_decoy": drec.get("candidate_nodes_min", np.nan),
                    "path_recoverability_decoy": drec.get("path_recoverability", np.nan),
                    "is_decoy_evaluable": True,
                })
            if n < 1:
                continue
            sampled = cand.copy().reset_index(drop=True)
            sampled["iteration"] = pd.NA
            sampled["decoy_triplet_id"] = sampled["triplet_id"].astype(str)
            sampled["triplet_id"] = original_triplet_id
            sampled["target_id"] = sampled["original_target_id"].astype(str)
            sampled["null_type"] = str(null_type)
            sampled["n_decoy_anchors_available"] = n
            parts.append(sampled)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    diag = pd.DataFrame(diag_rows)
    return out, diag

def write_target_decoy_diagnostics(
    *,
    out_dir: str | Path,
    assignments: pd.DataFrame,
    candidate_pair_counts: pd.DataFrame,
    candidate_joint_df: pd.DataFrame,
    global_joint_comparison: pd.DataFrame,
    comparison_triplet_sets: pd.DataFrame,
    observed_evaluable_ids_by_set: Mapping[str, set[str]],
    eval_set_types: Sequence[str],
    decoy_selection: str = TARGET_DECOY_SELECTION_MATCHED,
) -> dict[str, str]:
    """Write the required Target-Decoy matching / evaluability diagnostic tables.

    Every selected candidate stays in the evaluability table, including the
    non-evaluable ones, which keep a missing score instead of a zero.

    Matching balance and SMD acceptance describe how well the decoys were matched, so
    they are only produced for ``decoy_selection="matched"``; a random draw has no
    matching to judge and emits the evaluability tables alone.
    """
    out_dir = Path(out_dir)
    eval_types = [str(t) for t in eval_set_types]
    matched = str(decoy_selection) == TARGET_DECOY_SELECTION_MATCHED
    evaluability_parts: list[pd.DataFrame] = []
    triplet_evaluability_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    balance_parts: list[pd.DataFrame] = []
    delta_parts: list[pd.DataFrame] = []

    if matched:
        balance_parts.append(
            build_target_decoy_matching_balance(
                assignments,
                analysis_scope="all_profile_supported",
                stage="pre_inference_selected",
            ).assign(evaluable_set_type=pd.NA)
        )

    for eval_type in eval_types:
        counts = pd.DataFrame()
        if not candidate_pair_counts.empty:
            counts = candidate_pair_counts.loc[
                candidate_pair_counts["evaluable_set_type"].astype(str).eq(eval_type)
            ].copy()
        source = counts
        if not counts.empty:
            keep = [
                c for c in [
                    "triplet_id", "evaluable_set_type", "n_raw_cell_pairs",
                    "n_path_existing_cell_pairs", "n_drug_paths", "n_kd_paths",
                ] if c in counts.columns
            ]
            source = counts[keep].drop_duplicates("triplet_id").copy()
            if not candidate_joint_df.empty and "joint_jaccard" in candidate_joint_df.columns:
                joint = candidate_joint_df.loc[
                    candidate_joint_df["evaluable_set_type"].astype(str).eq(eval_type),
                    ["triplet_id", "joint_jaccard"],
                ].drop_duplicates("triplet_id")
                source = source.merge(joint, on="triplet_id", how="left")
        evaluability = build_target_decoy_evaluability(assignments, source, evaluable_set_type=eval_type)
        evaluability_parts.append(evaluability)
        triplet_evaluability_parts.append(build_target_decoy_triplet_evaluability(evaluability))

        observed_ids = {str(t) for t in observed_evaluable_ids_by_set.get(eval_type, set())}
        comparison_ids = _comparison_ids_for_group(
            comparison_triplet_sets, null_type=TARGET_DECOY_NULL_TYPE, eval_type=eval_type
        ) or set()
        summary_parts.append(
            summarize_target_decoy_evaluability(
                evaluability,
                evaluable_set_type=eval_type,
                observed_evaluable_triplet_ids=observed_ids,
                comparison_triplet_ids=comparison_ids,
            )
        )

        if matched:
            surviving = assignments.merge(
                evaluability[["pseudo_triplet_id", "is_decoy_evaluable"]].rename(
                    columns={"is_decoy_evaluable": "_is_surviving"}
                ),
                on="pseudo_triplet_id",
                how="left",
            )
            for scope, triplet_ids in [
                ("all_profile_supported", None),
                ("primary_path_evaluable", observed_ids or None),
            ]:
                if scope == "primary_path_evaluable" and not observed_ids:
                    continue
                if scope == "primary_path_evaluable":
                    # The pre-inference stage of the all_profile_supported scope does not
                    # depend on the evaluable set and is emitted once, above.
                    balance_parts.append(
                        build_target_decoy_matching_balance(
                            assignments,
                            analysis_scope=scope,
                            stage="pre_inference_selected",
                            triplet_ids=triplet_ids,
                        ).assign(evaluable_set_type=eval_type)
                    )
                balance_parts.append(
                    build_target_decoy_matching_balance(
                        surviving,
                        analysis_scope=scope,
                        stage="post_evaluability_surviving",
                        triplet_ids=triplet_ids,
                        surviving_col="_is_surviving",
                    ).assign(evaluable_set_type=eval_type)
                )

        if not global_joint_comparison.empty and "null_type" in global_joint_comparison.columns:
            delta_sub = global_joint_comparison.loc[
                global_joint_comparison["null_type"].astype(str).eq(TARGET_DECOY_NULL_TYPE)
                & global_joint_comparison["evaluable_set_type"].astype(str).eq(eval_type)
            ]
            delta_parts.append(
                summarize_target_decoy_delta_by_n_evaluable(delta_sub, evaluable_set_type=eval_type)
            )

    acceptance = pd.DataFrame()
    if matched:
        balance_frames = [p for p in balance_parts if p is not None and not p.empty]
        acceptance = build_target_decoy_matching_acceptance(
            pd.concat(balance_frames, ignore_index=True) if balance_frames else pd.DataFrame()
        )
        if not acceptance.empty and not bool(acceptance["all_matching_variables_within_threshold"].all()):
            failing = acceptance.loc[~acceptance["is_within_threshold"].astype(bool)]
            detail = ", ".join(
                f"{v}={x:.3f}" for v, x in zip(failing["variable"].astype(str), failing["abs_smd"])
            )
            print(
                "[WARN] Target-Decoy primary matching balance acceptance failed "
                f"(|SMD| >= {float(acceptance['abs_smd_threshold'].iloc[0])}): {detail}"
            )

    outputs: dict[str, str] = {}
    tables = {
        "decoy_candidate_evaluability": (out_dir / "decoy_candidate_evaluability.tsv", evaluability_parts),
        "decoy_triplet_evaluability": (out_dir / "decoy_triplet_evaluability.tsv", triplet_evaluability_parts),
        "decoy_evaluability_summary": (out_dir / "decoy_evaluability_summary.tsv", summary_parts),
        "decoy_matching_balance": (out_dir / "decoy_matching_balance.tsv", balance_parts),
        "decoy_matching_acceptance": (out_dir / "decoy_matching_acceptance.tsv", [acceptance]),
        "decoy_delta_by_n_evaluable_decoys": (out_dir / "decoy_delta_by_n_evaluable_decoys.tsv", delta_parts),
    }
    for name, (path, parts) in tables.items():
        frames = [p for p in parts if p is not None and not p.empty]
        if not frames:
            continue
        pd.concat(frames, ignore_index=True).to_csv(path, sep="\t", index=False)
        outputs[name] = str(path)
    return outputs


def target_decoy_summary_metadata(
    *,
    out_dir: str | Path,
    outputs: Mapping[str, str],
) -> dict[str, Any]:
    """Target-Decoy model definition and matching acceptance for the evaluation summary.

    The fixed model definition together with the actual run values (decoy selection
    rule, candidate-universe size, tie-break seed) belongs in the summary, not only in
    the matching-stage JSON, so that no consumer has to hard-code them.
    """
    out: dict[str, Any] = {}
    model_metadata_path = Path(out_dir) / "decoy_matching_summary.json"
    if model_metadata_path.exists():
        try:
            out["target_decoy_model"] = json.loads(model_metadata_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    acceptance_path = outputs.get("decoy_matching_acceptance")
    acceptance_df = (
        pd.read_csv(acceptance_path, sep="\t")
        if acceptance_path and Path(acceptance_path).exists()
        else pd.DataFrame()
    )
    out["target_decoy_matching_acceptance"] = target_decoy_acceptance_metadata(acceptance_df)
    return out


def _comparison_triplet_count_summary(comparison_triplet_sets: pd.DataFrame) -> dict[str, dict[str, int]]:
    if comparison_triplet_sets.empty:
        return {}
    counts = (
        comparison_triplet_sets.loc[comparison_triplet_sets["is_comparison_triplet"].astype(bool)]
        .groupby(["null_type", "evaluable_set_type"], dropna=False)
        .size()
        .reset_index(name="n_triplets")
    )
    out: dict[str, dict[str, int]] = {}
    for rec in counts.to_dict(orient="records"):
        null_type = str(rec["null_type"])
        eval_type = str(rec["evaluable_set_type"])
        out.setdefault(null_type, {})[eval_type] = int(rec["n_triplets"])
    return out


def _enabled_edge_scopes(evaluation_cfg: Mapping[str, Any]) -> list[str]:
    """Edge scopes to evaluate. ``all_edges`` is always on (enforced in config)."""
    scopes_cfg = (evaluation_cfg or {}).get("edge_scopes") or {}
    out = [EDGE_SCOPE_ALL]
    if bool((scopes_cfg.get(EDGE_SCOPE_FIRST_EDGE_EXCLUDED) or {}).get("enabled", False)):
        out.append(EDGE_SCOPE_FIRST_EDGE_EXCLUDED)
    return out


def baseline_legacy_pair_counters(
    drug_df: pd.DataFrame,
    pert_df: pd.DataFrame,
    *,
    cell_pair_mode: str,
    filters: Mapping[str, Any] | None,
) -> tuple[int, int]:
    """``(n_cell_pairs_baseline, n_triplets_baseline)`` for one null repeat.

    These two counters were the only consumers of the per-repeat legacy cell-pair
    and triplet metrics, so the set algebra behind them is skipped here. The triplet
    count reproduces ``len(aggregate_triplet_metrics(...))``: the number of filtered
    pair groups, or the diagnostics row count when no pair survives the filter.
    """
    pair_raw = build_cell_pairs(drug_df, pert_df, cell_pair_mode=cell_pair_mode)
    diag = cellpair_filter_diagnostics(pair_raw)
    if pair_raw.empty:
        return 0, 0
    filters = dict(filters or {})
    pair_df = filter_evaluable_cell_pairs(
        pair_raw,
        require_both_nonempty_inferred_paths=bool(filters.get("require_both_nonempty_inferred_paths", True)),
        require_both_nonempty_signed_edges=bool(filters.get("require_both_nonempty_signed_edges", False)),
    )
    if pair_df.empty:
        return 0, int(len(diag))
    group_cols = ["triplet_id"] + [c for c in ["drug_id", "target_id", "direction"] if c in pair_df.columns]
    return int(len(pair_df)), int(pair_df.groupby(group_cols, dropna=False).ngroups)


FIRST_EDGE_EXCLUSION_SUMMARY_COLUMNS = [
    "null_type",
    "evaluable_set_type",
    "statistic",
    "n_comparison_triplets",
    "observed_mean_all_edges",
    "observed_mean_first_edge_excluded",
    "null_median_all_edges",
    "null_median_first_edge_excluded",
    "delta_mean_all_edges",
    "delta_mean_first_edge_excluded",
    "delta_mean_ratio",
    "n_pairs_empty_after_exclusion",
    "n_edges_removed_median_drug",
    "n_edges_removed_median_pert",
]

# Column provenance for the cross-scope summary, kept in one place (spec §5.7).
# Keys are the summary's per-scope suffixes; values are the source columns in
# ``compute_global_joint_triplet_comparison`` output that are averaged per group.
FIRST_EDGE_EXCLUSION_COMPARISON_SOURCES = {
    "observed_mean": "observed_value",
    "null_median": "baseline_representative_value",
    "delta_mean": "delta",
}


def _comparison_group_means(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Per ``(null_type, evaluable_set_type, statistic)`` means of the cmp table."""
    if comparison_df is None or comparison_df.empty:
        return pd.DataFrame()
    df = comparison_df.copy()
    # ``baseline_representative`` holds the *name* of the representative; the value
    # lives in baseline_mean / baseline_median depending on that name. Current runs
    # record "median"; the branch stays so that a comparison table from a mean-based
    # run is still read with its own representative.
    representative = df.get("baseline_representative")
    baseline_mean = pd.to_numeric(df.get("baseline_mean"), errors="coerce")
    baseline_median = pd.to_numeric(df.get("baseline_median"), errors="coerce")
    if representative is None:
        df["baseline_representative_value"] = baseline_mean
    else:
        df["baseline_representative_value"] = baseline_mean.where(
            representative.astype(str).eq("mean"), baseline_median
        )
    df["observed_value"] = pd.to_numeric(df.get("observed_value"), errors="coerce")
    df["delta"] = pd.to_numeric(df.get("delta"), errors="coerce")
    keys = ["null_type", "evaluable_set_type", "statistic"]
    grouped = df.groupby(keys, dropna=False)
    out = grouped.agg(
        n_comparison_triplets=("triplet_id", "nunique"),
        observed_value=("observed_value", "mean"),
        baseline_representative_value=("baseline_representative_value", "mean"),
        delta=("delta", "mean"),
    ).reset_index()
    return out


def build_first_edge_exclusion_summary(
    *,
    comparison_by_scope: Mapping[str, pd.DataFrame],
    diagnostics: "FirstEdgeExclusionDiagnostics | None",
) -> pd.DataFrame:
    """Side-by-side all-edges vs first-edge-excluded sensitivity table (spec §5.7)."""
    base = _comparison_group_means(comparison_by_scope.get(EDGE_SCOPE_ALL, pd.DataFrame()))
    if base.empty:
        return pd.DataFrame(columns=FIRST_EDGE_EXCLUSION_SUMMARY_COLUMNS)
    excluded = _comparison_group_means(
        comparison_by_scope.get(EDGE_SCOPE_FIRST_EDGE_EXCLUDED, pd.DataFrame())
    )
    keys = ["null_type", "evaluable_set_type", "statistic"]
    out = base[keys + ["n_comparison_triplets"]].copy()
    for suffix, source in FIRST_EDGE_EXCLUSION_COMPARISON_SOURCES.items():
        out[f"{suffix}_{EDGE_SCOPE_ALL}"] = base[source].to_numpy()
    if excluded.empty:
        for suffix in FIRST_EDGE_EXCLUSION_COMPARISON_SOURCES:
            out[f"{suffix}_{EDGE_SCOPE_FIRST_EDGE_EXCLUDED}"] = np.nan
    else:
        merged = out.merge(
            excluded[keys + list(FIRST_EDGE_EXCLUSION_COMPARISON_SOURCES.values()) + ["n_comparison_triplets"]],
            on=keys,
            how="left",
            suffixes=("", "_excluded"),
        )
        mismatched = merged.loc[
            merged["n_comparison_triplets_excluded"].notna()
            & merged["n_comparison_triplets_excluded"].ne(merged["n_comparison_triplets"])
        ]
        if not mismatched.empty:
            raise AssertionError(
                "First-edge exclusion changed the comparison triplet set, which spec §5.3 forbids: "
                f"{mismatched[keys].to_dict(orient='records')[:5]}"
            )
        for suffix, source in FIRST_EDGE_EXCLUSION_COMPARISON_SOURCES.items():
            out[f"{suffix}_{EDGE_SCOPE_FIRST_EDGE_EXCLUDED}"] = merged[source].to_numpy()

    denominator = pd.to_numeric(out[f"delta_mean_{EDGE_SCOPE_ALL}"], errors="coerce")
    numerator = pd.to_numeric(out[f"delta_mean_{EDGE_SCOPE_FIRST_EDGE_EXCLUDED}"], errors="coerce")
    out["delta_mean_ratio"] = (numerator / denominator).where(denominator.ne(0) & denominator.notna())

    diag_frame = diagnostics.to_frame() if diagnostics is not None else pd.DataFrame()
    diag_columns = [
        "n_pairs_empty_after_exclusion",
        "n_edges_removed_median_drug",
        "n_edges_removed_median_pert",
    ]
    if diag_frame.empty:
        for column in diag_columns:
            out[column] = pd.NA
    else:
        out = out.merge(
            diag_frame[["null_type", "evaluable_set_type"] + diag_columns],
            on=["null_type", "evaluable_set_type"],
            how="left",
        )
    return out.reindex(columns=FIRST_EDGE_EXCLUSION_SUMMARY_COLUMNS)


def _write_edge_scope_statistics(
    *,
    scope: str,
    out_dir: Path,
    true_joint_triplet: pd.DataFrame,
    baseline_joint_triplet: pd.DataFrame,
    metrics: Sequence[str],
    global_options: Mapping[str, bool],
) -> dict[str, Any]:
    """Comparison and statistics for one edge scope.

    ``all_edges`` writes to the evaluation root with the historical filenames and
    no extra column; every other scope writes the same files into its own
    subdirectory with a leading ``edge_scope`` column.
    """
    is_primary_scope = scope == EDGE_SCOPE_ALL
    scope_dir = Path(out_dir) if is_primary_scope else ensure_dir(Path(out_dir) / scope)

    def _tag(df: pd.DataFrame) -> pd.DataFrame:
        if is_primary_scope or df is None or df.empty:
            return df
        tagged = df.copy()
        tagged.insert(0, "edge_scope", scope)
        return tagged

    n_shuffle_by_null_type: dict[str, int] = {}
    if not baseline_joint_triplet.empty and {"null_type", "iteration"} <= set(baseline_joint_triplet.columns):
        n_shuffle_by_null_type.update(
            baseline_joint_triplet.groupby("null_type", dropna=False)["iteration"].nunique().astype(int).to_dict()
        )
    comparison_triplet_sets = build_comparison_triplet_sets(
        true_joint_triplet,
        baseline_joint_triplet,
        n_shuffle_by_null_type=n_shuffle_by_null_type,
    )
    global_joint_comparison = compute_global_joint_triplet_comparison(
        true_joint_triplet,
        baseline_joint_triplet,
        metrics=metrics,
        comparison_triplet_sets=comparison_triplet_sets,
    )

    outputs: dict[str, str] = {}
    true_path = scope_dir / "global_joint_triplet_metrics_true.parquet"
    baseline_path = scope_dir / "global_joint_triplet_metrics_baseline.parquet"
    comparison_sets_path = scope_dir / "global_joint_triplet_comparison_sets.tsv"
    comparison_path = scope_dir / "global_joint_triplet_comparison.tsv"
    write_parquet_zstd(_tag(true_joint_triplet), true_path)
    write_parquet_zstd(_tag(prune_non_v23_null_types(baseline_joint_triplet)), baseline_path)
    _tag(prune_non_v23_null_types(comparison_triplet_sets)).to_csv(comparison_sets_path, sep="\t", index=False)
    _tag(prune_non_v23_null_types(global_joint_comparison)).to_csv(comparison_path, sep="\t", index=False)
    outputs["global_joint_triplet_metrics_true"] = str(true_path)
    outputs["global_joint_triplet_metrics_baseline"] = str(baseline_path)
    outputs["global_joint_triplet_comparison_sets"] = str(comparison_sets_path)
    outputs["global_joint_triplet_comparison"] = str(comparison_path)

    if global_options["global_effect_size"]:
        effect = prune_non_v23_null_types(
            compute_global_effect_size(
                true_joint_triplet, baseline_joint_triplet, metrics=metrics,
                comparison_triplet_sets=comparison_triplet_sets,
            )
        )
        effect_path = scope_dir / "global_effect_size.tsv"
        _tag(effect).to_csv(effect_path, sep="\t", index=False)
        outputs["global_effect_size"] = str(effect_path)

    return {
        "outputs": outputs,
        "comparison_triplet_sets": comparison_triplet_sets,
        "global_joint_comparison": global_joint_comparison,
        "n_shuffle_by_null_type": n_shuffle_by_null_type,
    }


def evaluate_direction(
    *,
    true_drug_infer_path: str | Path,
    true_pert_infer_path: str | Path,
    true_drug_input_path: str | Path | None = None,
    true_pert_input_path: str | Path | None = None,
    baseline_drug_input_path: str | Path | None = None,
    baseline_pert_input_path: str | Path | None = None,
    kg_tsv: str | Path | None = None,
    cutoff: int | None = None,
    gap: int | None = None,
    drug_mode: str,
    pert_mode: str,
    cell_pair_mode: str,
    evaluation_cfg: Mapping[str, Any],
    baseline_pairs: Sequence[Sequence[Any]],
    out_dir: str | Path,
    baseline_specs: Sequence[Mapping[str, Any]] | None = None,
    path_cache_dir: str | Path | None = None,
    path_strategy: str = "simple",
    decoy_candidate_drug_infer_path: str | Path | None = None,
    decoy_candidate_pert_infer_path: str | Path | None = None,
    decoy_candidate_table_path: str | Path | None = None,
    decoy_null_type: str | None = None,
    progress: bool = False,
    tqdm_kwargs: dict[str, Any] | None = None,
    baseline_cfg: Mapping[str, Any] | None = None,
    seed: int = 0,
    summary_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    cellpair_debug_writer = CellpairDebugWriter(Path(out_dir) / "cellpair_metrics_debug")
    filters = dict((evaluation_cfg or {}).get("filters") or {})
    metrics = [str(m) for m in (evaluation_cfg or {}).get("metrics", ["jaccard"])]
    if metrics != ["jaccard"]:
        raise ValueError("Val3 evaluation metrics must be exactly ['jaccard'].")
    baseline_cfg = dict(baseline_cfg or {})
    edge_scopes = _enabled_edge_scopes(evaluation_cfg)
    want_first_edge_excluded = EDGE_SCOPE_FIRST_EDGE_EXCLUDED in edge_scopes
    # Parallelism inside one Null 1 repeat, over source groups (spec §6.9).
    null1_max_workers = max(1, int((evaluation_cfg or {}).get("max_workers", 1) or 1))
    structural_enabled = _structural_null_enabled(baseline_cfg)
    # ``signed_edge_paths`` only feeds the structural null; skipping it removes a
    # per-row list of sets from every inference frame in the primary config.
    include_signed_edge_paths = bool(structural_enabled)
    first_edge_diagnostics = FirstEdgeExclusionDiagnostics() if want_first_edge_excluded else None
    iteration_timings: list[dict[str, Any]] = []

    true_drug_df = load_infer_signed_edge_df(
        true_drug_infer_path,
        mode=drug_mode,
        include_signed_edge_paths=include_signed_edge_paths,
        include_first_edge_excluded=want_first_edge_excluded,
    )
    true_pert_df = load_infer_signed_edge_df(
        true_pert_infer_path,
        mode=pert_mode,
        include_signed_edge_paths=include_signed_edge_paths,
        include_first_edge_excluded=want_first_edge_excluded,
    )

    cellpair_true, diag_true = compute_cellpair_metrics(
        true_drug_df, true_pert_df, cell_pair_mode=cell_pair_mode, filters=filters,
        progress=progress, tqdm_kwargs=tqdm_kwargs,
    )
    triplet_true = aggregate_triplet_metrics(cellpair_true, diag_true, include_iteration=False)

    normalized_pairs = [_normalize_baseline_pair(item) for item in baseline_pairs]
    inprocess_specs = list(baseline_specs or [])
    baseline_loaded: list[tuple[int, str, str, str]] = [
        (int(iteration), str(drug_path), str(pert_path), str(null_type))
        for iteration, drug_path, pert_path, null_type in normalized_pairs
    ] + [
        (
            int(spec["iteration"]),
            "<in-process>",
            "<in-process>",
            str(spec.get("null_type", "target_khop_restricted_paired_gene_label_permutation")),
        )
        for spec in inprocess_specs
    ]

    global_options = _enabled_global_options(evaluation_cfg)
    needs_global_summary = any(global_options.values())
    global_summary: dict[str, Any] = {"enabled": False}

    inference_null_types: list[str] = []
    inference_null_enabled = False
    decoy_enabled = False
    resolved_decoy_null_type = str(decoy_null_type or _enabled_decoy_null_type(baseline_cfg))
    eval_set_types: list[str] = []
    ev_cfg: dict[str, Any] = {}
    evaluable_df = pd.DataFrame()
    triplet_ids_by_set: dict[str, set[str]] = {}
    decoy_candidates = pd.DataFrame()

    if needs_global_summary:
        inference_null_types = _inference_null_types_enabled(baseline_cfg, baseline_loaded)
        inference_null_enabled = bool(inference_null_types)
        decoy_enabled = (
            _decoy_null_enabled(baseline_cfg)
            and decoy_candidate_drug_infer_path is not None
            and decoy_candidate_pert_infer_path is not None
        )
        if resolved_decoy_null_type not in DECOY_NULL_TYPES:
            raise ValueError(f"Unsupported decoy null type: {resolved_decoy_null_type}")
        resolved_decoy_null_type = canonical_decoy_null_type(resolved_decoy_null_type)
        if inference_null_enabled and not baseline_loaded:
            raise ValueError("Inference-based gene-label null requires at least one baseline iteration.")
        if not inference_null_enabled and not structural_enabled and not decoy_enabled:
            raise ValueError("At least one null model must be enabled when a global summary option is enabled.")
        eval_set_types = _enabled_evaluable_sets(evaluation_cfg)
        if not eval_set_types:
            raise ValueError("At least one evaluation.evaluable_sets.*.enabled must be true when a global summary option is enabled.")

        ev_cfg = dict((evaluation_cfg or {}).get("evaluable_sets") or {})
        # The evaluable triplet sets are fixed by the observed run, so they are
        # resolved before the baseline loop and every repeat reuses them.
        evaluable_parts: list[pd.DataFrame] = []
        if "fixed_coverage" in eval_set_types:
            fixed_cfg = dict(ev_cfg.get("fixed_coverage") or {})
            fixed_df = build_global_evaluable_triplets(
                true_drug_df,
                true_pert_df,
                cell_pair_mode=cell_pair_mode,
                min_candidate_nodes=int(fixed_cfg.get("min_candidate_nodes", 10)),
            )
            evaluable_parts.append(fixed_df)
            triplet_ids_by_set["fixed_coverage"] = set(fixed_df.loc[fixed_df["is_global_evaluable"], "triplet_id"].astype(str))
        if "path_existing_pairs" in eval_set_types:
            path_df = build_path_existing_evaluable_triplets(
                true_drug_df,
                true_pert_df,
                cell_pair_mode=cell_pair_mode,
            )
            evaluable_parts.append(path_df)
            triplet_ids_by_set["path_existing_pairs"] = set(path_df.loc[path_df["is_global_evaluable"], "triplet_id"].astype(str))
        evaluable_df = pd.concat(evaluable_parts, ignore_index=True) if evaluable_parts else pd.DataFrame()

        if decoy_candidate_drug_infer_path is not None and decoy_candidate_pert_infer_path is not None:
            if decoy_candidate_table_path is not None and Path(decoy_candidate_table_path).exists():
                decoy_candidates = pd.read_csv(decoy_candidate_table_path, sep="\t")

    true_joint_parts_by_scope: dict[str, list[pd.DataFrame]] = {scope: [] for scope in edge_scopes}
    baseline_joint_parts_by_scope: dict[str, list[pd.DataFrame]] = {scope: [] for scope in edge_scopes}

    def _scoped_diagnostics(scope: str, *, null_type: str, evaluable_set_type: str):
        if first_edge_diagnostics is None or scope != EDGE_SCOPE_FIRST_EDGE_EXCLUDED:
            return None
        return first_edge_diagnostics.scoped(null_type=null_type, evaluable_set_type=evaluable_set_type)

    for eval_type, fixed_triplet_ids in triplet_ids_by_set.items():
        for scope in edge_scopes:
            true_joint_parts_by_scope[scope].append(
                compute_joint_triplet_metrics(
                    true_drug_df,
                    true_pert_df,
                    cell_pair_mode=cell_pair_mode,
                    fixed_triplet_ids=fixed_triplet_ids,
                    metrics=metrics,
                    evaluable_set_type=eval_type,
                    edge_column_suffix=edge_column_suffix_for_scope(scope),
                    diagnostics_accumulator=_scoped_diagnostics(
                        scope, null_type="observed", evaluable_set_type=eval_type,
                    ),
                    # The cell-pair debug shard stays an all-edges artifact (spec §2.2).
                    debug_writer=cellpair_debug_writer if scope == EDGE_SCOPE_ALL else None,
                    debug_source="observed",
                    debug_null_type="observed",
                    debug_direction=summary_extra.get("direction") if summary_extra else None,
                )
            )

    baseline_cellpair_count = 0
    baseline_triplet_count = 0

    def _consume_baseline_iteration(
        *,
        iteration: int,
        null_type: str,
        b_drug: pd.DataFrame,
        b_pert: pd.DataFrame,
        started_at: float,
        stages: list[tuple[str, float, int]],
        joint_stage: str,
        joint_stage_offset_sec: float,
    ) -> None:
        nonlocal baseline_cellpair_count, baseline_triplet_count
        t0 = time.perf_counter()
        n_pairs, n_triplets = baseline_legacy_pair_counters(
            b_drug, b_pert, cell_pair_mode=cell_pair_mode, filters=filters,
        )
        baseline_cellpair_count += n_pairs
        baseline_triplet_count += n_triplets
        stages.append(("pair_counters", time.perf_counter() - t0, n_pairs))

        t0 = time.perf_counter()
        if inference_null_enabled:
            for eval_type, fixed_triplet_ids in triplet_ids_by_set.items():
                for scope in edge_scopes:
                    part = compute_joint_triplet_metrics(
                        b_drug,
                        b_pert,
                        cell_pair_mode=cell_pair_mode,
                        fixed_triplet_ids=fixed_triplet_ids,
                        metrics=metrics,
                        evaluable_set_type=eval_type,
                        iteration=iteration,
                        edge_column_suffix=edge_column_suffix_for_scope(scope),
                        diagnostics_accumulator=_scoped_diagnostics(
                            scope, null_type=str(null_type), evaluable_set_type=eval_type,
                        ),
                        debug_writer=cellpair_debug_writer if scope == EDGE_SCOPE_ALL else None,
                        debug_source="null",
                        debug_null_type=str(null_type),
                        debug_direction=summary_extra.get("direction") if summary_extra else None,
                    )
                    part["null_type"] = str(null_type)
                    baseline_joint_parts_by_scope[scope].append(part)
        stages.append((joint_stage, (time.perf_counter() - t0) + float(joint_stage_offset_sec), int(len(b_drug))))
        stages.append(("total", time.perf_counter() - started_at, int(len(b_drug))))
        for stage, duration, n_rows in stages:
            iteration_timings.append(
                {
                    "null_type": str(null_type),
                    "iteration": int(iteration),
                    "stage": str(stage),
                    "duration_sec": float(duration),
                    "n_rows": int(n_rows),
                }
            )

    pair_iter: Any = normalized_pairs
    if progress and normalized_pairs:
        pair_iter = tqdm(list(normalized_pairs), desc="baseline evaluation", unit="iter", **(tqdm_kwargs or {}))
    for iteration, drug_path, pert_path, null_type in pair_iter:
        started_at = time.perf_counter()
        stages: list[tuple[str, float, int]] = []
        t0 = time.perf_counter()
        b_drug = load_infer_signed_edge_df(
            drug_path, mode=drug_mode,
            include_signed_edge_paths=include_signed_edge_paths,
            include_first_edge_excluded=want_first_edge_excluded,
        )
        stages.append(("infer_drug", time.perf_counter() - t0, int(len(b_drug))))
        t0 = time.perf_counter()
        b_pert = load_infer_signed_edge_df(
            pert_path, mode=pert_mode,
            include_signed_edge_paths=include_signed_edge_paths,
            include_first_edge_excluded=want_first_edge_excluded,
        )
        stages.append(("infer_pert", time.perf_counter() - t0, int(len(b_pert))))
        _consume_baseline_iteration(
            iteration=int(iteration),
            null_type=str(null_type),
            b_drug=b_drug,
            b_pert=b_pert,
            started_at=started_at,
            stages=stages,
            joint_stage="joint_metrics",
            joint_stage_offset_sec=0.0,
        )
        del b_drug, b_pert

    if inprocess_specs:
        baseline_drug_input_path = baseline_drug_input_path or true_drug_input_path
        baseline_pert_input_path = baseline_pert_input_path or true_pert_input_path
        if baseline_drug_input_path is None or baseline_pert_input_path is None:
            raise ValueError("In-process baseline evaluation requires baseline or true drug/pert input paths.")
        if kg_tsv is None or cutoff is None or gap is None:
            raise ValueError("In-process baseline evaluation requires kg_tsv, cutoff, and gap.")
        graph_for_inprocess = _load_kg_graph(kg_tsv)
        prepared_drug = prepare_infer_input(baseline_drug_input_path, graph=graph_for_inprocess, mode=drug_mode)
        prepared_pert = prepare_infer_input(baseline_pert_input_path, graph=graph_for_inprocess, mode=pert_mode)
        trie_cache_drug = TrieCache(
            mode=drug_mode, path_strategy=str(path_strategy), cutoff=int(cutoff),
            valid_genes_digest=prepared_drug.valid_genes_digest,
        )
        trie_cache_pert = TrieCache(
            mode=pert_mode, path_strategy=str(path_strategy), cutoff=int(cutoff),
            valid_genes_digest=prepared_pert.valid_genes_digest,
        )
        parallel_evaluator = None
        if null1_max_workers > 1:
            parallel_evaluator = Null1ParallelEvaluator(
                sides={"drug": prepared_drug, "pert": prepared_pert},
                graph=graph_for_inprocess,
                gap=int(gap),
                cutoff=int(cutoff),
                path_strategy=str(path_strategy),
                path_cache_dir=path_cache_dir,
                max_workers=null1_max_workers,
            )
        spec_iter: Any = inprocess_specs
        if progress:
            spec_iter = tqdm(list(inprocess_specs), desc="baseline in-process evaluation", unit="iter", **(tqdm_kwargs or {}))
        try:
            for spec in spec_iter:
                started_at = time.perf_counter()
                stages = []
                iteration = int(spec["iteration"])
                null_type = str(spec.get("null_type", "target_khop_restricted_paired_gene_label_permutation"))
                t0 = time.perf_counter()
                permutation_specs = load_permutation_specs(spec.get("permutation_specs"))
                stages.append(("permutation_lookup", time.perf_counter() - t0, 0))
                t0 = time.perf_counter()
                if parallel_evaluator is not None:
                    frames = parallel_evaluator.signed_edge_frames(
                        permutation_specs=permutation_specs,
                        include_first_edge_excluded=want_first_edge_excluded,
                    )
                    b_drug = frames["drug"]
                    b_pert = frames["pert"]
                else:
                    b_drug = null1_signed_edge_df(
                        prepared_drug,
                        graph=graph_for_inprocess,
                        gap=int(gap),
                        permutation_specs=permutation_specs,
                        trie_cache=trie_cache_drug,
                        path_cache_dir=path_cache_dir,
                        include_first_edge_excluded=want_first_edge_excluded,
                    )
                    b_pert = null1_signed_edge_df(
                        prepared_pert,
                        graph=graph_for_inprocess,
                        gap=int(gap),
                        permutation_specs=permutation_specs,
                        trie_cache=trie_cache_pert,
                        path_cache_dir=path_cache_dir,
                        include_first_edge_excluded=want_first_edge_excluded,
                    )
                traversal_sec = time.perf_counter() - t0
                _consume_baseline_iteration(
                    iteration=iteration,
                    null_type=null_type,
                    b_drug=b_drug,
                    b_pert=b_pert,
                    started_at=started_at,
                    stages=stages,
                    joint_stage="fused_traversal_joint",
                    joint_stage_offset_sec=traversal_sec,
                )
                del b_drug, b_pert
                trie_cache_drug.release_row_caches()
                trie_cache_pert.release_row_caches()
        finally:
            if parallel_evaluator is not None:
                parallel_evaluator.close()

    structural_pool_diag_parts: list[pd.DataFrame] = []
    decoy_anchor_diagnostics = pd.DataFrame()
    decoy_candidate_pair_counts = pd.DataFrame()
    decoy_candidate_joint_by_scope: dict[str, pd.DataFrame] = {}

    if needs_global_summary:
        if structural_enabled:
            structural_pool_cache: dict[Any, Any] = {}
            for eval_type, fixed_triplet_ids in triplet_ids_by_set.items():
                structural_by_scope, structural_pool_diag = compute_structural_resampling_joint_metrics(
                    true_drug_df,
                    true_pert_df,
                    cell_pair_mode=cell_pair_mode,
                    fixed_triplet_ids=fixed_triplet_ids,
                    metrics=metrics,
                    evaluable_set_type=eval_type,
                    n_shuffle=int(baseline_cfg.get("n_shuffle", 100)),
                    rng=np.random.default_rng(int(seed) + 1000003),
                    kg_tsv=kg_tsv,
                    cutoff=cutoff,
                    drug_mode=drug_mode,
                    pert_mode=pert_mode,
                    path_strategy=path_strategy,
                    true_drug_input_path=true_drug_input_path,
                    true_pert_input_path=true_pert_input_path,
                    debug_writer=cellpair_debug_writer,
                    debug_direction=summary_extra.get("direction") if summary_extra else None,
                    edge_scopes=edge_scopes,
                    diagnostics_accumulator=first_edge_diagnostics,
                    pool_cache=structural_pool_cache,
                )
                for scope in edge_scopes:
                    part = structural_by_scope.get(scope)
                    if part is not None and not part.empty:
                        baseline_joint_parts_by_scope[scope].append(part)
                structural_pool_diag = structural_pool_diag.copy()
                structural_pool_diag["evaluable_set_type"] = eval_type
                structural_pool_diag_parts.append(structural_pool_diag)

        if decoy_enabled and not decoy_candidates.empty:
            decoy_candidate_joint_by_scope, decoy_candidate_pair_counts = compute_decoy_candidate_joint_metrics_streaming(
                drug_infer_path=decoy_candidate_drug_infer_path,
                pert_infer_path=decoy_candidate_pert_infer_path,
                decoy_candidates=decoy_candidates,
                drug_mode=drug_mode,
                pert_mode=pert_mode,
                cell_pair_mode=cell_pair_mode,
                eval_set_types=eval_set_types,
                ev_cfg=ev_cfg,
                metrics=metrics,
                edge_scopes=edge_scopes,
                diagnostics_accumulator=first_edge_diagnostics,
                debug_writer=cellpair_debug_writer,
                debug_direction=summary_extra.get("direction") if summary_extra else None,
            )
            primary_candidate_joint = decoy_candidate_joint_by_scope.get(EDGE_SCOPE_ALL, pd.DataFrame())
            if not primary_candidate_joint.empty and kg_tsv is not None and cutoff is not None:
                graph = _load_kg_graph(kg_tsv)
                deg_in = dict(graph.in_degree())
                deg_out = dict(graph.out_degree())
                reachable_n: dict[str, int] = {}
                for anchor in primary_candidate_joint["decoy_anchor_id"].dropna().astype(str).unique():
                    if anchor in graph:
                        reachable_n[anchor] = max(0, len(nx.single_source_shortest_path_length(graph, anchor, int(cutoff))) - 1)
                    else:
                        reachable_n[anchor] = 0
                for candidate_joint in decoy_candidate_joint_by_scope.values():
                    if candidate_joint.empty:
                        continue
                    anchors = candidate_joint["decoy_anchor_id"].astype(str)
                    candidate_joint["decoy_anchor_degree_in"] = anchors.map(lambda x: int(deg_in.get(x, 0)))
                    candidate_joint["decoy_anchor_degree_out"] = anchors.map(lambda x: int(deg_out.get(x, 0)))
                    candidate_joint["n_reachable_nodes_decoy"] = anchors.map(lambda x: int(reachable_n.get(x, 0)))
            for scope in edge_scopes:
                candidate_joint = decoy_candidate_joint_by_scope.get(scope, pd.DataFrame())
                if candidate_joint.empty:
                    continue
                scope_true_joint = (
                    pd.concat(true_joint_parts_by_scope[scope], ignore_index=True)
                    if true_joint_parts_by_scope[scope]
                    else pd.DataFrame()
                )
                # A fresh generator per scope keeps both scopes on the same anchors.
                decoy_baseline, scope_anchor_diagnostics = sample_decoy_anchor_joint_baseline(
                    candidate_joint,
                    scope_true_joint,
                    n_shuffle=int(baseline_cfg.get("n_shuffle", 100)),
                    rng=np.random.default_rng(int(seed) + 2000003),
                    evaluable_set_types=eval_set_types,
                    sample_with_replacement=False,
                    null_type=resolved_decoy_null_type,
                )
                if not decoy_baseline.empty:
                    baseline_joint_parts_by_scope[scope].append(decoy_baseline)
                if scope == EDGE_SCOPE_ALL:
                    decoy_anchor_diagnostics = scope_anchor_diagnostics

    if needs_global_summary:
        structural_pool_diag_all = (
            pd.concat(structural_pool_diag_parts, ignore_index=True) if structural_pool_diag_parts else pd.DataFrame()
        )
        global_evaluable_path = Path(out_dir) / "global_evaluable_triplets.tsv"
        evaluable_df.to_csv(global_evaluable_path, sep="\t", index=False)

        outputs_by_scope: dict[str, dict[str, str]] = {}
        comparison_by_scope: dict[str, pd.DataFrame] = {}
        n_shuffle_by_null_type: dict[str, int] = {}
        primary_comparison_sets = pd.DataFrame()
        primary_comparison = pd.DataFrame()
        for scope in edge_scopes:
            true_joint_triplet = canonical_sort_true_joint_triplet(
                pd.concat(true_joint_parts_by_scope[scope], ignore_index=True)
                if true_joint_parts_by_scope[scope]
                else pd.DataFrame()
            )
            baseline_joint_triplet = canonical_sort_baseline_joint_triplet(
                pd.concat(baseline_joint_parts_by_scope[scope], ignore_index=True)
                if baseline_joint_parts_by_scope[scope]
                else pd.DataFrame()
            )
            scope_result = _write_edge_scope_statistics(
                scope=scope,
                out_dir=out_dir,
                true_joint_triplet=true_joint_triplet,
                baseline_joint_triplet=baseline_joint_triplet,
                metrics=metrics,
                global_options=global_options,
            )
            outputs_by_scope[scope] = scope_result["outputs"]
            comparison_by_scope[scope] = scope_result["global_joint_comparison"]
            if scope == EDGE_SCOPE_ALL:
                n_shuffle_by_null_type = scope_result["n_shuffle_by_null_type"]
                primary_comparison_sets = scope_result["comparison_triplet_sets"]
                primary_comparison = scope_result["global_joint_comparison"]

        outputs: dict[str, str] = {"global_evaluable_triplets": str(global_evaluable_path)}
        outputs.update(outputs_by_scope[EDGE_SCOPE_ALL])
        structural_pool_diag_path = Path(out_dir) / "global_structural_pool_diagnostics.tsv"
        decoy_anchor_diag_path = Path(out_dir) / "decoy_anchor_diagnostics.tsv"
        if not structural_pool_diag_all.empty:
            structural_pool_diag_all.to_csv(structural_pool_diag_path, sep="\t", index=False)
            outputs["global_structural_pool_diagnostics"] = str(structural_pool_diag_path)
        # v3 final outputs do not retain the full decoy candidate joint metrics table.
        # It is kept in memory only long enough to sample Null 3 and summarize diagnostics.
        if not decoy_anchor_diagnostics.empty:
            decoy_anchor_diagnostics.to_csv(decoy_anchor_diag_path, sep="\t", index=False)
            outputs["decoy_anchor_diagnostics"] = str(decoy_anchor_diag_path)
        if decoy_enabled and resolved_decoy_null_type == TARGET_DECOY_NULL_TYPE and not decoy_candidates.empty:
            outputs.update(
                write_target_decoy_diagnostics(
                    out_dir=out_dir,
                    assignments=decoy_candidates,
                    candidate_pair_counts=decoy_candidate_pair_counts,
                    candidate_joint_df=decoy_candidate_joint_by_scope.get(EDGE_SCOPE_ALL, pd.DataFrame()),
                    global_joint_comparison=primary_comparison,
                    comparison_triplet_sets=primary_comparison_sets,
                    observed_evaluable_ids_by_set=triplet_ids_by_set,
                    eval_set_types=eval_set_types,
                    decoy_selection=str(
                        ((baseline_cfg.get("null_models") or {}).get(TARGET_DECOY_NULL_TYPE) or {})
                        .get("decoy_selection", TARGET_DECOY_SELECTION_MATCHED)
                    ),
                )
            )

        if want_first_edge_excluded:
            first_edge_summary = build_first_edge_exclusion_summary(
                comparison_by_scope=comparison_by_scope,
                diagnostics=first_edge_diagnostics,
            )
            first_edge_summary_path = Path(out_dir) / "first_edge_exclusion_summary.tsv"
            first_edge_summary.to_csv(first_edge_summary_path, sep="\t", index=False)
            outputs["first_edge_exclusion_summary"] = str(first_edge_summary_path)

        global_summary = {
            "enabled": True,
            "global_effect_size_enabled": bool(global_options["global_effect_size"]),
            "evaluable_set_types": eval_set_types,
            "enabled_null_models": inference_null_types + [
                name for name, enabled in [
                    ("path_count_length_matched_structural_resampling", structural_enabled),
                    (resolved_decoy_null_type, decoy_enabled),
                ] if enabled
            ],
            "decoy_null_type": resolved_decoy_null_type if decoy_enabled else None,
            "n_evaluable_triplets_by_set": {k: int(len(v)) for k, v in triplet_ids_by_set.items()},
            "n_comparison_triplets_by_null_and_set": _comparison_triplet_count_summary(primary_comparison_sets),
            "n_shuffle_by_null_type": {str(k): int(v) for k, v in n_shuffle_by_null_type.items()},
            "edge_scopes": edge_scopes,
            "outputs": outputs,
            "outputs_by_edge_scope": outputs_by_scope,
            "canonical_row_order_scope": CANONICAL_ROW_ORDER_SCOPE,
        }
        if decoy_enabled and resolved_decoy_null_type == TARGET_DECOY_NULL_TYPE:
            global_summary.update(target_decoy_summary_metadata(out_dir=out_dir, outputs=outputs))

    triplet_true_path = Path(out_dir) / "triplet_metrics_true.parquet"
    diag_true_path = Path(out_dir) / "cellpair_filter_diagnostics_true.parquet"

    write_parquet_zstd(triplet_true, triplet_true_path)
    write_parquet_zstd(diag_true, diag_true_path)

    iteration_timings_df = pd.DataFrame(
        iteration_timings, columns=["null_type", "iteration", "stage", "duration_sec", "n_rows"]
    )
    iteration_timings_path = Path(out_dir) / "evaluation_iteration_timings.tsv"
    iteration_timings_df.to_csv(iteration_timings_path, sep="\t", index=False)
    total_wall = iteration_timings_df.loc[iteration_timings_df["stage"].eq("total"), "duration_sec"] if not iteration_timings_df.empty else pd.Series(dtype=float)

    summary = {
        "n_cell_pairs_raw_total": int(diag_true["n_cell_pairs_raw"].sum()) if not diag_true.empty and "n_cell_pairs_raw" in diag_true else 0,
        "n_cell_pairs_evaluable_total": int(diag_true["n_cell_pairs_evaluable"].sum()) if not diag_true.empty and "n_cell_pairs_evaluable" in diag_true else 0,
        "n_cell_pairs_excluded_no_drug_path_total": int(diag_true["n_cell_pairs_excluded_no_drug_path"].sum()) if not diag_true.empty and "n_cell_pairs_excluded_no_drug_path" in diag_true else 0,
        "n_cell_pairs_excluded_no_pert_path_total": int(diag_true["n_cell_pairs_excluded_no_pert_path"].sum()) if not diag_true.empty and "n_cell_pairs_excluded_no_pert_path" in diag_true else 0,
        "n_cell_pairs_excluded_no_both_path_total": int(diag_true["n_cell_pairs_excluded_no_both_path"].sum()) if not diag_true.empty and "n_cell_pairs_excluded_no_both_path" in diag_true else 0,
        "n_triplets_true": int(len(triplet_true)),
        "n_cell_pairs_baseline": int(baseline_cellpair_count),
        "n_triplets_baseline": int(baseline_triplet_count),
        "n_shuffle": int((baseline_cfg or {}).get("n_shuffle", len(baseline_pairs))),
        "cell_pair_mode": cell_pair_mode,
        "true_drug_infer_path": str(true_drug_infer_path),
        "true_pert_infer_path": str(true_pert_infer_path),
        "triplet_metrics_true": str(triplet_true_path),
        "cellpair_filter_diagnostics_true": str(diag_true_path),
        "evaluation_iteration_timings": str(iteration_timings_path),
        "per_iteration_wall_sec_median": float(total_wall.median()) if not total_wall.empty else float("nan"),
        "per_iteration_wall_sec_max": float(total_wall.max()) if not total_wall.empty else float("nan"),
        "global_summary": global_summary,
        "cellpair_metrics_debug": cellpair_debug_writer.manifest(),
    }
    denom = summary["n_cell_pairs_raw_total"]
    summary["fraction_evaluable_cell_pairs"] = float(summary["n_cell_pairs_evaluable_total"] / denom) if denom else float("nan")
    if summary_extra:
        summary.update(dict(summary_extra))
    write_json(summary, Path(out_dir) / "summary.json")
    return summary
