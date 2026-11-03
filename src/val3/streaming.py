from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

ALLOWED_FINAL_NULL_TYPES = {
    "target_khop_restricted_paired_gene_label_permutation",
    "path_count_length_matched_structural_resampling",
    "topology_profile_matched_target_decoy_control",
    "observed_target_pool_decoy_control",
    "evaluable_decoy_anchor_resampling",
}
OBSERVED_NULL_TYPE = "observed"

# Both decoy controls carry the figure label ``Target-Decoy``. The two legacy names
# below are one and the same control, so they may coexist; the canonical
# topology- and profile-matched control must never share a report-facing table with
# them, because axis ticks, legends and markdown rows would then show two
# indistinguishable "Target-Decoy" entries.
LEGACY_TARGET_POOL_DECOY_NULL_TYPES = frozenset({
    "observed_target_pool_decoy_control",
    "evaluable_decoy_anchor_resampling",
})
TOPOLOGY_PROFILE_MATCHED_DECOY_NULL_TYPE = "topology_profile_matched_target_decoy_control"


def assert_distinguishable_null_types(df: pd.DataFrame, *, null_col: str = "null_type") -> None:
    """Reject report-facing tables that mix the two decoy controls.

    Enabling both in one config is already a config error, but a report or post-hoc
    step can be pointed at any directory, so the display path re-checks it instead of
    silently rendering two identically labelled Target-Decoy entries.
    """
    if df.empty or null_col not in df.columns:
        return
    present = set(df[null_col].dropna().astype(str))
    legacy = sorted(present & LEGACY_TARGET_POOL_DECOY_NULL_TYPES)
    if legacy and TOPOLOGY_PROFILE_MATCHED_DECOY_NULL_TYPE in present:
        raise ValueError(
            "Report-facing table mixes two decoy controls that share the figure label "
            f"'Target-Decoy': {legacy} and '{TOPOLOGY_PROFILE_MATCHED_DECOY_NULL_TYPE}'. "
            "Report each run separately."
        )


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_parquet_zstd(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> str:
    """Write a dataframe with deterministic zstd parquet compression.

    Falling back to the engine default compression is intentionally not silent: if zstd
    is unavailable in the local parquet engine, the caller should notice during test runs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=index, compression="zstd")
    return str(path)


def read_parquet_any_engine(path: str | Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a parquet file, preferring fastparquet when pyarrow cannot read it.

    Some Val3 outputs are written by a parquet writer that the locally installed
    pyarrow refuses to read back; fastparquet reads the same files.
    """
    try:
        return pd.read_parquet(path, engine="fastparquet", columns=columns)
    except Exception:
        return pd.read_parquet(path, engine="pyarrow", columns=columns)


def prune_non_v23_null_types(df: pd.DataFrame, *, null_col: str = "null_type") -> pd.DataFrame:
    """Keep only the three v2.3 null models in final/report-facing tables.

    ``global_paired_gene_label_permutation`` can still be present in legacy regression
    outputs but is excluded from v3 final tables by construction.
    """
    if df.empty or null_col not in df.columns:
        return df.copy()
    mask = df[null_col].astype(str).isin(ALLOWED_FINAL_NULL_TYPES)
    out = df.loc[mask].copy()
    assert_distinguishable_null_types(out, null_col=null_col)
    return out


def _first_available(row: Mapping[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in row:
            val = row.get(name)
            if val is None:
                continue
            if isinstance(val, float) and math.isnan(val):
                continue
            return val
    return default


def scalar_cellpair_debug_frame(
    cellpair_df: pd.DataFrame,
    *,
    source: str,
    null_type: str,
    evaluable_set_type: str,
    iteration: int | None = None,
    direction: str | None = None,
    anchor_id: str | None = None,
) -> pd.DataFrame:
    """Drop path/edge objects and keep only scalar raw cell-pair diagnostics.

    This is deliberately conservative: it never serializes full signed-edge sets or
    path JSON. It can be called for observed, gene-label null, and structural null
    pair-level tables without changing any metric calculation.
    """
    rows: list[dict[str, Any]] = []
    if cellpair_df is None or cellpair_df.empty:
        return pd.DataFrame(columns=_CELLPAIR_DEBUG_COLUMNS)
    for rec in cellpair_df.to_dict(orient="records"):
        both_paths = _first_available(rec, ["both_paths", "is_primary_evaluable_pair"], None)
        if both_paths is None:
            nd = _first_available(rec, ["n_drug_paths"], 0)
            npth = _first_available(rec, ["n_pert_paths"], 0)
            both_paths = int(nd or 0) > 0 and int(npth or 0) > 0
        rows.append({
            "source": source,
            "null_type": null_type,
            "iteration": iteration if iteration is not None else _first_available(rec, ["iteration"], None),
            "evaluable_set_type": evaluable_set_type,
            "triplet_id": _first_available(rec, ["triplet_id"]),
            "drug_id": _first_available(rec, ["drug_id", "drug_id_drug", "drug_id_pert"]),
            "target_id": _first_available(rec, ["target_id", "target_id_drug", "target_id_pert"], anchor_id),
            "original_target_id": _first_available(rec, ["original_target_id", "original_target_id_drug", "original_target_id_pert", "target_id"], anchor_id),
            "anchor_id": _first_available(rec, ["decoy_anchor_id", "anchor_id"], anchor_id),
            "disease_id": _first_available(rec, ["disease_id", "disease", "disease_drug", "disease_pert"]),
            "direction": _first_available(rec, ["direction", "direction_drug", "direction_pert"], direction),
            "cell_id_drug": _first_available(rec, ["cell_id_drug", "cell_id_x", "pert_cell_drug"]),
            "cell_id_pert": _first_available(rec, ["cell_id_pert", "cell_id_y", "pert_cell_pert"]),
            "n_drug_paths": int(_first_available(rec, ["n_drug_paths"], 0) or 0),
            "n_pert_paths": int(_first_available(rec, ["n_pert_paths"], 0) or 0),
            "n_drug_edges": int(_first_available(rec, ["n_drug_edges"], 0) or 0),
            "n_pert_edges": int(_first_available(rec, ["n_pert_edges"], 0) or 0),
            "n_intersection": int(_first_available(rec, ["n_intersection"], 0) or 0),
            "n_union": int(_first_available(rec, ["n_union"], 0) or 0),
            "jaccard": _first_available(rec, ["jaccard", "joint_jaccard"], None),
            "is_primary_evaluable_pair": bool(both_paths),
        })
    out = pd.DataFrame(rows)
    for col in _CELLPAIR_DEBUG_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[_CELLPAIR_DEBUG_COLUMNS]


_CELLPAIR_DEBUG_COLUMNS = [
    "source", "null_type", "iteration", "evaluable_set_type", "triplet_id", "drug_id",
    "target_id", "original_target_id", "anchor_id", "disease_id", "direction",
    "cell_id_drug", "cell_id_pert", "n_drug_paths", "n_pert_paths", "n_drug_edges",
    "n_pert_edges", "n_intersection", "n_union", "jaccard",
    "is_primary_evaluable_pair",
]


class CellpairDebugWriter:
    """Small append-only parquet writer for scalar cell-pair debug shards."""

    def __init__(self, root: str | Path):
        self.root = _ensure_dir(root)
        self._counts: dict[tuple[str, str, str, str], int] = {}
        self.paths: list[str] = []

    def write(
        self,
        df: pd.DataFrame,
        *,
        source: str,
        null_type: str,
        evaluable_set_type: str,
        iteration: int | None = None,
        direction: str | None = None,
    ) -> str | None:
        frame = scalar_cellpair_debug_frame(
            df,
            source=source,
            null_type=null_type,
            evaluable_set_type=evaluable_set_type,
            iteration=iteration,
            direction=direction,
        )
        if frame.empty:
            return None
        iter_label = "NA" if iteration is None else str(int(iteration))
        key = (source, null_type, evaluable_set_type, iter_label)
        part_idx = self._counts.get(key, 0)
        self._counts[key] = part_idx + 1
        part_dir = self.root / f"source={source}" / f"null_type={null_type}" / f"evaluable_set_type={evaluable_set_type}" / f"iteration={iter_label}"
        part_dir.mkdir(parents=True, exist_ok=True)
        path = part_dir / f"part-{part_idx:05d}.parquet"
        write_parquet_zstd(frame, path)
        self.paths.append(str(path))
        return str(path)

    def manifest(self) -> dict[str, Any]:
        return {"root": str(self.root), "n_parts": len(self.paths), "parts": list(self.paths)}
