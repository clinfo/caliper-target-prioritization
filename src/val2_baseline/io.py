"""Input loading for the val2 external-baseline comparison.

Reuses ``src/val2/io.load_data_bundle`` for the KG, the discretized profiles,
disease signs and labels (methodology §Step1). Adds two baseline-specific
inputs: the continuous (pre-discretization) KD profile (SI-03) and the val1
``k_selection.json`` (SI-02), plus SHA-256 manifesting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import hashlib
import json

import numpy as np
import pandas as pd

from val2.io import DataBundle, load_data_bundle  # reuse existing loader
from val2.registry import DatasetSpec, load_dataset_registry, resolve_dataset_spec


def sha256_of_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def file_manifest_entry(path: str | Path) -> Dict[str, object]:
    p = Path(path)
    exists = p.exists()
    return {
        "path": str(p),
        "exists": bool(exists),
        "size_bytes": int(p.stat().st_size) if exists and p.is_file() else None,
        "sha256": sha256_of_file(p) if exists and p.is_file() else None,
    }


def resolve_spec(cfg) -> DatasetSpec:
    registry = load_dataset_registry(cfg.registry_path)
    return resolve_dataset_spec(cfg.data, registry, cfg.registry_path.parent)


def load_bundle(dataset_spec: DatasetSpec) -> DataBundle:
    return load_data_bundle(dataset_spec)


def load_k_star(k_selection_json: str | Path, manual_k_star: Optional[int] = None) -> Tuple[int, Dict[str, object]]:
    """Read k_star from val1 selection JSON (SI-02).

    Raises if the file is missing, k_star is absent, or a manually supplied
    k_star disagrees with the JSON value (methodology §11 #1, SI-02 rule).
    """
    path = Path(k_selection_json)
    if not path.exists():
        raise FileNotFoundError(f"[SI-02] k_selection.json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or payload.get("k_star") is None:
        raise ValueError(f"[SI-02] k_selection.json does not contain a 'k_star': {path}")
    k_star = int(payload["k_star"])
    if manual_k_star is not None and int(manual_k_star) != k_star:
        raise ValueError(
            f"[SI-02] Manually supplied k_star={manual_k_star} disagrees with "
            f"k_selection.json k_star={k_star}."
        )
    return k_star, payload


def resolve_caliper_ks(
    k_selection_json: str | Path,
    caliper_ks: List[int] | None = None,
    manual_k_star: Optional[int] = None,
) -> Tuple[List[int], int, Dict[str, object]]:
    """Resolve CALIPER hop radii for scoring.

    When ``caliper_ks`` is provided, those values are used and ``manual_k_star``
    is not required to match ``k_selection.json``. Otherwise a single ``k_star``
    is read from the val1 selection JSON (SI-02).
    """
    enforce_manual = manual_k_star if caliper_ks is None else None
    k_star, payload = load_k_star(k_selection_json, enforce_manual)
    if caliper_ks is None:
        return [int(k_star)], int(k_star), payload
    ks = sorted(dict.fromkeys(int(k) for k in caliper_ks))
    if not ks or any(k <= 0 for k in ks):
        raise ValueError(
            f"external_baseline.caliper.ks must be a non-empty list of positive integers. got {ks!r}"
        )
    return ks, int(k_star), payload


def load_creeds_disease_signatures(
    parquet_path: str | Path,
    *,
    id_col: str = "doid",
) -> Dict[str, Dict[str, int]]:
    """Load full CREEDS disease signatures as sign dicts (methodology §3.3).

    The full CREEDS table is wide: one ``id_col`` (DOID) column plus one column
    per gene (``NCBIGENE:<n>``) holding the continuous disease signature value.
    The val2 discretized disease sign equals ``sign(value)`` at threshold 0
    (verified: identical gene set and 100% sign agreement), so the up/down query
    sets and seed set S_d are taken as the sign of the continuous value. This
    differs from the val2 bundle signatures only by NOT restricting the gene
    universe to the perturbation/KG intersection.
    """
    import numpy as _np

    parquet_path = str(parquet_path)

    def _read(columns=None):
        try:
            return pd.read_parquet(parquet_path, columns=columns, engine="pyarrow")
        except Exception:
            return pd.read_parquet(parquet_path, columns=columns, engine="fastparquet")

    df = _read()
    if id_col not in df.columns:
        raise ValueError(
            f"[full_creeds] disease signature parquet missing id column '{id_col}'. "
            f"existing={list(df.columns[:8])}"
        )
    gene_cols = [c for c in df.columns if c != id_col]
    if len(gene_cols) == 0:
        raise ValueError("[full_creeds] disease signature parquet has no gene columns.")

    sigs: Dict[str, Dict[str, int]] = {}
    gene_arr = list(map(str, gene_cols))
    for _, row in df.iterrows():
        did = str(row[id_col])
        vals = row[gene_cols].to_numpy(dtype=float)
        signs = _np.sign(vals).astype(int)
        nz = _np.nonzero(signs)[0]
        sigs[did] = {gene_arr[j]: int(signs[j]) for j in nz}
    return sigs


@dataclass
class ContinuousKD:
    """Continuous (target-cell averaged) KD reference profiles.

    Used as the NCS reference population (methodology §6.2). Per user direction
    the profiles are the target-cell averaged ``BING_knockdown_signature`` rows
    (one row per (ncbigene_id, cell_id)), not per-sig_id Level 5 signatures;
    see ``wtcs.py`` for the profile-unit deviation note.

    feature_genes : ordered BING feature space F (list of NCBI gene IDs)
    feature_index : gene -> column index in ``matrix``
    keys          : list of (ncbigene_id, cell_id) in row order of ``matrix``
    key_to_row    : (gene, cell) -> row index
    matrix        : float32 array [n_rows x n_features], r_{t,c}(g)
    """

    feature_genes: List[str]
    feature_index: Dict[str, int]
    keys: List[Tuple[str, str]]
    key_to_row: Dict[Tuple[str, str], int]
    matrix: np.ndarray

    def cells_for_target(self, target_gene: str) -> Set[str]:
        return {cell for (g, cell) in self.keys if g == target_gene}

    def profile(self, target_gene: str, cell_id: str) -> Optional[np.ndarray]:
        row = self.key_to_row.get((target_gene, cell_id))
        if row is None:
            return None
        return self.matrix[row]

    def rows_by_cell(self) -> Dict[str, List[int]]:
        """cell_id -> row indices of the full reference population R_{c} for
        that cell line (all target-cell profiles, candidate-independent)."""
        out: Dict[str, List[int]] = {}
        for row, (_g, cell) in enumerate(self.keys):
            out.setdefault(cell, []).append(row)
        return out


def load_continuous_kd(
    parquet_path: str | Path,
    *,
    id_col: str,
    cell_col: str,
    needed_genes: Optional[Set[str]] = None,
) -> ContinuousKD:
    """Load the continuous (target-cell averaged) KD profile parquet.

    Rows are keyed by (id_col, cell_col); all remaining columns form the BING
    feature space F. For the NCS baseline the full population must be loaded
    (``needed_genes=None``) so the per-cell normalization is fixed independently
    of the val2 candidate set (methodology §6.2.2, validation #13);
    ``needed_genes`` is retained only as an optional memory bound for other
    callers. The discretized profile must never be substituted here
    (validation #4); the caller checks discreteness.
    """
    import pyarrow.parquet as pq

    parquet_path = str(parquet_path)
    schema_names = pq.read_schema(parquet_path).names
    for col in (id_col, cell_col):
        if col not in schema_names:
            raise ValueError(f"[SI-03] continuous KD profile missing key column '{col}'. existing={schema_names[:10]}")

    feature_genes = [c for c in schema_names if c not in (id_col, cell_col)]
    if len(feature_genes) == 0:
        raise ValueError("[SI-03] continuous KD profile has no feature (gene) columns.")

    def _read(columns):
        try:
            return pd.read_parquet(parquet_path, columns=columns, engine="pyarrow")
        except Exception:
            return pd.read_parquet(parquet_path, columns=columns, engine="fastparquet")

    meta = _read([id_col, cell_col])
    meta[id_col] = meta[id_col].astype(str)
    meta[cell_col] = meta[cell_col].astype(str)
    if needed_genes is not None:
        keep_mask = meta[id_col].isin({str(g) for g in needed_genes}).to_numpy()
    else:
        keep_mask = np.ones(len(meta), dtype=bool)
    if not keep_mask.any():
        raise ValueError("[SI-03] No continuous KD rows remain after filtering by needed candidate targets.")

    keys = [
        (str(g), str(c))
        for g, c in zip(meta[id_col].to_numpy()[keep_mask], meta[cell_col].to_numpy()[keep_mask])
    ]

    # Read feature columns in chunks to limit peak memory, applying the row mask.
    col_chunk = 2000
    blocks: List[np.ndarray] = []
    for i in range(0, len(feature_genes), col_chunk):
        batch = feature_genes[i : i + col_chunk]
        chunk = _read(batch)
        arr = chunk.to_numpy(dtype=np.float32)[keep_mask]
        blocks.append(arr)
        del chunk
    matrix = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]

    feature_index = {g: j for j, g in enumerate(feature_genes)}
    key_to_row = {key: idx for idx, key in enumerate(keys)}
    return ContinuousKD(
        feature_genes=feature_genes,
        feature_index=feature_index,
        keys=keys,
        key_to_row=key_to_row,
        matrix=matrix,
    )
