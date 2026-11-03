"""Infer signature-reversal paths (correlated + anti-correlated constraints).

Gap handling (matches the current implementation):

1) Correlation side (KG polarity vs. perturbation profile)
   - Treat genes with pert_value == 0 as "gaps".
   - Drop a path if its terminal gene has pert_value == 0
     (the final correlation must be supported by RNA).

2) Anti-correlation side (perturbation profile vs. disease profile)
   - Treat genes with pert_value == 0 or disease_value == 0 as "gaps".
   - Drop a path if its terminal gene has pert_value == 0 or disease_value == 0
     (the final anti-correlation must be supported by RNA).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


META_COLUMNS_TO_KEEP = [
    "row_id",
    "triplet_id",
    "drug_id",
    "target_id",
    "direction",
    "input_side",
    "cell_id",
    "original_triplet_id",
    "original_target_id",
    "decoy_anchor_id",
]


def _to_jsonable(x: Any) -> Any:
    """Recursively convert nested structures for json.dumps (e.g. numpy scalars)."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        if np.isnan(f):
            return None
        return f
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, str):
        return x
    return str(x)


def _path_column_to_json_str(val: Any) -> str:
    """Serialize one cell of signature_reversal_paths / _polarities for Parquet storage."""
    if val is None:
        return "[]"
    if isinstance(val, float) and pd.isna(val):
        return "[]"
    if isinstance(val, str):
        s = val.strip()
        return s if s else "[]"
    return json.dumps(_to_jsonable(val), ensure_ascii=False, separators=(",", ":"))


def serialize_signature_path_columns_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Store nested path columns as JSON strings for reliable Parquet round-trips."""
    out = df.copy()
    for col in ("signature_reversal_paths", "signature_reversal_polarities"):
        if col not in out.columns:
            continue
        out[col] = out[col].map(_path_column_to_json_str)
    return out


def _infer_polarity_along_path(
    path: List[Tuple[str, str, int]],
    graph: nx.MultiDiGraph,
    initial_polarity: int,
) -> Dict[str, int]:
    current_polarity = initial_polarity
    node_to_polarity: Dict[str, int] = {}

    g = graph
    for u, v, key in path:
        current_polarity *= g[u][v][key]["polarity"]
        node_to_polarity[v] = current_polarity
    return node_to_polarity



def _collect_path_polarities_from_source(
    source: str,
    graph: nx.MultiDiGraph,
    initial_polarity: int,
    cutoff: int,
    path_strategy: str = "simple",
) -> List[Dict[str, int]]:
    if path_strategy not in {"simple", "shortest"}:
        raise ValueError("path_strategy must be 'simple' or 'shortest'")
    lengths = nx.single_source_shortest_path_length(graph, source, cutoff)
    reachable: Set[str] = set(lengths)
    reachable.discard(source)

    if not reachable:
        return []

    path_polarities: List[Dict[str, int]] = []
    append = path_polarities.append
    infer = _infer_polarity_along_path

    if path_strategy == "simple":
        for path in nx.all_simple_edge_paths(graph, source=source, target=reachable, cutoff=cutoff):
            append(infer(path, graph, initial_polarity))
    else:
        for target in reachable:
            shortest_len = int(lengths[target])
            for path in nx.all_simple_edge_paths(graph, source=source, target=target, cutoff=shortest_len):
                if len(path) == shortest_len:
                    append(infer(path, graph, initial_polarity))
    return path_polarities


def _cache_file_for_source(cache_dir: Path, *, mode: str, cutoff: int, source: str, path_strategy: str = "simple") -> Path:
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()
    if path_strategy != "simple":
        return cache_dir / f"mode={mode}" / f"path_strategy={path_strategy}" / f"cutoff={int(cutoff)}" / f"{digest}.pkl"
    return cache_dir / f"mode={mode}" / f"cutoff={int(cutoff)}" / f"{digest}.pkl"


def _collect_path_polarities_cached(
    *,
    source: str,
    graph: nx.MultiDiGraph,
    initial_polarity: int,
    cutoff: int,
    mode: str,
    cache_dir: Path | None,
    path_strategy: str = "simple",
    memory_cache: MutableMapping[tuple[str, str, int, str], List[Dict[str, int]]] | None = None,
) -> List[Dict[str, int]]:
    cache_key = (str(mode), str(path_strategy), int(cutoff), str(source))
    if memory_cache is not None and cache_key in memory_cache:
        return memory_cache[cache_key]

    if cache_dir is None:
        path_polarities = _collect_path_polarities_from_source(
            source=source,
            graph=graph,
            initial_polarity=initial_polarity,
            cutoff=cutoff,
            path_strategy=path_strategy,
        )
        if memory_cache is not None:
            memory_cache[cache_key] = path_polarities
        return path_polarities

    cache_path = _cache_file_for_source(cache_dir, mode=mode, cutoff=cutoff, source=source, path_strategy=path_strategy)
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                payload = pickle.load(f)
            if (
                payload.get("source") == source
                and payload.get("mode") == mode
                and int(payload.get("cutoff")) == int(cutoff)
                and str(payload.get("path_strategy", "simple")) == str(path_strategy)
            ):
                path_polarities = payload["path_polarities"]
                if memory_cache is not None:
                    memory_cache[cache_key] = path_polarities
                return path_polarities
        except Exception:
            pass

    path_polarities = _collect_path_polarities_from_source(
        source=source,
        graph=graph,
        initial_polarity=initial_polarity,
        cutoff=cutoff,
        path_strategy=path_strategy,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
    try:
        with tmp_path.open("wb") as f:
            pickle.dump(
                {
                    "source": source,
                    "mode": mode,
                    "cutoff": int(cutoff),
                    "path_strategy": str(path_strategy),
                    "path_polarities": path_polarities,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        tmp_path.replace(cache_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    if memory_cache is not None:
        memory_cache[cache_key] = path_polarities
    return path_polarities


def load_permutation_specs(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        if p.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(f)
        else:
            payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Permutation specs must be a mapping: {p}")
    specs = payload.get("permutation_specs", payload)
    if not isinstance(specs, dict) or "gene_order" not in specs or "per_triplet" not in specs:
        raise ValueError(f"Invalid permutation specs: {p}")
    return specs


def _build_permutation_source_positions(
    *,
    genes: list[str],
    permutation_specs: dict[str, Any] | None,
) -> dict[str, np.ndarray]:
    if permutation_specs is None:
        return {}
    gene_order = [str(g) for g in permutation_specs["gene_order"]]
    gene_to_pos = {str(g): i for i, g in enumerate(genes)}
    missing = [g for g in gene_order if g not in gene_to_pos]
    if missing:
        raise ValueError(f"Permutation specs reference genes absent from infer input: {missing[:5]}")

    base = np.arange(len(genes), dtype=np.intp)
    target_pos = np.asarray([gene_to_pos[g] for g in gene_order], dtype=np.intp)
    per_triplet = permutation_specs.get("per_triplet") or {}
    out: dict[str, np.ndarray] = {}
    for group, perm_list in per_triplet.items():
        perm = np.asarray(perm_list, dtype=np.intp)
        if perm.shape[0] != len(gene_order):
            raise ValueError(f"Permutation length mismatch for group {group!r}: {perm.shape[0]} != {len(gene_order)}")
        source_pos = base.copy()
        source_pos[target_pos] = target_pos[perm]
        out[str(group)] = source_pos
    return out



def prune_prefix_paths_with_idx(items: List[tuple[List[str], int]]) -> List[tuple[List[str], int]]:
    keyed = [(p, i, tuple(p)) for p, i in items]
    keyed.sort(key=lambda x: len(x[2]), reverse=True)

    maximal: List[tuple[List[str], int]] = []
    prefix_set: Set[tuple[str, ...]] = set()

    for p, i, ks in keyed:
        if ks in prefix_set:
            continue
        maximal.append((p, i))
        for L in range(1, len(ks) + 1):
            prefix_set.add(ks[:L])
    return maximal



def _evaluate_single_row_arrays(
    disease: str,
    pert_gene: str,
    pert_cell: str,
    label: int,
    disease_values,
    pert_values,
    disease_gene_to_pos: Dict[str, int],
    pert_gene_to_pos: Dict[str, int],
    path_polarities: List[Dict[str, int]],
    num_nodes_in_paths: int,
    gap: int,
    meta_dict: Dict[str, object] | None = None,
    disease_source_pos: np.ndarray | None = None,
    pert_source_pos: np.ndarray | None = None,
) -> Dict[str, object]:
    signature_reversal = []

    gap_local = gap
    dpos = disease_gene_to_pos
    ppos = pert_gene_to_pos
    dsrc = disease_source_pos
    psrc = pert_source_pos

    for i, pol_dict in enumerate(path_polarities):
        is_path_correlated = True
        is_path_signature_reversal = True

        zero_corr = 0
        corr_ok_last = False
        zero_anti = 0
        anti_ok_last = False

        last_idx = len(pol_dict) - 1

        for idx, (gene, pol) in enumerate(pol_dict.items()):
            pert_idx = ppos[gene]
            disease_idx = dpos[gene]
            if psrc is not None:
                pert_idx = int(psrc[pert_idx])
            if dsrc is not None:
                disease_idx = int(dsrc[disease_idx])
            pert_value = pert_values[pert_idx]
            disease_value = disease_values[disease_idx]
            is_last = idx == last_idx

            if pert_value == 0:
                if is_last:
                    is_path_correlated = False
                    break
                zero_corr += 1
            else:
                if pert_value != pol:
                    is_path_correlated = False
                    break
                if is_last:
                    corr_ok_last = True

            if disease_value == 0 or pert_value == 0:
                if is_last:
                    is_path_signature_reversal = False
                else:
                    zero_anti += 1
            else:
                if disease_value * pert_value != -1:
                    is_path_signature_reversal = False
                if is_last:
                    anti_ok_last = True

        if not is_path_correlated:
            continue
        if zero_corr > gap_local or not corr_ok_last:
            continue
        if not is_path_signature_reversal:
            continue
        if zero_anti > gap_local or not anti_ok_last:
            continue

        signature_reversal.append((list(pol_dict.keys()), i))

    signature_reversal = prune_prefix_paths_with_idx(signature_reversal)
    signature_reversal_paths = [p for p, _ in signature_reversal]
    signature_reversal_paths_polarities = [list(path_polarities[idx].values()) for _, idx in signature_reversal]

    out = {
        "pert_id": pert_gene,
        "pert_cell": pert_cell,
        "disease": disease,
        "label": label,
        "num_nodes_in_paths": num_nodes_in_paths,
        "signature_reversal_paths": signature_reversal_paths,
        "signature_reversal_polarities": signature_reversal_paths_polarities,
    }
    if meta_dict:
        out.update(meta_dict)
    return out



def signature_reversal_paths_from_graph(
    df: pd.DataFrame,
    graph: nx.MultiDiGraph,
    mode: str,
    cutoff: int,
    gap: int,
    permutation_specs: dict[str, Any] | None = None,
    path_cache_dir: str | Path | None = None,
    path_strategy: str = "simple",
    path_pool_cache: MutableMapping[tuple[str, str, int, str], List[Dict[str, int]]] | None = None,
) -> pd.DataFrame:
    mode_configs = {
        "knockdown": {"polarity": -1},
        "overexpression": {"polarity": 1},
    }
    if mode not in mode_configs:
        raise ValueError("Error: 'mode' must be 'knockdown' or 'overexpression'")

    initial_polarity = mode_configs[mode]["polarity"]

    disease_cols = [col for col in df.columns if col.endswith("_disease")]
    disease_genes = [col.replace("_disease", "") for col in disease_cols]

    pert_cols = [col for col in df.columns if col.endswith("_gene")]
    pert_genes = [col.replace("_gene", "") for col in pert_cols]

    kg_nodes = set(graph.nodes)
    valid_genes: Set[str] = set(disease_genes) & set(pert_genes) & kg_nodes
    print(f"[info]Number of valid genes for analysis: {len(valid_genes)}")

    disease_gene_to_pos = {g: i for i, g in enumerate(disease_genes)}
    pert_gene_to_pos = {g: i for i, g in enumerate(pert_genes)}
    disease_perm_positions = _build_permutation_source_positions(
        genes=disease_genes,
        permutation_specs=permutation_specs,
    )
    pert_perm_positions = _build_permutation_source_positions(
        genes=pert_genes,
        permutation_specs=permutation_specs,
    )
    global_disease_pos = disease_perm_positions.get("__global__")
    global_pert_pos = pert_perm_positions.get("__global__")
    cache_dir = Path(path_cache_dir) if path_cache_dir is not None else None

    results_list: List[Dict[str, object]] = []
    missing_pert_ids = set(df["pert_id"].unique()) - kg_nodes
    if missing_pert_ids:
        print(f"[warn] {len(missing_pert_ids)} pert_id(s) not in the knowledge graph — they will be skipped.")
        df = df[~df["pert_id"].isin(missing_pert_ids)].copy()

    n_pert = df["pert_id"].nunique()
    group_iter = df.groupby("pert_id", sort=False)

    gc_every = 50
    group_count = 0

    for pert_id, sub_df in tqdm(group_iter, desc="pert_id groups", total=n_pert):
        if pert_id in graph.nodes:
            path_polarities = _collect_path_polarities_cached(
                source=pert_id,
                graph=graph,
                initial_polarity=initial_polarity,
                cutoff=cutoff,
                mode=mode,
                cache_dir=cache_dir,
                path_strategy=path_strategy,
                memory_cache=path_pool_cache,
            )
        else:
            path_polarities = []

        if path_polarities:
            vp = valid_genes
            path_polarities = [
                pdict for pdict in path_polarities if len(pdict) <= cutoff and all((g in vp) for g in pdict.keys())
            ]

        nodes_all: Set[str] = set()
        for pdict in path_polarities:
            nodes_all.update(pdict.keys())
        num_nodes_in_paths = len(nodes_all)

        do_ids = sub_df["do_id"].to_numpy()
        cell_ids = sub_df["cell_id"].to_numpy()
        labels = sub_df["label"].to_numpy()
        disease_mat = sub_df[disease_cols].to_numpy()
        pert_mat = sub_df[pert_cols].to_numpy()
        triplet_ids = sub_df["triplet_id"].astype(str).to_numpy() if permutation_specs is not None and "triplet_id" in sub_df.columns else None

        meta_arrays = {c: sub_df[c].to_numpy() for c in META_COLUMNS_TO_KEEP if c in sub_df.columns}

        for r in range(len(sub_df)):
            meta_dict = {c: meta_arrays[c][r] for c in meta_arrays}
            disease_source_pos = global_disease_pos
            pert_source_pos = global_pert_pos
            if triplet_ids is not None and global_disease_pos is None and global_pert_pos is None:
                group = str(triplet_ids[r])
                disease_source_pos = disease_perm_positions.get(group)
                pert_source_pos = pert_perm_positions.get(group)
            result = _evaluate_single_row_arrays(
                disease=do_ids[r],
                pert_gene=pert_id,
                pert_cell=cell_ids[r],
                label=labels[r],
                disease_values=disease_mat[r],
                pert_values=pert_mat[r],
                disease_gene_to_pos=disease_gene_to_pos,
                pert_gene_to_pos=pert_gene_to_pos,
                path_polarities=path_polarities,
                num_nodes_in_paths=num_nodes_in_paths,
                gap=gap,
                meta_dict=meta_dict,
                disease_source_pos=disease_source_pos,
                pert_source_pos=pert_source_pos,
            )
            results_list.append(result)

        del path_polarities, disease_mat, pert_mat, do_ids, cell_ids, labels, meta_arrays
        group_count += 1
        if (group_count % gc_every) == 0:
            gc.collect()

    return pd.DataFrame(results_list)



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Infer signature-reversal paths using a config file (.yml/.yaml/.json). "
            "All parameters are loaded from the config file."
        )
    )
    p.add_argument("--config", type=str, required=True, help="config file path (.yml/.yaml/.json)")
    return p.parse_args()



def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping/dict")
    return cfg



def req_any(cfg: dict, keys: List[str]):
    for k in keys:
        if k in cfg:
            return cfg[k]
    raise ValueError(f"Missing required key in config (any of {keys}): {keys[0]}")



def resolve_input_parquet(config: dict, config_path: str | Path, mode: str) -> Path:
    cfg_dir = Path(config_path).resolve().parent
    input_parquet = config.get("input_parquet") or config.get("INPUT_PARQUET")
    if input_parquet:
        p = Path(input_parquet)
        if not p.is_absolute():
            p = (cfg_dir / p).resolve()
        return p

    data_dir = Path(str(req_any(config, ["data_dir", "DATA_DIR"])))
    if not data_dir.is_absolute():
        data_dir = (cfg_dir / data_dir).resolve()
    kd_parquet = str(req_any(config, ["kd_parquet", "KD_PARQUET"]))
    oe_parquet = str(req_any(config, ["oe_parquet", "OE_PARQUET"]))
    return (data_dir / (oe_parquet if mode == "overexpression" else kd_parquet)).resolve()



def resolve_kg_path(config: dict, config_path: str | Path) -> Path:
    cfg_dir = Path(config_path).resolve().parent
    kg_tsv = req_any(config, ["kg_tsv", "KG_TSV"])
    p = Path(str(kg_tsv))
    if p.is_absolute():
        return p

    if "data_dir" in config or "DATA_DIR" in config:
        data_dir = Path(str(config.get("data_dir") or config.get("DATA_DIR")))
        if not data_dir.is_absolute():
            data_dir = (cfg_dir / data_dir).resolve()
        candidate = (data_dir / p).resolve()
        if candidate.exists():
            return candidate
    return (cfg_dir / p).resolve()



def resolve_output_dir(config: dict, config_path: str | Path) -> Path:
    cfg_dir = Path(config_path).resolve().parent
    out = Path(str(req_any(config, ["output_dir", "OUTPUT_DIR"])))
    if out.is_absolute():
        return out
    return (cfg_dir / out).resolve()


def resolve_optional_path(config: dict, config_path: str | Path, key: str) -> Path | None:
    value = config.get(key)
    if value is None:
        return None
    p = Path(str(value))
    if p.is_absolute():
        return p
    return (Path(config_path).resolve().parent / p).resolve()


def _iter_parquet_batches(path: str | Path):
    """Yield row groups as DataFrames so large inputs are not loaded at once."""
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        for i in range(pf.metadata.num_row_groups):
            yield pf.read_row_group(i).to_pandas()
    except Exception:
        yield pd.read_parquet(path)



def _write_infer_parts_streaming(
    *,
    parquet_path: str | Path,
    out_path: str | Path,
    graph: nx.MultiDiGraph,
    mode: str,
    cutoff: int,
    gap: int,
    permutation_specs: dict[str, Any] | None,
    path_cache_dir: str | Path | None,
    path_strategy: str,
) -> int:
    """Evaluate row-group batches and stream directly to a zstd parquet file.

    This preserves the existing infer columns and path acceptance logic, but avoids
    retaining all batch results in memory before writing.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    n_rows = 0
    wrote = False
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        for df in _iter_parquet_batches(parquet_path):
            if df.empty:
                continue
            part = signature_reversal_paths_from_graph(
                df=df,
                graph=graph,
                mode=mode,
                cutoff=cutoff,
                gap=gap,
                permutation_specs=permutation_specs,
                path_cache_dir=path_cache_dir,
                path_strategy=path_strategy,
            )
            part = serialize_signature_path_columns_for_parquet(part)
            table = pa.Table.from_pandas(part, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
            writer.write_table(table)
            n_rows += int(len(part))
            wrote = True
            del table, part, df
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
    if wrote:
        return n_rows

    # Empty input/output fallback. Keep a real parquet file so downstream skip checks work.
    pd.DataFrame().to_parquet(out_path, index=False, compression="zstd")
    return 0


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    mode = str(req_any(config, ["mode", "MODE"]))
    cutoff = int(req_any(config, ["cutoff", "CUTOFF"]))
    gap = int(req_any(config, ["gap", "GAP"]))

    parquet_path = resolve_input_parquet(config, args.config, mode)
    kg_path = resolve_kg_path(config, args.config)
    output_dir = resolve_output_dir(config, args.config)
    permutation_specs_path = resolve_optional_path(config, args.config, "permutation_specs")
    path_cache_dir = resolve_optional_path(config, args.config, "path_cache_dir")
    path_strategy = str(config.get("path_strategy", "simple"))
    if path_strategy not in {"simple", "shortest"}:
        raise ValueError("Config key path_strategy must be 'simple' or 'shortest'.")
    permutation_specs = load_permutation_specs(permutation_specs_path)

    kg = pd.read_csv(kg_path, sep="\t")
    if "polarity" not in kg.columns:
        raise ValueError(f"KG file must contain 'polarity': {kg_path}")
    kg["polarity"] = kg["polarity"].astype(int)
    graph = nx.from_pandas_edgelist(
        kg,
        source="source",
        target="target",
        edge_attr=["polarity"],
        create_using=nx.MultiDiGraph(),
    )

    print(f"Analyzing mode={mode}, cutoff={cutoff}, gap={gap}")
    print(f"[info]path_strategy={path_strategy}")
    print(f"[info]input_parquet={parquet_path}")
    print(f"[info]kg_tsv={kg_path}")
    if permutation_specs_path is not None:
        print(f"[info]permutation_specs={permutation_specs_path}")
    if path_cache_dir is not None:
        print(f"[info]path_cache_dir={path_cache_dir}")

    out_subdir = output_dir / f"{mode}" / f"cutoff={cutoff}"
    os.makedirs(out_subdir, exist_ok=True)
    out_path = out_subdir / f"signature_reversal_paths_gap_{gap}.parquet"
    n_rows = _write_infer_parts_streaming(
        parquet_path=parquet_path,
        out_path=out_path,
        graph=graph,
        mode=mode,
        cutoff=cutoff,
        gap=gap,
        permutation_specs=permutation_specs,
        path_cache_dir=path_cache_dir,
        path_strategy=path_strategy,
    )
    print(f"[info]saved={out_path}")
    print(f"[info]n_rows={n_rows}")


if __name__ == "__main__":
    main()
