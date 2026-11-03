"""Equivalence tests for the Val3 performance work (spec §7, T0-T9).

Every optimization in Part C is required to leave the analysis output bit-identical,
so each test here pins an optimized routine against the straightforward
implementation it replaced rather than against a stored expected value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from inference import signature_reversal_paths as srp  # noqa: E402
from inference.prefix_trie_traversal import (  # noqa: E402
    accepted_paths_in_reference_order,
    build_path_trie,
    evaluate_trie_row,
    expand_trie_paths,
)
from val3 import metrics as m  # noqa: E402
from val3 import null1_engine as engine  # noqa: E402
from val3 import null1_parallel as parallel  # noqa: E402
from val3.streaming import CellpairDebugWriter  # noqa: E402

GENE_NODE_NULL = "target_khop_restricted_paired_gene_label_permutation"
PATH_SHUFFLE_NULL = "path_count_length_matched_structural_resampling"
TARGET_DECOY_NULL = "topology_profile_matched_target_decoy_control"


# ---------------------------------------------------------------------------
# shared toy inputs
# ---------------------------------------------------------------------------

GENES = ["g1", "g2", "g3", "g4", "g5", "g6"]
KG_EDGES = [
    ("g1", "g2", 1),
    ("g1", "g3", -1),
    ("g2", "g3", 1),
    ("g2", "g4", -1),
    ("g3", "g4", 1),
    ("g3", "g5", -1),
    ("g4", "g5", 1),
    ("g5", "g6", -1),
    ("g1", "g4", 1),
]


def _graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(GENES)
    for source, target, polarity in KG_EDGES:
        graph.add_edge(source, target, polarity=int(polarity))
    return graph


def _profile_df(*, n_rows: int = 12, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sources = ["g1", "g2", "g3", "g1", "g2", "g3"]
    rows = []
    for i in range(n_rows):
        pert_id = sources[i % len(sources)]
        row = {
            "pert_id": pert_id,
            "cell_id": f"C{i % 3}",
            "do_id": f"DOID:{i % 2}",
            "label": int(i % 2),
            "triplet_id": f"T{i % 4}",
            "drug_id": f"DRUG{i % 4}",
            "target_id": pert_id,
            "direction": "inhibitory",
        }
        for gene in GENES:
            row[f"{gene}_gene"] = int(rng.integers(-1, 2))
            row[f"{gene}_disease"] = int(rng.integers(-1, 2))
        rows.append(row)
    return pd.DataFrame(rows)


def _permutation_specs(*, per_triplet: bool, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    gene_order = list(GENES)
    if per_triplet:
        groups = [f"T{i}" for i in range(4)]
    else:
        groups = [engine.GLOBAL_PERMUTATION_GROUP]
    return {
        "gene_order": gene_order,
        "per_triplet": {g: rng.permutation(len(gene_order)).tolist() for g in groups},
    }


# ---------------------------------------------------------------------------
# T0: canonical row order
# ---------------------------------------------------------------------------

def _true_joint_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"evaluable_set_type": "path_existing_pairs", "triplet_id": tid, "joint_jaccard": v}
        for tid, v in [("T3", 0.3), ("T1", 0.1), ("T2", 0.2), ("T10", 0.4)]
    ])


def _baseline_joint_frame() -> pd.DataFrame:
    rows = []
    for null_type in [GENE_NODE_NULL, PATH_SHUFFLE_NULL, TARGET_DECOY_NULL]:
        for iteration in [2, 1]:
            for tid in ["T2", "T1"]:
                rows.append({
                    "evaluable_set_type": "path_existing_pairs",
                    "null_type": null_type,
                    "iteration": iteration,
                    "triplet_id": tid,
                    "joint_jaccard": 0.1,
                })
    return pd.DataFrame(rows)


def test_canonical_sort_is_idempotent_and_order_only() -> None:
    true_df = _true_joint_frame()
    once = m.canonical_sort_true_joint_triplet(true_df)
    twice = m.canonical_sort_true_joint_triplet(once)
    pd.testing.assert_frame_equal(once, twice)
    assert sorted(once["triplet_id"]) == sorted(true_df["triplet_id"])
    assert list(once["triplet_id"]) == ["T1", "T10", "T2", "T3"]

    base = _baseline_joint_frame()
    b_once = m.canonical_sort_baseline_joint_triplet(base)
    pd.testing.assert_frame_equal(b_once, m.canonical_sort_baseline_joint_triplet(b_once))
    assert len(b_once) == len(base)


def test_canonical_sort_keeps_null_type_blocks_in_generation_order() -> None:
    base = _baseline_joint_frame()
    out = m.canonical_sort_baseline_joint_triplet(base)
    assert list(dict.fromkeys(out["null_type"])) == list(dict.fromkeys(base["null_type"]))
    for null_type, block in out.groupby("null_type", sort=False):
        assert list(zip(block["iteration"], block["triplet_id"])) == sorted(
            zip(block["iteration"], block["triplet_id"])
        ), null_type


def test_canonical_sort_is_invariant_to_input_row_order() -> None:
    base = _baseline_joint_frame()
    shuffled_within_blocks = pd.concat(
        [
            block.sample(frac=1.0, random_state=3)
            for _, block in base.groupby("null_type", sort=False)
        ],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(
        m.canonical_sort_baseline_joint_triplet(base),
        m.canonical_sort_baseline_joint_triplet(shuffled_within_blocks),
    )


# ---------------------------------------------------------------------------
# T1: streaming baseline accumulation
# ---------------------------------------------------------------------------

def test_streaming_iteration_parts_equal_materialized_baseline() -> None:
    base = _baseline_joint_frame()
    materialized = m.canonical_sort_baseline_joint_triplet(base)
    # The streaming loop appends one part per (null_type, iteration) instead of
    # holding every iteration frame until the end.
    parts = [
        block
        for _, block in base.groupby(["null_type", "iteration"], sort=False)
    ]
    streamed = m.canonical_sort_baseline_joint_triplet(pd.concat(parts, ignore_index=True))
    pd.testing.assert_frame_equal(materialized, streamed)


# ---------------------------------------------------------------------------
# T2: baseline legacy pair counters
# ---------------------------------------------------------------------------

def _null1_side_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    def _rows(prefix: str, n_paths_by_cell: dict[str, int]) -> pd.DataFrame:
        rows = []
        for cell, n_paths in n_paths_by_cell.items():
            rows.append({
                "triplet_id": "T1" if prefix == "a" else "T2",
                "drug_id": f"DRUG_{prefix}",
                "target_id": f"TGT_{prefix}",
                "direction": "inhibitory",
                "cell_id": cell,
                "n_paths": n_paths,
                "num_nodes_in_paths": 2,
                "signed_edges": {("g1", "g2", 1)} if n_paths else set(),
                "signed_edge_paths": [{("g1", "g2", 1)}] if n_paths else [],
            })
        return pd.DataFrame(rows)

    drug = pd.concat([_rows("a", {"C1": 2, "C2": 0}), _rows("b", {"C1": 1})], ignore_index=True)
    pert = pd.concat([_rows("a", {"C1": 1, "C3": 3}), _rows("b", {"C1": 0})], ignore_index=True)
    return drug, pert


@pytest.mark.parametrize("filters", [
    {"require_both_nonempty_inferred_paths": True},
    {"require_both_nonempty_inferred_paths": True, "require_both_nonempty_signed_edges": True},
    {"require_both_nonempty_inferred_paths": False},
])
def test_legacy_pair_counters_match_the_removed_metric_computation(filters: dict) -> None:
    drug, pert = _null1_side_frames()
    n_pairs, n_triplets = m.baseline_legacy_pair_counters(
        drug, pert, cell_pair_mode="cartesian", filters=filters
    )
    pair_df, diag = m.compute_cellpair_metrics(
        drug, pert, cell_pair_mode="cartesian", filters=filters, iteration=1
    )
    triplet_df = m.aggregate_triplet_metrics(pair_df, diag, include_iteration=True)
    assert n_pairs == int(len(pair_df))
    assert n_triplets == int(len(triplet_df))


def test_legacy_pair_counters_handle_an_empty_cell_pair_table() -> None:
    drug, pert = _null1_side_frames()
    disjoint = pert.assign(triplet_id="ZZZ")
    assert m.baseline_legacy_pair_counters(
        drug, disjoint, cell_pair_mode="cartesian", filters={}
    ) == (0, 0)


# ---------------------------------------------------------------------------
# T3: unique edge-set pair memoization
# ---------------------------------------------------------------------------

def test_memoized_pair_metrics_equal_naive_per_row_set_algebra() -> None:
    drug, pert = _null1_side_frames()
    pair = m.build_cell_pairs(drug, pert, cell_pair_mode="cartesian")
    pair["both_paths"] = pd.to_numeric(pair["n_drug_paths"]).gt(0) & pd.to_numeric(pair["n_pert_paths"]).gt(0)
    metrics = ["jaccard"]
    out = m._joint_pair_values(pair, metrics)

    for i, rec in enumerate(pair.to_dict(orient="records")):
        ea, eb = set(rec["signed_edges_drug"]), set(rec["signed_edges_pert"])
        assert int(out["n_intersection"].iloc[i]) == len(ea & eb)
        assert int(out["n_union"].iloc[i]) == len(ea | eb)
        for metric, fn in [("jaccard", m.metric_jaccard)]:
            naive = fn(ea, eb)
            if not bool(rec["both_paths"]) or pd.isna(naive):
                naive = 0.0
            assert float(out[f"joint_{metric}"].iloc[i]) == pytest.approx(naive)


def test_weighted_and_expanded_joint_aggregation_agree(tmp_path: Path) -> None:
    drug, pert = _null1_side_frames()
    kwargs = dict(
        cell_pair_mode="cartesian",
        fixed_triplet_ids={"T1", "T2"},
        metrics=["jaccard"],
    )
    # Passing a debug writer is the only way to force the expanded cell-pair route,
    # which the weighted aggregation replaces for cartesian pairing.
    debug_writer = CellpairDebugWriter(tmp_path / "debug")
    for eval_type in ["fixed_coverage", "path_existing_pairs"]:
        weighted = m.compute_joint_triplet_metrics_weighted(drug, pert, evaluable_set_type=eval_type, **kwargs)
        expanded = m.compute_joint_triplet_metrics(
            drug, pert, evaluable_set_type=eval_type, debug_writer=debug_writer, **kwargs,
        )
        shared = [c for c in weighted.columns if c in expanded.columns]
        pd.testing.assert_frame_equal(
            weighted[shared].sort_values("triplet_id").reset_index(drop=True),
            expanded[shared].sort_values("triplet_id").reset_index(drop=True),
            check_dtype=False,
        )


# ---------------------------------------------------------------------------
# T6: sparse permutation lookups
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("per_triplet", [False, True])
def test_sparse_permutation_lookup_matches_dense_source_positions(per_triplet: bool) -> None:
    specs = _permutation_specs(per_triplet=per_triplet)
    genes = GENES + ["extra1", "extra2"]
    dense = srp._build_permutation_source_positions(genes=genes, permutation_specs=specs)
    sparse = engine.build_sparse_permutation_lookups(genes, specs)
    assert set(dense) == set(sparse)
    for group, dense_positions in dense.items():
        np.testing.assert_array_equal(sparse[group].as_dense(), dense_positions)
        positions = list(range(len(genes)))
        assert sparse[group].gather(positions) == [int(v) for v in dense_positions]
        # The identity entries are exactly what the sparse form drops.
        assert len(sparse[group]) == int((dense_positions != np.arange(len(genes))).sum())


def test_sparse_permutation_lookups_are_empty_without_specs() -> None:
    assert engine.build_sparse_permutation_lookups(GENES, None) == {}


def test_sparse_permutation_lookup_rejects_unknown_genes() -> None:
    specs = _permutation_specs(per_triplet=False)
    with pytest.raises(ValueError, match="absent from infer input"):
        engine.build_sparse_permutation_lookups(["g1"], specs)


# ---------------------------------------------------------------------------
# T7: prefix trie traversal and the fused Null 1 route
# ---------------------------------------------------------------------------

def _path_polarities(source: str, *, initial_polarity: int, cutoff: int) -> list[dict[str, int]]:
    return srp._collect_path_polarities_from_source(
        source=source, graph=_graph(), initial_polarity=initial_polarity, cutoff=cutoff
    )


def test_trie_fold_is_lossless() -> None:
    polarities = _path_polarities("g1", initial_polarity=-1, cutoff=4)
    trie = build_path_trie(polarities, source="g1", initial_polarity=-1)
    expanded = expand_trie_paths(trie)
    assert [i for i, _, _ in expanded] == list(range(len(polarities)))
    for path_index, names, pols in expanded:
        assert names == list(polarities[path_index].keys())
        assert pols == list(polarities[path_index].values())
    assert trie.num_nodes_in_paths == len({g for p in polarities for g in p})


def test_trie_node_edges_match_the_reference_signed_edge_reconstruction() -> None:
    polarities = _path_polarities("g1", initial_polarity=-1, cutoff=4)
    trie = build_path_trie(polarities, source="g1", initial_polarity=-1)
    for path_index, names, pols in expand_trie_paths(trie):
        row = {
            "pert_id": "g1",
            "signature_reversal_paths": [names],
            "signature_reversal_polarities": [pols],
        }
        reference = m.signed_edges_from_infer_row(row, mode="knockdown")
        node = next(n for n, idx in trie.terminal_paths.items() if path_index in idx)
        assert set(trie.node_edges(node)) == reference


@pytest.mark.parametrize("gap", [0, 1, 2])
def test_trie_traversal_accepts_the_same_paths_as_the_row_evaluator(gap: int) -> None:
    graph = _graph()
    df = _profile_df()
    polarities_by_source = {
        source: [
            p for p in _path_polarities(source, initial_polarity=-1, cutoff=4)
            if len(p) <= 4 and all(g in set(GENES) for g in p)
        ]
        for source in df["pert_id"].unique()
    }
    pert_pos = {g: i for i, g in enumerate(GENES)}
    for rec in df.to_dict(orient="records"):
        polarities = polarities_by_source[rec["pert_id"]]
        trie = build_path_trie(polarities, source=rec["pert_id"], initial_polarity=-1)
        pert_values = [rec[f"{g}_gene"] for g in GENES]
        disease_values = [rec[f"{g}_disease"] for g in GENES]
        reference = srp._evaluate_single_row_arrays(
            disease=rec["do_id"], pert_gene=rec["pert_id"], pert_cell=rec["cell_id"], label=rec["label"],
            disease_values=disease_values, pert_values=pert_values,
            disease_gene_to_pos=pert_pos, pert_gene_to_pos=pert_pos,
            path_polarities=polarities, num_nodes_in_paths=len(GENES), gap=gap,
        )
        accepted = evaluate_trie_row(
            trie,
            [pert_values[pert_pos[g]] for g in trie.genes],
            [disease_values[pert_pos[g]] for g in trie.genes],
            gap=gap,
        )
        ordered = accepted_paths_in_reference_order(accepted)
        pruned = srp.prune_prefix_paths_with_idx(
            [(trie.node_names(node), path_index) for path_index, node in ordered]
        )
        assert [p for p, _ in pruned] == reference["signature_reversal_paths"]
    assert graph.number_of_edges() == len(KG_EDGES)


@pytest.mark.parametrize("permutation_mode", ["none", "global", "per_triplet"])
def test_fused_null1_route_reproduces_the_reference_infer_frame(tmp_path: Path, permutation_mode: str) -> None:
    graph = _graph()
    df = _profile_df()
    input_path = tmp_path / "input.parquet"
    df.to_parquet(input_path, index=False)
    specs = {
        "none": None,
        "global": _permutation_specs(per_triplet=False),
        "per_triplet": _permutation_specs(per_triplet=True),
    }[permutation_mode]

    reference_infer = srp.signature_reversal_paths_from_graph(
        df.copy(), graph, mode="knockdown", cutoff=4, gap=1, permutation_specs=specs,
    )
    reference = m.signed_edge_df_from_infer_df(
        reference_infer, mode="knockdown", include_first_edge_excluded=True,
    )

    prepared = engine.prepare_infer_input(input_path, graph=graph, mode="knockdown")
    fused = engine.null1_signed_edge_df(
        prepared,
        graph=graph,
        gap=1,
        permutation_specs=specs,
        trie_cache=engine.TrieCache(
            mode="knockdown", path_strategy="simple", cutoff=4,
            valid_genes_digest=prepared.valid_genes_digest,
        ),
        path_cache_dir=None,
        include_first_edge_excluded=True,
    )

    shared = [c for c in reference.columns if c in fused.columns]
    assert "signed_edges" in shared and "n_paths" in shared
    assert "signed_edges" + m.FIRST_EDGE_EXCLUDED_SUFFIX in shared
    pd.testing.assert_frame_equal(
        reference[shared].reset_index(drop=True),
        fused[shared].reset_index(drop=True),
        check_dtype=False,
    )
    assert int(fused["n_paths"].sum()) > 0


def test_trie_cache_reuses_one_trie_per_source(tmp_path: Path) -> None:
    graph = _graph()
    df = _profile_df()
    input_path = tmp_path / "input.parquet"
    df.to_parquet(input_path, index=False)
    prepared = engine.prepare_infer_input(input_path, graph=graph, mode="knockdown")
    cache = engine.TrieCache(
        mode="knockdown", path_strategy="simple", cutoff=4,
        valid_genes_digest=prepared.valid_genes_digest,
    )
    frames = [
        engine.null1_signed_edge_df(
            prepared, graph=graph, gap=1,
            permutation_specs=_permutation_specs(per_triplet=False),
            trie_cache=cache, path_cache_dir=None,
        )
        for _ in range(3)
    ]
    stats = cache.stats()
    n_sources = int(pd.Series(prepared.pert_id_values).nunique())
    assert stats["n_tries_built"] == n_sources
    assert stats["n_tries_reused"] == 2 * n_sources
    for frame in frames[1:]:
        pd.testing.assert_frame_equal(frames[0], frame)


# ---------------------------------------------------------------------------
# T8: source-parallel Null 1 traversal
# ---------------------------------------------------------------------------

def _serial_null1_frame(prepared, graph, specs, *, include_first_edge_excluded: bool) -> pd.DataFrame:
    return engine.null1_signed_edge_df(
        prepared,
        graph=graph,
        gap=1,
        permutation_specs=specs,
        trie_cache=engine.TrieCache(
            mode=prepared.mode, path_strategy="simple", cutoff=4,
            valid_genes_digest=prepared.valid_genes_digest,
        ),
        path_cache_dir=None,
        include_first_edge_excluded=include_first_edge_excluded,
    )


@pytest.mark.parametrize("max_workers", [2, 3])
@pytest.mark.parametrize("per_triplet", [False, True])
def test_source_parallel_traversal_matches_the_serial_frame(
    tmp_path: Path, max_workers: int, per_triplet: bool
) -> None:
    graph = _graph()
    input_path = tmp_path / "input.parquet"
    _profile_df().to_parquet(input_path, index=False)
    prepared_drug = engine.prepare_infer_input(input_path, graph=graph, mode="knockdown")
    prepared_pert = engine.prepare_infer_input(input_path, graph=graph, mode="overexpression")
    specs = _permutation_specs(per_triplet=per_triplet)

    expected = {
        "drug": _serial_null1_frame(prepared_drug, graph, specs, include_first_edge_excluded=True),
        "pert": _serial_null1_frame(prepared_pert, graph, specs, include_first_edge_excluded=True),
    }
    with parallel.Null1ParallelEvaluator(
        sides={"drug": prepared_drug, "pert": prepared_pert},
        graph=graph, gap=1, cutoff=4, path_strategy="simple", path_cache_dir=None,
        max_workers=max_workers, scratch_root=tmp_path,
    ) as evaluator:
        actual = evaluator.signed_edge_frames(permutation_specs=specs, include_first_edge_excluded=True)
        # The second wave reuses warm worker tries and a cost-ordered schedule.
        again = evaluator.signed_edge_frames(permutation_specs=specs, include_first_edge_excluded=True)

    for side in ("drug", "pert"):
        pd.testing.assert_frame_equal(actual[side], expected[side])
        pd.testing.assert_frame_equal(again[side], expected[side])
    assert int(expected["drug"]["n_paths"].sum()) > 0


def test_source_parallel_traversal_matches_the_serial_frame_without_permutations(tmp_path: Path) -> None:
    graph = _graph()
    input_path = tmp_path / "input.parquet"
    _profile_df().to_parquet(input_path, index=False)
    prepared = engine.prepare_infer_input(input_path, graph=graph, mode="knockdown")
    expected = _serial_null1_frame(prepared, graph, None, include_first_edge_excluded=False)
    with parallel.Null1ParallelEvaluator(
        sides={"drug": prepared},
        graph=graph, gap=1, cutoff=4, path_strategy="simple", path_cache_dir=None,
        max_workers=2, scratch_root=tmp_path,
    ) as evaluator:
        actual = evaluator.signed_edge_frames(permutation_specs=None, include_first_edge_excluded=False)
    pd.testing.assert_frame_equal(actual["drug"], expected)


def test_published_profile_matrix_round_trips_read_only(tmp_path: Path) -> None:
    mat = np.arange(-6, 6, dtype=np.int8).reshape(3, 4)
    ref = parallel._publish_matrix(mat, tmp_path / "mat.bin")
    attached = ref.attach()
    np.testing.assert_array_equal(attached, mat)
    with pytest.raises(ValueError):
        attached[0, 0] = 1


def test_permutation_wire_round_trips_the_sparse_lookup() -> None:
    lookup = engine.build_sparse_permutation_lookups(
        GENES, _permutation_specs(per_triplet=False)
    )[engine.GLOBAL_PERMUTATION_GROUP]
    restored = parallel._PermutationWire.of(lookup).to_lookup(len(GENES))
    positions = list(range(len(GENES)))
    assert restored.gather(positions) == lookup.gather(positions)
    np.testing.assert_array_equal(restored.as_dense(), lookup.as_dense())
    assert parallel._PermutationWire.of(None) is None


# ---------------------------------------------------------------------------
# T9: shared structural path pool
# ---------------------------------------------------------------------------

def test_shared_structural_pool_matches_the_per_side_build() -> None:
    graph = _graph()
    targets = {"g1", "g2", "g3"}
    common = dict(
        graph=graph, targets=targets, mode="knockdown", cutoff=4,
        valid_genes=set(GENES), path_strategy="simple",
    )
    drug_pool, drug_diag = m._build_target_khop_path_pool(side="drug", **common)
    pert_pool, pert_diag = m._build_target_khop_path_pool(side="pert", **common)

    cache: dict = {}
    shared_drug_pool, shared_drug_diag = m.build_shared_target_khop_path_pool(
        side="drug", pool_cache=cache, **common
    )
    shared_pert_pool, shared_pert_diag = m.build_shared_target_khop_path_pool(
        side="pert", pool_cache=cache, **common
    )
    assert len(cache) == 1
    assert shared_drug_pool == drug_pool
    assert shared_pert_pool == pert_pool
    pd.testing.assert_frame_equal(shared_drug_diag.reset_index(drop=True), drug_diag)
    pd.testing.assert_frame_equal(shared_pert_diag.reset_index(drop=True), pert_diag)
    # The candidate order must survive the sharing, since the sampler indexes into it.
    for target, by_length in shared_pert_pool.items():
        for length, paths in by_length.items():
            assert paths == pert_pool[target][length]


def test_shared_structural_pool_separates_different_valid_gene_sets() -> None:
    graph = _graph()
    cache: dict = {}
    common = dict(graph=graph, targets={"g1"}, mode="knockdown", cutoff=4, path_strategy="simple", side="drug")
    full, _ = m.build_shared_target_khop_path_pool(valid_genes=set(GENES), pool_cache=cache, **common)
    restricted, _ = m.build_shared_target_khop_path_pool(valid_genes={"g2"}, pool_cache=cache, **common)
    assert len(cache) == 2
    assert full != restricted
