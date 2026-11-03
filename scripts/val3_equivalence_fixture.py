#!/usr/bin/env python
"""Build, freeze and re-check the small Val3 equivalence fixture (spec §7.1).

The fixture is a self-contained 20-node knowledge graph with three observed
triplets, Null 1 permutation specs and a Target-Decoy assignment, small enough to
run ``evaluate_direction`` end to end in a few seconds. Two variants are covered:
``null2_off`` (the primary null set) and ``null2_on`` (structural resampling also
enabled).

Usage::

    python scripts/val3_equivalence_fixture.py build     # write the raw inputs
    python scripts/val3_equivalence_fixture.py golden    # freeze the golden outputs
    python scripts/val3_equivalence_fixture.py check     # re-run and diff vs golden

``golden`` must be run on the reference implementation before a change lands;
``check`` is what every later commit re-runs. ``tests/test_val3_equivalence_fixture.py``
calls ``check`` and skips when no golden has been frozen yet.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from inference.signature_reversal_paths import (  # noqa: E402
    serialize_signature_path_columns_for_parquet,
    signature_reversal_paths_from_graph,
)
from val3 import decoy as td  # noqa: E402
from val3.io import ensure_dir, read_parquet_any, write_parquet  # noqa: E402
from val3.metrics import evaluate_direction  # noqa: E402
from val3.prepare import build_drug_input, build_pert_input  # noqa: E402
from val3.shuffle import (  # noqa: E402
    build_target_khop_restricted_paired_gene_label_permutation_context,
    make_target_khop_restricted_paired_gene_label_permutation_specs_from_context,
    save_null1_permutation_specs,
)

FIXTURE_ROOT = REPO_ROOT / "tests" / "data" / "val3_equivalence_fixture"
INPUT_DIR = FIXTURE_ROOT / "inputs"
GOLDEN_DIR = FIXTURE_ROOT / "golden"

DIRECTION = "inhibitory"
K_STAR = 4
GAP = 1
SEED = 0
N_SHUFFLE = 3
N_DECOYS = 3

GENE_NODE_NULL = "target_khop_restricted_paired_gene_label_permutation"
PATH_SHUFFLE_NULL = "path_count_length_matched_structural_resampling"
TARGET_DECOY_NULL = td.TARGET_DECOY_NULL_TYPE

N_NODES = 20
NODES = [f"NCBIGENE:{i}" for i in range(1, N_NODES + 1)]
CELLS = ["MCF7", "PC3", "A375"]
DISEASES = ["DOID:1", "DOID:2"]

# Float columns are compared with the spec §7.1 tolerance; everything else must match.
RTOL = 1e-10
ATOL = 1e-12


# ---------------------------------------------------------------------------
# fixture inputs
# ---------------------------------------------------------------------------

def _kg_edges() -> pd.DataFrame:
    """A small directed graph where every observed target reaches several nodes."""
    rng = np.random.default_rng(4242)
    rows: list[dict[str, Any]] = []
    for i, source in enumerate(NODES):
        for offset in (1, 2, 5):
            j = (i + offset) % N_NODES
            if j == i:
                continue
            rows.append({
                "source": source,
                "target": NODES[j],
                "polarity": int(rng.choice([-1, 1])),
            })
    return pd.DataFrame(rows).drop_duplicates(["source", "target"]).reset_index(drop=True)


def _ternary(rng: np.random.Generator, n: int, *, p_zero: float) -> np.ndarray:
    draws = rng.random(n)
    signs = rng.choice([-1, 1], size=n)
    return np.where(draws < p_zero, 0, signs).astype(np.int8)


def _triplets() -> pd.DataFrame:
    return pd.DataFrame([
        {"triplet_id": "T1", "drug_id": "PUBCHEM.COMPOUND:11", "target_id": NODES[0],
         "disease_id": DISEASES[0], "label": 1, "direction": DIRECTION},
        {"triplet_id": "T2", "drug_id": "PUBCHEM.COMPOUND:12", "target_id": NODES[3],
         "disease_id": DISEASES[0], "label": 1, "direction": DIRECTION},
        {"triplet_id": "T3", "drug_id": "PUBCHEM.COMPOUND:13", "target_id": NODES[7],
         "disease_id": DISEASES[1], "label": 1, "direction": DIRECTION},
        # Unsupported on purpose: no drug profile, so the universe filter drops it.
        {"triplet_id": "T4", "drug_id": "PUBCHEM.COMPOUND:99", "target_id": NODES[9],
         "disease_id": DISEASES[1], "label": 1, "direction": DIRECTION},
    ])


def _drug_profiles() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for drug_id in ["PUBCHEM.COMPOUND:11", "PUBCHEM.COMPOUND:12", "PUBCHEM.COMPOUND:13"]:
        for cell_id in CELLS[:2]:
            values = _ternary(rng, N_NODES, p_zero=0.45)
            rows.append({"drug_id": drug_id, "cell_id": cell_id,
                         **{f"{g}_gene": int(v) for g, v in zip(NODES, values)}})
    return pd.DataFrame(rows)


def _target_profiles() -> pd.DataFrame:
    rng = np.random.default_rng(22)
    rows = []
    for pert_id in NODES:
        for cell_id in CELLS:
            values = _ternary(rng, N_NODES, p_zero=0.55)
            rows.append({"pert_id": pert_id, "cell_id": cell_id,
                         **{f"{g}_gene": int(v) for g, v in zip(NODES, values)}})
    return pd.DataFrame(rows)


def _disease_profiles() -> pd.DataFrame:
    rng = np.random.default_rng(33)
    rows = []
    for disease_id in DISEASES:
        values = _ternary(rng, N_NODES, p_zero=0.35)
        rows.append({"disease_id": disease_id,
                     **{f"{g}_disease": int(v) for g, v in zip(NODES, values)}})
    return pd.DataFrame(rows)


def build_fixture_inputs() -> Path:
    ensure_dir(INPUT_DIR)
    _kg_edges().to_csv(INPUT_DIR / "kg.tsv", sep="\t", index=False)
    _triplets().to_csv(INPUT_DIR / "triplets.tsv", sep="\t", index=False)
    write_parquet(_drug_profiles(), INPUT_DIR / "drug_profiles.parquet")
    write_parquet(_target_profiles(), INPUT_DIR / "target_profiles.parquet")
    write_parquet(_disease_profiles(), INPUT_DIR / "disease_profiles.parquet")
    print(f"[OK] fixture inputs written to {INPUT_DIR}")
    return INPUT_DIR


def _load_fixture_inputs() -> dict[str, Any]:
    if not (INPUT_DIR / "kg.tsv").exists():
        raise FileNotFoundError(
            f"Fixture inputs are missing. Run 'python {Path(__file__).name} build' first."
        )
    return {
        "kg_path": INPUT_DIR / "kg.tsv",
        "triplets": pd.read_csv(INPUT_DIR / "triplets.tsv", sep="\t"),
        "drug_df": read_parquet_any(INPUT_DIR / "drug_profiles.parquet"),
        "target_df": read_parquet_any(INPUT_DIR / "target_profiles.parquet"),
        "disease_df": read_parquet_any(INPUT_DIR / "disease_profiles.parquet"),
    }


# ---------------------------------------------------------------------------
# running evaluate_direction over the fixture
# ---------------------------------------------------------------------------

def _write_infer(df: pd.DataFrame, graph, *, mode: str, path: Path) -> Path:
    inferred = signature_reversal_paths_from_graph(
        df, graph, mode=mode, cutoff=K_STAR, gap=GAP,
    )
    write_parquet(serialize_signature_path_columns_for_parquet(inferred), path)
    return path


def _evaluation_cfg(*, first_edge_excluded: bool, max_workers: int = 1) -> dict[str, Any]:
    return {
        "metrics": ["jaccard"],
        "filters": {"require_both_nonempty_inferred_paths": True},
        "evaluable_sets": {"path_existing_pairs": {"enabled": True}},
        "global_effect_size": {"enabled": True},
        "edge_scopes": {
            "all_edges": {"enabled": True},
            "first_edge_excluded": {"enabled": bool(first_edge_excluded)},
        },
        "max_workers": int(max_workers),
    }


def _baseline_cfg(*, structural: bool) -> dict[str, Any]:
    return {
        "n_shuffle": N_SHUFFLE,
        "is_primary_spec": False,
        "null_models": {
            GENE_NODE_NULL: {"enabled": True, "min_local_bg_genes": 0},
            PATH_SHUFFLE_NULL: {"enabled": bool(structural)},
            TARGET_DECOY_NULL: {
                "enabled": True,
                "n_decoys_per_triplet": N_DECOYS,
                "exclude_known_positive_targets": True,
                "positive_exclusion_scope": "same_drug_disease",
                "is_primary_spec": False,
            },
        },
    }


def run_variant(variant: str, out_root: Path, *, max_workers: int = 1) -> Path:
    """Run ``evaluate_direction`` for one fixture variant into ``out_root``.

    ``max_workers`` only selects how the Null 1 traversal is scheduled (spec §6.9);
    every worker count must produce the same outputs.
    """
    if variant not in {"null2_off", "null2_on"}:
        raise ValueError(f"Unknown fixture variant: {variant}")
    structural = variant == "null2_on"
    inputs = _load_fixture_inputs()
    graph = td.load_directed_kg_graph(inputs["kg_path"])
    kg_nodes = set(map(str, graph.nodes()))

    work = ensure_dir(out_root / "work")
    eval_dir = ensure_dir(out_root / "evaluation" / DIRECTION)

    triplets = inputs["triplets"].copy()
    triplets["triplet_id"] = triplets["triplet_id"].astype(str)
    supported = triplets.loc[
        triplets["drug_id"].isin(set(inputs["drug_df"]["drug_id"].astype(str)))
        & triplets["target_id"].isin(kg_nodes)
    ].reset_index(drop=True)

    drug_input, _ = build_drug_input(supported, inputs["drug_df"], inputs["disease_df"])
    pert_input, _ = build_pert_input(supported, inputs["target_df"], inputs["disease_df"])
    write_parquet(drug_input, work / "drug_input.parquet")
    write_parquet(pert_input, work / "pert_input.parquet")

    observed_drug_infer = _write_infer(
        drug_input.copy(), graph, mode="knockdown", path=work / "infer_drug.parquet"
    )
    observed_pert_infer = _write_infer(
        pert_input.copy(), graph, mode="knockdown", path=work / "infer_pert.parquet"
    )

    # Null 1 permutation specs, generated in the same iteration order as the runner.
    rng = np.random.default_rng(SEED)
    context = build_target_khop_restricted_paired_gene_label_permutation_context(
        drug_input, pert_input, kg_tsv=inputs["kg_path"], cutoff=K_STAR, kg_nodes=kg_nodes,
    )
    baseline_specs = []
    for iteration in range(1, N_SHUFFLE + 1):
        perm_specs, _ = make_target_khop_restricted_paired_gene_label_permutation_specs_from_context(
            context, rng=rng,
        )
        saved = save_null1_permutation_specs(
            out_dir=work / "permutations", iteration=iteration, permutation_specs=perm_specs,
        )
        baseline_specs.append({
            "iteration": iteration,
            "null_type": GENE_NODE_NULL,
            "inprocess": True,
            **saved,
        })

    # Target-Decoy assignments and the matching pseudo-triplet inference.
    universe = td.build_target_decoy_candidate_universe(inputs["target_df"], kg_nodes=kg_nodes)
    features = td.build_target_decoy_matching_features(
        universe, graph=graph, target_df=inputs["target_df"], reach_cutoff=K_STAR,
    )
    assignments = td.build_target_decoy_assignments(
        supported,
        feature_table=features,
        positive_targets_by_drug_disease=td.build_positive_targets_by_drug_disease(triplets),
        direction=DIRECTION,
        n_decoys_per_triplet=N_DECOYS,
        seed=SEED,
        exclude_known_positive_targets=True,
    )
    decoy_drug_input, decoy_pert_input, _ = td.build_decoy_inputs(
        assignments, drug_input, inputs["target_df"], inputs["disease_df"],
    )
    candidate_table = work / "decoy_candidate_table.tsv"
    assignments.to_csv(candidate_table, sep="\t", index=False)
    decoy_drug_infer = _write_infer(
        decoy_drug_input.copy(), graph, mode="knockdown", path=work / "infer_decoy_drug.parquet"
    )
    decoy_pert_infer = _write_infer(
        decoy_pert_input.copy(), graph, mode="knockdown", path=work / "infer_decoy_pert.parquet"
    )

    summary = evaluate_direction(
        true_drug_infer_path=observed_drug_infer,
        true_pert_infer_path=observed_pert_infer,
        true_drug_input_path=work / "drug_input.parquet",
        true_pert_input_path=work / "pert_input.parquet",
        baseline_drug_input_path=work / "drug_input.parquet",
        baseline_pert_input_path=work / "pert_input.parquet",
        kg_tsv=inputs["kg_path"],
        cutoff=K_STAR,
        gap=GAP,
        drug_mode="knockdown",
        pert_mode="knockdown",
        cell_pair_mode="cartesian",
        evaluation_cfg=_evaluation_cfg(first_edge_excluded=True, max_workers=max_workers),
        baseline_pairs=[],
        baseline_specs=baseline_specs,
        out_dir=eval_dir,
        decoy_candidate_drug_infer_path=decoy_drug_infer,
        decoy_candidate_pert_infer_path=decoy_pert_infer,
        decoy_candidate_table_path=candidate_table,
        decoy_null_type=TARGET_DECOY_NULL,
        baseline_cfg=_baseline_cfg(structural=structural),
        seed=SEED,
        summary_extra={"direction": DIRECTION, "fixture_variant": variant},
    )
    (eval_dir / "fixture_summary.json").write_text(
        json.dumps({"variant": variant, "n_shuffle": summary["n_shuffle"]}, indent=2),
        encoding="utf-8",
    )
    return eval_dir


# ---------------------------------------------------------------------------
# golden comparison
# ---------------------------------------------------------------------------

# Wall-clock values legitimately differ between runs and are excluded from the diff.
VOLATILE_SUMMARY_KEYS = {
    "per_iteration_wall_sec_median",
    "per_iteration_wall_sec_max",
}
VOLATILE_FILES = {
    "evaluation_iteration_timings.tsv",
    "fixture_summary.json",
}


def _iter_comparable_files(root: Path) -> list[Path]:
    return sorted(
        p.relative_to(root)
        for p in root.rglob("*")
        if p.is_file() and p.name not in VOLATILE_FILES
    )


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return read_parquet_any(path)
    return pd.read_csv(path, sep="\t")


def _strip_volatile(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            k: _strip_volatile(v)
            for k, v in payload.items()
            if k not in VOLATILE_SUMMARY_KEYS and not str(k).endswith("_sec")
        }
    if isinstance(payload, list):
        return [_strip_volatile(v) for v in payload]
    return payload


def _compare_frames(actual: pd.DataFrame, expected: pd.DataFrame, label: str) -> list[str]:
    problems: list[str] = []
    if list(actual.columns) != list(expected.columns):
        return [f"{label}: column mismatch {list(actual.columns)} != {list(expected.columns)}"]
    if len(actual) != len(expected):
        return [f"{label}: row count {len(actual)} != {len(expected)}"]
    for column in expected.columns:
        exp = expected[column]
        act = actual[column]
        if pd.api.types.is_float_dtype(exp) or pd.api.types.is_float_dtype(act):
            try:
                np.testing.assert_allclose(
                    pd.to_numeric(act, errors="coerce").to_numpy(dtype=float),
                    pd.to_numeric(exp, errors="coerce").to_numpy(dtype=float),
                    rtol=RTOL, atol=ATOL, equal_nan=True,
                )
            except AssertionError as err:
                problems.append(f"{label}[{column}]: {str(err).splitlines()[0]}")
        elif not act.astype(object).equals(exp.astype(object)):
            n_diff = int((act.astype(object) != exp.astype(object)).sum())
            problems.append(f"{label}[{column}]: {n_diff} differing values")
    return problems


def compare_against_golden(actual_root: Path, golden_root: Path) -> list[str]:
    problems: list[str] = []
    actual_files = set(_iter_comparable_files(actual_root))
    golden_files = set(_iter_comparable_files(golden_root))
    for missing in sorted(golden_files - actual_files):
        problems.append(f"missing output: {missing}")
    for extra in sorted(actual_files - golden_files):
        problems.append(f"unexpected output: {extra}")
    for rel in sorted(actual_files & golden_files):
        actual, expected = actual_root / rel, golden_root / rel
        if rel.suffix == ".json":
            a = _strip_volatile(json.loads(actual.read_text(encoding="utf-8")))
            b = _strip_volatile(json.loads(expected.read_text(encoding="utf-8")))
            if a != b:
                problems.append(f"{rel}: json payload differs")
        elif rel.suffix in {".tsv", ".parquet"}:
            problems.extend(_compare_frames(_read_frame(actual), _read_frame(expected), str(rel)))
        elif actual.read_bytes() != expected.read_bytes():
            problems.append(f"{rel}: bytes differ")
    return problems


def _rewrite_run_paths(root: Path) -> None:
    """Drop absolute paths from summary.json so goldens stay location independent."""
    for summary_path in root.rglob("summary.json"):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))

        def _relativize(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _relativize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_relativize(v) for v in obj]
            if isinstance(obj, str) and str(root) in obj:
                return obj.replace(str(root), "<run_root>")
            # The fixture can run under either /tmp or a sandbox-specific
            # temporary directory.  Summary paths are runtime metadata, not
            # numerical outputs, so normalize every absolute path here.
            if isinstance(obj, str) and Path(obj).is_absolute():
                return "<scratch>/" + Path(obj).name
            return obj

        summary_path.write_text(json.dumps(_relativize(payload), indent=2), encoding="utf-8")


def generate_golden(variants: list[str]) -> None:
    for variant in variants:
        target = GOLDEN_DIR / variant
        if target.exists():
            shutil.rmtree(target)
        ensure_dir(target)
        with tempfile.TemporaryDirectory() as tmp:
            eval_dir = run_variant(variant, Path(tmp))
            shutil.copytree(eval_dir, target, dirs_exist_ok=True)
        _rewrite_run_paths(target)
        print(f"[OK] golden written: {target}")


def check_golden(variants: list[str]) -> int:
    failures = 0
    for variant in variants:
        golden = GOLDEN_DIR / variant
        if not golden.exists():
            print(f"[SKIP] no golden for variant {variant}; run 'golden' first")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            eval_dir = run_variant(variant, Path(tmp))
            scratch = Path(tmp) / "compare"
            shutil.copytree(eval_dir, scratch)
            _rewrite_run_paths(scratch)
            problems = compare_against_golden(scratch, golden)
        if problems:
            failures += 1
            print(f"[FAIL] {variant}: {len(problems)} difference(s)")
            for problem in problems[:20]:
                print(f"  - {problem}")
        else:
            print(f"[OK] {variant} matches golden")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "run", "golden", "check"])
    parser.add_argument("--variant", action="append", choices=["null2_off", "null2_on"])
    parser.add_argument("--out", type=Path, help="Output root for the 'run' command.")
    args = parser.parse_args()
    variants = args.variant or ["null2_off", "null2_on"]

    if args.command == "build":
        build_fixture_inputs()
        return 0
    if args.command == "run":
        if args.out is None:
            parser.error("--out is required for 'run'")
        for variant in variants:
            print(run_variant(variant, ensure_dir(args.out / variant)))
        return 0
    if args.command == "golden":
        generate_golden(variants)
        return 0
    return 1 if check_golden(variants) else 0


if __name__ == "__main__":
    raise SystemExit(main())
