"""End-to-end golden comparison on the small Val3 fixture (spec §7.1).

The fixture and its golden outputs are produced by
``scripts/val3_equivalence_fixture.py``. When no golden has been frozen yet the
tests skip, so a fresh checkout does not fail before the golden is generated.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

_SPEC = importlib.util.spec_from_file_location(
    "val3_equivalence_fixture", REPO_ROOT / "scripts" / "val3_equivalence_fixture.py"
)
fixture = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fixture)


@pytest.mark.parametrize("variant", ["null2_off", "null2_on"])
def test_fixture_run_matches_golden(variant: str) -> None:
    golden = fixture.GOLDEN_DIR / variant
    if not golden.exists():
        pytest.skip(
            f"No golden for {variant}. Run "
            "'python scripts/val3_equivalence_fixture.py golden' to freeze one."
        )
    with tempfile.TemporaryDirectory() as tmp:
        eval_dir = fixture.run_variant(variant, Path(tmp))
        scratch = Path(tmp) / "compare"
        shutil.copytree(eval_dir, scratch)
        fixture._rewrite_run_paths(scratch)
        problems = fixture.compare_against_golden(scratch, golden)
    assert not problems, "\n".join(problems[:20])


@pytest.mark.parametrize("max_workers", [2, 3])
def test_fixture_run_matches_golden_under_source_parallelism(max_workers: int) -> None:
    """C8 (spec §6.9, §7.2): worker count must not reach any artifact."""
    golden = fixture.GOLDEN_DIR / "null2_off"
    if not golden.exists():
        pytest.skip("No golden frozen yet.")
    with tempfile.TemporaryDirectory() as tmp:
        eval_dir = fixture.run_variant("null2_off", Path(tmp), max_workers=max_workers)
        scratch = Path(tmp) / "compare"
        shutil.copytree(eval_dir, scratch)
        fixture._rewrite_run_paths(scratch)
        problems = fixture.compare_against_golden(scratch, golden)
    assert not problems, "\n".join(problems[:20])


def test_target_decoy_writes_all_edges_cellpair_shards_per_pseudo_triplet() -> None:
    """The same-cell subset of the Target-Decoy control is rebuilt from these shards."""
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmp:
        eval_dir = fixture.run_variant("null2_off", Path(tmp))
        decoy_root = (
            eval_dir / "cellpair_metrics_debug"
            / f"source=null/null_type={fixture.TARGET_DECOY_NULL}"
        )
        assert decoy_root.is_dir()
        parts = sorted(decoy_root.glob("evaluable_set_type=*/iteration=*/*.parquet"))
        assert parts
        # Decoy candidates are not repeats, so they are stored under iteration=NA.
        assert {p.parent.name for p in parts} == {"iteration=NA"}
        rows = pd.concat([fixture._read_frame(p) for p in parts], ignore_index=True)
        assert rows["triplet_id"].str.contains("__matched_decoy__").all()
        # Only the all-edges scope writes a shard: a second scope would duplicate every pair.
        assert not rows.duplicated(["triplet_id", "cell_id_drug", "cell_id_pert"]).any()

        assignments = pd.read_csv(eval_dir.parent.parent / "work" / "decoy_candidate_table.tsv", sep="\t")
        assert set(rows["triplet_id"]) <= set(assignments["pseudo_triplet_id"].astype(str))


def test_fixture_emits_every_first_edge_scope_output() -> None:
    golden = fixture.GOLDEN_DIR / "null2_off"
    if not golden.exists():
        pytest.skip("No golden frozen yet.")
    expected = {
        "global_joint_triplet_metrics_true.parquet",
        "global_joint_triplet_metrics_baseline.parquet",
        "global_joint_triplet_comparison_sets.tsv",
        "global_joint_triplet_comparison.tsv",
        "global_effect_size.tsv",
    }
    scope_dir = golden / "first_edge_excluded"
    assert expected <= {p.name for p in scope_dir.iterdir()}
    assert (golden / "first_edge_exclusion_summary.tsv").exists()
