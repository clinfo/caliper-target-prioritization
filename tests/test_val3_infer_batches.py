"""Batch scheduling of the Target-Decoy candidate inference.

``run_infer_dataframe_batches`` drives one infer subprocess per candidate batch.
The batches are independent and each part file is named from its position in the
input stream, so raising ``max_workers`` may only change how long the loop takes.
These tests replace the subprocess with a deterministic stub so the scheduling
itself is what is under test.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from val3 import infer_runner  # noqa: E402


class _StubInfer:
    """Stands in for one infer subprocess and records observed concurrency."""

    def __init__(self, *, delay: float = 0.02) -> None:
        self.delay = delay
        self._lock = threading.Lock()
        self._running = 0
        self.max_concurrent = 0
        self.n_calls = 0

    def __call__(self, *, output_dir, mode, cutoff, gap, input_parquet, **_: object) -> Path:
        with self._lock:
            self._running += 1
            self.n_calls += 1
            self.max_concurrent = max(self.max_concurrent, self._running)
        try:
            time.sleep(self.delay)
            out = infer_runner.infer_output_path(output_dir, mode=mode, cutoff=cutoff, gap=gap)
            out.parent.mkdir(parents=True, exist_ok=True)
            frame = pd.read_parquet(input_parquet)
            frame.assign(mode=str(mode)).to_parquet(out, index=False)
            return out
        finally:
            with self._lock:
                self._running -= 1


def _batches(n_batches: int, rows_per_batch: int = 3) -> list[pd.DataFrame]:
    return [
        pd.DataFrame({"pert_id": [f"g{idx}_{row}" for row in range(rows_per_batch)]})
        for idx in range(n_batches)
    ]


def _run(tmp_path: Path, stub: _StubInfer, *, max_workers: int, n_batches: int) -> Path:
    monkeypatched = infer_runner.run_infer_subprocess
    infer_runner.run_infer_subprocess = stub  # type: ignore[assignment]
    try:
        return infer_runner.run_infer_dataframe_batches(
            repo_root=REPO_ROOT,
            mode="knockdown",
            cutoff=4,
            gap=1,
            input_batches=iter(_batches(n_batches)),
            kg_tsv=tmp_path / "kg.tsv",
            output_dir=tmp_path / f"out_w{max_workers}",
            max_workers=max_workers,
        )
    finally:
        infer_runner.run_infer_subprocess = monkeypatched  # type: ignore[assignment]


@pytest.mark.parametrize("max_workers", [2, 4, 8])
def test_parallel_batches_produce_the_same_dataset_as_the_serial_loop(
    tmp_path: Path, max_workers: int
) -> None:
    n_batches = 9
    serial_dir = _run(tmp_path, _StubInfer(delay=0.0), max_workers=1, n_batches=n_batches)
    parallel_dir = _run(tmp_path, _StubInfer(delay=0.0), max_workers=max_workers, n_batches=n_batches)

    serial_parts = sorted(p.name for p in serial_dir.glob("part-*.parquet"))
    assert serial_parts == sorted(p.name for p in parallel_dir.glob("part-*.parquet"))
    assert len(serial_parts) == n_batches
    for name in serial_parts:
        pd.testing.assert_frame_equal(
            pd.read_parquet(serial_dir / name), pd.read_parquet(parallel_dir / name)
        )


def test_batches_run_concurrently_but_never_exceed_max_workers(tmp_path: Path) -> None:
    stub = _StubInfer(delay=0.05)
    _run(tmp_path, stub, max_workers=3, n_batches=12)
    assert stub.n_calls == 12
    assert 1 < stub.max_concurrent <= 3


def test_empty_batch_stream_still_writes_one_part(tmp_path: Path) -> None:
    stub = _StubInfer(delay=0.0)
    dataset_dir = _run(tmp_path, stub, max_workers=4, n_batches=0)
    assert stub.n_calls == 0
    assert [p.name for p in dataset_dir.glob("part-*.parquet")] == ["part-000000.parquet"]


def test_scratch_batch_files_are_cleaned_up(tmp_path: Path) -> None:
    _run(tmp_path, _StubInfer(delay=0.0), max_workers=4, n_batches=5)
    assert not (tmp_path / "out_w4" / "_tmp_batches").exists()


# ---------------------------------------------------------------------------
# observed input split into pert_id group batches
# ---------------------------------------------------------------------------

def _wide_observed_input(tmp_path: Path) -> Path:
    """One wide infer input with skewed, interleaved ``pert_id`` groups."""
    sizes = {"g_a": 7, "g_b": 1, "g_c": 5, "g_d": 1, "g_e": 9, "g_f": 2, "g_g": 1}
    rows = [
        {"pert_id": pert_id, "row_in_group": i, "X_gene": i % 3 - 1, "X_disease": i % 2}
        for pert_id, size in sizes.items()
        for i in range(size)
    ]
    path = tmp_path / "observed.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_group_batches_keep_every_pert_id_group_whole_and_in_order(tmp_path: Path) -> None:
    path = _wide_observed_input(tmp_path)
    original = pd.read_parquet(path)
    group_order = list(dict.fromkeys(original["pert_id"]))

    for target_batches in (1, 2, 3, 5, 8, 50):
        batches = list(infer_runner.iter_pert_id_group_batches(path, target_batches=target_batches))
        # No group is ever split across two batches.
        seen: list[str] = []
        for batch in batches:
            seen.extend(dict.fromkeys(batch["pert_id"]))
        assert seen == group_order, target_batches
        # The concatenated batches are the original rows, in the original order.
        pd.testing.assert_frame_equal(
            pd.concat(batches).reset_index(drop=True), original.reset_index(drop=True)
        )


def test_group_batches_never_split_a_group_larger_than_the_budget(tmp_path: Path) -> None:
    path = _wide_observed_input(tmp_path)
    batches = list(infer_runner.iter_pert_id_group_batches(path, target_batches=50))
    # target_batches=50 asks for a 1-row budget, but the 9-row group stays whole.
    assert max(len(b) for b in batches) == 9
    assert all(b["pert_id"].nunique() == 1 for b in batches)


def test_group_batches_drop_sources_absent_from_the_kg(tmp_path: Path) -> None:
    path = _wide_observed_input(tmp_path)
    known = {"g_a", "g_c", "g_e"}
    batches = list(
        infer_runner.iter_pert_id_group_batches(path, target_batches=3, kg_nodes=known)
    )
    kept = pd.concat(batches).reset_index(drop=True)
    assert set(kept["pert_id"]) == known
    # Every emitted batch still holds rows, so no part can end up column-less.
    assert all(not batch.empty for batch in batches)
    original = pd.read_parquet(path)
    pd.testing.assert_frame_equal(
        kept, original.loc[original["pert_id"].isin(known)].reset_index(drop=True)
    )


def test_missing_pert_id_column_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "no_pert_id.parquet"
    pd.DataFrame({"other": [1, 2]}).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="pert_id"):
        list(infer_runner.iter_pert_id_group_batches(path, target_batches=2))


class _GroupMajorInfer:
    """Stub infer that mirrors the group-major output of the real path inference.

    ``signature_reversal_paths_from_graph`` appends its results one
    ``groupby("pert_id", sort=False)`` group at a time and derives per-source values
    inside that loop. ``n_rows_in_group`` stands in for such a value, so a batch split
    that cut a group in half would change it and fail the equivalence assertion.
    """

    def __call__(self, *, output_dir, mode, cutoff, gap, input_parquet, **_: object) -> Path:
        frame = pd.read_parquet(input_parquet)
        parts = [
            sub.assign(n_rows_in_group=len(sub), mode=str(mode))
            for _, sub in frame.groupby("pert_id", sort=False)
        ]
        out = infer_runner.infer_output_path(output_dir, mode=mode, cutoff=cutoff, gap=gap)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(parts, ignore_index=True).to_parquet(out, index=False)
        return out


@pytest.mark.parametrize(("target_batches", "max_workers"), [(1, 1), (3, 2), (8, 4), (50, 8)])
def test_batched_observed_input_reproduces_the_single_task_rows(
    tmp_path: Path, target_batches: int, max_workers: int
) -> None:
    """The batched route must be row-for-row identical to inferring the whole input."""
    path = _wide_observed_input(tmp_path)
    monkeypatched = infer_runner.run_infer_subprocess
    infer_runner.run_infer_subprocess = _GroupMajorInfer()  # type: ignore[assignment]
    try:
        reference_dir = infer_runner.run_infer_dataframe_batches(
            repo_root=REPO_ROOT,
            mode="knockdown",
            cutoff=4,
            gap=1,
            input_batches=iter([pd.read_parquet(path)]),
            kg_tsv=tmp_path / "kg.tsv",
            output_dir=tmp_path / f"ref_{target_batches}_{max_workers}",
            max_workers=1,
        )
        batched_dir = infer_runner.run_infer_dataframe_batches(
            repo_root=REPO_ROOT,
            mode="knockdown",
            cutoff=4,
            gap=1,
            input_batches=infer_runner.iter_pert_id_group_batches(path, target_batches=target_batches),
            kg_tsv=tmp_path / "kg.tsv",
            output_dir=tmp_path / f"batched_{target_batches}_{max_workers}",
            max_workers=max_workers,
        )
    finally:
        infer_runner.run_infer_subprocess = monkeypatched  # type: ignore[assignment]

    pd.testing.assert_frame_equal(
        pd.read_parquet(batched_dir).reset_index(drop=True),
        pd.read_parquet(reference_dir).reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# paired drug/pert batch stream
# ---------------------------------------------------------------------------

def _paired_batches(n_batches: int) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    return [
        (
            pd.DataFrame({"pert_id": [f"d{idx}"], "side_marker": ["drug"]}),
            pd.DataFrame({"pert_id": [f"p{idx}"], "side_marker": ["pert"]}),
        )
        for idx in range(n_batches)
    ]


def _run_paired(tmp_path: Path, stub: _StubInfer, *, max_workers: int, n_batches: int, tag: str = ""):
    monkeypatched = infer_runner.run_infer_subprocess
    infer_runner.run_infer_subprocess = stub  # type: ignore[assignment]
    try:
        return infer_runner.run_infer_paired_dataframe_batches(
            repo_root=REPO_ROOT,
            cutoff=4,
            gap=1,
            input_batches=iter(_paired_batches(n_batches)),
            drug_mode="knockdown",
            pert_mode="overexpression",
            drug_output_dir=tmp_path / f"paired{tag}_w{max_workers}" / "drug",
            pert_output_dir=tmp_path / f"paired{tag}_w{max_workers}" / "pert",
            kg_tsv=tmp_path / "kg.tsv",
            log_label="decoy_candidate",
            max_workers=max_workers,
        )
    finally:
        infer_runner.run_infer_subprocess = monkeypatched  # type: ignore[assignment]


def test_paired_batches_write_matching_part_names_per_side(tmp_path: Path) -> None:
    n_batches = 6
    drug_dir, pert_dir = _run_paired(tmp_path, _StubInfer(delay=0.0), max_workers=4, n_batches=n_batches)
    drug_parts = sorted(p.name for p in drug_dir.glob("part-*.parquet"))
    pert_parts = sorted(p.name for p in pert_dir.glob("part-*.parquet"))
    # _paired_infer_part_paths pairs the two sides by part name, so both sides must
    # carry the same names and each name must hold its own side's rows.
    assert drug_parts == pert_parts
    assert len(drug_parts) == n_batches
    for name in drug_parts:
        assert pd.read_parquet(drug_dir / name)["side_marker"].eq("drug").all()
        assert pd.read_parquet(pert_dir / name)["side_marker"].eq("pert").all()
        assert pd.read_parquet(drug_dir / name)["mode"].eq("knockdown").all()
        assert pd.read_parquet(pert_dir / name)["mode"].eq("overexpression").all()


def test_paired_batches_share_one_pool_across_both_sides(tmp_path: Path) -> None:
    stub = _StubInfer(delay=0.05)
    _run_paired(tmp_path, stub, max_workers=3, n_batches=5)
    # Both sides of every batch are submitted, and the bound spans them together.
    assert stub.n_calls == 10
    assert 1 < stub.max_concurrent <= 3


def test_paired_batches_match_running_each_side_separately(tmp_path: Path) -> None:
    n_batches = 5
    drug_dir, pert_dir = _run_paired(tmp_path, _StubInfer(delay=0.0), max_workers=4, n_batches=n_batches, tag="_par")
    serial_drug, serial_pert = _run_paired(
        tmp_path, _StubInfer(delay=0.0), max_workers=1, n_batches=n_batches, tag="_ser"
    )
    for parallel_dir, serial_dir in ((drug_dir, serial_drug), (pert_dir, serial_pert)):
        names = sorted(p.name for p in serial_dir.glob("part-*.parquet"))
        assert names == sorted(p.name for p in parallel_dir.glob("part-*.parquet"))
        for name in names:
            pd.testing.assert_frame_equal(
                pd.read_parquet(parallel_dir / name), pd.read_parquet(serial_dir / name)
            )


def test_paired_empty_stream_still_writes_one_part_per_side(tmp_path: Path) -> None:
    stub = _StubInfer(delay=0.0)
    drug_dir, pert_dir = _run_paired(tmp_path, stub, max_workers=4, n_batches=0)
    assert stub.n_calls == 0
    for dataset_dir in (drug_dir, pert_dir):
        assert [p.name for p in dataset_dir.glob("part-*.parquet")] == ["part-000000.parquet"]
        assert not (dataset_dir.parent / dataset_dir.name / "_tmp_batches").exists()
