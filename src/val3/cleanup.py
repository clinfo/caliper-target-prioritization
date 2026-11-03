from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += int(p.stat().st_size)
            except OSError:
                pass
    return total


def _safe_remove(path: Path, *, run_root: Path) -> dict[str, Any]:
    rec = {"path": str(path), "existed": path.exists(), "bytes": _dir_size(path), "removed": False, "error": ""}
    if not path.exists():
        return rec
    try:
        # Refuse paths outside run_root; this is a storage cleanup, not a generic rm -rf.
        path.resolve().relative_to(run_root.resolve())
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        rec["removed"] = True
    except Exception as exc:  # pragma: no cover - defensive logging
        rec["error"] = str(exc)
    return rec


def cleanup_intermediates(
    *,
    run_root: str | Path,
    retention: str,
    after_stage: str,
    extra_paths: Iterable[str | Path] | None = None,
) -> pd.DataFrame:
    """Apply the val3 v3 intermediate retention policy.

    ``evaluation`` deletes path exploration/infer intermediates after evaluation has
    succeeded; ``report`` delays that deletion until the report step has succeeded;
    ``path_exploration`` and ``none`` retain all intermediates.
    """
    run_root = Path(run_root)
    retention = str(retention or "evaluation")
    if retention not in {"path_exploration", "evaluation", "report", "none"}:
        raise ValueError("storage.intermediate_retention must be one of path_exploration/evaluation/report/none")
    if retention in {"none", "path_exploration"}:
        return pd.DataFrame(columns=["path", "existed", "bytes", "removed", "error", "retention", "after_stage"])
    if retention == "evaluation" and after_stage not in {"evaluation", "report"}:
        return pd.DataFrame(columns=["path", "existed", "bytes", "removed", "error", "retention", "after_stage"])
    if retention == "report" and after_stage != "report":
        return pd.DataFrame(columns=["path", "existed", "bytes", "removed", "error", "retention", "after_stage"])

    targets = [
        run_root / "infer_outputs",
        run_root / "prepared_inputs" / "baseline",
        run_root / "prepared_inputs" / "decoy_candidate",
    ]
    if extra_paths:
        targets.extend(Path(p) for p in extra_paths)
    rows = []
    for path in targets:
        rec = _safe_remove(path, run_root=run_root)
        rec["retention"] = retention
        rec["after_stage"] = after_stage
        rows.append(rec)
    out = pd.DataFrame(rows)
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(log_dir / f"cleanup_{after_stage}.tsv", sep="\t", index=False)
    return out
