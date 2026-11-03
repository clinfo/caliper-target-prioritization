from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from common.metadata import json_safe as _json_safe, write_json_safe

TOKYO = ZoneInfo("Asia/Tokyo")


def _now() -> str:
    return datetime.now(TOKYO).isoformat()


def _public_context(ctx: dict[str, Any]) -> dict[str, Any]:
    return {str(k): v for k, v in ctx.items() if not str(k).startswith("_")}


def write_metadata_snapshot(run_context: dict[str, Any], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_context["last_updated_at"] = _now()
    return write_json_safe(_json_safe(_public_context(run_context)), output_dir / "metadata.json")


def _summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    keys = ["family", "f_idx", "rep", "no_op"]
    return {key: result[key] for key in keys if key in result}


class H2ProgressRecorder:
    """Persist H2 condition-level progress into metadata.json."""

    def __init__(self, run_context: dict[str, Any], output_dir: str | Path, *, enabled: bool = True):
        self.run_context = run_context
        self.output_dir = Path(output_dir)
        self.enabled = bool(enabled)

    def _append_event(self, event: dict[str, Any]) -> None:
        events = self.run_context.setdefault("progress_events", [])
        events.append(event)
        if len(events) > 500:
            del events[:-500]

    def start(
        self,
        section: str,
        *,
        total: int,
        unit: str,
        desc: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        now = _now()
        status = {
            "status": "running",
            "desc": str(desc),
            "unit": str(unit),
            "completed": 0,
            "total": int(total),
            "started_at": now,
            "last_updated_at": now,
        }
        if extra:
            status.update(extra)
        progress_status = self.run_context.setdefault("progress_status", {})
        progress_status[str(section)] = status
        progress_status["current"] = str(section)
        self._append_event(
            {"time": now, "section": str(section), "event": "start", "completed": 0, "total": int(total)}
        )
        write_metadata_snapshot(self.run_context, self.output_dir)

    def update(
        self,
        section: str,
        *,
        completed: int,
        total: int,
        result: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        now = _now()
        status = self.run_context.setdefault("progress_status", {}).setdefault(str(section), {})
        status.update(
            {
                "status": "running",
                "completed": int(completed),
                "total": int(total),
                "last_updated_at": now,
            }
        )
        if result is not None:
            status["last_result"] = _summarize_result(result)
        self._append_event(
            {
                "time": now,
                "section": str(section),
                "event": "update",
                "completed": int(completed),
                "total": int(total),
                "last_result": status.get("last_result"),
            }
        )
        write_metadata_snapshot(self.run_context, self.output_dir)

    def complete(self, section: str) -> None:
        if not self.enabled:
            return
        now = _now()
        status = self.run_context.setdefault("progress_status", {}).setdefault(str(section), {})
        status["status"] = "completed"
        status["finished_at"] = now
        status["last_updated_at"] = now
        self._append_event(
            {
                "time": now,
                "section": str(section),
                "event": "completed",
                "completed": status.get("completed"),
                "total": status.get("total"),
            }
        )
        write_metadata_snapshot(self.run_context, self.output_dir)

    def fail(
        self,
        section: str,
        exc: BaseException,
        *,
        completed: int | None = None,
        total: int | None = None,
    ) -> None:
        if not self.enabled:
            return
        now = _now()
        status = self.run_context.setdefault("progress_status", {}).setdefault(str(section), {})
        status["status"] = "failed"
        status["error_type"] = type(exc).__name__
        status["error_message"] = str(exc)
        status["last_updated_at"] = now
        if completed is not None:
            status["completed"] = int(completed)
        if total is not None:
            status["total"] = int(total)
        self._append_event(
            {
                "time": now,
                "section": str(section),
                "event": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "completed": completed,
                "total": total,
            }
        )
        write_metadata_snapshot(self.run_context, self.output_dir)

    def callback(self, section: str):
        def _cb(result: Any, completed: int, total: int) -> None:
            result_dict = result if isinstance(result, dict) else None
            self.update(section, completed=completed, total=total, result=result_dict)

        return _cb

    def error_callback(self, section: str):
        def _cb(exc: BaseException, completed: int, total: int) -> None:
            self.fail(section, exc, completed=completed, total=total)

        return _cb
