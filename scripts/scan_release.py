#!/usr/bin/env python3
"""Scan a CALIPER release tree for common public-surface leakage risks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


MAX_BYTES = 10 * 1024 * 1024
EXCLUDED_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
BANNED_PATH_PARTS = {".claude", ".codex"}
BANNED_FILE_NAMES = {"claude.md", "agents.md"}
SELF_EXEMPT = {"scripts/scan_release.py"}
MARKERS = re.compile(r"(?i)(\bcodex\b|\bllm\b|chatgpt|claude|openai[ _-]?(prompt|conversation)|agent[ _-]?(log|trace))")
ABSOLUTE_PATH = re.compile(r"/(?:home|users|private)/", re.IGNORECASE)
SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?key|password)\b\s*[:=]\s*[^\s#]{8,}"
)


def text_content(path: Path) -> str | None:
    if path.suffix.lower() in {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".cff", ".tsv"} or not path.suffix:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return None
    return None


def scan_notebook(path: Path, findings: list[dict[str, str]]) -> None:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive release scan
        findings.append({"kind": "invalid_notebook", "path": str(path), "detail": str(exc)})
        return
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("execution_count") is not None:
            findings.append({"kind": "notebook_execution_count", "path": str(path), "detail": f"cell={index}"})
        if cell.get("outputs"):
            findings.append({"kind": "notebook_output", "path": str(path), "detail": f"cell={index}"})
        source = "".join(cell.get("source", []))
        if ABSOLUTE_PATH.search(source):
            findings.append({"kind": "absolute_path", "path": str(path), "detail": f"cell={index}"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    findings: list[dict[str, str]] = []

    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        if any(part.lower() in BANNED_PATH_PARTS for part in Path(relative).parts):
            findings.append({"kind": "banned_path", "path": relative, "detail": "development artifact directory"})
        if path.is_dir():
            continue
        if path.name.lower() in BANNED_FILE_NAMES:
            findings.append({"kind": "banned_file", "path": relative, "detail": "development instruction file"})
        if re.search(r"(?i)(prompt|handoff|conversation|agent[_-]?(log|trace))", path.name):
            findings.append({"kind": "development_trace_path", "path": relative, "detail": "file name"})
        if path.stat().st_size > MAX_BYTES:
            findings.append({"kind": "large_file", "path": relative, "detail": str(path.stat().st_size)})
        if path.suffix.lower() == ".ipynb":
            scan_notebook(path, findings)
        content = None if relative in SELF_EXEMPT else text_content(path)
        if content is not None:
            if ABSOLUTE_PATH.search(content):
                findings.append({"kind": "absolute_path", "path": relative, "detail": "content"})
            if MARKERS.search(content):
                findings.append({"kind": "development_trace", "path": relative, "detail": "content"})
            if SECRET_ASSIGNMENT.search(content):
                findings.append({"kind": "possible_secret", "path": relative, "detail": "assignment pattern"})

    report = {"repo_root": str(root), "finding_count": len(findings), "findings": findings}
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
