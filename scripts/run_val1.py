#!/usr/bin/env python3
"""Val1 launcher. Calls val1.runner.main; --config defaults to repo config/val1.yaml when omitted."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from val1.runner import main


if __name__ == "__main__":
    main()
