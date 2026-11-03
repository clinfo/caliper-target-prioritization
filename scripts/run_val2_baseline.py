#!/usr/bin/env python3
"""val2_baseline launcher. --config must point to a val2_baseline config YAML
(no repo-wide default file exists; see config/val2_baseline.*.yaml for the
available primary_analysis variants)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from val2_baseline.runner import main


if __name__ == "__main__":
    main()
