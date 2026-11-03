from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SharedCache:
    proposed_base_metrics: Dict[int, tuple[float, float]] = field(default_factory=dict)
    proposed_base_by_disease: Dict[int, object] = field(default_factory=dict)
    proposed_score_cache: dict | None = None
