"""val2 external-baseline comparison package.

Implements docs/val2/val2_external_baseline_methodology_ncs.md: compares CALIPER
against selected external baselines (L1000-NCS, Network proximity, Classical
RWR) on a single shared disease / candidate / label universe.

The L1000-NCS baseline normalizes per-cell WTCS to NCS (§6.2.6) using the
target-cell averaged BING_knockdown_signature as the reference population (a
deliberate deviation from the per-sig_id unit of §6.2.1 / validation #4; see
``wtcs.py``).

Existing val2 implementation is reused for data loading, CALIPER scoring and
tie-aware ranking metrics; only the external baselines, the common-set logic
and the evaluation aggregation are new.
"""

from __future__ import annotations
