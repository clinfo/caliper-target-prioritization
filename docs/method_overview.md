# Method overview

CALIPER selects a target-centered causal reasoning locality without using therapeutic target labels, combines signed-graph inference with observed knockdown profiles to rank target-disease pairs, and extracts explanatory paths. The final primary analysis used the SIGNOR Jan 2026 signed causal graph, LINCS L1000 profiles, CREEDS disease signatures, and Open Targets 25.12 labels reserved for retrospective evaluation.

The label-free locality procedure balances transcriptomic coverage against shortest-path sign unambiguity. The completed primary analysis selected `k* = 4`. Ranking combines edge polarity and knockdown alignment within this locality. Path support compares signed edge overlap between drug-perturbation-derived and target-KD-derived paths against gene-node, path-structure, and Target-Decoy controls.

The public release preserves these analysis families:

- **Val1:** label-free effective-locality selection and path-budget diagnostics;
- **Val2:** CALIPER target ranking and ablations;
- **Val2 baseline:** L1000-NCS, network proximity, classical RWR, and DADA comparators;
- **Val3:** drug-target-disease path-overlap support and null models;
- **Val4-H1:** regulatory mechanism-family RNA observability;
- **Val4-H2:** leave-one-family-out ranking contribution.

OpenBioLink2020 is used for a graph-choice robustness analysis. It is not presented as an independent biological cohort. The repository omits the historical Val4-H3 and other legacy analysis paths because they are not part of the final paper route.

For algorithms, inputs, outputs, and manuscript components, see `analysis_to_script_map.tsv`. For data roles and access conditions, see `data_provenance.md`.
