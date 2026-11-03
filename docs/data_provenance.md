# Data provenance

CALIPER uses external resources with distinct analytical roles. The code repository contains no third-party raw data, large profile matrices, complete knowledge graphs, or publication-facing biological result tables. A separate source-data archive contains only compact, canonical CALIPER-derived tables for the manuscript figures and tables; it contains no raw external payload.

| Resource | Release or accession | Role in CALIPER | Acquisition boundary |
| --- | --- | --- | --- |
| SIGNOR | Jan 2026 release; acquired 2026-02-17 | Primary directed signed causal knowledge graph | Obtain from the SIGNOR Downloads service; do not bundle the complete release here. |
| LINCS L1000 | GEO GSE92742 | Level 5 KD profiles for locality selection and ranking; Level 5 drug profiles for path support | Obtain from GEO; do not bundle GCTX files or derived profile matrices here. |
| CREEDS | `disease_signatures-v1.0` | Disease-expression signatures | Obtain from the CREEDS download service; do not bundle the complete signature release here. |
| Open Targets Platform | 25.12; released 2025-12-10 | Inhibitory target-disease labels and drug-target-disease triplets for retrospective evaluation | Obtain from the Platform release; do not bundle complete source datasets here. |
| OpenBioLink2020 | v1.0.1; DOI 10.5281/zenodo.5361324 | Graph-choice robustness analysis | Obtain from its Zenodo record; do not bundle the graph or derivatives here. |
| HGNC | archived mapping snapshot used in preprocessing | Gene-symbol and identifier normalization | Use an appropriate archived HGNC download; only mappings needed for a cleared derived asset may be redistributed. |
| PubChem | compound identifiers used in Val3 | Identifier namespace for reported triplets | No PubChem chemical-record content is redistributed; source-data tables retain identifiers only where needed for the derived triplet metrics. |

The final primary cohort reported in the manuscript contains 430 targets, 145 diseases, 16 cell lines, 3,473 target-cell profiles, and 61,527 target-disease pairs. The path-support analysis reports 1,570 evaluable inhibitory drug-target-disease triplets. These manuscript-level counts describe completed analyses; they are not bundled datasets.

The `targetnexus_db` keys in code and configuration are compatibility identifiers for the Open Targets 25.12-derived label construction. They do not identify a data package distributed by this repository.

For licence and redistribution conditions, see `data_licenses.md`. For source-data assets, including their canonical run alias, source file, extraction method, checksum, and asset-level release status, see `source_data_manifest.tsv`.
