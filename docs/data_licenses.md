# Data licences and redistribution

CALIPER-authored software is released under the MIT License. That software licence does not apply to external datasets, external identifiers, or future source-data assets.

| Resource | Current licence or terms | Public repository policy |
| --- | --- | --- |
| SIGNOR 4.0 / Jan 2026 | SIGNOR 4.0 is licensed CC BY 4.0. | Do not bundle the complete release or an edge inventory. Compact CALIPER-derived metrics retain attribution and are assessed per asset. |
| LINCS L1000 GSE92742 | Publicly accessible through GEO. No Creative Commons licence is asserted here. | Do not bundle GCTX files, metadata archives, or large derived profile matrices. Provide the accession and acquisition instructions. |
| CREEDS | The CREEDS site states that its contents are CC BY 4.0. | Do not bundle the complete signature release. Compact CALIPER-derived metrics retain attribution and are assessed per asset. |
| Open Targets Platform | Platform data are CC0 1.0; Platform code is Apache 2.0. Individual upstream-source notices remain relevant. | Do not bundle complete Platform release datasets. Treat publication-facing derivatives as asset-level release decisions. |
| OpenBioLink2020 | The benchmark integrates external sources with heterogeneous licences; its record directs users to the external-resource licence information. | Do not bundle the graph, nodes, edges, or a blanket graph derivative. The source-data archive includes only compact aggregate CALIPER metrics for the reported robustness panels, with no OpenBioLink records and no new licence assertion. |
| HGNC | HGNC states that its data are CC0. | No HGNC bulk file is bundled. A small mapping derivative may be considered only with provenance retained. |
| PubChem | Individual PubChem records can have source-specific terms. | No PubChem record content is bundled; only PubChem compound identifiers occur where necessary in CALIPER-derived triplet metrics. |

The separate source-data archive contains canonical publication-facing derived assets only. Each asset is marked `PUBLIC_OK`, `EXCLUDE`, or `BLOCKED` in `source_data_manifest.tsv`. Public assets are not blanket-relicensed: the archive preserves upstream attribution and terms, and its source tables are not a redistribution of raw external datasets.
