# Data boundary

This directory intentionally contains no third-party raw data or large analysis matrices. CALIPER requires external resources for manuscript-scale analyses:

- SIGNOR signed causal interactions;
- LINCS L1000 perturbational profiles from GEO accession GSE92742;
- CREEDS disease signatures;
- Open Targets Platform target-disease evidence and labels;
- OpenBioLink2020 only for the documented robustness route.

Obtain external resources directly from their authoritative providers under their own licences and terms. Configure their local locations in a local copy of `config/dataset_registry.example.yaml`; do not commit that local copy.

Publication-facing source data are separate from this code repository. The manuscript-associated Zenodo archive contains both the code and the separate source-data archive. Its DOI is [10.5281/zenodo.23917060](https://doi.org/10.5281/zenodo.23917060), which resolves after Zenodo publication. See `docs/source_data_manifest.tsv` and `docs/data_provenance.md` for the assets and their provenance.

The only bundled data are the small, entirely synthetic inputs and golden outputs in `tests/data/val3_equivalence_fixture/`. They support Route A implementation validation and are not biological results.
