# CALIPER

CALIPER (Causality-Aware Locality Identification for Perturbation-Effect Reasoning) integrates perturbational transcriptomes and signed causal graphs for locality-constrained therapeutic target prioritization.

## What this repository contains

This is the software release for the CALIPER analyses. It contains the manuscript analysis modules, paper-aligned configuration templates, synthetic validation data, preprocessing-boundary documentation, and release documentation. It does not contain third-party raw datasets, large profile matrices, publication-facing derived source data, or exploratory preprocessing notebooks.

## Repository structure

```text
config/       Paper-aligned configuration templates and external-data registry
data/         Instructions for obtaining external data and source-data archives
docs/         Methods, provenance, licences, software versions, and route map
env/          Tested public environment definitions
preprocess/   Preprocessing boundary and external-data preparation notes
scripts/      Analysis entry points and release validation commands
src/          CALIPER analysis modules
tests/        Unit tests and a synthetic Val3 equivalence fixture
```

## Installation

The canonical public environment is `env/environment-public.yml`. The paper reported Python 3.12.3 with NumPy 1.26.4, pandas 2.2.2, and NetworkX 3.3; the remaining pins in the public environment are release-test pins rather than a claim about every historical package version.

```bash
conda env create -f env/environment-public.yml
conda activate caliper-public
python -m pip install -e .
```

Alternatively, use `python -m pip install -r env/requirements.txt` followed by `python -m pip install -e .`.

## External data requirements

Full manuscript analyses require externally obtained SIGNOR, LINCS L1000, CREEDS, Open Targets Platform, and (where applicable) OpenBioLink resources, together with the publication-facing source-data archive. Obtain each resource from its authoritative provider under its own terms. The repository does not redistribute those inputs.

See [data/README.md](data/README.md), [docs/data_provenance.md](docs/data_provenance.md), and [docs/data_licenses.md](docs/data_licenses.md).

## Reproducing the paper analyses

Two routes are intentionally distinct.

- **Route A — public implementation validation.** This route requires no external data and runs imports, configuration parsing, unit tests, and the bundled synthetic Val3 equivalence fixture. It validates the public implementation surface; it is not a full paper reproduction.
- **Route B — manuscript analysis.** This route requires the external inputs and the separate source-data archive. Run the paper templates in `config/paper/` after setting the paths in `config/dataset_registry.example.yaml` or in a local, untracked copy.

```bash
# Route A
python scripts/validate_release.py --repo-root . --run-fixture
pytest -q

# Route B: examples after external inputs have been supplied locally
python scripts/run_val1.py --config config/paper/val1.yaml
python scripts/run_val2.py --config config/paper/val2.yaml
python scripts/run_val2_baseline.py --config config/paper/val2_baseline.yaml
python scripts/run_val3.py --config config/paper/val3.yaml
python scripts/run_val4_h1.py --config config/paper/val4/h1.yaml
python scripts/run_val4_h2.py --config config/paper/val4/h2.yaml
```

The publication-facing source-data archive is deliberately separate from the code repository. It contains compact canonical derived tables for Main Figs. 2–5, Figs. S1–S5, and Table 1/Tables S1–S4; it does not contain raw external resources. Its file-level release status, canonical source, and checksum are described by `docs/source_data_manifest.tsv`. The manuscript-associated Zenodo archive, containing the code and the separate source-data archive, has DOI [10.5281/zenodo.23917060](https://doi.org/10.5281/zenodo.23917060). The DOI resolves after Zenodo publication.

## Reproducing publication-facing outputs

Use the Route B configurations together with the separately staged publication-facing source data. `docs/analysis_to_script_map.tsv` identifies the analysis entry point, required inputs, and expected output family for each manuscript component. Fig. 1 is conceptual and has no numerical source-data asset; Main Fig. 4a is an explanatory schematic and likewise has no numerical source-data asset.

## Tests and validation

```bash
python -m compileall src scripts
python scripts/scan_release.py --repo-root .
python scripts/validate_release.py --repo-root . --run-fixture
pytest -q
```

The tests cover runtime behaviour and release-surface integrity. They do not independently establish the manuscript's scientific conclusions.

## Data provenance and licensing

External datasets retain their original licences and terms. CALIPER code is MIT-licensed; that licence does not apply to external data. See [docs/data_provenance.md](docs/data_provenance.md) and [docs/data_licenses.md](docs/data_licenses.md).

## Citation

Please cite the accompanying manuscript and the manuscript-associated code and source-data archive: [10.5281/zenodo.23917060](https://doi.org/10.5281/zenodo.23917060). The DOI resolves after Zenodo publication. Software citation metadata are in [CITATION.cff](CITATION.cff).

## License

CALIPER-authored software is released under the [MIT License](LICENSE).
