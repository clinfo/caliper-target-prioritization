# Reproducibility

## Scope

CALIPER provides a code-focused release. It separates implementation validation from manuscript-scale execution because the latter needs large external resources and publication-facing derived data that are not redistributed here.

## Route A: public implementation validation

Route A requires no external biological data. It checks imports, configuration parsing, the release surface, and the synthetic Val3 equivalence fixture.

```bash
python -m pip install -r env/requirements.txt
python -m pip install -e .
python -m compileall src scripts
python scripts/scan_release.py --repo-root .
python scripts/validate_release.py --repo-root . --run-fixture
pytest -q
```

The fixture is a small synthetic directed graph with synthetic profiles and fixed golden outputs. It validates deterministic implementation behaviour; it does not reproduce manuscript figures or establish biological results.

## Route B: manuscript analyses

Route B requires locally acquired external resources and, where figure regeneration is required, the separately staged publication-facing source-data archive. Do not place those raw resources in this repository. Start by creating a local, untracked copy of `config/dataset_registry.example.yaml` and point it to the following prepared inputs:

1. signed protein-protein edge tables from SIGNOR Jan 2026 and, for robustness, OpenBioLink2020 v1.0.1;
2. LINCS L1000 Level 5 KD (`trt_sh.cgs`) and drug (`trt_cp`) profiles, aggregated by perturbagen and cell line at NCBI Gene ID level;
3. CREEDS `disease_signatures-v1.0`, mapped to NCBI Gene IDs and aggregated by disease;
4. Open Targets Platform 25.12 target-disease labels and inhibitory drug-target-disease triplets, mapped to NCBI Gene IDs and DOIDs;
5. the derived analysis tables required for publication-facing panels.

Run the documented routes in this order:

```bash
python scripts/run_val1.py --config config/paper/val1.yaml
python scripts/run_val2.py --config config/paper/val2.yaml
python scripts/run_val2_baseline.py --config config/paper/val2_baseline.yaml
python scripts/run_val3.py --config config/paper/val3.yaml
python scripts/run_val4_h1.py --config config/paper/val4/h1.yaml
python scripts/run_val4_h2.py --config config/paper/val4/h2.yaml
```

The wrappers preserve the source module entry points. The `targetnexus_db` configuration key is retained for code compatibility; in this release it denotes the Open Targets 25.12-derived target-disease label construction described in the manuscript, not a separately redistributed database.

## Paper-aligned settings

The paper templates record the completed-analysis settings reported in Supplementary Table S4:

- effective locality `k* = 4`, with `k = 3` and `k = 5` as adjacent sensitivity settings;
- accepted directed simple paths with gap tolerance 1, maximum 100,000 paths per source and `k`, and a 600-second safeguard;
- 100 size-matched baseline repetitions, 1,000 network-proximity null draws, 100 structured-null repetitions, up to 10 matched Target-Decoy controls per observed triplet, and 1,000 path-overlap bootstrap resamples;
- 1,000 RNA-observability null and bootstrap repetitions; 100 LOFO random edge-drop repetitions and 1,000 bootstrap resamples;
- primary seed 0; external-baseline seed 42; RWR restart probability 0.5, L1 tolerance `1e-6`, and maximum 100,000 iterations.

The source development configuration used smaller values for selected exploratory settings. The public templates use the manuscript values without changing the analysis algorithms.

## Expected boundaries

Route B has not been rerun from raw downloads in this release-preparation workspace. The separate publication-facing source-data archive was prepared from completed canonical result tables; its contents were not recalculated here. Route B must therefore be treated as a documented execution route, not a newly verified raw-to-paper reproduction. See `source_data_manifest.tsv` for asset-level canonical provenance and release status.
