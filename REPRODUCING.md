# Reproducing the results

> **Research data record (Zenodo): in preparation, TBD.** The record and its DOI do not exist yet and `DATA_PACKAGE.json` has no address for it. Until it is published, `00_fetch_data.py` needs `--source` (a folder that holds the files of the record), and the commands below describe the published record. The predictor (`src/ligterm/`, `models/`) and its regression test ([Verification](docs/VERIFICATION.md)) do not depend on it.

| Level | Meaning | How |
|---|---|---|
| S | The software runs and reproduces its expected output | [Verification](docs/VERIFICATION.md) |
| R1 | The stored results are intact and equal the expected tables | `90_check_reproduction.py` on the downloaded results (seconds) |
| R2 | An experiment is re-run from the data and equals the expected tables | the numbered scripts of [`reproduction/`](reproduction/README.md) (minutes per study) |

## Set up

Python 3.12 with the packages of [`reproduction/requirements.txt`](reproduction/requirements.txt) (CPU only; the versions that were run are in [`reproduction/environment.md`](reproduction/environment.md)), and the `zstd` command. Everything below runs from `reproduction/`.

```
python3 scripts/00_fetch_data.py --data-dir ../data                          # datasets and results of the Zenodo record, verified by SHA-256
python3 scripts/00_fetch_data.py --netcleave-dir ../third_party/netcleave    # only for scripts 20, 21 and 22
```

## R1: check the stored results

```
python3 scripts/90_check_reproduction.py --results ../data/results --require-all
```

Every cell of every per-study record (study, terminus, analysis, model, fit, counts, metrics and their intervals) must be present exactly once and equal the expected table, with equal types; the studies must be those of the expected table; the number of non-finite values must be equal. The summaries (the comparison with MHCII-NP, the external evaluation, the audits of the external cohort, the peptide ranking) are compared value by value. With `--require-all` a check whose result is missing is an error.

## R2: re-run an experiment

```
python3 scripts/10_terminal_predictability.py --data ../data/datasets --out out/terminal_predictability --study 41315082
python3 scripts/90_check_reproduction.py --results out --only terminal_predictability --subset
```

`--all` runs the 20 studies; `--subset` compares only the studies that were run. Use 2 BLAS threads as in [`environment.md`](reproduction/environment.md) when the result must equal the expected table exactly. Everything must be equal except the retrained NetCleave-style networks (scripts 21 and 22), which may differ by 1e-4: a torch run is not bitwise reproducible across machines (observed: 2.7e-06). The per-script commands, run times and needs are in [`reproduction/README.md`](reproduction/README.md).

The release weights are not changed by any script: `41_train_ligterm_gb.py` writes a new folder and compares it with `models/gb`, `43_assemble_ligterm.py` assembles LigTerm from LigTerm-GB and the stored window-MLP heads and compares it with `models/ens`.

## What is not reproduced here

- **The derived inputs.** The IEDB export of 2026-11-03 is no longer served and the UniProt release was not recorded; the derived tables (units, sequences, sequence groups, the external cohort tables) are in the data package. The selection of the external cohort and its homology classes are stored tables; the searches that produced them are not repeated. Reproducing the evaluation of the cohort is not a verification of its independence (see [Evaluation](docs/EVALUATION.md) for the criteria and their limits).
- **GPU training.** The window-MLP heads and the chains of the development stages that used them were trained on GPUs. The stored heads are checked (`42_train_window_mlp.py check`) and the aggregates are rebuilt from the stored records; no GPU training is repeated, and a refit is not expected to equal the shipped heads bit for bit (training is reproducible from its seed on one device and software version, not across devices). `42_train_window_mlp.py fit` is the training code of the heads: it does not rebuild its inputs but reads the stored training rows and the cohort file of the data package (`intermediates/predictor_development/window_mlp_inputs/`), and it was only run on a CPU as a smoke test.
- **The comparison with MHCII-NP from the tool's responses.** The raw responses of the web service are not redistributed; the comparison is rebuilt from the stored per-protein table, which has 10 significant digits, so its numbers are compared at 1e-8. The comparison was pre-specified on an exposed development subset: it is pre-specified, not confirmatory.
- **Embedding stores** of the third-party protein language model (25.9 GB) and the GPU environment.
- **NetCleave.** Scripts 20, 21 and 22 need the descriptor table, the class-II weights and the training sequences of the NetCleave repository; `00_fetch_data.py --netcleave-dir` downloads them from their own repository at a pinned commit and verifies their SHA-256. Script 20 reads the NetCleave training windows only to keep the random stream of the shared bootstrap aligned.
