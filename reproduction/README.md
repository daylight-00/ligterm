# Reproduction

Scripts, configuration, expected tables and tests that regenerate the evaluation of LigTerm from the research data package (Zenodo, in preparation: TBD). The predictor (`src/ligterm/`, `models/`) needs none of it. Start with [REPRODUCING.md](../REPRODUCING.md): what is reproduced, from what, and what is not.

Folders: `scripts/` (numbered, 00 first), `lib/` (the library `ligterm_repro`), `configs/` (one JSON per experiment), `expected/` (the tables the scripts must reproduce) and `tests/` (`python3 tests/run_all.py`).

## Scripts

Times are measured with 2 CPUs unless stated; "per study" is the median of the 20 studies. Every script prints a usage line (`--help`); the per-study scripts take `--study PMID [...]` or `--all` and write `study_<pmid>.json` into `--out`.

| Script | What it does | Time |
|---|---|---|
| `00_fetch_data.py` | Downloads and verifies the data package; fetches the NetCleave files (pinned commit, SHA-256) | minutes |
| `10_terminal_predictability.py` | Ranking of the observed terminal bonds of a held-out study by sequence models | 1.3 min per study |
| `20_directional_decomposition.py` | Interior-facing and outward-facing parts of the signal | 0.6 min per study |
| `21_published_method_comparison.py` | Paul-style motif and NetCleave against the window models (torch) | 3.8 min per study |
| `22_exact_window_control.py` | The complete standard 4+3 window of NetCleave as a control (torch) | 2.6 min per study |
| `30_composition_position.py --family linear\|boosted` | Contribution of composition and position of the eight interior residues | 2.5 / 26 min per study |
| `40_peptide_ranking_metrics.py` | Peptide-level ranking metrics; `--summarize` aggregates the 20 studies | about 10 min per study |
| `41_train_ligterm_gb.py` | Refits LigTerm-GB on all verified units and compares it with `models/gb` | 7 min, 8 CPUs |
| `42_train_window_mlp.py` | `check`: the stored window-MLP heads in numpy and torch; `fit`: their training code (GPU optional) | seconds (`check`) |
| `43_assemble_ligterm.py` | Assembles LigTerm from LigTerm-GB and the stored heads and compares it with `models/ens` | seconds |
| `44_development_benchmark.py <part>` | Rebuilds the stored aggregate tables of the development stages; `historical-test run` refits the reserved historical test (8 CPUs, 8 BLAS threads) | seconds; refit 3 to 5.5 min per cell |
| `45_mhcii_np_comparison.py` | The pre-specified comparison with MHCII-NP and, with `--supplement`, the post hoc supplement | seconds |
| `50_external_evaluation.py` | The released models scored on the external test cohort | 7.6 min, 4 CPUs |
| `51_external_cohort_audit.py` | Independence check and local-window similarity audit of the external cohort | 1.6 min, 8 CPUs |
| `90_check_reproduction.py` | Compares regenerated results with `expected/` | seconds |

## Data

`00_fetch_data.py --data-dir <folder>` unpacks `datasets/` and `results/` (25 MB; enough for the checks of the stored results and for scripts 10, 30, 40, 41, 50 and 51) and, with `--intermediates NAME ...` or `all` (0.72 GB), `intermediates/<name>/`, which scripts 42 to 45 start from. Scripts 20 to 22 also need the NetCleave files (`--netcleave-dir`). Put a regenerated result in a folder of the same name under your own `results` folder; `90_check_reproduction.py --results <folder>` finds it there. Limits: [REPRODUCING.md](../REPRODUCING.md).
