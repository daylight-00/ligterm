#!/usr/bin/env python3
"""The development benchmark and the historical test: record-level aggregation of the stored development records and the refit of the historical test.

usage: 44_development_benchmark.py <part> [options]      (the parts are listed below; `<part> --help` lists the options of a part)

One entry point, five parts. Four of them read stored records and rebuild the stored aggregate tables (no model is fitted, no GPU run is repeated); one refits models on the CPU:

  representation-comparison   the Protein-representation comparison (ESMC frozen and LoRA): the aggregation of the stored per-study records of the window MLP, the frozen-ESMC arm and the LoRA-adapted arm against
                              LigTerm-GB and LR-Both16-CV (medians over the 19 unseen development studies, paired contrasts with group-bootstrap intervals, gates and pre-specified readings, extension rule)
                              --records <folder of study_<pmid>_window_mlp_and_frozen.json and study_<pmid>_lora.json> --out <folder> [--config FILE]
  ensemble-scaling            the ensemble score scaling (LigTerm-1S, fixed scale 1) from its stored per-study records
                              --records <folder of study_<pmid>.json> --out <folder> [--config FILE]
  ensemble-seed-robustness    the ensemble seed robustness (three seeds of the window MLP, LigTerm) from its stored per-study records
                              --records <folder of study_<pmid>.json> --out <folder> [--config FILE]
  historical-test             the previously reserved, once-evaluated historical test (two studies, three regimes): `run` refits LigTerm-GB, LR-Both8, LR-Both16, LR-Both16 with the penalty chosen on the training folds and
                              the Paul-style motif on the CPU for one study and one regime; `summarize` aggregates the six records. The historical test is not independent evidence for the final predictor.
                              run --data <datasets> --out <folder> --study <pmid> --regime <name> | summarize --records <folder> --out <folder>
  overlap-strata              the development studies split by whether an observed ligand-terminal bond of the evaluation set is also a terminal bond of a training unit (post hoc, descriptive): `summarize` aggregates
                              the stored per-study records, `study` recomputes a per-study record from the stored per-bond logits
                              summarize --records <folder> --out <folder> | study --data <datasets> --logits <folder> --study <pmid> --out <folder>

The options of a part are listed by `44_development_benchmark.py <part> --help`. Configurations are in ../configs/. Python >= 3.10 with numpy, scipy and scikit-learn (the frozen environment of the project: Python 3.12.3, numpy 2.5.2, scipy 1.18.1,
scikit-learn 1.9.1); the aggregations run in seconds to minutes on one CPU, the historical-test refits take a few minutes per cell on 8 CPUs (the thread count is part of the numerical result: set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS to 8).
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

PARTS = {"representation-comparison": "representation_comparison", "ensemble-scaling": "ensemble_stages", "ensemble-seed-robustness": "ensemble_stages",
         "historical-test": "historical_test", "overlap-strata": "overlap_strata"}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") or sys.argv[1] not in PARTS:
        print(__doc__)
        sys.exit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2)
    part, argv = sys.argv[1], sys.argv[2:]
    mod = importlib.import_module(f"ligterm_repro.{PARTS[part]}")
    if part == "ensemble-scaling":
        mod.main(argv, mod.STAGE_SCALING)
    elif part == "ensemble-seed-robustness":
        mod.main(argv, mod.STAGE_SEEDS)
    else:
        mod.main(argv)


if __name__ == "__main__":
    main()
