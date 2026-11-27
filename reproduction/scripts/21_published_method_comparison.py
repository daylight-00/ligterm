#!/usr/bin/env python3
"""Published-method comparison: simple window models against retrained and released published methods at the terminal bond.

For one held-out study: the Paul-style cleavage motif (both termini), the NetCleave-style network retrained with the released decoys and with the protocol
negatives (C terminus), and the NetCleave predictor as released (C terminus), against the window models LR-Both4, LR-Both8 and Window-NB4. Primary labels, and
the label sets complete_standard_4plus3_window (every bond has the 4+3 window of NetCleave) and netcleave_unseen_window (that set without windows of the released
NetCleave training data), with paired differences on both. The overlap of the evaluation windows with the released training windows is measured and stored.
The retrained and released methods are third-party material: the NetCleave fork is read from --netcleave-dir (see 00_fetch_data.py) and not redistributed.

The random stream of the bootstrap is shared with 20_directional_decomposition.py (see ligterm_repro/directional_and_published.py).

usage: 21_published_method_comparison.py --data <datasets folder> --netcleave-dir <folder> --out <results folder> (--study PMID [PMID ...] | --all) [--config FILE]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from ligterm_repro import comparators, directional_and_published, models, records, training  # noqa: E402


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="datasets folder")
    ap.add_argument("--netcleave-dir", required=True, help="folder of the NetCleave fork (code/QSAR_table.csv, weights/, training/)")
    ap.add_argument("--out", required=True, help="results folder (one study_<pmid>.json per study)")
    ap.add_argument("--config", default=os.path.join(here, "..", "configs", "directional_decomposition.json"))
    ap.add_argument("--study", nargs="+")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    cfg = records.load_config(a.config)
    studies = cfg["studies"] if a.all else a.study
    if not studies:
        ap.error("give --study or --all")
    models.configure(cfg["lr"])
    comparators.configure(a.netcleave_dir)
    assert cfg["kmer_overlap"] == training.KMER_OVERLAP
    ctx, _ = records.load_study_context(a.data)
    for S in studies:
        directional_and_published.run_study("published", S, cfg, ctx, a.out)


if __name__ == "__main__":
    main()
