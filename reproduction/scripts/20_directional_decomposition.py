#!/usr/bin/env python3
"""Directional decomposition: which side of the terminal bond carries the signal.

For one held-out study and each terminus: logistic regressions on the residues of the outward-facing side only (the side facing away from the ligand), of the interior-facing
side only (the side facing into the ligand) and of both sides, at half-widths 2, 4, 8, 12 and 16, with Bond-AUROC, Bond-AUPRC, Bond recall at the top 10 %, Local-AUC (also split by the
side of the non-observed neighbour) and the group bootstrap; paired differences between them; and the study-exclusive label set (positives only for
bonds of ligands that no other study lists). Same cohort, strict training set and primary labels as 10_terminal_predictability.py.

The random stream of the bootstrap is shared with 21_published_method_comparison.py (see ligterm_repro/directional_and_published.py), so this entry point also
reads the training windows of the released NetCleave data (--netcleave-dir), only to keep the stream aligned.

usage: 20_directional_decomposition.py --data <datasets folder> --netcleave-dir <folder> --out <results folder> (--study PMID [PMID ...] | --all) [--config FILE]
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
    ap.add_argument("--netcleave-dir", required=True, help="folder of the NetCleave fork (training/ is read; see 00_fetch_data.py)")
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
        directional_and_published.run_study("directional", S, cfg, ctx, a.out)


if __name__ == "__main__":
    main()
