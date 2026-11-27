"""Reading of the datasets and configuration, and the public schema of the per-study result records.

A result record is {"study", "study_info"/"strict", "diag", "cells", ...}; a cell is one row of a long table:
study, seen, terminus, analysis, kind ("model" or "paired_difference"), model (or model_a / model_b), optional fit
("cv" or "fixed"), counts and metrics (bond_auroc, bond_auprc, bond_recall_at_10pct, local_auc, with _lo and _hi).
"""
import csv
import json
import os
import sys
import time

from .data import load_ctx
from .training import lab_key

BASE_METRICS = ("bond_auroc", "bond_auprc", "bond_recall_at_10pct", "local_auc")
SIDE_METRICS = ("local_auc_interior", "local_auc_outward")


def load_config(path):
    return json.load(open(path))


def load_study_context(data_dir):
    """The context of the terminal-bond analyses from the public datasets folder: (context, last-author keys of the studies)."""
    csv.field_size_limit(10**8)
    keys = {r["pmid"]: lab_key(r["authors"]) for r in csv.DictReader(open(os.path.join(data_dir, "studies.tsv"), encoding="utf-8"), delimiter="\t")}
    ctx = load_ctx(os.path.join(data_dir, "units.tsv.gz"), os.path.join(data_dir, "protein_sequences.json"), os.path.join(data_dir, "sequence_groups.tsv.gz"))
    return ctx, keys


def terminus_stream(seed, study, T):
    """the seed sequence of the random stream of a (study, terminus): the training bonds and the bootstrap of the stored runs depend on it."""
    return [seed, int(study), 0 if T == "N" else 1]


def cell(study, seen, T, analysis, model, bt, extra=None, post_date=None, fit=None, side_split=False):
    d = dict(study=study, seen=seen)
    if post_date is not None:
        d["post_date"] = post_date
    d.update(terminus=T, analysis=analysis, kind="model", model=model)
    if fit:
        d["fit"] = fit
    d.update(n_proteins=bt["n_proteins"], n_groups=bt["n_groups"], n_pos=bt["n_pos"])
    if side_split:
        d["local_pairs_interior"], d["local_pairs_outward"] = bt["local_pairs_interior"], bt["local_pairs_outward"]
    for k in BASE_METRICS + (SIDE_METRICS if side_split else ()):
        d[k], d[k + "_lo"], d[k + "_hi"] = bt[k]
    if extra:
        d.update(extra)
    return d


def paired(study, seen, T, analysis, a, b, est, lo, hi, post_date=None, fit_a=None, fit_b=None):
    d = dict(study=study, seen=seen)
    if post_date is not None:
        d["post_date"] = post_date
    d.update(terminus=T, analysis=analysis, kind="paired_difference", model_a=a)
    if fit_a:
        d["fit_a"] = fit_a
    d["model_b"] = b
    if fit_b:
        d["fit_b"] = fit_b
    d.update(bond_auroc=est, bond_auroc_lo=lo, bond_auroc_hi=hi)
    return d


def write_study(out_dir, study, record):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"study_{study}.json")
    with open(path + ".tmp", "w") as fh:
        json.dump(record, fh, indent=1, default=float)
        fh.write("\n")
    os.replace(path + ".tmp", path)                    # atomic: an interrupted run leaves no half-written file
    return path


class Logger:
    def __init__(self, study):
        self.t0, self.study = time.time(), study

    def __call__(self, *a):
        print(f"[{time.time() - self.t0:5.0f}s] study={self.study}", *a, flush=True)


def add_lib_to_path(script_file):
    """scripts/ sits next to lib/: make `ligterm_repro` importable without installation."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(script_file)), "..", "lib"))
