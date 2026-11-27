#!/usr/bin/env python3
"""Exact-window control: does the window of NetCleave itself explain the comparison with the window models?

C terminus only (NetCleave is a C-terminal method), one held-out study. Same cohort, strict training set, training stream, label set and per-protein metrics as
21_published_method_comparison.py (the label set complete_standard_4plus3_window: every evaluated bond has the 4+3 window of NetCleave). Models: LR-Both(4+3) =
the window-model recipe on the seven residues of the NetCleave window (the last four ligand residues and the next three), LR-Both4 = the model on 4+4 residues, and
the NetCleave-style network trained on the protocol negatives (refitted here because the pairing needs its per-protein values). Nothing is tuned.

The NetCleave fork is third-party material read from --netcleave-dir (see 00_fetch_data.py).

usage: 22_exact_window_control.py --data <datasets folder> --netcleave-dir <folder> --out <results folder> (--study PMID [PMID ...] | --all) [--config FILE]
"""
import argparse
import os
import platform
import sys

import numpy as np
import scipy
import sklearn
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from ligterm_repro import comparators, data, evaluation, metrics, models, records, training  # noqa: E402


def run_study(S, cfg, ctx, out_dir):
    log = records.Logger(S)
    T, PRIM, B_ = cfg["terminus"], cfg["primary_grouping"], cfg["bootstrap"]
    ANALYSIS = cfg["analysis"]
    SEEN = S in cfg["seen"]
    ENV = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, torch=torch.__version__)
    rows, sinfo = training.strict_rows(ctx, S, PRIM)
    training.check_disjoint(ctx, rows, S, PRIM)
    rng_tr = np.random.default_rng(records.terminus_stream(cfg["seed_training"], S, T))        # the seeding of the C terminus in the earlier experiments: the same training bonds
    rng_bs = np.random.default_rng(records.terminus_stream(cfg["seed_bootstrap"], S, T))
    es = evaluation.eval_setup(ctx, S, T)
    etok = data.tokens(ctx, es["bp"], es["bi"])
    epf = data.posfeat(es["bi"], ctx.L[es["bp"]])
    gid = ctx.G[PRIM][es["eprot"]]
    training.check_disjoint(ctx, rows, S, PRIM, eprot=es["eprot"], peptides=False)
    P, B, y, tinfo = training.build_train(ctx, rows, T, rng_tr, cfg["neg_ratio"])
    primary = es["labs"]["primary"]
    complete = evaluation.complete_window(es["bi"], ctx.L[es["bp"]], etok)
    VW = np.where(complete, primary, -1).astype(np.int8)
    D = dict(terminus=T, eval_proteins=int(len(es["eprot"])), eval_bonds=int(es["nb"].sum()), n_pos=es["n_pos"], vw_pos_bonds=int((VW == 1).sum()), vw_neg_bonds=int((VW == 0).sum()),
             vw_excluded_bonds=int((VW == -1).sum()), n_eval_groups=int(len(np.unique(gid))), train=dict(pos=tinfo["n_pos"], neg=tinfo["n_neg"], proteins=tinfo["n_train_proteins"], units=int(len(rows))))
    # ---- models
    nc = cfg["netcleave"]
    m_ref, m_new = models.window_model(4), models.exact_window_model()
    m_nc = comparators.NetCleaveNet(T, "protocol", nc["max_epochs"], nc["seed"])
    fitted = {m.name: m for m in (m_ref, m_new, m_nc)}
    assert list(cfg["models"]) == [m_new.name, m_ref.name, m_nc.name] and set(fitted) == set(cfg["models"])
    log(D, list(fitted))
    scores, fi = models.fit_score(ctx, P, B, y, [m_ref, m_new, m_nc], etok, epf)
    RES = []
    pps = {}
    for n in cfg["models"]:
        pps[n] = metrics.per_protein(scores[n], VW, es["ebase"], es["nb"], T=T)
        RES.append(records.cell(S, SEEN, T, ANALYSIS, n, metrics.boot(pps[n], gid, rng_bs, B_)))
    for d in cfg["pairs"]:
        est, lo, hi = metrics.boot_diff(pps[d["a"]], pps[d["b"]], gid, rng_bs, B_)
        RES.append(records.paired(S, SEEN, T, ANALYSIS, d["a"], d["b"], est, lo, hi))
    WEIGHTS = [dict(terminus=T, model=m_new.name, value=np.round(np.asarray(m_new.W, np.float64), 4).tolist())]
    D["fit_info"] = fi
    path = records.write_study(out_dir, S, dict(study=S, diag=[D], cells=RES, weights=WEIGHTS, strict=sinfo, env=ENV))
    log("written", path)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="datasets folder")
    ap.add_argument("--netcleave-dir", required=True, help="folder of the NetCleave fork (code/QSAR_table.csv, training/)")
    ap.add_argument("--out", required=True, help="results folder (one study_<pmid>.json per study)")
    ap.add_argument("--config", default=os.path.join(here, "..", "configs", "exact_window_control.json"))
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
        run_study(S, cfg, ctx, a.out)


if __name__ == "__main__":
    main()
