#!/usr/bin/env python3
"""Composition-position decomposition: what the composition and the position of the eight interior residues contribute to the terminal-bond signal.

For one held-out study and both termini, on the eight interior residues of the bond: six models (LR-Full8 position-specific, LR-Comp8 composition only, LR-Cys8
the cysteine count alone, LR-Prox2 the two positions next to the bond, LR-Dist6 the six distal positions, LR-Comp8+Prox2) in the linear family, or eight
models in the boosted-tree family (GB-In8 = positions and counts of the interior residues, the reference; GB-Both8, GB-Out8, GB-Pos8, GB-Comp8, GB-Prox2,
GB-Comp8+Prox2, GB-Dist6), each at a hyper-parameter fixed in advance (fit "fixed") and chosen on the training rows only (fit "cv"), against the window models
LR-Both8, LR-Out8 and LR-In8. Bond-AUROC, Bond-AUPRC, Bond recall at the top 10 %, Local-AUC with the group bootstrap, paired differences, and the same without the
evaluation proteins of the largest sequence group. Same cohort, strict training set and primary labels as 10_terminal_predictability.py.

usage: 30_composition_position.py --family {linear,boosted} --data <datasets folder> --out <results folder> (--study PMID [PMID ...] | --all) [--config FILE]
"""
import argparse
import os
import platform
import sys

import numpy as np
import scipy
import sklearn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from ligterm_repro import boosted, composition as c1, data, evaluation, metrics, models, records, training  # noqa: E402

NG = "primary_without_largest_group"


def run_study(family, S, cfg, ctx, out_dir):
    fam = cfg[family]
    log = records.Logger(S)
    PRIM, B_ = cfg["primary_grouping"], cfg["bootstrap"]
    SEEN = S in cfg["seen"]
    rows, sinfo = training.strict_rows(ctx, S, PRIM)
    training.check_disjoint(ctx, rows, S, PRIM)
    cg = np.bincount(ctx.G[PRIM])
    GIANT = int(np.argmax(cg))
    assert int(cg[GIANT]) == cfg["largest_group_size"], f"the largest sequence group has {int(cg[GIANT])} proteins, expected {cfg['largest_group_size']}"
    bases = c1.BASES if family == "linear" else boosted.BASES
    refs_names = cfg["ref_models"]
    NEW = [(b, r) for b in bases for r in ("fixed", "cv") if not (family == "linear" and b == "LR-Full8" and r == "fixed")]   # LR-Full8 at the fixed constant is the window model LR-In8
    EVAL = [(n, None) for n in refs_names] + NEW
    RES, DIAG, WEIGHTS, CVINFO, FITINFO, NIT = [], [], [], [], [], {}

    def rec(T, analysis, key, bt):
        name, fit = key
        extra = {"n_iter": int(NIT[key])} if family == "boosted" and key in NEW else None
        RES.append(records.cell(S, SEEN, T, analysis, name, bt, extra=extra, fit=fit, side_split=True))

    for T in ("N", "C"):
        rng_tr = np.random.default_rng(records.terminus_stream(cfg["seed_training"], S, T))        # the seeding of the earlier experiments: the same training bonds
        rng_bs = np.random.default_rng(records.terminus_stream(fam["seed_bootstrap"], S, T))
        es = evaluation.eval_setup(ctx, S, T)
        etok = data.tokens(ctx, es["bp"], es["bi"])
        epf = data.posfeat(es["bi"], ctx.L[es["bp"]])
        gid = ctx.G[PRIM][es["eprot"]]
        keep = gid != GIANT
        training.check_disjoint(ctx, rows, S, PRIM, eprot=es["eprot"], peptides=False)
        P, B, y, tinfo = training.build_train(ctx, rows, T, rng_tr, cfg["neg_ratio"])
        primary = es["labs"]["primary"]
        D = dict(terminus=T, eval_proteins=int(len(es["eprot"])), eval_bonds=int(es["nb"].sum()), n_pos=es["n_pos"], n_eval_groups=int(len(np.unique(gid))), eval_proteins_in_largest_group=int((~keep).sum()),
                 train=dict(pos=tinfo["n_pos"], neg=tinfo["n_neg"], proteins=tinfo["n_train_proteins"], units=int(len(rows))))
        assert D["eval_proteins_in_largest_group"] == int((gid == GIANT).sum())
        tok = data.tokens(ctx, P, B)
        grp = ctx.G[PRIM][P]
        fid = c1.fold_ids(grp)
        for f in range(c1.NFOLD):
            assert not np.isin(grp[fid != f], grp[fid == f]).any() and 0 < y[fid == f].sum() < (fid == f).sum(), "a CV fold shares a group with its training part or has a single class"
        if family == "boosted":
            boosted.assert_folds(fid, grp, y)
        D["cv_fold_rows"] = [int((fid == f).sum()) for f in range(c1.NFOLD)]
        D["cv_fold_pos"] = [int(y[fid == f].sum()) for f in range(c1.NFOLD)]
        log(T, D)
        # ---- the window models (the references, fixed constant)
        refs = [models.window_model(8), models.side_model(T, "outward", 8), models.side_model(T, "interior", 8)]
        assert [m.name for m in refs] == list(refs_names)
        sc, fi = models.fit_score(ctx, P, B, y, refs, etok, epf)
        scores = {(n, None): sc[n] for n in refs_names}
        rows_all = np.arange(len(y))
        # ---- the new models
        if family == "linear":
            chk_rows = np.random.default_rng(records.terminus_stream(fam["seed_bootstrap"], S, T) + [7]).choice(len(y), size=min(20000, len(y)), replace=False)   # training rows for the scoring self-check
            mP = c1.FeatLR("LR-Full8", T)
            rawP = mP.raw(tok)
            stP = mP.fit_rows(rawP, y, rows_all, c1.FIXC)
            refIN = refs[2]                                          # LR-In8, fitted above on the same rows: LR-Full8 at the fixed constant must have its weights
            assert float(np.max(np.abs(stP["W"] - refIN.W.astype(np.float64)))) < 1e-6 and abs(stP["b"] - float(refIN.b)) < 1e-6, "LR-Full8 at the fixed constant differs from LR-In8"
            for b in bases:
                m = c1.FeatLR(b, T)
                raw = rawP if b == "LR-Full8" else m.raw(tok)
                if b != "LR-Full8":
                    st = m.fit_rows(raw, y, rows_all, c1.FIXC)
                    assert st["C"] == c1.FIXC
                    scores[(b, "fixed")] = m.score_tok(st, etok)
                    assert np.isfinite(scores[(b, "fixed")]).all()
                    m.check_scoring(raw, st, tok, chk_rows)
                    FITINFO.append(dict(terminus=T, model=b, fit="fixed", value=dict(C=c1.FIXC, n_iter=st["n_iter"], hit_limit=st["hit_limit"])))
                    if b in fam["weights_for"]:
                        WEIGHTS.append(dict(terminus=T, model=b, fit="fixed", value=dict(wc=None if st["wc"] is None else np.round(st["wc"], 4).tolist(), W=None if st["W"] is None else np.round(st["W"], 4).tolist(), b=round(st["b"], 4))))
                Cs, tab, cv_hits = c1.select_C(m, raw, tok, y, grp)
                st = m.fit_rows(raw, y, rows_all, Cs)
                assert st["C"] == Cs and Cs in c1.GRID
                scores[(b, "cv")] = m.score_tok(st, etok)
                assert np.isfinite(scores[(b, "cv")]).all()
                CVINFO.append(dict(terminus=T, model=b, value=dict(C=Cs, grid=list(c1.GRID), loss=[float(v) for v in tab], at_edge=bool(Cs in (c1.GRID[0], c1.GRID[-1])), cv_fits_at_limit=int(cv_hits))))
                m.check_scoring(raw, st, tok, chk_rows)
                FITINFO.append(dict(terminus=T, model=b, fit="cv", value=dict(C=Cs, n_iter=st["n_iter"], hit_limit=st["hit_limit"])))
                if b in fam["weights_for"]:
                    WEIGHTS.append(dict(terminus=T, model=b, fit="cv", value=dict(wc=None if st["wc"] is None else np.round(st["wc"], 4).tolist(), W=None if st["W"] is None else np.round(st["W"], 4).tolist(), b=round(st["b"], 4))))
                log(T, f"{b}: C*={Cs}; lbfgs iterations {st['n_iter']}")
            for n, f_ in fi.items():
                FITINFO.append(dict(terminus=T, model=n, value=f_))
        else:
            for b in bases:
                r = boosted.run_base(T, b, tok, y, fid, etok, grid=boosted.GRID, fix_iter=boosted.FIXIT, maxit=boosted.MAXIT)
                for suf, (n_it, z) in r["regimes"].items():
                    scores[(b, suf)], NIT[(b, suf)] = z, n_it                      # the number of iterations travels with the scores
                    assert np.isfinite(z).all()
                nstar = int(r["regimes"]["cv"][0])
                CVINFO.append(dict(terminus=T, model=b, value=dict(n_iter=nstar, grid=list(boosted.GRID), loss=[float(v) for v in r["loss"]], at_edge=bool(nstar in (boosted.GRID[0], boosted.GRID[-1])), fix_iter=boosted.FIXIT)))
                log(T, f"{b}: iterations*={nstar} of {list(boosted.GRID)}")
        # ---- metrics, intervals and paired differences: the primary labels, and without the evaluation proteins of the largest sequence group
        pps, ppg = {}, {}
        for key in EVAL:
            pps[key] = metrics.per_protein(scores[key], primary, es["ebase"], es["nb"], T=T)
            rec(T, "primary", key, metrics.boot(pps[key], gid, rng_bs, B_))
            ppg[key] = c1.mask_pp(pps[key], keep)
            n_out = int((np.isfinite(pps[key]["bond_auroc"]) & (gid == GIANT)).sum())
            if np.isfinite(ppg[key]["bond_auroc"]).any():
                btg = metrics.boot(ppg[key], gid, rng_bs, B_)
                assert btg["n_proteins"] == int(np.isfinite(pps[key]["bond_auroc"]).sum()) - n_out, "the largest-group evaluation does not remove exactly the proteins of the largest group"
                rec(T, NG, key, btg)
        ref_in = ("LR-In8", None)
        D["pos_without_largest_group"] = int(ppg[ref_in]["npos"][np.isfinite(ppg[ref_in]["bond_auroc"])].sum())
        for d in fam["pairs"]:
            a, b = (d["a"], d.get("fit_a")), (d["b"], d.get("fit_b"))
            est, lo, hi = metrics.boot_diff(pps[a], pps[b], gid, rng_bs, B_)
            RES.append(records.paired(S, SEEN, T, "primary", a[0], b[0], est, lo, hi, fit_a=a[1], fit_b=b[1]))
        for d in fam["pairs_without_largest_group"]:
            a, b = (d["a"], d.get("fit_a")), (d["b"], d.get("fit_b"))
            if np.isfinite(ppg[a]["bond_auroc"]).any():
                est, lo, hi = metrics.boot_diff(ppg[a], ppg[b], gid, rng_bs, B_)
                RES.append(records.paired(S, SEEN, T, NG, a[0], b[0], est, lo, hi, fit_a=a[1], fit_b=b[1]))
        DIAG.append(D)
        log(T, "done")
    ENV = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__)
    rec_out = dict(study=S, diag=DIAG, cells=RES)
    if family == "linear":
        rec_out["weights"] = WEIGHTS
    rec_out["cv"] = CVINFO
    if family == "linear":
        rec_out["fit_diagnostics"] = FITINFO
    rec_out.update(strict=sinfo, env=ENV)
    path = records.write_study(out_dir, S, rec_out)
    log("written", path)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--family", required=True, choices=("linear", "boosted"))
    ap.add_argument("--data", required=True, help="datasets folder")
    ap.add_argument("--out", required=True, help="results folder (one study_<pmid>.json per study)")
    ap.add_argument("--config", default=os.path.join(here, "..", "configs", "composition_position.json"))
    ap.add_argument("--study", nargs="+")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    cfg = records.load_config(a.config)
    studies = cfg["studies"] if a.all else a.study
    if not studies:
        ap.error("give --study or --all")
    models.configure(cfg["lr"])
    assert cfg["kmer_overlap"] == training.KMER_OVERLAP
    assert (c1.FIXC, c1.MAXITER, c1.TOL) == (cfg["lr"]["C_window_lr"], cfg["lr"]["max_iter"], cfg["lr"]["tol"]), "the fixed constant of the linear family is the one of the window models"
    ctx, _ = records.load_study_context(a.data)
    for S in studies:
        run_study(a.family, S, cfg, ctx, a.out)


if __name__ == "__main__":
    main()
