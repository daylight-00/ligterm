#!/usr/bin/env python3
"""Peptide-level ranking metrics: how well do the bond scores of the models rank the observed peptides of a held-out study among the candidate peptides of the same protein?

For one held-out study (run) and, afterwards, for all 20 studies (summarize). Each model (LR-Both8, LR-Both16, LR-Comp8+Prox2 chosen on the training rows, GB-Both8 chosen on the training rows, and the
Paul-style motif) is refitted at both termini on the strict training set of the study (the training bonds are those of the bond-level experiments) and its bond scores are turned into peptide scores.
A peptide (start, end) has its N-terminal bond at start - 1 and its C-terminal bond at end; both bonds must lie inside the protein, and the length is 9 to 30 residues. Score variants: length_only
(a length prior fitted on the strict training units), n_bond, c_bond, both_bonds (the sum of the two bond scores) and both_bonds_and_length. Per evaluation protein and variant:
  peptide_auroc          the protein AUROC of the observed peptides against all negatives (ties 0.5); proteins without a positive or a negative are left out of the study mean;
  length_matched_auc     each observed peptide against the negatives of its own length (share of pairs won, ties 0.5); the study value is the macro mean over proteins;
  near_miss_auc          each observed peptide against its near-miss negatives (start and end each shifted by at most 3 residues); the pairs are pooled over the study.
Negatives: clean_negatives (primary: the negatives whose two boundaries carry the primary bond label 0) or all_negatives (every non-excluded peptide; sensitivity). Excluded, never negatives: peptides
equal to an observed unit of another study on the protein and other occurrences of an observed peptide inside the protein. Guards (a failure stops the study before anything is written): the training set
equals the bond-level one; with --reference-results, the bond-level macro Bond-AUROC and Local-AUC of every model reproduce the stored cells of the bond-level experiments (below 1e-5); the fast
peptide metrics equal an independent brute-force implementation on real proteins; the positives scored equal the candidate units.
The metric is described, not gated: the reference levels of the summary (length_matched_auc >= 0.60 in median, ...) are those of the protocol; a level that is not reached is reported as 'not reached'.

usage: 40_peptide_ranking_metrics.py --data <datasets folder> --out <results folder> (--study PMID [PMID ...] | --all) [--reference-results <results folder of 10, 20, 21, 30>] [--config FILE]
       40_peptide_ranking_metrics.py --summarize <results folder with the study files> --out <folder> [--config FILE]
"""
import argparse
import csv
import json
import math
import os
import platform
import sys
import time

import numpy as np
import scipy
import sklearn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from ligterm_repro import boosted, comparators, composition as c1, data, evaluation, metrics, models, records, training  # noqa: E402
from ligterm_repro import peptide_ranking as PE  # noqa: E402
from ligterm_repro import peptide_reference as PR  # noqa: E402

NEGATIVES = ("clean_negatives", "all_negatives")
MODE_OF = {"clean_negatives": "clean", "all_negatives": "all"}
VARIANTS = ("length_only", "n_bond", "c_bond", "both_bonds", "both_bonds_and_length")
VARIANT_OF = dict(zip(VARIANTS, PE.VARIANTS))                       # public variant -> the code of the metric module (LEN, N, C, NC, NCL)
KEYS = ("peptide_auroc", "length_matched_num", "length_matched_den", "near_miss_num", "near_miss_den")
KEY_OF = dict(zip(KEYS, ("F", "M_num", "M_den", "X_num", "X_den")))
# where the stored bond-level cell of a model lies (folder of the results, model name in the cell, fit)
REFERENCE = {("LR-Both8", None): "terminal_predictability", ("LR-Both16", None): "directional_decomposition", ("Paul-style motif", None): "published_method_comparison",
             ("LR-Comp8+Prox2", "cv"): "composition_position/linear", ("GB-Both8", "cv"): "composition_position/boosted"}


def mkey(m):
    return (m["model"], m.get("fit"))


def mrow(m):
    return {"model": m["model"], **({"fit": m["fit"]} if m.get("fit") else {})}


def lst(a):
    return [None if (isinstance(x, float) and not np.isfinite(x)) else (int(x) if isinstance(x, (int, np.integer)) else round(float(x), 10)) for x in a.tolist()]


def reference_cell(ref_dir, S, T, m):
    d = os.path.join(ref_dir, REFERENCE[mkey(m)], f"study_{S}.json")
    for r in json.load(open(d))["cells"]:
        if r["terminus"] == T and r["analysis"] == "primary" and r["kind"] == "model" and r["model"] == m["model"] and r.get("fit") == m.get("fit"):
            return r
    raise AssertionError(f"no stored bond-level cell of {mkey(m)} for study {S} terminus {T} in {d}")


def run_study(S, cfg, ctx, ref_dir, out_dir):
    log = records.Logger(S)
    t0 = time.time()
    PRIM, SEEN = cfg["primary_grouping"], S in cfg["seen"]
    LO, HI, DELTA = cfg["peptide_length"]["min"], cfg["peptide_length"]["max"], cfg["near_miss_shift"]
    MODELS = cfg["models"]
    assert S in cfg["studies"], f"{S} is not one of the {len(cfg['studies'])} studies of the analysis"
    ENV = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__)
    rows, sinfo = training.strict_rows(ctx, S, PRIM)
    training.check_disjoint(ctx, rows, S, PRIM)
    log(f"context: {ctx.NP} proteins, {ctx.nv} verified units, the study has {len(ctx.stud[S])}; strict training rows {len(rows)}", sinfo)
    # ---- the peptide-level populations of the study: candidates, positives, exclusions
    SR = ctx.stud[S]
    vp, vs, ve = ctx.vp[SR], ctx.vs[SR], ctx.ve[SR]
    ln = ve - vs + 1
    okb = (vs >= 2) & (ve <= ctx.L[vp] - 1)
    okl = (ln >= LO) & (ln <= HI)
    cand = okb & okl
    UN = dict(total=int(len(SR)), candidate=int(cand.sum()), dropped_bond_outside=int((~okb).sum()), dropped_length_only=int((okb & ~okl).sum()))
    dprot = np.unique(vp[cand])
    isd = np.zeros(ctx.NP, bool)
    isd[dprot] = True
    inS = np.zeros(ctx.nv, bool)
    inS[SR] = True
    posd, excl_other, excl_rep = {}, {}, {}
    for r in SR.tolist():
        p = int(ctx.vp[r])
        if isd[p]:
            posd.setdefault(p, []).append((int(ctx.vs[r]), int(ctx.ve[r])))
    for r in np.flatnonzero(isd[ctx.vp] & ~inS).tolist():                    # verified units of OTHER studies on the evaluation proteins
        excl_other.setdefault(int(ctx.vp[r]), set()).add((int(ctx.vs[r]), int(ctx.ve[r])))
    for r in SR.tolist():                                                   # other occurrences of a peptide of the study inside the same protein
        p = int(ctx.vp[r])
        if not isd[p]:
            continue
        sq, pep = ctx.seqs[ctx.names[p]], ctx.vseq[r]
        k = sq.find(pep)
        while k >= 0:
            if k + 1 != int(ctx.vs[r]):
                excl_rep.setdefault(p, set()).add((k + 1, k + len(pep)))
            k = sq.find(pep, k + 1)
    excl = {p: excl_other.get(p, set()) | excl_rep.get(p, set()) for p in set(excl_other) | set(excl_rep)}
    Ltr = ctx.ve[rows] - ctx.vs[rows] + 1
    cnt = np.array([int((Ltr == ell).sum()) for ell in range(LO, HI + 1)], np.float64)
    g = np.log((cnt + 1.0) / (cnt.sum() + len(cnt)))                       # the length prior, from the strict training units of this study
    UN.update(evaluation_proteins=int(len(dprot)), other_study_exclusions=int(sum(len(v) for v in excl_other.values())), repeat_exclusions=int(sum(len(v) for v in excl_rep.values())))
    log("peptide-level populations:", UN)
    assert UN["candidate"] > 0 and len(dprot) > 0, "no candidate positive peptide in this study"
    gid = ctx.G[PRIM][dprot]
    # ---- the bond scores of every model at both termini
    SC, ES, GUARD_REF, FI = {}, {}, [], []
    for T in "NC":
        rng_tr = np.random.default_rng(records.terminus_stream(cfg["seed_training"], S, T))        # the seeding of the bond-level experiments: the same training bonds
        es = evaluation.eval_setup(ctx, S, T)
        etok = data.tokens(ctx, es["bp"], es["bi"])
        epf = data.posfeat(es["bi"], ctx.L[es["bp"]])
        training.check_disjoint(ctx, rows, S, PRIM, eprot=es["eprot"], peptides=False)
        P, B, y, tinfo = training.build_train(ctx, rows, T, rng_tr, cfg["neg_ratio"])
        assert np.isin(dprot, es["eprot"]).all(), f"{T}: an evaluation protein of the peptide-level populations is not an evaluation protein of the bond-level run"
        if ref_dir:
            c0 = reference_cell(ref_dir, S, T, {"model": "LR-Both8"})
            assert (c0["train_pos"], c0["train_neg"], c0["train_proteins"], c0["train_units"]) == (tinfo["n_pos"], tinfo["n_neg"], tinfo["n_train_proteins"], int(len(rows))), "the training set differs from the bond-level run"
        ES[T] = es
        log(T, f"training {tinfo}; evaluation proteins {len(es['eprot'])}, bonds {int(es['nb'].sum())}")
        tok = data.tokens(ctx, P, B)
        pf = data.posfeat(B, ctx.L[P])
        grp = ctx.G[PRIM][P]
        sc = {}
        for m in MODELS:
            t1 = time.time()
            k, name = mkey(m), m["model"]
            if name in ("LR-Both8", "LR-Both16"):
                mm = models.window_model(int(name[7:]))
                sc[k] = mm.fit(tok, pf, y).score(etok, epf)
                info = dict(n_iter=mm.n_iter, hit_limit=bool(mm.n_iter >= cfg["lr"]["max_iter"]))
            elif name == "Paul-style motif":
                sc[k] = comparators.PaulMotif(T).fit(tok, pf, y).score(etok, epf)
                info = None
            elif name == "LR-Comp8+Prox2":
                mm = c1.FeatLR(name, T)
                raw = mm.raw(tok)
                Cs, _tab, hits = c1.select_C(mm, raw, tok, y, grp)
                st = mm.fit_rows(raw, y, np.arange(len(y)), Cs)
                assert st["C"] == Cs and Cs in c1.GRID
                sc[k] = mm.score_tok(st, etok)
                info = dict(C=Cs, n_iter=st["n_iter"], hit_limit=st["hit_limit"], cv_fits_at_limit=int(hits))
            elif name == "GB-Both8":
                fid = c1.fold_ids(grp)
                boosted.assert_folds(fid, grp, y)
                r = boosted.run_base(T, name, tok, y, fid, etok, boosted.GRID, boosted.FIXIT, boosted.MAXIT)
                n_it, z = r["regimes"]["cv"]
                sc[k] = z
                info = dict(n_iter=int(n_it), grid=list(boosted.GRID), at_edge=bool(n_it in (boosted.GRID[0], boosted.GRID[-1])))
            else:
                raise ValueError(f"unknown model {k}")
            assert np.isfinite(sc[k]).all(), f"non-finite scores of {k}"
            if info is not None:
                FI.append(dict(terminus=T, **mrow(m), value=info))
            log(T, f"{name} fitted and scored ({time.time() - t1:.0f}s)", info or "")
        if ref_dir:                                                         # guard: the bond-level macro AUROC and Local-AUC reproduce the stored cells of the bond-level experiments
            for m in MODELS:
                pp = metrics.per_protein(sc[mkey(m)], es["labs"]["primary"], es["ebase"], es["nb"])
                mine = dict(bond_auroc=float(np.nanmean(pp["bond_auroc"])), local_auc=float(pp["local_n"].sum() / pp["local_d"].sum()))
                fr = reference_cell(ref_dir, S, T, m)
                dd = {k_: abs(mine[k_] - fr[k_]) for k_ in mine}
                GUARD_REF.append(dict(terminus=T, **mrow(m), bond_auroc_difference=dd["bond_auroc"], local_auc_difference=dd["local_auc"]))
                assert all(v < cfg["bond_level_guard_abort"] for v in dd.values()), f"reproduction of the stored {mkey(m)} cell failed for {S} {T}: {mine} vs {[fr[k_] for k_ in mine]}"
            log(T, "guard: the bond-level cells of every model are reproduced (largest differences):", {r["model"]: float(f"{max(r['bond_auroc_difference'], r['local_auc_difference']):.2e}") for r in GUARD_REF if r["terminus"] == T})
        SC[T] = sc
    # ---- the label structure of the evaluation proteins (no score is read here)
    EB = {T: np.searchsorted(ES[T]["eprot"], dprot) for T in "NC"}
    for T in "NC":
        assert np.array_equal(ES[T]["eprot"][EB[T]], dprot)
    NPOS, NDROP = np.zeros(len(dprot), np.int64), np.zeros(len(dprot), np.int64)
    LP = ctx.L[dprot]

    def bond_slice(arr, T, j):
        """the bonds 1..L-1 of the j-th evaluation protein out of an array over the evaluation bonds of terminus T"""
        es, k = ES[T], EB[T][j]
        return arr[es["ebase"][k]:es["ebase"][k] + es["nb"][k]]

    def prep_protein(j):
        p = int(dprot[j])
        L = int(ctx.L[p])
        lab = {T: bond_slice(ES[T]["labs"]["primary"], T, j) for T in "NC"}
        assert len(lab["N"]) == len(lab["C"]) == L - 1
        prep = {"clean": PE.prepare(L, posd[p], excl.get(p, ()), LO, HI, lab["N"], lab["C"]), "all": PE.prepare(L, posd[p], excl.get(p, ()), LO, HI)}     # shared by every model
        assert prep["clean"]["pos"] == prep["all"]["pos"]
        assert all(lab["N"][s - 2] == 1 and lab["C"][e - 1] == 1 for s, e in prep["clean"]["pos"]), f"a positive peptide's own terminal bond does not carry the label 1 ({ctx.names[p]}): the bond slices do not match the peptide coordinates"
        return p, L, lab, prep

    for j in range(len(dprot)):
        prep_j = prep_protein(j)[3]
        NPOS[j], NDROP[j] = len(prep_j["clean"]["pos"]), prep_j["clean"]["dropped"]
    assert int(NPOS.sum()) == UN["candidate"], "the positives scored in the evaluation proteins differ from the candidate units"
    log(f"label structure of the {len(dprot)} evaluation proteins: {int(NPOS.sum())} positives (= the candidate units), {int(NDROP.sum())} other units of the study dropped by the candidate rule")
    # ---- the brute-force guard on real proteins: the two shortest evaluation proteins and the proteins of at most 200 residues with the most positives (a reserve is drawn until the clean
    # length-matched and near-miss populations have been compared), three models, both modes, all five variants. The fast implementation must equal the independent reference.
    GM = cfg["guard_models"]
    base_idx, reserve_idx = PR.select_guard_proteins(LP, NPOS)
    cmp_tot = {mo: dict(F_pairs=0.0, M_den=0.0, X_den=0.0, proteins=0) for mo in NEGATIVES}
    GUARDED = []

    def guard_protein(j, from_reserve):
        p, L, lab, prep = prep_protein(j)
        GUARDED.append(dict(acc=ctx.names[p], L=L, n_pos=len(prep["clean"]["pos"]), reserve=bool(from_reserve)))
        for gm in GM:
            sl = {T: bond_slice(SC[T][mkey(gm)], T, j) for T in "NC"}
            for neg in NEGATIVES:
                mode = MODE_OF[neg]
                res = PE.protein_metrics(sl["N"], sl["C"], L, None, None, g, LO, HI, DELTA, prepared=prep[mode])
                kw = dict(labN=lab["N"], labC=lab["C"]) if mode == "clean" else {}
                for v in PE.VARIANTS:
                    ref = PR.brute(sl["N"], sl["C"], L, posd[p], excl.get(p, ()), g, v, LO, HI, DELTA, **kw)
                    got = res[v]
                    ck = ("F", "M_num", "M_den", "X_num", "X_den")
                    bad = [k_ for k_ in ck if not ((np.isnan(got[k_]) and np.isnan(ref[k_])) or abs(got[k_] - ref[k_]) < 1e-9)] + [k_ for k_ in ("n_pos", "n_pos_dropped") if got[k_] != ref[k_]]
                    assert not bad, f"{mkey(gm)} {v} {mode}: protein {ctx.names[p]} (L={L}) differs from the brute-force reference in {bad}"
                    if gm is GM[0] and v == "NC":                           # the denominators depend neither on the model nor on the variant: count them once
                        cmp_tot[neg]["F_pairs"] += ref["n_pos"] * ref["n_neg"]
                        cmp_tot[neg]["M_den"] += ref["M_den"]
                        cmp_tot[neg]["X_den"] += ref["X_den"]
                        cmp_tot[neg]["proteins"] += 1

    def clean_compared():
        c_ = cmp_tot["clean_negatives"]
        return c_["F_pairs"] > 0 and c_["M_den"] > 0 and c_["X_den"] > 0

    t1 = time.time()
    for j in base_idx:
        guard_protein(j, False)
    for j in reserve_idx:
        if clean_compared():
            break
        guard_protein(j, True)
    assert all(cmp_tot["all_negatives"][k_] > 0 for k_ in ("F_pairs", "M_den", "X_den")), f"the brute-force guard compared nothing with all negatives: {cmp_tot['all_negatives']}"
    log(f"brute-force guard: {len(GUARDED)} proteins x {len(GM)} models x 2 negative sets x 5 variants equal the independent reference ({time.time() - t1:.0f}s); compared {cmp_tot}")
    # ---- metrics per evaluation protein
    MET = {neg: {mkey(m): {v: {k_: np.full(len(dprot), np.nan) for k_ in KEYS} for v in VARIANTS} for m in MODELS} for neg in NEGATIVES}
    t1 = time.time()
    for j in range(len(dprot)):
        p, L, lab, prep = prep_protein(j)
        for m in MODELS:
            sl = {T: bond_slice(SC[T][mkey(m)], T, j) for T in "NC"}
            assert len(sl["N"]) == len(sl["C"]) == L - 1
            for neg in NEGATIVES:
                res = PE.protein_metrics(sl["N"], sl["C"], L, None, None, g, LO, HI, DELTA, prepared=prep[MODE_OF[neg]])
                for v in VARIANTS:
                    for k_ in KEYS:
                        MET[neg][mkey(m)][v][k_][j] = res[VARIANT_OF[v]][KEY_OF[k_]]
        if (j + 1) % 1000 == 0 or j + 1 == len(dprot):
            log(f"metrics: {j + 1} of {len(dprot)} proteins, {len(MODELS)} models, both negative sets ({time.time() - t1:.0f}s)")
    SUMMARY = []
    for neg in NEGATIVES:
        for m in MODELS:
            for v in VARIANTS:
                m_ = MET[neg][mkey(m)][v]
                okF = np.isfinite(m_["peptide_auroc"])
                okM = m_["length_matched_den"] > 0
                SUMMARY.append(dict(negatives=neg, **mrow(m), variant=v, peptide_auroc=float(np.mean(m_["peptide_auroc"][okF])) if okF.any() else None,
                                    length_matched_auc=float(np.mean(m_["length_matched_num"][okM] / m_["length_matched_den"][okM])) if okM.any() else None,
                                    near_miss_auc=float(m_["near_miss_num"].sum() / m_["near_miss_den"].sum()) if m_["near_miss_den"].sum() > 0 else None,
                                    n_proteins_peptide_auroc=int(okF.sum()), n_proteins_length_matched=int(okM.sum())))
    rec = dict(study=S, seen=SEEN, units=UN, length_prior=[round(float(x), 10) for x in g], models=[mrow(m) for m in MODELS], variants=list(VARIANTS),
               proteins=dict(acc=[ctx.names[p] for p in dprot], group=[int(x) for x in gid], length=[int(x) for x in LP], n_pos=[int(x) for x in NPOS], n_pos_dropped=[int(x) for x in NDROP]),
               metrics=[dict(negatives=neg, **mrow(m), variant=v, **{k_: lst(MET[neg][mkey(m)][v][k_]) for k_ in KEYS}) for neg in NEGATIVES for m in MODELS for v in VARIANTS],
               summary=SUMMARY,
               guards=dict(positives_in_proteins=int(NPOS.sum()), units_candidate=UN["candidate"],
                           brute_force=dict(proteins=GUARDED, models=[mrow(m) for m in GM], compared=cmp_tot, clean_populations_compared=bool(clean_compared())),
                           **({"bond_level_reference": GUARD_REF} if ref_dir else {})),
               fit_info=FI, strict=sinfo, env=ENV)
    os.makedirs(out_dir, exist_ok=True)                                     # the vectors have 250,000 numbers: compact JSON, written atomically
    path = os.path.join(out_dir, f"study_{S}.json")
    with open(path + ".tmp", "w") as fh:
        json.dump(rec, fh, separators=(",", ":"), default=float)
        fh.write("\n")
    os.replace(path + ".tmp", path)
    log("written", path, f"({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------------------------------------------------------------------- summarize
def vec(rec, neg, m, var, key):
    row = rec["_index"][(neg, mkey(m), var)]
    return np.array([np.nan if x is None else x for x in row[key]], np.float64)


def study_stats(rec, neg, m, var):
    """study-level quantities of one model and variant"""
    F, Mn, Md, Xn, Xd = (vec(rec, neg, m, var, k) for k in KEYS)
    okF = np.isfinite(F)
    okM = Md > 0
    return dict(peptide_auroc=float(F[okF].mean()) if okF.any() else math.nan, length_matched_auc=float((Mn[okM] / Md[okM]).mean()) if okM.any() else math.nan,
                near_miss_auc=float(Xn.sum() / Xd.sum()) if Xd.sum() > 0 else math.nan, n_proteins_peptide_auroc=int(okF.sum()), n_proteins_length_matched=int(okM.sum()), n_near_miss_pairs=float(Xd.sum()))


def _ratio_boot(inv, G, num, den, draw):
    sn = np.bincount(inv, weights=num, minlength=G)
    sd = np.bincount(inv, weights=den, minlength=G)
    return sn[draw].sum(1) / np.maximum(sd[draw].sum(1), 1e-300)


def study_boot(rec, neg, m, B, seed):
    """percentile intervals (2.5, 97.5) of the three metrics for both_bonds, both_bonds_and_length and length_only, and of the paired difference peptide_auroc(both_bonds_and_length) -
    peptide_auroc(length_only), over resampled sequence groups (the same resamples for every quantity)"""
    gid = np.array(rec["proteins"]["group"])
    ug, inv = np.unique(gid, return_inverse=True)
    G = len(ug)
    rng = np.random.default_rng([seed, 0, int(rec["study"])])
    draw = rng.integers(0, G, size=(B, G))
    out = {}
    Fv = {v: vec(rec, neg, m, v, "peptide_auroc") for v in ("both_bonds", "both_bonds_and_length", "length_only")}
    for v in ("both_bonds", "both_bonds_and_length", "length_only"):
        F = Fv[v]
        okF = np.isfinite(F)
        Mn, Md, Xn, Xd = (vec(rec, neg, m, v, k) for k in ("length_matched_num", "length_matched_den", "near_miss_num", "near_miss_den"))
        okM = Md > 0
        out[("peptide_auroc", v)] = np.percentile(_ratio_boot(inv, G, np.where(okF, F, 0.0), okF.astype(float), draw), [2.5, 97.5]).tolist()
        out[("length_matched_auc", v)] = np.percentile(_ratio_boot(inv, G, np.where(okM, Mn / np.where(okM, Md, 1.0), 0.0), okM.astype(float), draw), [2.5, 97.5]).tolist()
        out[("near_miss_auc", v)] = np.percentile(_ratio_boot(inv, G, Xn, Xd, draw), [2.5, 97.5]).tolist()
    both = np.isfinite(Fv["both_bonds_and_length"]) & np.isfinite(Fv["length_only"])
    d = np.where(both, Fv["both_bonds_and_length"] - Fv["length_only"], 0.0)
    out[("peptide_auroc_gain", "both_bonds_and_length")] = np.percentile(_ratio_boot(inv, G, d, both.astype(float), draw), [2.5, 97.5]).tolist()
    return out


def check_records(recs, cfg, tol=1e-9):
    """consistency of the per-study records, before anything is aggregated -> list of problems (an empty list: all consistent)"""
    bad = []
    models_cfg = [mkey(m) for m in cfg["models"]]
    if set(recs) != set(cfg["studies"]):
        bad.append(f"the studies run differ from the {len(cfg['studies'])} studies: missing {sorted(set(cfg['studies']) - set(recs))}, extra {sorted(set(recs) - set(cfg['studies']))}")
    for s in sorted(recs):
        rec = recs[s]
        if str(rec.get("study")) != str(s):
            bad.append(f"{s}: the record says study {rec.get('study')}")
        if bool(rec.get("seen")) != (s in cfg["seen"]):
            bad.append(f"{s}: the seen flag differs from the configuration")
        if [mkey(m) for m in rec["models"]] != models_cfg or tuple(rec["variants"]) != VARIANTS:
            bad.append(f"{s}: the models or variants differ from the configuration")
        pr, u, g = rec["proteins"], rec["units"], rec["guards"]
        n = len(pr["group"])
        for k in ("acc", "length", "n_pos", "n_pos_dropped"):
            if len(pr[k]) != n:
                bad.append(f"{s}: proteins.{k} has {len(pr[k])} entries for {n} proteins")
        if len(set(pr["acc"])) != n:
            bad.append(f"{s}: an evaluation protein appears twice")
        if u["evaluation_proteins"] != n:
            bad.append(f"{s}: units.evaluation_proteins {u['evaluation_proteins']} against {n} proteins")
        if u["total"] != u["candidate"] + u["dropped_bond_outside"] + u["dropped_length_only"]:
            bad.append(f"{s}: the units are not candidate + dropped (bond outside) + dropped (length)")
        if sum(pr["n_pos"]) != u["candidate"] or g["positives_in_proteins"] != u["candidate"] or g["units_candidate"] != u["candidate"]:
            bad.append(f"{s}: the positives of the proteins differ from the candidate units")
        bf = g["brute_force"]
        if not bf["proteins"] or any(not (bf["compared"]["all_negatives"][k] > 0) for k in ("F_pairs", "M_den", "X_den")):
            bad.append(f"{s}: the brute-force guard is missing or compared no pairs with all negatives")
        index = {}
        for row in rec["metrics"]:
            index[(row["negatives"], (row["model"], row.get("fit")), row["variant"])] = row
        rec["_index"] = index
        ref = {}
        for neg in NEGATIVES:
            for m in cfg["models"]:
                for v in VARIANTS:
                    if (neg, mkey(m), v) not in index:
                        bad.append(f"{s}: no metrics of {neg} {mkey(m)} {v}")
                        continue
                    mm = {k: vec(rec, neg, m, v, k) for k in KEYS}
                    tag = f"{neg} {mkey(m)} {v}"
                    if any(len(a) != n for a in mm.values()):
                        bad.append(f"{s}: {tag}: a metric vector does not have {n} entries")
                        continue
                    F, Mn, Md, Xn, Xd = (mm[k] for k in KEYS)
                    fin = np.isfinite(F)
                    if not all(np.isfinite(a).all() for a in (Mn, Md, Xn, Xd)):
                        bad.append(f"{s}: {tag}: a non-finite pair count")
                        continue
                    if (F[fin] < -tol).any() or (F[fin] > 1 + tol).any() or (Md < 0).any() or (Xd < 0).any() or (Mn < -tol).any() or (Xn < -tol).any() or (Mn > Md + tol).any() or (Xn > Xd + tol).any():
                        bad.append(f"{s}: {tag}: a metric is outside its bounds")
                    if neg not in ref:
                        ref[neg] = (Md, Xd, fin)
                    elif not (np.array_equal(Md, ref[neg][0]) and np.array_equal(Xd, ref[neg][1]) and np.array_equal(fin, ref[neg][2])):
                        bad.append(f"{s}: {tag}: the pair counts or the undefined-AUROC pattern differ from the first model and variant (they depend on the labels and the exclusions only)")
                    if v == "length_only" and not np.allclose(Mn, 0.5 * Md, rtol=0, atol=tol):
                        bad.append(f"{s}: {tag}: the length-only score must tie within a length (length-matched AUC = 0.5 exactly)")
        if "clean_negatives" in ref and "all_negatives" in ref:
            if (ref["all_negatives"][0] < ref["clean_negatives"][0]).any() or (ref["all_negatives"][1] < ref["clean_negatives"][1]).any() or (ref["clean_negatives"][2] & ~ref["all_negatives"][2]).any():
                bad.append(f"{s}: the clean negatives are not a subset of all negatives")
        for row in rec["summary"]:
            st = study_stats(rec, row["negatives"], row, row["variant"])
            for a in ("peptide_auroc", "length_matched_auc", "near_miss_auc"):
                x, y = row[a], st[a]
                if (x is None) != (not math.isfinite(y)) or (x is not None and abs(x - y) > tol):
                    bad.append(f"{s}: {row['negatives']} {mkey(row)} {row['variant']}: the stored {a} {x} differs from the recomputed {y}")
    return bad


def summarize(results_dir, out_dir, cfg):
    studies = cfg["studies"]
    recs = {}
    for s in studies:
        with open(os.path.join(results_dir, f"study_{s}.json"), encoding="utf-8") as fh:
            recs[s] = json.load(fh)
    extra = sorted(f for f in os.listdir(results_dir) if f.startswith("study_") and f.endswith(".json") and f[len("study_"):-len(".json")] not in set(studies))
    bad = check_records(recs, cfg)
    if extra:
        bad.append(f"the results folder holds records of studies that are not in the configuration: {extra}")
    if bad:
        sys.exit("the study records fail the consistency checks; nothing is aggregated:\n  " + "\n  ".join(bad))
    unseen = [s for s in studies if s not in set(cfg["seen"])]
    ag = cfg["aggregation"]
    lv_cfg = ag["levels"]
    REF = cfg["reference_model"]
    B, SEED0 = ag["bootstrap"], ag["seed"]
    per_study, boot, coverage, guards = [], [], [], []
    for s in sorted(studies):                                              # the rows of the tables are in the order of the study identifiers (unseen and seen keep the order of the configuration)
        rec = recs[s]
        for neg in NEGATIVES:
            for m in cfg["models"]:
                for v in VARIANTS:
                    per_study.append(dict(study=s, seen=bool(rec["seen"]), negatives=neg, **mrow(m), variant=v, **study_stats(rec, neg, m, v)))
        for neg in NEGATIVES:
            for (metric, v), (lo, hi) in study_boot(rec, neg, REF, B, SEED0).items():
                boot.append(dict(study=s, negatives=neg, **mrow(REF), metric=metric, variant=v, lo=lo, hi=hi))
        u = rec["units"]
        coverage.append(dict(study=s, units_total=u["total"], candidate_positives=u["candidate"], dropped_bond_outside=u["dropped_bond_outside"], dropped_length_only=u["dropped_length_only"],
                             coverage=u["candidate"] / u["total"] if u["total"] else math.nan, evaluation_proteins=u["evaluation_proteins"], other_study_exclusions=u["other_study_exclusions"],
                             repeat_exclusions=u["repeat_exclusions"], seen=bool(rec["seen"])))
        gd = rec["guards"]
        bf = gd["brute_force"]
        gr = gd.get("bond_level_reference")
        guards.append(dict(study=s, max_bond_auroc_difference=max(r["bond_auroc_difference"] for r in gr) if gr else None, max_local_auc_difference=max(r["local_auc_difference"] for r in gr) if gr else None,
                           brute_force_proteins=len(bf["proteins"]), brute_force_from_reserve=sum(1 for p in bf["proteins"] if p.get("reserve")), compared_all_negatives=bf["compared"]["all_negatives"],
                           compared_clean_negatives=bf["compared"]["clean_negatives"], clean_populations_compared=bf["clean_populations_compared"], positives_equal_candidates=bool(gd["positives_in_proteins"] == gd["units_candidate"])))
    tot = {k: sum(c[k] for c in coverage) for k in ("units_total", "candidate_positives", "dropped_bond_outside", "dropped_length_only")}
    tot["coverage"] = tot["candidate_positives"] / tot["units_total"]

    def med(vals):
        a = np.array(vals, np.float64)
        a = a[np.isfinite(a)]
        return float(np.median(a)) if len(a) else math.nan

    ps = {(r["study"], r["negatives"], mkey(r), r["variant"]): r for r in per_study}
    medians, comparison = [], []
    for neg in NEGATIVES:
        for m in cfg["models"]:
            for v in VARIANTS:
                medians.append(dict(negatives=neg, **mrow(m), variant=v, **{q: med([ps[(s, neg, mkey(m), v)][q] for s in unseen]) for q in ("peptide_auroc", "length_matched_auc", "near_miss_auc")}))
        for m in cfg["models"]:
            for v, q in (("both_bonds_and_length", "peptide_auroc"), ("both_bonds", "length_matched_auc"), ("both_bonds", "near_miss_auc")):
                diffs = [ps[(s, neg, mkey(m), v)][q] - ps[(s, neg, mkey(REF), v)][q] for s in unseen]
                comparison.append(dict(negatives=neg, **mrow(m), variant=v, metric=q, median_difference_to_reference=med(diffs)))
    # the reference levels: the reference model, clean negatives, the unseen studies
    Mv = [ps[(s, "clean_negatives", mkey(REF), "both_bonds")]["length_matched_auc"] for s in unseen]
    FD = [ps[(s, "clean_negatives", mkey(REF), "both_bonds_and_length")]["peptide_auroc"] - ps[(s, "clean_negatives", mkey(REF), "length_only")]["peptide_auroc"] for s in unseen]
    Xv = [ps[(s, "clean_negatives", mkey(REF), "both_bonds")]["near_miss_auc"] for s in unseen]
    n = len(unseen)
    ok19 = n == 19
    lv = dict(length_matched_auc_median=med(Mv), length_matched_auc_studies_at_level=int(sum(1 for x in Mv if x >= lv_cfg["length_matched_auc_per_study"])), peptide_auroc_gain_median=med(FD),
              peptide_auroc_gain_studies_at_level=int(sum(1 for x in FD if x >= lv_cfg["peptide_auroc_gain_per_study"])), near_miss_auc_median=med(Xv), n_unseen=n)
    lv["length_matched_auc_reached"] = bool(ok19 and lv["length_matched_auc_median"] >= lv_cfg["length_matched_auc_median"] and lv["length_matched_auc_studies_at_level"] >= lv_cfg["min_studies"])
    lv["peptide_auroc_gain_reached"] = bool(ok19 and lv["peptide_auroc_gain_median"] >= lv_cfg["peptide_auroc_gain_median"] and lv["peptide_auroc_gain_studies_at_level"] >= lv_cfg["min_studies"])
    lv["near_miss_auc_reached"] = bool(ok19 and lv["near_miss_auc_median"] >= lv_cfg["near_miss_auc_median"])
    lv["evaluable"] = ok19
    res = dict(studies=sorted(studies), unseen=unseen, seen=[s for s in studies if s in set(cfg["seen"])], models=[mrow(m) for m in cfg["models"]], variants=list(VARIANTS), reference_model=mrow(REF), levels=lv_cfg,
               reference_levels=lv, per_study=per_study, bootstrap_intervals=boot, medians=medians, comparison_with_reference_model=comparison, coverage=coverage, coverage_pooled=tot, guards=guards)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "peptide_ranking_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
        fh.write("\n")
    with open(os.path.join(out_dir, "peptide_ranking_per_study.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["study", "seen", "negatives", "model", "fit", "variant", "peptide_auroc", "length_matched_auc", "near_miss_auc", "n_proteins_peptide_auroc", "n_proteins_length_matched", "n_near_miss_pairs"])
        w.writeheader()
        w.writerows({"fit": "", **r} for r in per_study)
    print("wrote peptide_ranking_summary.json and peptide_ranking_per_study.csv")
    print("reference levels (reference model, clean negatives):", lv)
    return res


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", help="datasets folder (run)")
    ap.add_argument("--out", required=True, help="results folder (run: one study_<pmid>.json per study) or the output folder of the summary")
    ap.add_argument("--config", default=os.path.join(here, "..", "configs", "peptide_ranking.json"))
    ap.add_argument("--study", nargs="+")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--reference-results", help="results folder with the bond-level results of scripts 10, 20, 21 and 30 (the bond-level guard)")
    ap.add_argument("--summarize", metavar="RESULTS", help="aggregate the study files of this folder into the summary tables")
    a = ap.parse_args()
    cfg = records.load_config(a.config)
    if a.summarize:
        summarize(a.summarize, a.out, cfg)
        return
    studies = cfg["studies"] if a.all else a.study
    if not studies or not a.data:
        ap.error("give --data and --study or --all (or --summarize)")
    models.configure(cfg["lr"])
    assert cfg["kmer_overlap"] == training.KMER_OVERLAP
    ctx, _ = records.load_study_context(a.data)
    for S in studies:
        if os.path.exists(os.path.join(a.out, f"study_{S}.json")):
            sys.exit(f"refusing to overwrite {a.out}/study_{S}.json")
        run_study(S, cfg, ctx, a.reference_results, a.out)


if __name__ == "__main__":
    main()
