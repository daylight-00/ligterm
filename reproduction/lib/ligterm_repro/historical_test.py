"""Historical test: the models of the final predictor and their comparators, refitted for two studies that were reserved from model development and evaluated once, and the aggregation of the records.

The two studies are a previously reserved, once-evaluated historical test. The result is the measured performance of the evaluated refits beside the development results; it is NOT independent evidence for the final
predictor (both studies share peptides with the development studies and with each other, and their units are ordinary training units of every development run), and there is no pass rule: paired differences are descriptions.

`run`: ONE study in ONE training regime (group_disjoint: sequence groups u30 plus the 9-mer exclusion, the regime of every development run; lab_excluded: the same plus the exclusion of studies of the same laboratory;
identity_50_grouping: groups at 50% identity plus the 9-mer and the laboratory exclusion, the sensitivity regime). The training rows are those of `training.strict_rows` (and `training.lab_exclusion`); their unit list must equal the label-only list of the
configuration (its sizes and SHA-256). Five models are refitted at both termini on the training bonds and scored on every bond of the evaluation proteins of the study: LigTerm-GB (boosted, one-hot window of 16 residues each side as 32
categorical columns plus the residue counts of both sides, the number of iterations chosen on three group-disjoint folds), LR-Both8 and LR-Both16 (logistic, fixed penalty), LR-Both16 with the penalty chosen on the folds (printed LR-Both16-CV),
and the Paul-style motif. The peptide-level populations are those of the peptide-ranking metrics (clean negatives; score variants length_only, both_bonds and both_bonds_and_length); the brute-force guard on real proteins runs before any metric is stored.
The record is written once, as a compact JSON file, with the per-protein vectors rounded to 10 decimals.

`summarize`: the consistency checks of the six records (any failure stops the aggregation), then per study and pooled values (the pooled value of a cell is the unweighted mean of the two study-level values), their 95% percentile bootstrap
intervals over sequence groups across both studies (2,000 resamples; seed [20261116, 3, slot, 0 / 1 / 2 for the N terminus / the C terminus / the peptide populations], slot 0 for the pooled interval and the PMID for the interval of one study),
five descriptive paired differences, the selections, the development medians beside them (inputs: the public results of the bond-level experiments and of the peptide-ranking metrics, and the stored medians of the other two models).

The arithmetic, the order of the random draws and the rounding are those of the stored records.

usage: 44_development_benchmark.py historical-test run --data <datasets folder> --out <folder> --study PMID --regime NAME [--config FILE]
       44_development_benchmark.py historical-test summarize --records <folder> --out <folder> [--config FILE] [--reference-results <results folder>] [--reference-medians FILE]
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import resource
import sys
import time

import numpy as np
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression

from . import boosted, comparators, composition as c1, data, evaluation, metrics, models, records, training
from . import peptide_ranking as PE
from . import peptide_reference as PR

# ------------------------------------------------------------------------------------------------------------------------------ names
# a model is addressed inside this module by a string: its name, followed by '/cv' when its penalty is chosen on the folds (the printed name of that model is LR-Both16-CV); records and tables carry model and fit as separate fields
MODELS = ("LigTerm-GB", "LR-Both8", "LR-Both16", "LR-Both16/cv", "Paul-style motif")
REGIMES = ("group_disjoint", "lab_excluded", "identity_50_grouping")
VARIANTS = ("length_only", "both_bonds", "both_bonds_and_length")
VARIANT_CODE = dict(zip(VARIANTS, ("LEN", "NC", "NCL")))                       # the codes of the metric module peptide_ranking
KEYS = ("peptide_auroc", "length_matched_num", "length_matched_den", "near_miss_num", "near_miss_den")
KEY_CODE = dict(zip(KEYS, ("F", "M_num", "M_den", "X_num", "X_den")))
BOND_Q = (("N", "bond_auroc"), ("C", "bond_auroc"), ("N", "local_auc"), ("C", "local_auc"))
PEP_Q = tuple((v, k) for v in VARIANTS for k in ("peptide_auroc", "length_matched_auc", "near_miss_auc"))
QUANT = BOND_Q + PEP_Q
GRID, ITER_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0), (50, 100, 200, 400, 800)
B_BOOT, SEED0, STAGE = 2000, 20261116, 3
CONTRASTS = (("LigTerm-GB", "LR-Both16"), ("LigTerm-GB", "LR-Both16/cv"), ("LR-Both16/cv", "LR-Both16"), ("LR-Both16", "LR-Both8"), ("LR-Both16", "Paul-style motif"))
NEG_RATIO = 6                                           # negatives per positive of every training set
LR16_CV, GB = "LR-Both16/cv", "LigTerm-GB"                   # the two models with a recorded selection
WIN = 16
CSV_FIELDS = ["regime", "study", "model", "fit", "terminus", "variant", "metric", "estimate", "ci_lo", "ci_hi"]


def mrow(key):
    name, _, fit = key.partition("/")
    return {"model": name, **({"fit": fit} if fit else {})}


def mkey(m):
    return m["model"] + ("/" + m["fit"] if m.get("fit") else "")


def pair_row(a, b):
    ra, rb = mrow(a), mrow(b)
    d = {"model_a": ra["model"]}
    if "fit" in ra:
        d["fit_a"] = ra["fit"]
    d["model_b"] = rb["model"]
    if "fit" in rb:
        d["fit_b"] = rb["fit"]
    return d


def qfields(q):
    return {"terminus": q[0], "metric": q[1]} if q[0] in ("N", "C") else {"variant": q[0], "metric": q[1]}


def display(key):
    name, _, fit = key.partition("/")
    return name + ("-CV" if fit == "cv" else "")


def lst(a):
    return [None if (isinstance(x, float) and not np.isfinite(x)) else (int(x) if isinstance(x, (int, np.integer)) else round(float(x), 10)) for x in np.asarray(a).tolist()]


def gkey(regime):
    return "group_m50" if regime == "identity_50_grouping" else "group_u30"


# ------------------------------------------------------------------------------------------------------ the label-only training lists
def unit_keys(c, rows):
    return sorted(f"{c.names[int(c.vp[r])]}:{int(c.vs[r])}:{int(c.ve[r])}:{c.vseq[r]}" for r in np.asarray(rows).tolist())


def list_hash(keys):
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def regime_rows(c, S, regime, inS, keys):
    """-> (rows, info) of one regime; the same-laboratory exclusion is the rule of training.lab_exclusion (5% share) applied to the rows of the grouping"""
    grouping = regime["grouping"]
    rows, sinfo = training.strict_rows(c, S, grouping)
    info = dict(grouping=grouping, strict=sinfo, n_after_strict=int(len(rows)))
    if regime["lab_exclusion"]:
        X, why = training.lab_exclusion(c, S, inS, keys)
        bad = np.zeros(c.nv, bool)
        for s2 in X:
            bad[c.stud[s2]] = True
        r2 = np.setdiff1d(rows, np.flatnonzero(bad))
        info.update(lab_excluded_studies=[dict(study=k, **v) for k, v in sorted(why.items())], lab_excluded_units_removed=int(len(rows) - len(r2)))
        rows = r2
    return rows, info


# ------------------------------------------------------------------------------------------- the boosted model and the sparse penalty head
class SparseHead:
    """LR-Both16 with the penalty chosen on the folds, for composition.select_C: lbfgs logistic regression on the sparse one-hot design (the window-model recipe at a given C).
    select_C is called as select_C(head, X, X, y, groups): the second X plays the part of the token rows."""

    def __init__(self, maxiter, tol):
        self.maxiter, self.tol = maxiter, tol

    def fit_rows(self, raw, y, rows, C):
        m = LogisticRegression(C=C, solver="lbfgs", max_iter=self.maxiter, tol=self.tol).fit(raw[rows], y[rows])
        n_iter = int(np.max(m.n_iter_))
        return dict(W=m.coef_.ravel().astype(np.float64), b=float(m.intercept_[0]), n_iter=n_iter, hit_limit=bool(n_iter >= self.maxiter), model=m)

    def score_tok(self, st, tok):
        return np.asarray(tok @ st["W"]).ravel() + st["b"]


NSYM = 21


def loc_scores(coef, b, tok):
    """the score of the one-hot window logistic model on token rows: b + the sum over the 32 positions of W[position, token]"""
    W = np.asarray(coef, np.float64).reshape(2 * WIN, NSYM)
    return float(b) + W[np.arange(2 * WIN)[None, :], tok[:, :2 * WIN].astype(np.int64)].sum(1)


def gb_features(tok, T, w=WIN):
    """(X float32, categorical column indices): the residue token at each of the 2w positions (categorical), then the twenty residue counts of the interior-facing and of the outward-facing w residues (numeric); w = 16: 32 categorical columns and 40 counts = 72 features."""
    cols = boosted.full_cols(T, w)
    parts = [tok[:, cols].astype(np.float32), c1.counts20(tok, c1.interior_cols(T, w)).astype(np.float32), c1.counts20(tok, boosted.outward_cols(T, w)).astype(np.float32)]
    return np.concatenate(parts, axis=1), list(range(len(cols)))


def fit_gb(X, y, fid, cat, grid=ITER_GRID):
    """the number of iterations by boosted.select_iter on the folds `fid` of the training rows, then one fit of the largest grid value of iterations; the fitted model is checked against the specified settings.
    Returns (model, chosen iterations, pooled out-of-fold log-loss per grid value)."""
    nstar, loss = boosted.select_iter(X, y, fid, cat, grid)
    assert nstar in grid, "the chosen number of iterations is not on the grid"
    model = boosted.new_model(cat, grid[-1]).fit(X, y)
    boosted.check_model(model, len(cat), X.shape[1], grid[-1])
    return model, int(nstar), loss


def score_gb(model, nstar, X):
    return boosted.staged_scores(model, X, [nstar], chunk=len(X))[nstar]


def blocks(n, chunk):
    for i in range(0, n, chunk):
        yield i, min(i + chunk, n)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


# ======================================================================================================================================== run
def run_cell(S, regime, cfg, ctx, keys, out_dir):
    t0 = time.time()
    name = regime["name"]
    log = lambda *a: print(f"[{time.time() - t0:5.0f}s] study={S} regime={name}", *a, flush=True)          # noqa: E731
    outfile = os.path.join(out_dir, f"study_{S}_{name}.json")
    if os.path.exists(outfile):
        sys.exit(f"refusing to overwrite {outfile}: a finished cell is never overwritten")
    models.configure(cfg["lr"])
    assert S in cfg["studies"], f"{S} is not a study of the historical test"
    GROUPING = regime["grouping"]
    grid, iter_grid, maxiter, tol = tuple(cfg["penalty_grid"]), tuple(cfg["iteration_grid"]), cfg["max_iter"], cfg["tol"]
    assert cfg["kmer_overlap"] == training.KMER_OVERLAP and cfg["neg_ratio"] == NEG_RATIO and grid == c1.GRID == GRID and iter_grid == boosted.GRID == ITER_GRID, "the configuration and the library disagree on the overlap k, the negatives or the grids"
    assert maxiter == c1.MAXITER == models.LRCFG["max_iter"] and tol == c1.TOL == models.LRCFG["tol"] and cfg["n_folds"] == c1.NFOLD == boosted.NFOLD == 3, "the lbfgs settings or the folds disagree"
    assert tuple(mkey(m) for m in cfg["models"]) == MODELS and tuple(cfg["variants"]) == VARIANTS and tuple(r["name"] for r in cfg["regimes"]) == REGIMES, "the models, variants or regimes of the configuration differ from the module"
    LO, HI, DELTA = cfg["peptide_length"]["min"], cfg["peptide_length"]["max"], cfg["near_miss_shift"]
    ENV = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__)
    log("environment:", ENV)
    inS = np.zeros(ctx.nv, bool)
    inS[ctx.stud[S]] = True
    rows, rinfo = regime_rows(ctx, S, regime, inS, keys)
    training.check_disjoint(ctx, rows, S, GROUPING)
    ent = next(e for e in cfg["training_lists"] if e["study"] == S and e["regime"] == name)
    ent_pos = {x["terminus"]: x["n_pos"] for x in ent["train_pos"]}
    assert ent["units"] == len(rows) and ent["sha256"] == list_hash(unit_keys(ctx, rows)), f"the training list of {S} {name} is not the label-only list of the configuration"
    log(f"context: {ctx.NP} proteins, {ctx.nv} verified units, the study has {len(ctx.stud[S])}; the training list ({len(rows)} units, {ent['proteins']} proteins) equals the label-only list (sha256 {ent['sha256'][:12]}...)")

    # ---- the peptide-level populations of the study (candidates, positives, exclusions, the length prior of the strict training units of this regime)
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
    posd, excl_other, excl_rep = {}, {}, {}
    for r in SR.tolist():
        p = int(ctx.vp[r])
        if isd[p]:
            posd.setdefault(p, []).append((int(ctx.vs[r]), int(ctx.ve[r])))
    for r in np.flatnonzero(isd[ctx.vp] & ~inS).tolist():
        excl_other.setdefault(int(ctx.vp[r]), set()).add((int(ctx.vs[r]), int(ctx.ve[r])))
    for r in SR.tolist():
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
    g = np.log((cnt + 1.0) / (cnt.sum() + len(cnt)))
    UN.update(evaluation_proteins=int(len(dprot)))
    log("peptide-level populations:", UN)
    assert UN["candidate"] > 0 and len(dprot) > 0, "no candidate positive peptide in this study"
    LP = ctx.L[dprot]

    ES, SC, FI, SEL, TINFO = {}, {m: {} for m in MODELS}, {}, {}, {}
    SKH = SparseHead(maxiter, tol)

    def fit_score_terminus(T):
        """the rows of one terminus (the training stream seeded as in the bond-level experiments), the five models fitted on them and the scores of every evaluation bond"""
        rng_tr = np.random.default_rng(records.terminus_stream(cfg["seed_training"], S, T))
        es = evaluation.eval_setup(ctx, S, T)
        etok = data.tokens(ctx, es["bp"], es["bi"])
        epf = data.posfeat(es["bi"], ctx.L[es["bp"]])
        training.check_disjoint(ctx, rows, S, GROUPING, eprot=es["eprot"], peptides=False)
        P, B, y, tinfo = training.build_train(ctx, rows, T, rng_tr, cfg["neg_ratio"])
        assert (tinfo["n_pos"], tinfo["n_train_proteins"]) == (ent_pos[T], ent["proteins"]) and tinfo["n_neg"] == cfg["neg_ratio"] * tinfo["n_pos"], "the training bonds differ from the label-only list"
        assert np.isin(dprot, es["eprot"]).all(), f"{T}: an evaluation protein of the peptide-level populations is not an evaluation protein of the bond-level run"
        ES[T], TINFO[T] = es, dict(tinfo, train_units=int(len(rows)), n_eval_proteins=int(len(es["eprot"])), n_eval_bonds=int(len(es["bp"])))
        log(T, f"training {tinfo}; evaluation proteins {len(es['eprot'])}, bonds {len(es['bp'])}")
        tok = data.tokens(ctx, P, B)
        pf = data.posfeat(B, ctx.L[P])
        grp = ctx.G[GROUPING][P]
        fid = c1.fold_ids(grp)
        boosted.assert_folds(fid, grp, y)
        fi, sel = {}, {}
        for m_ in ("LR-Both8", "LR-Both16"):
            t1 = time.time()
            mdl = models.window_model(int(m_[7:])).fit(tok, pf, y)
            SC[m_][T] = mdl.score(etok, epf)
            fi[m_] = dict(n_iter=mdl.n_iter, hit_limit=bool(mdl.n_iter >= models.LRCFG["max_iter"]))
            log(T, f"{m_} fitted and scored ({time.time() - t1:.0f}s)")
        SC["Paul-style motif"][T] = comparators.PaulMotif(T).fit(tok, pf, y).score(etok, epf)
        t1 = time.time()
        Xo = models.window_model(WIN)._X(tok)                                                        # the one-hot design (float32, sparse)
        Cs, tab, hits = c1.select_C(SKH, Xo, Xo, y, grp)                                               # the rule of the choice of the penalty, verbatim
        m = LogisticRegression(C=Cs, solver="lbfgs", max_iter=maxiter, tol=tol).fit(Xo, y)
        n_iter = int(np.max(m.n_iter_))
        SC[LR16_CV][T] = loc_scores(m.coef_.ravel().astype(np.float64), float(m.intercept_[0]), etok)
        sel[LR16_CV] = dict(C=float(Cs), C_loss=[float(x) for x in tab], cv_fits_at_limit=int(hits))
        fi[LR16_CV] = dict(n_iter=n_iter, hit_limit=bool(n_iter >= maxiter))
        del Xo
        log(T, f"{display(LR16_CV)} fitted and scored ({time.time() - t1:.0f}s)")
        t1 = time.time()
        X, cat = gb_features(tok, T)
        assert X.shape[1] == 72 and len(cat) == 32, "LigTerm-GB: wrong number of features"
        model, nstar, loss = fit_gb(X, y, fid, cat)
        del X
        z = np.full(len(etok), np.nan, np.float64)
        for i, j in blocks(len(etok), cfg["evaluation_block"]):
            Xb, _ = gb_features(etok[i:j], T)
            z[i:j] = score_gb(model, nstar, Xb)
        SC[GB][T] = z
        sel[GB] = dict(iterations=int(nstar), loss=[float(x) for x in loss])
        fi[GB] = dict(n_iter=int(nstar), at_edge=bool(nstar in (iter_grid[0], iter_grid[-1])))
        log(T, f"LigTerm-GB fitted and scored ({time.time() - t1:.0f}s); max RSS {rss_gb():.1f} GB")
        assert all(np.isfinite(SC[m_][T]).all() for m_ in MODELS), f"{T}: a score of an evaluation bond is not finite"
        FI[T], SEL[T] = fi, sel

    for T in "NC":
        fit_score_terminus(T)
    EB = {T: np.searchsorted(ES[T]["eprot"], dprot) for T in "NC"}
    for T in "NC":
        assert np.array_equal(ES[T]["eprot"][EB[T]], dprot)

    def bond_slice(arr, T, j):
        es_, k = ES[T], EB[T][j]
        return arr[es_["ebase"][k]:es_["ebase"][k] + es_["nb"][k]]

    def prep_protein(j):
        p = int(dprot[j])
        L = int(ctx.L[p])
        lab = {T: bond_slice(ES[T]["labs"]["primary"], T, j) for T in "NC"}
        assert len(lab["N"]) == len(lab["C"]) == L - 1
        prep = PE.prepare(L, posd[p], excl.get(p, ()), LO, HI, lab["N"], lab["C"])
        assert all(lab["N"][s - 2] == 1 and lab["C"][e - 1] == 1 for s, e in prep["pos"]), f"a positive peptide's own terminal bond does not carry the label 1 ({ctx.names[p]})"
        return p, L, lab, prep

    NPOS = np.zeros(len(dprot), np.int64)
    for j in range(len(dprot)):
        NPOS[j] = len(prep_protein(j)[3]["pos"])
    assert int(NPOS.sum()) == UN["candidate"], "the positives scored in the evaluation proteins differ from the candidate units"

    def per_prot(T, z):
        return metrics.per_protein(z, ES[T]["labs"]["primary"], ES[T]["ebase"], ES[T]["nb"])

    def summ(pp):
        return dict(bond_auroc=float(np.nanmean(pp["bond_auroc"])), local_auc=float(pp["local_n"].sum() / pp["local_d"].sum()))

    PPS = {T: {m: per_prot(T, SC[m][T]) for m in MODELS} for T in "NC"}
    SUMM = {T: {m: summ(PPS[T][m]) for m in MODELS} for T in "NC"}

    # ---- the brute-force guard on real proteins (two implementations of one quantity) runs before any peptide metric is stored
    G = {}
    base_idx, reserve_idx = PR.select_guard_proteins(LP, NPOS)
    cmp_tot, GUARDED = dict(F_pairs=0.0, M_den=0.0, X_den=0.0, proteins=0), []
    GM = [mkey(m) for m in cfg["guard_models"]]

    def guard_protein(j, from_reserve):
        p, L, lab, prep = prep_protein(j)
        GUARDED.append(dict(acc=ctx.names[p], L=L, n_pos=len(prep["pos"]), reserve=bool(from_reserve)))
        for fam in GM:
            sl = {T: bond_slice(SC[fam][T], T, j) for T in "NC"}
            res = PE.protein_metrics(sl["N"], sl["C"], L, None, None, g, LO, HI, DELTA, prepared=prep)
            for v in PE.VARIANTS:
                ref = PR.brute(sl["N"], sl["C"], L, posd[p], excl.get(p, ()), g, v, LO, HI, DELTA, labN=lab["N"], labC=lab["C"])
                got = res[v]
                bad = [k_ for k_ in KEY_CODE.values() if not ((np.isnan(got[k_]) and np.isnan(ref[k_])) or abs(got[k_] - ref[k_]) < cfg["brute_force_tolerance"])] + [k_ for k_ in ("n_pos", "n_pos_dropped") if got[k_] != ref[k_]]
                assert not bad, f"{fam} {v}: protein {ctx.names[p]} (L={L}) differs from the brute-force reference in {bad}"
                if fam == GM[0] and v == "NC":
                    cmp_tot["F_pairs"] += ref["n_pos"] * ref["n_neg"]
                    cmp_tot["M_den"] += ref["M_den"]
                    cmp_tot["X_den"] += ref["X_den"]
                    cmp_tot["proteins"] += 1

    t1 = time.time()
    for j in base_idx:
        guard_protein(j, False)
    for j in reserve_idx:
        if cmp_tot["F_pairs"] > 0 and cmp_tot["M_den"] > 0 and cmp_tot["X_den"] > 0:
            break
        guard_protein(j, True)
    assert len(GUARDED) >= 1, "the brute-force guard compared no protein"            # an empty clean population is recorded and noted, not an abort
    G["brute_force"] = dict(proteins=GUARDED, models=[mrow(m) for m in GM], compared=cmp_tot, clean_populations_compared=bool(cmp_tot["F_pairs"] > 0 and cmp_tot["M_den"] > 0 and cmp_tot["X_den"] > 0))
    log(f"brute-force guard: {len(GUARDED)} proteins x {len(GM)} models x {len(PE.VARIANTS)} variants equal the independent reference ({time.time() - t1:.0f}s); compared denominators {cmp_tot}")
    MET = {m: {v: {k_: np.full(len(dprot), np.nan) for k_ in KEYS} for v in VARIANTS} for m in MODELS}
    t1 = time.time()
    for j in range(len(dprot)):
        p, L, lab, prep = prep_protein(j)
        for m in MODELS:
            sl = {T: bond_slice(SC[m][T], T, j) for T in "NC"}
            res = PE.protein_metrics(sl["N"], sl["C"], L, None, None, g, LO, HI, DELTA, prepared=prep)
            for v in VARIANTS:
                for k_ in KEYS:
                    MET[m][v][k_][j] = res[VARIANT_CODE[v]][KEY_CODE[k_]]
        if (j + 1) % 2000 == 0 or j + 1 == len(dprot):
            log(f"peptide metrics: {j + 1} of {len(dprot)} proteins, {len(MODELS)} models ({time.time() - t1:.0f}s)")

    def pep_summary(F, Mn, Md, Xn, Xd):
        okF, okM = np.isfinite(F), Md > 0
        return dict(peptide_auroc=float(F[okF].mean()) if okF.any() else float("nan"), length_matched_auc=float((Mn[okM] / Md[okM]).mean()) if okM.any() else float("nan"),
                    near_miss_auc=float(Xn.sum() / Xd.sum()) if Xd.sum() > 0 else float("nan"))

    PSUM = {m: {v: pep_summary(*(MET[m][v][k_] for k_ in KEYS)) for v in VARIANTS} for m in MODELS}
    log("every guard has passed")

    bp_rows = [dict(terminus=T, acc=[ctx.names[int(p)] for p in ES[T]["eprot"]], group_u30=[int(x) for x in ctx.G["u30"][ES[T]["eprot"]]], group_m50=[int(x) for x in ctx.G["m50"][ES[T]["eprot"]]],
                    length=[int(ctx.L[int(p)]) for p in ES[T]["eprot"]]) for T in "NC"]
    rec = dict(study=S, regime=name, grouping=GROUPING, units=UN, models=[mrow(m) for m in MODELS], variants=list(VARIANTS),
               training_list=dict(sha256=ent["sha256"], units=ent["units"], proteins=ent["proteins"], train_pos=[dict(terminus=T, n_pos=ent_pos[T]) for T in "NC"], info=rinfo),
               rows=[dict(terminus=T, **TINFO[T]) for T in "NC"],
               selection=[dict(terminus=T, **mrow(m), value=SEL[T][m]) for T in "NC" for m in SEL[T]],
               fit_info=[dict(terminus=T, **mrow(m), value=FI[T][m]) for T in "NC" for m in FI[T]],
               bond_proteins=bp_rows,
               bond=[dict(**mrow(m), terminus=T, bond_auroc=lst(PPS[T][m]["bond_auroc"]), local_pairs_won=lst(PPS[T][m]["local_n"]), local_pairs=lst(PPS[T][m]["local_d"]), n_positive_bonds=lst(PPS[T][m]["npos"]))
                     for m in MODELS for T in "NC"],
               bond_summary=[dict(**mrow(m), terminus=T, **SUMM[T][m]) for m in MODELS for T in "NC"],
               peptide_proteins=dict(acc=[ctx.names[int(p)] for p in dprot], group_u30=[int(x) for x in ctx.G["u30"][dprot]], group_m50=[int(x) for x in ctx.G["m50"][dprot]], length=[int(x) for x in LP], n_pos=[int(x) for x in NPOS]),
               peptide=[dict(**mrow(m), variant=v, **{k_: lst(MET[m][v][k_]) for k_ in KEYS}) for m in MODELS for v in VARIANTS],
               peptide_summary=[dict(**mrow(m), variant=v, **PSUM[m][v]) for m in MODELS for v in VARIANTS],
               guards=G, env=ENV)
    os.makedirs(out_dir, exist_ok=True)
    tmp = f"{outfile}.{platform.node()}.{os.getpid()}.tmp"                                           # a name of its own: two tasks of one cell never share a temporary file
    with open(tmp, "w") as fh:
        json.dump(rec, fh, separators=(",", ":"), default=float)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.link(tmp, outfile)                                                                            # atomic; REFUSES (FileExistsError) if another task wrote this cell meanwhile
    os.unlink(tmp)
    log("written", outfile)


# ============================================================================================================================ summarize
def read_view(path):
    """a public record -> the nested view the aggregation works on: model keys (see MODELS), vectors by their public names"""
    r = json.load(open(path, encoding="utf-8"))
    v = dict(study=r["study"], regime=r["regime"], grouping=r["grouping"], units=r["units"], models=[mkey(m) for m in r["models"]], variants=r["variants"], env=r["env"], guards=r["guards"])
    tl = r["training_list"]
    v["training_list"] = dict(sha256=tl["sha256"], units=tl["units"], proteins=tl["proteins"], train_pos={x["terminus"]: x["n_pos"] for x in tl["train_pos"]}, info=tl["info"])
    v["rows"] = {x["terminus"]: {k: val for k, val in x.items() if k != "terminus"} for x in r["rows"]}
    v["sel"] = {"N": {}, "C": {}}
    for x in r["selection"]:
        v["sel"][x["terminus"]][mkey(x)] = x["value"]
    v["fit_info"] = {"N": {}, "C": {}}
    for x in r["fit_info"]:
        v["fit_info"][x["terminus"]][mkey(x)] = x["value"]
    v["bond_proteins"] = {x["terminus"]: {k: val for k, val in x.items() if k != "terminus"} for x in r["bond_proteins"]}
    v["bond"] = {}
    for x in r["bond"]:
        v["bond"].setdefault(mkey(x), {})[x["terminus"]] = {k: x[k] for k in ("bond_auroc", "local_pairs_won", "local_pairs", "n_positive_bonds")}
    v["bond_summary"] = {"N": {}, "C": {}}
    for x in r["bond_summary"]:
        v["bond_summary"][x["terminus"]][mkey(x)] = {k: x[k] for k in ("bond_auroc", "local_auc")}
    v["peptide_proteins"] = r["peptide_proteins"]
    v["pep"] = {}
    for x in r["peptide"]:
        v["pep"].setdefault(mkey(x), {})[x["variant"]] = {k: x[k] for k in KEYS}
    v["pep_summary"] = {}
    for x in r["peptide_summary"]:
        v["pep_summary"].setdefault(mkey(x), {})[x["variant"]] = {k: x[k] for k in ("peptide_auroc", "length_matched_auc", "near_miss_auc")}
    return v


def arrays(rec, regime):
    """numpy view of one record: bond[T] = dict(gid, bond_auroc, local_pairs_won, local_pairs) (models x proteins), pep = dict(gid, v -> dict(key -> models x proteins))"""
    nan = lambda xs: [np.nan if x is None else x for x in xs]                                                    # noqa: E731
    out = dict(bond={}, pep={})
    for T in "NC":
        out["bond"][T] = dict(gid=np.array(rec["bond_proteins"][T][gkey(regime)], np.int64),
                              **{k: np.array([nan(rec["bond"][m][T][k]) for m in MODELS], np.float64) for k in ("bond_auroc", "local_pairs_won", "local_pairs")})
    out["pep"] = dict(gid=np.array(rec["peptide_proteins"][gkey(regime)], np.int64), **{v: {k: np.array([nan(rec["pep"][m][v][k]) for m in MODELS], np.float64) for k in KEYS} for v in VARIANTS})
    return out


def q_bond(A, w):
    ok = np.isfinite(A["bond_auroc"])
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.where(ok, A["bond_auroc"], 0.0) * w).sum(1) / (ok * w).sum(1), (A["local_pairs_won"] * w).sum(1) / (A["local_pairs"] * w).sum(1)


def q_pep(P, w):
    okF, okM = np.isfinite(P["peptide_auroc"]), P["length_matched_den"] > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        Mr = np.where(okM, P["length_matched_num"] / np.where(okM, P["length_matched_den"], 1.0), 0.0)
        return ((np.where(okF, P["peptide_auroc"], 0.0) * w).sum(1) / (okF * w).sum(1), (Mr * w).sum(1) / (okM * w).sum(1),
                (P["near_miss_num"] * w).sum(1) / (P["near_miss_den"] * w).sum(1))


def vals_bond(arr, T, w=None):
    A = arr["bond"][T]
    w = np.ones(A["bond_auroc"].shape[1]) if w is None else w
    a, e = q_bond(A, w)
    return {(T, "bond_auroc"): a, (T, "local_auc"): e}


def vals_pep(arr, w=None):
    w = np.ones(len(arr["pep"]["gid"])) if w is None else w
    out = {}
    for v in VARIANTS:
        F, M, X = q_pep(arr["pep"][v], w)
        out[(v, "peptide_auroc")], out[(v, "length_matched_auc")], out[(v, "near_miss_auc")] = F, M, X
    return out


def study_values(arr):
    """-> {quantity: vector over the models} of one study (every protein counted once)"""
    out = {}
    for T in "NC":
        out.update(vals_bond(arr, T))
    out.update(vals_pep(arr))
    return out


def pooled_values(per_study):
    return {q: np.mean([sv[q] for sv in per_study], axis=0) for q in QUANT}


def scope_boot(arrs, studies, slot, B=B_BOOT, seed0=SEED0):
    """point values and bootstrap draws of the pooled (mean over `studies`) quantities: -> (point {q: vec}, draws {q: (B, models)}). Groups are resampled with replacement from the UNION of the groups of the studies' proteins of that terminus (or of the peptide populations)."""
    point = pooled_values([study_values(arrs[s]) for s in studies])
    draws = {q: np.empty((B, len(MODELS))) for q in QUANT}
    for k in (0, 1, 2):
        gids = {s: (arrs[s]["bond"]["NC"[k]]["gid"] if k < 2 else arrs[s]["pep"]["gid"]) for s in studies}
        U = np.unique(np.concatenate(list(gids.values())))
        pos = {s: np.searchsorted(U, gids[s]) for s in studies}
        rng = np.random.default_rng([seed0, STAGE, int(slot), k])
        for b in range(B):
            cnt = np.bincount(rng.integers(0, len(U), size=len(U)), minlength=len(U)).astype(np.float64)
            sv = [vals_bond(arrs[s], "NC"[k], cnt[pos[s]]) if k < 2 else vals_pep(arrs[s], cnt[pos[s]]) for s in studies]
            for q in sv[0]:
                draws[q][b] = np.mean([x[q] for x in sv], axis=0)
    return point, draws


def interval(x):
    return float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))


def nonfinite_count(draws):
    """the number of bootstrap draws (over all quantities and models) that are not finite: `interval` leaves them out of the percentiles, so they are COUNTED here and reported, never dropped silently"""
    return int(sum(int((~np.isfinite(v)).sum()) for v in draws.values()))


def check_runs(recs, studies, regimes, lists=None, tol=1e-9):
    """consistency of the records with the configuration and with one another, before anything is aggregated -> (fatal, notes). Fatal: the set of (study, regime) cells; the identity of every record; one software environment; the training list hash,
    size and positives equal to the label-only lists of the configuration; vector lengths and counts; every metric within its bounds; the quantities that depend on the labels only (local pair counts, positives, length-matched and near-miss
    denominators, the undefined-AUROC pattern) equal across the models; the penalty and the number of iterations equal to the minimisers of the stored tables; the brute-force guard; the summaries of the record equal to the recomputed ones.
    Notes (not fatal): fits at the iteration limit, a choice at the edge of a grid (rows)."""
    fatal, notes = [], []

    def bad(c, msg):
        fatal.append(f"{c}: {msg}")
    fin = lambda x: isinstance(x, (int, float)) and math.isfinite(x)                                              # noqa: E731
    want = {(s, r) for s in studies for r in regimes}
    if set(recs) != want:
        fatal.append(f"the cells run differ from the expected {len(want)}: missing {sorted(want - set(recs))}, extra {sorted(set(recs) - want)}")
    first = None
    for c in sorted(recs):
        s, r = c
        rec = recs[c]
        if str(rec.get("study")) != s or rec.get("regime") != r:
            bad(c, f"the record says study {rec.get('study')} regime {rec.get('regime')}")
        if tuple(rec["models"]) != MODELS or tuple(rec["variants"]) != VARIANTS:
            bad(c, "models or variants differ from the specification")
        env = rec.get("env", {})
        key = {k: env.get(k) for k in ("python", "numpy", "scipy", "sklearn")}
        first = key if first is None else first
        if key != first:
            bad(c, f"the software differs from the other cells: {key} against {first}")
        if lists is not None:
            fl = lists[(s, r)]
            li = rec["training_list"]
            if li["sha256"] != fl["sha256"] or li["units"] != fl["units"] or li["proteins"] != fl["proteins"] or li["train_pos"] != fl["train_pos"]:
                bad(c, "the training list of the record differs from the label-only list")
            for T in "NC":
                if rec["rows"][T]["n_pos"] != fl["train_pos"][T] or rec["rows"][T]["train_units"] != fl["units"]:
                    bad(c, f"{T}: the training rows differ from the label-only list")
        u = rec["units"]
        if u["total"] != u["candidate"] + u["dropped_bond_outside"] + u["dropped_length_only"]:
            bad(c, "the units are not candidate + dropped (bond outside) + dropped (length)")
        for T in "NC":
            if rec["rows"][T]["n_neg"] != NEG_RATIO * rec["rows"][T]["n_pos"]:
                bad(c, f"{T}: the negatives are not {NEG_RATIO} per positive")
        pp = rec["peptide_proteins"]
        npro = len(pp["group_u30"])
        if any(len(pp[k]) != npro for k in ("acc", "length", "n_pos", "group_m50")) or npro != u["evaluation_proteins"] or sum(pp["n_pos"]) != u["candidate"]:
            bad(c, "the peptide-level protein list does not match the evaluation proteins or the candidate units")
        for T in "NC":
            bp = rec["bond_proteins"][T]
            nb = len(bp["group_u30"])
            if len(bp["acc"]) != nb or len(set(bp["acc"])) != nb or len(bp["length"]) != nb or len(bp["group_m50"]) != nb:
                bad(c, f"{T}: the bond-level protein list is inconsistent")
            ref = None
            for m in MODELS:
                vs = {k: np.array([np.nan if x is None else x for x in rec["bond"][m][T][k]], float) for k in ("bond_auroc", "local_pairs_won", "local_pairs", "n_positive_bonds")}
                if any(len(a) != nb for a in vs.values()):
                    bad(c, f"{T} {m}: a vector does not have {nb} entries")
                    continue
                ok = np.isfinite(vs["bond_auroc"])
                if (vs["bond_auroc"][ok] < -tol).any() or (vs["bond_auroc"][ok] > 1 + tol).any() or (vs["local_pairs_won"] < -tol).any() or (vs["local_pairs_won"] > vs["local_pairs"] + tol).any() or (vs["local_pairs"] < 0).any():
                    bad(c, f"{T} {m}: a metric is outside its bounds")
                if ref is None:
                    ref = (vs["local_pairs"], vs["n_positive_bonds"], ok)
                elif not (np.array_equal(vs["local_pairs"], ref[0]) and np.array_equal(vs["n_positive_bonds"], ref[1]) and np.array_equal(ok, ref[2])):
                    bad(c, f"{T} {m}: the local pair counts, the positives or the pattern of the defined AUROC differ from the first model (they depend on the labels only)")
        refp, Mkeep = None, {}
        for m in MODELS:
            for v in VARIANTS:
                mm = {k: np.array([np.nan if x is None else x for x in rec["pep"][m][v][k]], float) for k in KEYS}
                if any(len(a) != npro for a in mm.values()):
                    bad(c, f"{m} {v}: a peptide vector does not have {npro} entries")
                    continue
                F, Mn, Md, Xn, Xd = (mm[k] for k in KEYS)
                if not all(np.isfinite(a).all() for a in (Mn, Md, Xn, Xd)):
                    bad(c, f"{m} {v}: a non-finite pair count")
                    continue
                okF = np.isfinite(F)
                if (F[okF] < -tol).any() or (F[okF] > 1 + tol).any() or (Md < 0).any() or (Xd < 0).any() or (Mn < -tol).any() or (Xn < -tol).any() or (Mn > Md + tol).any() or (Xn > Xd + tol).any():
                    bad(c, f"{m} {v}: a metric is outside its bounds")
                if v == "length_only" and (Md > 0).any() and not np.allclose(Mn[Md > 0], 0.5 * Md[Md > 0], rtol=0, atol=1e-9):
                    bad(c, f"{m} length_only: the length-matched AUC is not exactly 0.5 (the length prior alone ties every length-matched pair)")
                Mkeep[(m, v)] = (Mn, Md)
                if refp is None:
                    refp = (Md, Xd, okF)
                elif not (np.array_equal(Md, refp[0]) and np.array_equal(Xd, refp[1]) and np.array_equal(okF, refp[2])):
                    bad(c, f"{m} {v}: the length-matched and near-miss denominators or the undefined-AUROC pattern differ from the first model and variant (they depend on the labels only)")
        for m in MODELS:
            if (m, "both_bonds") in Mkeep and (m, "both_bonds_and_length") in Mkeep and not (np.array_equal(Mkeep[(m, "both_bonds")][0], Mkeep[(m, "both_bonds_and_length")][0])
                                                                                          and np.array_equal(Mkeep[(m, "both_bonds")][1], Mkeep[(m, "both_bonds_and_length")][1])):
                bad(c, f"{m}: the length-matched counts differ between both_bonds and both_bonds_and_length (the length prior is constant within a length, so the two rank every length-matched pair alike)")
        for T in "NC":
            sel, fi = rec["sel"][T], rec["fit_info"][T]
            if LR16_CV not in sel or sel[LR16_CV]["C"] not in GRID or int(np.argmin(sel[LR16_CV]["C_loss"])) != GRID.index(sel[LR16_CV]["C"]):
                bad(c, f"{T} {display(LR16_CV)}: C is not the smallest minimiser of the stored pooled out-of-fold log-loss")
            else:
                if fi[LR16_CV]["hit_limit"] or sel[LR16_CV]["cv_fits_at_limit"] > 0:
                    notes.append(dict(study=s, regime=r, terminus=T, **mrow(LR16_CV), note="penalty_choice_at_iteration_limit", final_fit=bool(fi[LR16_CV]["hit_limit"]), cv_fits_at_limit=int(sel[LR16_CV]["cv_fits_at_limit"]), cv_fits=21))
                if sel[LR16_CV]["C"] in (GRID[0], GRID[-1]):
                    notes.append(dict(study=s, regime=r, terminus=T, **mrow(LR16_CV), note="penalty_at_grid_edge", penalty=sel[LR16_CV]["C"]))
            if GB not in sel or sel[GB]["iterations"] not in ITER_GRID or int(np.argmin(sel[GB]["loss"])) != ITER_GRID.index(sel[GB]["iterations"]):
                bad(c, f"{T} LigTerm-GB: the number of iterations is not the smallest minimiser of the stored pooled out-of-fold log-loss")
            elif sel[GB]["iterations"] in (ITER_GRID[0], ITER_GRID[-1]):
                notes.append(dict(study=s, regime=r, terminus=T, **mrow(GB), note="iterations_at_grid_edge", iterations=sel[GB]["iterations"]))
            for m in ("LR-Both8", "LR-Both16"):
                if fi[m]["hit_limit"]:
                    notes.append(dict(study=s, regime=r, terminus=T, **mrow(m), note="fit_at_iteration_limit"))
        bf = rec["guards"]["brute_force"]
        if len(bf["proteins"]) < 1:
            bad(c, "the brute-force guard compared no protein")
        for k_ in ("F_pairs", "M_den", "X_den"):
            if not bf["compared"][k_] > 0:
                notes.append(dict(study=s, regime=r, note="brute_force_guard_compared_no_pairs", population=k_))
        arr = arrays(rec, r)
        sv = study_values(arr)
        for T in "NC":
            for i, m in enumerate(MODELS):
                for k in ("bond_auroc", "local_auc"):
                    if not (fin(rec["bond_summary"][T][m][k]) and abs(rec["bond_summary"][T][m][k] - sv[(T, k)][i]) < 1e-9):
                        bad(c, f"{T} {m}: the record's bond summary ({k}) differs from the recomputed value")
        for i, m in enumerate(MODELS):
            for v in VARIANTS:
                for k in ("peptide_auroc", "length_matched_auc", "near_miss_auc"):
                    a, b = rec["pep_summary"][m][v][k], sv[(v, k)][i]
                    if not ((a is None or (isinstance(a, float) and math.isnan(a))) and math.isnan(b)) and not (fin(a) and abs(a - b) < 1e-9):
                        bad(c, f"{m} {v}: the record's peptide summary ({k}) differs from the recomputed value")
    for s in studies:                                                                                             # the evaluation setup (proteins, groups, label-only counts) is the same in every regime of a study
        cells = [(r, recs[(s, r)]) for r in regimes if (s, r) in recs]
        if len(cells) < 2:
            continue
        r0, ref = cells[0]
        for r, rec in cells[1:]:
            same = ref["units"] == rec["units"] and ref["peptide_proteins"] == rec["peptide_proteins"] and all(ref["bond_proteins"][T] == rec["bond_proteins"][T] for T in "NC")
            same = same and all(ref["bond"][MODELS[0]][T][k] == rec["bond"][MODELS[0]][T][k] for T in "NC" for k in ("local_pairs", "n_positive_bonds"))
            same = same and all(ref["pep"][MODELS[0]][v][k] == rec["pep"][MODELS[0]][v][k] for v in VARIANTS for k in ("length_matched_den", "near_miss_den"))
            if not same:
                bad((s, r), f"the evaluation setup (proteins, groups, label-only counts) differs from that of regime {r0} of the same study (it does not depend on the regime)")
    return fatal, notes


def dev_reference(ref_dir, ref_medians_path):
    """the development medians (the 19 unseen development studies) of every model, for the display beside the historical-test values: LR-Both8 from the bond-level results of the terminal-predictability experiment, LR-Both16 from the
    directional decomposition, the Paul-style motif from the published-method comparison; the peptide level of these three from the peptide-ranking summary (clean negatives); LR-Both16-CV and LigTerm-GB from the stored table of medians
    (`--reference-medians`: it cannot be recomputed here)."""
    rm = json.load(open(ref_medians_path, encoding="utf-8"))
    un = rm["unseen"]
    pr = json.load(open(os.path.join(ref_dir, "peptide_ranking", "peptide_ranking_summary.json"), encoding="utf-8"))
    assert sorted(pr["unseen"]) == sorted(un), "the unseen studies of the peptide-ranking summary and of the stored medians differ"
    med = lambda v: float(np.median(v))                                                                           # noqa: E731
    src = {"LR-Both8": "terminal_predictability", "LR-Both16": "directional_decomposition", "Paul-style motif": "published_method_comparison"}
    out = {m: {} for m in MODELS}
    pmed = {(r["model"], r["variant"]): r for r in pr["medians"] if r["negatives"] == "clean_negatives" and "fit" not in r}
    for m, folder in src.items():
        cells = {}
        for p in un:
            for cc in json.load(open(os.path.join(ref_dir, folder, f"study_{p}.json"), encoding="utf-8"))["cells"]:
                if cc["kind"] == "model" and cc["model"] == m and "fit" not in cc and cc["analysis"] == "primary":
                    cells[(p, cc["terminus"])] = cc
        for T in "NC":
            out[m][(T, "bond_auroc")], out[m][(T, "local_auc")] = med([cells[(p, T)]["bond_auroc"] for p in un]), med([cells[(p, T)]["local_auc"] for p in un])
        for v in ("both_bonds", "both_bonds_and_length", "length_only"):
            for k in ("peptide_auroc", "length_matched_auc", "near_miss_auc"):
                out[m][(v, k)] = float(pmed[(m, v)][k])
    for row in rm["medians"]:
        out[mkey(row)][(row["terminus"] if "terminus" in row else row["variant"], row["metric"])] = row["median"]
    return out


def aggregate(recs, studies, regimes):
    arrs = {r: {s: arrays(recs[(s, r)], r) for s in studies} for r in regimes}
    res, nf = {}, 0
    for r in regimes:
        pt, dr = scope_boot(arrs[r], list(studies), 0)
        nf += nonfinite_count(dr)
        per = {}
        for s in studies:
            ps, ds = scope_boot(arrs[r], [s], int(s))
            nf += nonfinite_count(ds)
            per[s] = (ps, ds)
        res[r] = dict(pooled=(pt, dr), per_study=per)
    res["_nonfinite_draws"] = nf
    return res


def est_row(est, ci):
    return dict(estimate=float(est), ci_lo=ci[0], ci_hi=ci[1])


def build_output(res, studies, regimes, recs, dev):
    """the public summary: tables as rows (regime, [study], model/fit or model_a/model_b, terminus or variant, metric, estimate, ci_lo, ci_hi)"""
    out = dict(studies=list(studies), regimes=list(regimes), models=[mrow(m) for m in MODELS], contrasts=[pair_row(a, b) for a, b in CONTRASTS], bootstrap=B_BOOT, seed=[SEED0, STAGE],
               sizes=[], pooled=[], per_study=[], contrast_pooled=[], contrast_per_study=[], selection=[], development_reference_medians=None)
    for r in regimes:
        pt, dr = res[r]["pooled"]
        for s in studies:
            rc = recs[(s, r)]
            out["sizes"].append(dict(regime=r, study=s, units=rc["training_list"]["units"], proteins=rc["training_list"]["proteins"], train_pos=[dict(terminus=T, n_pos=rc["training_list"]["train_pos"][T]) for T in "NC"],
                                     eval=[dict(terminus=T, **rc["rows"][T]) for T in "NC"], candidates=rc["units"]["candidate"], evaluation_proteins=rc["units"]["evaluation_proteins"]))
        for i, m in enumerate(MODELS):
            for q in QUANT:
                out["pooled"].append(dict(regime=r, **mrow(m), **qfields(q), **est_row(pt[q][i], interval(dr[q][:, i]))))
        for i, m in enumerate(MODELS):
            for s in studies:
                ps, ds = res[r]["per_study"][s]
                for q in QUANT:
                    out["per_study"].append(dict(regime=r, study=s, **mrow(m), **qfields(q), **est_row(ps[q][i], interval(ds[q][:, i]))))
        for a, b in CONTRASTS:
            ia, ib = MODELS.index(a), MODELS.index(b)
            for q in QUANT:
                out["contrast_pooled"].append(dict(regime=r, **pair_row(a, b), **qfields(q), **est_row(pt[q][ia] - pt[q][ib], interval(dr[q][:, ia] - dr[q][:, ib]))))
            for s in studies:
                ps, ds = res[r]["per_study"][s]
                for q in QUANT:
                    out["contrast_per_study"].append(dict(regime=r, study=s, **pair_row(a, b), **qfields(q), **est_row(ps[q][ia] - ps[q][ib], interval(ds[q][:, ia] - ds[q][:, ib]))))
        for s in studies:
            for T in "NC":
                out["selection"].append(dict(regime=r, study=s, terminus=T, **mrow(LR16_CV), penalty=recs[(s, r)]["sel"][T][LR16_CV]["C"]))
                out["selection"].append(dict(regime=r, study=s, terminus=T, **mrow(GB), iterations=recs[(s, r)]["sel"][T][GB]["iterations"]))
    if dev:
        out["development_reference_medians"] = [dict(**mrow(m), **qfields(q), median=dev[m][q]) for m in MODELS for q in QUANT if q in dev[m]]
    out["nonfinite_bootstrap_draws"] = int(res.get("_nonfinite_draws", 0))
    return out


def fmt(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def _index(rows, *fields):
    idx = {}
    for r in rows:
        idx[tuple(r.get(f) for f in fields)] = r
    return idx


def _qkey(q):
    return (q[0] if q[0] in ("N", "C") else None, q[0] if q[0] not in ("N", "C") else None, q[1])


NOTE_TEXT = {
    "penalty_at_grid_edge": lambda n: f"{n['study']} {n['regime']} {n['terminus']} {display(mkey(n))}: the penalty C = {n['penalty']} is at the edge of the grid",
    "iterations_at_grid_edge": lambda n: f"{n['study']} {n['regime']} {n['terminus']} {display(mkey(n))}: {n['iterations']} iterations is at the edge of the grid",
    "penalty_choice_at_iteration_limit": lambda n: f"{n['study']} {n['regime']} {n['terminus']} {display(mkey(n))}: fits at the lbfgs limit (final fit {n['final_fit']}, {n['cv_fits_at_limit']} of the {n['cv_fits']} fits of the choice of the penalty)",
    "fit_at_iteration_limit": lambda n: f"{n['study']} {n['regime']} {n['terminus']} {display(mkey(n))}: the fit reached the lbfgs limit",
    "brute_force_guard_compared_no_pairs": lambda n: f"{n['study']} {n['regime']}: the brute-force guard compared no pairs of the population {n['population']} (the fast metrics of that population were not checked against the independent reference on this cell)",
    "nonfinite_bootstrap_draws": lambda n: f"{n['count']} bootstrap draws were not finite (a resample without a defined protein of a quantity): they are left out of the percentiles of their cell",
}


def csv_text(out):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_FIELDS)
    idx = _index(out["per_study"], "regime", "study", "model", "fit", "terminus", "variant", "metric")
    for r in out["regimes"]:
        for m in MODELS:
            for s in out["studies"]:
                for q in QUANT:
                    d = idx[(r, s, mrow(m)["model"], mrow(m).get("fit")) + _qkey(q)[:2] + (q[1],)]
                    w.writerow([r, s, d["model"], d.get("fit", ""), d.get("terminus", ""), d.get("variant", ""), d["metric"], f"{d['estimate']:.6f}", f"{d['ci_lo']:.6f}", f"{d['ci_hi']:.6f}"])
    return buf.getvalue()


def render_report(out):
    """the text report of a summary (rows as written by build_output)"""
    L = ["# Historical test: the measured performance of the evaluated refits", "",
         "The two studies are a previously reserved, once-evaluated historical test: they were held out of model development and evaluated once, after the models and the evaluation rules were fixed. The result is the measured performance of the evaluated refits "
         "(group-disjoint and lab-excluded regimes, with the 50%-identity grouping as the sensitivity) beside the development results. It is not independent evidence for the final predictor, there is no pass rule, and the paired differences are descriptions. "
         "The pooled value of a cell is the unweighted mean of the two study-level values; the interval is the 95% percentile bootstrap over sequence groups across both studies. Both studies share peptides with the development studies and with each other, "
         "and their units are ordinary training units of every development run.", ""]
    pooled = _index(out["pooled"], "regime", "model", "fit", "terminus", "variant", "metric")
    pstudy = _index(out["per_study"], "regime", "study", "model", "fit", "terminus", "variant", "metric")
    cpool = _index(out["contrast_pooled"], "regime", "model_a", "fit_a", "model_b", "fit_b", "terminus", "variant", "metric")
    cstudy = _index(out["contrast_per_study"], "regime", "study", "model_a", "fit_a", "model_b", "fit_b", "terminus", "variant", "metric")
    sizes = _index(out["sizes"], "regime", "study")
    cols = (("N", "bond_auroc"), ("C", "bond_auroc"), ("N", "local_auc"), ("C", "local_auc"), ("both_bonds", "peptide_auroc"), ("both_bonds", "length_matched_auc"), ("both_bonds", "near_miss_auc"),
            ("both_bonds_and_length", "peptide_auroc"), ("both_bonds_and_length", "length_matched_auc"), ("both_bonds_and_length", "near_miss_auc"))
    heads = "| model | Bond-AUROC N | Bond-AUROC C | Local-AUC N | Local-AUC C | Peptide-AUROC (both_bonds) | Length-matched AUC (both_bonds) | Near-miss AUC (both_bonds) | Peptide-AUROC (both_bonds_and_length) | Length-matched AUC (both_bonds_and_length) | Near-miss AUC (both_bonds_and_length) |"
    for r in out["regimes"]:
        L += [f"## Regime {r}", "", "| study | training units | proteins | training positives N / C | evaluation proteins (bond level N / C) | candidate positives |", "|---|---|---|---|---|---|"]
        for s in out["studies"]:
            z = sizes[(r, s)]
            tp = {x["terminus"]: x["n_pos"] for x in z["train_pos"]}
            ev = {x["terminus"]: x for x in z["eval"]}
            L.append(f"| {s} | {z['units']} | {z['proteins']} | {tp['N']} / {tp['C']} | {ev['N']['n_eval_proteins']} / {ev['C']['n_eval_proteins']} | {z['candidates']} |")
        L += ["", "Pooled values with 95% intervals (study values below):", "", heads, "|---|" + "---|" * len(cols)]
        for m in MODELS:
            mr = mrow(m)
            cells = []
            for q in cols:
                d = pooled[(r, mr["model"], mr.get("fit")) + _qkey(q)]
                cells.append(f"{fmt(d['estimate'])} [{fmt(d['ci_lo'])}, {fmt(d['ci_hi'])}]")
            L.append(f"| {display(m)} | " + " | ".join(cells) + " |")
        for s in out["studies"]:
            L += ["", f"Study {s}:", "", heads, "|---|" + "---|" * len(cols)]
            for m in MODELS:
                mr = mrow(m)
                L.append(f"| {display(m)} | " + " | ".join(fmt(pstudy[(r, s, mr["model"], mr.get("fit")) + _qkey(q)]["estimate"]) for q in cols) + " |")
        cc = (("N", "bond_auroc"), ("C", "bond_auroc"), ("N", "local_auc"), ("C", "local_auc"), ("both_bonds", "length_matched_auc"), ("both_bonds", "near_miss_auc"), ("both_bonds", "peptide_auroc"))
        L += ["", "Descriptive paired differences (first minus second; pooled, with the interval; the per-study values follow):", "",
              "| contrast | Bond-AUROC N | Bond-AUROC C | Local-AUC N | Local-AUC C | Length-matched AUC (both_bonds) | Near-miss AUC (both_bonds) | Peptide-AUROC (both_bonds) |", "|---|---|---|---|---|---|---|---|"]
        for a, b in CONTRASTS:
            pr = pair_row(a, b)
            key = (r, pr["model_a"], pr.get("fit_a"), pr["model_b"], pr.get("fit_b"))
            cells = []
            for q in cc:
                d = cpool[key + _qkey(q)]
                cells.append(f"{fmt(d['estimate'])} [{fmt(d['ci_lo'])}, {fmt(d['ci_hi'])}]")
            L.append(f"| {display(a)} minus {display(b)} | " + " | ".join(cells) + " |")
            for s in out["studies"]:
                skey = (r, s, pr["model_a"], pr.get("fit_a"), pr["model_b"], pr.get("fit_b"))
                L.append(f"| &nbsp;&nbsp;{s} | " + " | ".join(fmt(cstudy[skey + _qkey(q)]["estimate"]) for q in cc) + " |")
        sel = _index(out["selection"], "regime", "study", "terminus", "model", "fit")
        parts = []
        for s in out["studies"]:
            part = []
            for T in "NC":
                lc, gb = sel[(r, s, T, "LR-Both16", "cv")], sel[(r, s, T, "LigTerm-GB", None)]
                part.append(f"{T} C={lc['penalty']}, {gb['iterations']} it")
            parts.append(f"{s}: " + "; ".join(part))
        L += ["", "Selection (LR-Both16-CV: penalty by the rule on the training folds; LigTerm-GB: boosting iterations by the rule on the training folds): " + "; ".join(parts), ""]
    if out["development_reference_medians"]:
        dm = _index(out["development_reference_medians"], "model", "fit", "terminus", "variant", "metric")
        cd = (("N", "bond_auroc"), ("C", "bond_auroc"), ("N", "local_auc"), ("C", "local_auc"), ("both_bonds", "length_matched_auc"), ("both_bonds", "near_miss_auc"), ("both_bonds", "peptide_auroc"))
        L += ["## Development medians beside the historical-test values (19 unseen development studies; the group-disjoint regime of every development run)", "",
              "| model | Bond-AUROC N | Bond-AUROC C | Local-AUC N | Local-AUC C | Length-matched AUC (both_bonds) | Near-miss AUC (both_bonds) | Peptide-AUROC (both_bonds) |", "|---|---|---|---|---|---|---|---|"]
        for m in MODELS:
            mr = mrow(m)
            L.append(f"| {display(m)} | " + " | ".join(fmt(dm[(mr["model"], mr.get("fit")) + _qkey(q)]["median"]) for q in cd) + " |")
        L.append("")
    L += ["## Consistency checks of the run records (before the aggregation; a failure stops it)", "",
          "all passed; notes (not fatal): " + (str(len(out["notes"])) if out["notes"] else "none") + f"; bootstrap draws that were not finite (left out of the percentiles and counted): {out.get('nonfinite_bootstrap_draws', 0)}"] + [f"- note: {NOTE_TEXT[n['note']](n)}" for n in out["notes"]]
    return "\n".join(L) + "\n"


def summarize(records_dir, out_dir, cfg, ref_dir=None, ref_medians=None):
    studies = tuple(cfg["studies"])
    regimes = tuple(r["name"] for r in cfg["regimes"])
    assert regimes == REGIMES and tuple(mkey(m) for m in cfg["models"]) == MODELS and tuple(cfg["variants"]) == VARIANTS, "the configuration differs from the module"
    assert cfg["bootstrap"]["resamples"] == B_BOOT and tuple(cfg["bootstrap"]["seed"]) == (SEED0, STAGE) and tuple(cfg["penalty_grid"]) == GRID and tuple(cfg["iteration_grid"]) == ITER_GRID and cfg["neg_ratio"] == NEG_RATIO
    pairs = [pair_row(a, b) for a, b in CONTRASTS]
    assert [(p["model_a"], p.get("fit_a"), p["model_b"], p.get("fit_b")) for p in pairs] == [(c["a"], c.get("fit_a"), c["b"], c.get("fit_b")) for c in cfg["contrasts"]], "the contrasts of the configuration differ from the module"
    lists = {(e["study"], e["regime"]): dict(sha256=e["sha256"], units=e["units"], proteins=e["proteins"], train_pos={x["terminus"]: x["n_pos"] for x in e["train_pos"]}) for e in cfg["training_lists"]}
    for suffix in ("historical_test_summary.json", "historical_test_per_study.csv", "historical_test_report.md"):                    # the aggregation is made once: refuse before any work is done
        assert not os.path.exists(os.path.join(out_dir, suffix)), f"{os.path.join(out_dir, suffix)} exists: the aggregation is made once"
    expected_files = {f"study_{s}_{r}.json" for s in studies for r in regimes}
    stray = sorted(set(os.listdir(records_dir)) - expected_files)
    if stray:                                                                                                    # a temporary file of a task that did not finish counts
        raise SystemExit(f"the records folder holds files that are not records of this aggregation: {stray[:6]}")
    recs = {}
    for s in studies:
        for r in regimes:
            p = os.path.join(records_dir, f"study_{s}_{r}.json")
            if os.path.exists(p):
                recs[(s, r)] = read_view(p)
    fatal, notes = check_runs(recs, studies, regimes, lists=lists)
    if fatal:
        raise SystemExit("consistency checks FAILED, nothing is aggregated:\n  " + "\n  ".join(fatal))
    dev = dev_reference(ref_dir, ref_medians) if ref_dir and ref_medians else None
    res = aggregate(recs, studies, regimes)
    out = build_output(res, studies, regimes, recs, dev)
    if out["nonfinite_bootstrap_draws"]:
        notes.append(dict(note="nonfinite_bootstrap_draws", count=out["nonfinite_bootstrap_draws"]))
    out["notes"] = notes
    texts = {"historical_test_summary.json": json.dumps(out, indent=1), "historical_test_per_study.csv": csv_text(out), "historical_test_report.md": render_report(out)}
    os.makedirs(out_dir, exist_ok=True)
    tmps = {}
    for name, text in texts.items():                                                                             # all or nothing: the files are written under temporary names first and linked afterwards
        tmps[name] = os.path.join(out_dir, name + ".tmp")
        with open(tmps[name], "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
    for name, tmp in tmps.items():
        os.link(tmp, os.path.join(out_dir, name))                                                                # refuses if the file exists
        os.unlink(tmp)
    print("wrote", " ".join(os.path.join(out_dir, n) for n in texts))
    return out


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    default_cfg = os.path.join(here, "..", "..", "configs", "historical_test.json")
    ap = argparse.ArgumentParser(prog="44_development_benchmark.py historical-test", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a_run = sub.add_parser("run", help="refit the five models for one study in one regime and write the record")
    a_run.add_argument("--data", required=True, help="datasets folder")
    a_run.add_argument("--out", required=True, help="folder of the records (study_<pmid>_<regime>.json)")
    a_run.add_argument("--study", required=True)
    a_run.add_argument("--regime", required=True, choices=REGIMES)
    a_run.add_argument("--config", default=default_cfg)
    a_sum = sub.add_parser("summarize", help="aggregate the six records")
    a_sum.add_argument("--records", required=True)
    a_sum.add_argument("--out", required=True)
    a_sum.add_argument("--config", default=default_cfg)
    a_sum.add_argument("--reference-results", help="results folder with terminal_predictability, directional_decomposition, published_method_comparison and peptide_ranking (the development medians beside the values)")
    a_sum.add_argument("--reference-medians", help="stored medians of LR-Both16-CV and LigTerm-GB over the unseen development studies (JSON)")
    a = ap.parse_args(argv)
    cfg = records.load_config(a.config)
    if a.cmd == "run":
        regime = next(r for r in cfg["regimes"] if r["name"] == a.regime)
        models.configure(cfg["lr"])
        ctx, keys = records.load_study_context(a.data)
        run_cell(a.study, regime, cfg, ctx, keys, a.out)
    else:
        if bool(a.reference_results) != bool(a.reference_medians):
            ap.error("give --reference-results and --reference-medians together (or neither: no development medians)")
        summarize(a.records, a.out, cfg, a.reference_results, a.reference_medians)


if __name__ == "__main__":
    main()
