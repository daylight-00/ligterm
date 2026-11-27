"""Composition versus position within the interior side of the bond: the boosted-tree family (a sensitivity analysis of the linear family).

One HistGradientBoostingClassifier per feature set, with the number of boosting iterations chosen on the training rows only (group-disjoint folds, as the penalty
of the linear family; the fold ids are computed once by the caller and validated with assert_folds). The evaluation bonds are scored in blocks at the chosen number
of iterations (fit "cv") and at a number fixed in advance (fit "fixed"). Token matrix and column sets as in composition.py.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import composition as c1
from .data import WMAX

HGB = dict(learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, max_bins=255, early_stopping=False, random_state=0)
GRID = (50, 100, 200, 400, 800)                    # numbers of boosting iterations considered ('cv': the smallest pooled out-of-fold log-loss on the training rows)
MAXIT = GRID[-1]
FIXIT = 200                                        # 'fixed': the same number for every model
NFOLD = c1.NFOLD
CHUNK = 400_000                                    # rows per block when the evaluation bonds are scored


def outward_cols(T, w=8):
    """token columns of the outward-facing w residues (facing away from the ligand): N-bond P_w..P1 (columns 8..15), C-bond P1'..P_w' (columns 16..23)"""
    if T == "N":
        return np.arange(WMAX - w, WMAX)
    if T == "C":
        return np.arange(WMAX, WMAX + w)
    raise ValueError(T)


def full_cols(T, w=8):
    return np.arange(WMAX - w, WMAX + w)


def colsets(T):
    return {"full": full_cols(T), "interior": c1.interior_cols(T), "outward": outward_cols(T), "proximal": c1.proximal_cols(T), "distal": c1.distal_cols(T)}


# model -> (categorical column sets (residue at each position, 21 categories), column sets whose twenty residue counts are numeric features)
SPECS = {
    "GB-Both8": (("full",), ("interior", "outward")),      # the +-8 window: positions of both sides and the counts of each side
    "GB-In8": (("interior",), ("interior",)),              # the eight interior residues: positions and counts = everything the linear LR-Full8 can use (the reference of the decomposition)
    "GB-Out8": (("outward",), ("outward",)),               # the eight outward residues: positions and counts
    "GB-Pos8": (("interior",), ()),                        # the interior positions only (no counts)
    "GB-Comp8": ((), ("interior",)),                       # composition only: the twenty counts of the eight interior residues
    "GB-Prox2": (("proximal",), ()),                       # the two positions next to the bond
    "GB-Comp8+Prox2": (("proximal",), ("interior",)),      # composition and the two positions next to the bond
    "GB-Dist6": (("distal",), ()),                         # interior positions 3-8
}
BASES = tuple(SPECS)


def features(tok, T, base):
    """(X float32, indices of the categorical columns): the residue token at each position of the categorical sets (categorical for the trees: 21 categories incl. pad), then the residue counts (numeric)."""
    cats, cnts = SPECS[base]
    cs = colsets(T)
    parts, ncat = [], 0
    for c in cats:
        parts.append(tok[:, cs[c]].astype(np.float32))
        ncat += len(cs[c])
    for c in cnts:
        parts.append(c1.counts20(tok, cs[c]).astype(np.float32))
    return np.concatenate(parts, axis=1), list(range(ncat))


def n_features(T, base):
    cats, cnts = SPECS[base]
    cs = colsets(T)
    return sum(len(cs[c]) for c in cats) + 20 * len(cnts)


def new_model(cat_idx, max_iter):
    return HistGradientBoostingClassifier(**HGB, max_iter=int(max_iter), categorical_features=(list(cat_idx) if len(cat_idx) else None))


def staged_scores(model, X, stages, chunk=CHUNK):
    """raw scores (log-odds) after each of the boosting iteration counts in `stages` (ascending), for the rows of X, computed in blocks of `chunk` rows: {k: float64 array}. The iteration stops at the last stage requested."""
    stages = [int(k) for k in stages]
    out = {k: np.full(len(X), np.nan, np.float64) for k in stages}          # NaN, not empty: a stage that is never produced must not pass as zeros
    want = set(stages)
    for i in range(0, len(X), chunk):
        for k, raw in enumerate(model.staged_decision_function(X[i:i + chunk]), 1):
            if k in want:
                out[k][i:i + chunk] = raw
            if k >= stages[-1]:
                break
    assert all(np.isfinite(v).all() for v in out.values()), "a requested boosting stage was not produced (the model has fewer iterations than the last stage requested) or a score is not finite"
    return out


def score_eval(model, etok, T, base, stages, chunk=CHUNK):
    """scores of the evaluation bonds (token rows `etok`) after each of the iteration counts in `stages`: the features are built block by block, so that no feature matrix of all evaluation bonds exists"""
    stages = sorted({int(k) for k in stages})
    out = {k: np.full(len(etok), np.nan, np.float64) for k in stages}
    for i in range(0, len(etok), chunk):
        Xc, _ = features(etok[i:i + chunk], T, base)
        st = staged_scores(model, Xc, stages, chunk=len(Xc))
        for k in stages:
            out[k][i:i + chunk] = st[k]
    assert all(np.isfinite(v).all() for v in out.values()), "a score of an evaluation bond is missing or not finite"
    return out


def check_model(m, ncat, nfeat, max_iter):
    """the fitted model has the settings of HGB, was fitted for exactly `max_iter` iterations, and its categorical mask is the first `ncat` of the `nfeat` features"""
    p = m.get_params()
    assert all(p[k] == v for k, v in HGB.items()), f"the settings of the fitted model differ from the specified ones: {({k: p[k] for k in HGB if p[k] != HGB[k]})}"
    assert int(m.n_iter_) == int(max_iter), f"the model was fitted for {m.n_iter_} iterations, expected {max_iter}"
    ic = getattr(m, "is_categorical_", None)
    if ncat == 0:
        assert ic is None or not np.any(ic), "a model without positions has categorical features"
    else:
        assert ic is not None and len(ic) == nfeat and bool(np.all(ic[:ncat])) and not bool(np.any(ic[ncat:])), "the categorical mask of the fitted model is not 'the position columns, then the counts'"


def assert_folds(fid, groups, y, k=NFOLD):
    """the folds of the training rows are valid: k non-empty folds, no protein group in a fold and in its training part, both classes in every validation part and training part"""
    assert len(fid) == len(y) == len(groups), "fold ids, labels and groups have different lengths"
    assert sorted(np.unique(fid).tolist()) == list(range(k)), "fewer non-empty folds than requested"
    for f in range(k):
        va, tr = fid == f, fid != f
        assert not np.isin(groups[tr], groups[va]).any(), "a protein group lies in the training part and in the validation part of a fold"
        assert 0 < y[va].sum() < va.sum() and 0 < y[tr].sum() < tr.sum(), "a fold or its training part has a single class"


def select_iter(X, y, fid, cat_idx, grid=GRID, k=NFOLD):
    """training-only choice of the number of iterations: the pooled out-of-fold log-loss over the group-disjoint folds `fid` of the training rows; the smallest minimiser of the grid.
    Returns (iterations*, pooled loss per grid value)."""
    assert len(fid) == len(y) == len(X) and sorted(np.unique(fid).tolist()) == list(range(k)), "the fold ids do not describe k non-empty folds of the rows"
    loss = np.zeros(len(grid))
    yf = y.astype(np.float64)
    for f in range(k):
        tr, va = np.flatnonzero(fid != f), np.flatnonzero(fid == f)
        m = new_model(cat_idx, grid[-1]).fit(X[tr], y[tr])
        check_model(m, len(cat_idx), X.shape[1], grid[-1])
        z = staged_scores(m, X[va], list(grid))
        for gi, it in enumerate(grid):
            zz = z[it]
            loss[gi] += float(np.sum(np.logaddexp(0.0, zz) - yf[va] * zz))
    loss /= len(y)
    assert np.isfinite(loss).all(), "a non-finite cross-validated loss"
    return int(grid[int(np.argmin(loss))]), loss


def run_base(T, base, tok, y, fid, etok, grid, fix_iter, maxit, chunk=CHUNK):
    """ONE model, start to end, for terminus T: features of the training tokens, the number of iterations chosen on the folds `fid`, the final fit of `maxit` iterations, the scores of the
    evaluation tokens at the chosen number ('cv') and at `fix_iter` ('fixed'). The final fit has exactly the largest grid value of iterations.
    Returns dict(loss=pooled out-of-fold loss per grid value, regimes={'cv': (iterations, scores), 'fixed': (iterations, scores)})."""
    assert int(maxit) == int(grid[-1]), "the final fit must have exactly the largest number of iterations of the grid"
    X, cat = features(tok, T, base)
    assert X.shape[1] == n_features(T, base), f"{base}: wrong number of features"
    nstar, loss = select_iter(X, y, fid, cat, grid)
    assert nstar in grid, f"{base}: the chosen number of iterations is not on the grid"
    model = new_model(cat, maxit).fit(X, y)
    check_model(model, len(cat), X.shape[1], maxit)
    z = score_eval(model, etok, T, base, [nstar, fix_iter], chunk)
    return dict(loss=loss, regimes={"cv": (int(nstar), z[nstar]), "fixed": (int(fix_iter), z[fix_iter])})
