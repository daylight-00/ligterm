"""Composition versus position within the interior side of the bond: the linear family.

Six logistic regressions on the eight interior residues of the terminal bond: LR-Full8 (position-specific one-hot), LR-Comp8 (the 20 residue counts only),
LR-Cys8 (the cysteine count alone), LR-Prox2 (the two positions next to the bond), LR-Dist6 (the six distal positions) and LR-Comp8+Prox2. Each is fitted at a
constant fixed in advance (fit "fixed") and at a constant chosen on the training rows only (fit "cv"). The count features are standardised with training statistics
only; the choice of C uses group-disjoint folds of the training rows.
Token matrix: columns 0..15 = P16..P1, 16..31 = P1'..P16'; token 0..19 = the 20 residues (order AA), 20 = pad (outside the protein, or non-standard).
The interior side of the T-bond: N-bond P1'..P8' = columns 16..23; C-bond P8..P1 = columns 8..15. The two proximal positions are the two interior positions nearest the bond.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

from .data import AA, WMAX

W8 = 8
FIXC = 0.1                                        # the constant of the window models, fixed in advance
GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
NFOLD = 3
MAXITER, TOL = 1000, 1e-4
CYS = AA.index("C")
CHUNK = 500_000                                   # rows per block when the evaluation bonds are scored


def interior_cols(T, w=W8):
    """token columns of the interior-facing w residues, ordered from the bond outward for the N-bond (P1'..P_w') and from the far end to the bond for the C-bond (P_w..P1)."""
    if T == "N":
        return np.arange(WMAX, WMAX + w)
    if T == "C":
        return np.arange(WMAX - w, WMAX)
    raise ValueError(T)


def proximal_cols(T):
    ic = interior_cols(T)
    return ic[:2] if T == "N" else ic[-2:]


def distal_cols(T):
    ic = interior_cols(T)
    return ic[2:] if T == "N" else ic[:-2]


# model -> (columns of the count features or None, columns of the one-hot features or None, residues kept by the count part or None = all 20)
SPECS = {
    "LR-Full8": (None, "interior", None),
    "LR-Comp8": ("interior", None, None),
    "LR-Cys8": ("interior", None, [CYS]),
    "LR-Prox2": (None, "proximal", None),
    "LR-Dist6": (None, "distal", None),
    "LR-Comp8+Prox2": ("interior", "proximal", None),
}
BASES = tuple(SPECS)
COLSETS = {"interior": interior_cols, "proximal": proximal_cols, "distal": distal_cols}


def counts20(tok, cols):
    """(n, 20) float64: how many of the residues at `cols` are residue a (pad and non-standard positions count for none)."""
    t = tok[:, cols]
    out = np.zeros((len(tok), 20), np.float64)
    for a in range(20):
        out[:, a] = (t == a).sum(1)
    return out


def onehot(tok, cols):
    """sparse one-hot, 21 categories per column (the construction of models.ColLR, so that LR-Full8 at the fixed constant equals LR-In8)."""
    t = tok[:, cols].astype(np.int64)
    n, k = t.shape
    ind = (np.arange(k)[None, :] * 21 + t).ravel()
    return sp.csr_matrix((np.ones(len(ind), np.float32), ind, np.arange(0, n * k + 1, k)), shape=(n, k * 21))


def fold_ids(groups, k=NFOLD):
    """deterministic group-disjoint folds: the groups, largest first (ties by group id), each goes to the fold with the fewest rows so far. Returns the fold of every row."""
    ug, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True)
    order = np.lexsort((ug, -cnt))
    load = np.zeros(k)
    fog = np.empty(len(ug), np.int64)
    for gi in order:
        f = int(np.argmin(load))
        fog[gi] = f
        load[f] += cnt[gi]
    return fog[inv]


class FeatLR:
    """One of the six models. fit_rows() fits on a subset of the training rows (for the folds); score_tok() scores token rows from the raw-scale weights, so that no design matrix of the evaluation bonds is built."""

    def __init__(self, base, T):
        self.base, self.T = base, T
        cc, oc, only = SPECS[base]
        self.ccols = COLSETS[cc](T) if cc else None
        self.ocols = COLSETS[oc](T) if oc else None
        self.only = only

    def raw(self, tok):
        cnt = counts20(tok, self.ccols) if self.ccols is not None else None
        if cnt is not None and self.only is not None:
            cnt = cnt[:, self.only]
        oh = onehot(tok, self.ocols) if self.ocols is not None else None
        return cnt, oh

    def fit_rows(self, raw, y, rows, C):
        cnt, oh = raw
        mu = sd = cz = None
        if cnt is not None:
            c = cnt[rows]
            mu, sd = c.mean(0), c.std(0) + 1e-9               # training rows only (the training part of a fold in the cross-validation)
            cz = (c - mu) / sd
        if cnt is not None and oh is None:
            X = cz
        elif cnt is None:
            X = oh[rows]
        else:
            X = sp.hstack([sp.csr_matrix(cz), oh[rows]], format="csr")
        m = LogisticRegression(C=C, solver="lbfgs", max_iter=MAXITER, tol=TOL).fit(X, y[rows])
        coef, b = m.coef_.ravel().astype(np.float64), float(m.intercept_[0])
        nc = 0 if cnt is None else cnt.shape[1]
        wc = None
        if cnt is not None:
            wz = coef[:nc] / sd                                  # back to the raw count scale
            b -= float((coef[:nc] * mu / sd).sum())
            wc = np.zeros(20)
            wc[self.only if self.only is not None else np.arange(20)] = wz
        W = coef[nc:].reshape(len(self.ocols), 21) if oh is not None else None
        n_iter = int(np.max(m.n_iter_))
        return dict(b=b, wc=wc, W=W, C=float(C), n_iter=n_iter, hit_limit=bool(n_iter >= MAXITER), _coef=coef, _b0=float(m.intercept_[0]), _mu=mu, _sd=sd)

    def score_tok(self, st, tok):
        s = np.full(len(tok), st["b"], np.float64)
        if self.ccols is not None:
            for i in range(0, len(tok), CHUNK):                  # the counts times the raw-scale weights, accumulated in a fixed order over the 20 residues (no BLAS blocking): the score
                cn, part = counts20(tok[i:i + CHUNK], self.ccols), s[i:i + CHUNK]          # is a deterministic function of the count vector of the row, so equal compositions get bitwise equal scores
                for a in range(20):
                    part += cn[:, a] * st["wc"][a]
        if self.ocols is not None:
            s += st["W"][np.arange(len(self.ocols))[None, :], tok[:, self.ocols].astype(np.int64)].sum(1)
        return s

    def check_scoring(self, raw, st, tok, rows, tol=1e-9):
        """Self-check on training rows only: score_tok (raw-scale weights) must equal the prediction in design space (standardised counts and one-hot columns times the fitted coefficients plus the intercept)."""
        cnt, oh = raw
        parts = []
        if cnt is not None:
            parts.append(sp.csr_matrix((cnt[rows] - st["_mu"]) / st["_sd"]))
        if oh is not None:
            parts.append(oh[rows].astype(np.float64))
        X = parts[0] if len(parts) == 1 else sp.hstack(parts, format="csr")
        z_ref = np.asarray(X @ st["_coef"]).ravel() + st["_b0"]
        d = float(np.max(np.abs(z_ref - self.score_tok(st, tok[rows]))))
        assert d < tol, f"{self.base}: the raw-scale scoring differs from the design-space prediction by {d:.2e}"
        return d


def select_C(model, raw, tok, y, groups, grid=GRID, k=NFOLD):
    """Training-only choice of C: the pooled out-of-fold log-loss over group-disjoint folds of the training rows; the smallest minimiser of the grid.
    Returns (C*, pooled loss per grid value, number of cross-validation fits that reached the lbfgs iteration limit)."""
    fid = fold_ids(groups, k)
    assert len(np.unique(fid)) == k, "fewer non-empty folds than requested"
    loss = np.zeros(len(grid))
    hits = 0
    yf = y.astype(np.float64)
    for f in range(k):
        tr, va = np.flatnonzero(fid != f), np.flatnonzero(fid == f)
        assert not np.isin(groups[tr], groups[va]).any(), "a protein group lies in the training part and in the validation part of a fold"
        for gi, C in enumerate(grid):
            st = model.fit_rows(raw, y, tr, C)
            hits += int(st["hit_limit"])
            z = model.score_tok(st, tok[va])
            loss[gi] += float(np.sum(np.logaddexp(0.0, z) - yf[va] * z))
    loss /= len(y)
    assert np.isfinite(loss).all(), "a non-finite cross-validated loss"
    return float(grid[int(np.argmin(loss))]), loss, hits


def mask_pp(pp, keep):
    """per-protein arrays with the proteins outside `keep` removed from the macro averages (the same arrays boot reads)."""
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in pp.items()}
    for k in ("bond_auroc", "bond_auprc", "bond_recall_at_10pct"):
        out[k][~keep] = np.nan
    for k in ("local_n", "local_d", "local_interior_n", "local_interior_d", "local_outward_n", "local_outward_d"):
        out[k][~keep] = 0.0
    out["npos"][~keep] = 0
    return out
