"""Window models on the token matrix: the position-only model, the bond-pair and window naive-Bayes baselines, and the one-hot
logistic regressions on a set of token columns (both sides of the bond, only the outward side, only the interior side, the
exact NetCleave window). Nothing is tuned: the settings come from the configuration of an experiment (`configure`)."""
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

from .data import WMAX, posfeat, tokens

LRCFG = dict(C_position_lr=1.0, C_window_lr=0.1, max_iter=1000, tol=1e-4)


def configure(lr):
    LRCFG.update(lr)


class PositionLR:
    name = "Position-LR"

    def fit(self, tok, pf, y):
        self.mu, self.sd = pf.mean(0), pf.std(0) + 1e-9
        self.m = LogisticRegression(C=LRCFG["C_position_lr"], solver="lbfgs", max_iter=LRCFG["max_iter"], tol=LRCFG["tol"]).fit((pf - self.mu) / self.sd, y)
        self.n_iter = int(np.max(self.m.n_iter_))
        return self

    def score(self, tok, pf):
        return self.m.decision_function((pf - self.mu) / self.sd)


class BondPair:
    name = "BondPair"

    def fit(self, tok, pf, y):
        a, b = tok[:, WMAX - 1].astype(int), tok[:, WMAX].astype(int)
        cp = np.bincount((a * 21 + b)[y == 1], minlength=441) + 1.0
        cn = np.bincount((a * 21 + b)[y == 0], minlength=441) + 1.0
        self.t = np.log(cp / cp.sum()) - np.log(cn / cn.sum())
        return self

    def score(self, tok, pf):
        return self.t[tok[:, WMAX - 1].astype(int) * 21 + tok[:, WMAX].astype(int)]


class WindowNB4:
    name = "Window-NB4"
    cols = np.arange(WMAX - 4, WMAX + 4)             # P4..P1, P1'..P4'

    def fit(self, tok, pf, y):
        self.t = np.zeros((8, 21))
        for j, col in enumerate(self.cols):
            cp = np.bincount(tok[y == 1, col].astype(int), minlength=21) + 1.0
            cn = np.bincount(tok[y == 0, col].astype(int), minlength=21) + 1.0
            self.t[j] = np.log(cp / cp.sum()) - np.log(cn / cn.sum())
        return self

    def score(self, tok, pf):
        return sum(self.t[j][tok[:, col].astype(int)] for j, col in enumerate(self.cols))


class ColLR:
    """One-hot logistic regression on a set of token columns (21 categories per column; the window-model recipe: lbfgs, C_window_lr, tolerance)."""

    def __init__(self, name, cols):
        self.name, self.cols = name, np.asarray(cols)

    def _X(self, tok):
        t = tok[:, self.cols].astype(np.int64)
        n, k = t.shape
        ind = (np.arange(k)[None, :] * 21 + t).ravel()
        return sp.csr_matrix((np.ones(len(ind), np.float32), ind, np.arange(0, n * k + 1, k)), shape=(n, k * 21))

    def fit(self, tok, pf, y):
        self.m = LogisticRegression(C=LRCFG["C_window_lr"], solver="lbfgs", max_iter=LRCFG["max_iter"], tol=LRCFG["tol"]).fit(self._X(tok), y)
        self.n_iter = int(np.max(self.m.n_iter_))
        self.W = self.m.coef_.reshape(len(self.cols), 21)
        self.b = float(self.m.intercept_[0])
        return self

    def score(self, tok, pf):
        return self.b + self.W[np.arange(len(self.cols))[None, :], tok[:, self.cols].astype(np.int64)].sum(1)


class WindowLR:
    """The window model on both sides of the bond, half-width w (P_w..P1 and P1'..P_w'): the recipe of ColLR with the columns taken as a slice of the token matrix.
    It is a class of its own and not a ColLR on the same columns because the two do not give identical scores: with a ColLR on these columns, cells of LR-Both8 of one
    study differed from the stored results in the seventh decimal place of the AUROC (26 of the 162 cells of a three-study re-run of the first experiment; every one of them an
    LR-Both8 cell or a difference with LR-Both8), and with this class none differed. The cause was not investigated."""

    def __init__(self, w):
        self.w, self.name = w, f"LR-Both{w}"

    def _X(self, tok):
        w = self.w
        t = tok[:, WMAX - w:WMAX + w].astype(np.int64)
        n = len(t)
        ind = (np.arange(2 * w)[None, :] * 21 + t).ravel()
        return sp.csr_matrix((np.ones(len(ind), np.float32), ind, np.arange(0, n * 2 * w + 1, 2 * w)), shape=(n, 2 * w * 21))

    def fit(self, tok, pf, y):
        self.m = LogisticRegression(C=LRCFG["C_window_lr"], solver="lbfgs", max_iter=LRCFG["max_iter"], tol=LRCFG["tol"]).fit(self._X(tok), y)
        self.n_iter = int(np.max(self.m.n_iter_))
        self.W = self.m.coef_.reshape(2 * self.w, 21)
        self.b = float(self.m.intercept_[0])
        return self

    def score(self, tok, pf):
        t = tok[:, WMAX - self.w:WMAX + self.w].astype(np.int64)
        return self.b + self.W[np.arange(2 * self.w)[None, :], t].sum(1)


def window_model(w):
    """both sides of the bond, half-width w."""
    return WindowLR(w)


def nc_window_cols():
    """token columns of the NetCleave C-bond window: the last 4 residues of the ligand P4..P1 and the next 3 residues P1'..P3'."""
    return np.arange(WMAX - 4, WMAX + 3)


def exact_window_model():
    return ColLR("LR-Both(4+3)", nc_window_cols())


def side_cols(T, side, w):
    """Columns on the outward-facing side (facing away from the ligand) or on the interior-facing side (facing into the ligand) of the T-bond, half-width w.
    N-bond: P1..Pw (upstream of the ligand) are outward, P1'..Pw' (the first residues of the ligand) are interior.
    C-bond: P1..Pw (the last residues of the ligand) are interior, P1'..Pw' (downstream) are outward.
    For w larger than the ligand length (median about 15) the interior-facing residues are not all residues of the ligand."""
    p_side = np.arange(WMAX - w, WMAX)                       # P_w .. P1
    pp_side = np.arange(WMAX, WMAX + w)                      # P1' .. P_w'
    if T == "N":
        outward, interior = p_side, pp_side
    elif T == "C":
        interior, outward = p_side, pp_side
    else:
        raise ValueError(T)
    if side == "outward":
        return outward
    if side == "interior":
        return interior
    raise ValueError(side)


def side_model(T, side, w):
    return ColLR(f"{'LR-Out' if side == 'outward' else 'LR-In'}{w}", side_cols(T, side, w))


def make_models(names):
    """models of the terminal-predictability experiment by public name: Position-LR, BondPair, Window-NB4, LR-Both<w>."""
    out = []
    for n in names:
        if n == "Position-LR":
            out.append(PositionLR())
        elif n == "BondPair":
            out.append(BondPair())
        elif n == "Window-NB4":
            out.append(WindowNB4())
        elif n.startswith("LR-Both") and n[7:].isdigit():
            out.append(window_model(int(n[7:])))
        else:
            raise ValueError(f"unknown model {n!r}")
    return out


def fit_score(c, P, B, y, models, eval_tok, eval_pf):
    """Fit every model on the training bonds and score the evaluation bonds. Models with needs_bonds build their own training rows
    (fit_bonds). Returns (scores, fit_info): the lbfgs iterations and whether the limit was reached; for networks the epochs run and
    whether the epoch cap, not early stopping, ended the training."""
    tok = tokens(c, P, B)
    pf = posfeat(B, c.L[P])
    scores, info = {}, {}
    for m in models:
        if getattr(m, "needs_bonds", False):
            scores[m.name] = m.fit_bonds(c, P, B, y).score(eval_tok, eval_pf)
        else:
            scores[m.name] = m.fit(tok, pf, y).score(eval_tok, eval_pf)
        if hasattr(m, "capped"):
            info[m.name] = dict(n_iter=m.n_iter, hit_limit=bool(m.capped), kind="nn", train_rows=m.n_train_rows)
        elif hasattr(m, "n_iter"):
            info[m.name] = dict(n_iter=m.n_iter, hit_limit=bool(m.n_iter >= LRCFG["max_iter"]))
    return scores, info
