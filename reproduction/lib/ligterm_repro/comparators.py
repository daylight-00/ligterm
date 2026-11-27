"""The published methods of the comparison, as model objects on the token matrix of `data`.

Two kinds, never to be mixed:
  * retrained methods, fitted on the strict training set of the held-out study with the published features, architecture and
    training rules: the Paul-style motif (both termini) and the NetCleave-style network (C terminus; 'released decoys' or
    'protocol negatives');
  * the NetCleave predictor as released (C terminus; the class-II weights of the BSC fork; fit() does nothing), whose training data
    overlap with the held-out studies is measured, not assumed.
Third-party material (weights, descriptor table, training sets) is read from the NetCleave directory given to `configure`;
it is not redistributed with this package (see scripts/00_fetch_data.py).
"""
import csv

import numpy as np

from .data import AA, WMAX, tokens
from .metrics import auroc

EXT = None                                           # directory of the NetCleave fork (code/, weights/, training/)
NCW = slice(WMAX - 4, WMAX + 3)                      # token columns 12..18 = P4..P1, P1'..P3' (NetCleave: last 4 ligand residues and the next 3; a C-bond window)


def configure(netcleave_dir):
    global EXT, DESC
    EXT = netcleave_dir
    DESC = None


# ====================================================================== Paul et al. 2018 cleavage motif, retrained
class PaulMotif:
    """Each terminus is scored by the 3 residues (1 flank + 2 ligand) around it: N-motif = (P1, P1', P2') of the N-bond, C-motif = (P2, P1, P1') of the C-bond;
    motif score = sum over the three positions of log10(F_terminus / F_background), F_terminus = residue frequency among the training ligand termini
    (unique training positives), F_background = residue frequency among the uniformly sampled non-terminal bonds of the training proteins. Add-one pseudo-counts.
    Not reproduced: the length term, the binning of the motif score into an empirical probability and the product with the other terminus (they need
    candidate peptides, not bonds); the data rules of the paper (ligand lengths 13-23, alleles with at least 50 entries, 199 potential false-positive antigens)."""

    def __init__(self, T):
        self.T = T
        self.name = "Paul-style motif"
        self.cols = np.array([WMAX - 1, WMAX, WMAX + 1] if T == "N" else [WMAX - 2, WMAX - 1, WMAX])

    def fit(self, tok, pf, y):
        self.t = np.zeros((3, 21))
        for j, col in enumerate(self.cols):
            cp = np.bincount(tok[y == 1, col].astype(int), minlength=21)[:20] + 1.0
            cn = np.bincount(tok[y == 0, col].astype(int), minlength=21)[:20] + 1.0
            self.t[j, :20] = np.log10(cp / cp.sum()) - np.log10(cn / cn.sum())
        return self

    def score(self, tok, pf):
        return sum(self.t[j][tok[:, col].astype(int)] for j, col in enumerate(self.cols))


# ====================================================================== NetCleave: descriptors, encoding, network
def load_descriptors():
    """The 20 x 48 QSAR descriptor table of the NetCleave repository, standardised across the 20 residues like sklearn's StandardScaler (population std)."""
    rows = list(csv.reader(open(f"{EXT}/code/QSAR_table.csv")))
    body = rows[1:]
    order = [r[0] for r in body]
    X = np.array([[float(v) for v in r[1:]] for r in body], np.float64)
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd
    out = np.zeros((21, X.shape[1]), np.float32)                # token ids follow AA; row 20 (pad) stays 0
    for i, a in enumerate(AA):
        out[i] = Z[order.index(a)]
    return out


DESC = None


def encode(tok7):
    """(n, 7) int tokens to the (n, 336) float32 descriptor matrix (position-major, descriptor order of the table)."""
    global DESC
    if DESC is None:
        DESC = load_descriptors()
    return DESC[tok7.astype(np.int64)].reshape(len(tok7), -1)


def window_complete(tok7):
    return (tok7 < 20).all(1)


class NetCleaveFrozen:
    """The class-II NetCleave predictor as released (BSC-CNS-EAPM fork, weights converted to .npz): 336 -> 336 tanh -> 112 tanh -> 1 sigmoid on the
    7-residue window P4..P1, P1'..P3'. Scores are logits (monotone in the published probability). Bonds whose window is incomplete get -1e9."""

    def __init__(self, which="II_mass-spectrometry_HLA", name="NetCleave (published weights)"):
        self.name, self.which = name, which
        z = np.load(f"{EXT}/weights/{which}.npz")
        self.W1, self.b1 = z["dense__dense__kernel:0"].astype(np.float32), z["dense__dense__bias:0"].astype(np.float32)
        self.W2, self.b2 = z["dense_1__dense_1__kernel:0"].astype(np.float32), z["dense_1__dense_1__bias:0"].astype(np.float32)
        self.W3, self.b3 = z["dense_2__dense_2__kernel:0"].astype(np.float32), z["dense_2__dense_2__bias:0"].astype(np.float32)

    def fit(self, tok, pf, y):
        return self

    def logit7(self, tok7):
        X = encode(tok7)
        h1 = np.tanh(X @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        return (h2 @ self.W3 + self.b3)[:, 0]

    def score(self, tok, pf):
        w = tok[:, NCW].astype(np.int64)
        code = np.zeros(len(w), np.int64)
        for k in range(7):
            code = code * 21 + w[:, k]
        uc, inv = np.unique(code, return_inverse=True)
        uw = np.zeros((len(uc), 7), np.int64)
        c = uc.copy()
        for k in range(6, -1, -1):
            uw[:, k] = c % 21
            c //= 21
        out = np.full(len(uc), -1e9, np.float64)
        ok = window_complete(uw)
        idx = np.flatnonzero(ok)
        for s in range(0, len(idx), 100000):
            sel = idx[s:s + 100000]
            out[sel] = self.logit7(uw[sel])
        return out[inv]


def netcleave_training_windows(which="II_mass-spectrometry_HLA"):
    """The 7-mers of the released class-II training set of the fork: (set of positive windows, set of all windows) as base-21 codes over the token ids."""
    idx = {a: i for i, a in enumerate(AA)}
    pos, allw = set(), set()
    with open(f"{EXT}/training/{which}_sequence_class.txt") as f:
        next(f)
        for line in f:
            s, c = line.rstrip("\n").split("\t")
            if len(s) != 7 or any(ch not in idx for ch in s):
                continue
            code = 0
            for ch in s:
                code = code * 21 + idx[ch]
            allw.add(code)
            if c == "1":
                pos.add(code)
    return pos, allw


def window_codes(tok):
    w = tok[:, NCW].astype(np.int64)
    code = np.zeros(len(w), np.int64)
    for k in range(7):
        code = code * 21 + w[:, k]
    return code


# ====================================================================== NetCleave architecture, retrained (torch)
class NetCleaveNet:
    """The NetCleave network retrained on the strict training set of the held-out study (C terminus only, as published).
    Architecture and optimiser as in the released code: 336 -> 336 tanh -> 112 tanh -> dropout 0.5 -> 1 sigmoid, Glorot-normal weights, SGD lr 0.01 without
    momentum, binary cross-entropy, batch 64. decoys='adjacent' (name 'NetCleave-style (released decoys)'): training negatives as released = exactly the two
    windows shifted by -1 and +1 around each positive (no de-duplication), early stopping on the validation MCC at 0.5 (patience 7, last-epoch weights).
    decoys='protocol' (name 'NetCleave-style (protocol negatives)'): the sampled non-terminal bonds of the training set, early stopping on the validation
    AUROC (the 1:6 class ratio makes the MCC at 0.5 useless). Deviations from the code: at most `max_epochs` epochs, a random 70/30 train / validation split
    of the training rows, torch instead of Keras."""

    def __init__(self, T, decoys, max_epochs=100, seed=0, batch=64, lr=0.01, dropout=0.5):
        assert decoys in ("adjacent", "protocol")
        self.T, self.decoys, self.max_epochs, self.seed, self.batch, self.lr, self.dropout = T, decoys, max_epochs, seed, batch, lr, dropout
        self.name = "NetCleave-style (released decoys)" if decoys == "adjacent" else "NetCleave-style (protocol negatives)"
        self.needs_bonds = True

    def _rows(self, c, P, B, y):
        if self.decoys == "protocol":
            tok = tokens(c, P, B)
            keep = window_complete(tok[:, NCW])
            return tok[keep], y[keep]
        pos = y == 1
        pp, bb = P[pos], B[pos]
        L = c.L[pp]
        dp, db = [], []
        for d in (-1, 1):                                      # as released: exactly the two shifted windows of every positive; no de-duplication and no removal of decoys that are themselves positive
            b2 = bb + d
            ok = (b2 >= 1) & (b2 <= L - 1)
            dp.append(pp[ok])
            db.append(b2[ok])
        dp, db = np.concatenate(dp), np.concatenate(db)
        tok_pos = tokens(c, pp, bb)
        tok_dec = tokens(c, dp, db)
        tok = np.concatenate([tok_pos, tok_dec])
        yy = np.concatenate([np.ones(len(tok_pos), np.int8), np.zeros(len(tok_dec), np.int8)])
        keep = window_complete(tok[:, NCW])
        return tok[keep], yy[keep]

    def fit_bonds(self, c, P, B, y):
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        tok, yy = self._rows(c, P, B, y)
        X = encode(tok[:, NCW])
        perm = rng.permutation(len(X))
        nv = int(0.3 * len(X))
        va, tr = perm[:nv], perm[nv:]
        Xtr, ytr = torch.from_numpy(X[tr]), torch.from_numpy(yy[tr].astype(np.float32))
        Xva, yva = torch.from_numpy(X[va]), yy[va]
        net = torch.nn.Sequential(torch.nn.Linear(336, 336), torch.nn.Tanh(), torch.nn.Linear(336, 112), torch.nn.Tanh(), torch.nn.Dropout(self.dropout), torch.nn.Linear(112, 1))
        lin = [m for m in net if isinstance(m, torch.nn.Linear)]
        for k_, m in enumerate(lin):                           # glorot_normal for the two hidden layers, glorot_uniform for the output layer; all biases zero
            (torch.nn.init.xavier_normal_ if k_ < 2 else torch.nn.init.xavier_uniform_)(m.weight)
            torch.nn.init.zeros_(m.bias)
        opt = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=0.0)
        lossf = torch.nn.BCEWithLogitsLoss()
        best, bad, epochs, hist = -2.0, 0, 0, []
        for epoch in range(self.max_epochs):
            net.train()
            order = torch.from_numpy(rng.permutation(len(Xtr)))
            for s in range(0, len(order), self.batch):
                ix = order[s:s + self.batch]
                opt.zero_grad()
                lossf(net(Xtr[ix])[:, 0], ytr[ix]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                z = net(Xva)[:, 0].numpy()
            if self.decoys == "adjacent":                                    # as published: validation MCC at 0.5
                pr = (z > 0).astype(np.int64)
                tp = float(((pr == 1) & (yva == 1)).sum())
                tn = float(((pr == 0) & (yva == 0)).sum())
                fp = float(((pr == 1) & (yva == 0)).sum())
                fn = float(((pr == 0) & (yva == 1)).sum())
                den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                crit = (tp * tn - fp * fn) / (den + 1e-7)
            else:
                crit = auroc(z, yva.astype(np.int8))
            hist.append(float(crit))
            epochs = epoch + 1
            if crit > best:
                best, bad = crit, 0
            else:
                bad += 1
                if bad >= 7:                                                  # early stopping with patience 7; the weights of the last epoch are kept
                    break
        self.net = net.eval()
        self.n_iter = epochs
        self.capped = bool(epochs >= self.max_epochs)
        self.hist = hist
        self.n_train_rows = int(len(tr))
        return self

    def fit(self, tok, pf, y):
        raise RuntimeError("NetCleaveNet is fitted with fit_bonds (it builds its own training rows)")

    def score(self, tok, pf):
        import torch
        w = tok[:, NCW]
        ok = window_complete(w)
        out = np.full(len(w), -1e9, np.float64)
        idx = np.flatnonzero(ok)
        with torch.no_grad():
            for s in range(0, len(idx), 100000):
                sel = idx[s:s + 100000]
                out[sel] = self.net(torch.from_numpy(encode(w[sel])))[:, 0].numpy()
        return out


def make_models(T, nc):
    """The comparators of terminus T. NetCleave is a C-terminal method: no N-terminal NetCleave model exists and none is invented."""
    ms = [PaulMotif(T)]
    if T == "C":
        ms += [NetCleaveNet(T, "adjacent", nc["max_epochs"], nc["seed"]), NetCleaveNet(T, "protocol", nc["max_epochs"], nc["seed"]), NetCleaveFrozen(nc["frozen_weights"])]
    return ms
