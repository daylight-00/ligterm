"""Metrics of the terminal-bond analyses, per protein and macro-averaged, with group bootstrap intervals.

Bond-AUROC (protein-macro; ties 0.5), Bond-AUPRC, Bond recall at the top 10 % of bonds (ties split proportionally) and the
Local-AUC: the share of pairs in which an observed terminal bond scores above a non-observed bond 1 to 3 positions away,
pooled over proteins (ties 0.5). Labels: 1 observed terminal bond, 0 other bond, -1 excluded. The order in which the bootstrap
draws are taken is part of the result: functions that take `rng` consume it in the order of the calls.
"""
import math

import numpy as np
from scipy.stats import rankdata


def auroc(ss, yy):
    n1 = int(yy.sum())
    n0 = len(yy) - n1
    r = rankdata(ss)
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def ap_ties(sc, y):
    order = np.argsort(-sc, kind="mergesort")
    s_, y_ = sc[order], y[order]
    tp = np.cumsum(y_)
    fp = np.cumsum(1 - y_)
    d = np.r_[np.flatnonzero(np.diff(s_)), len(s_) - 1]
    prec = tp[d] / (tp[d] + fp[d])
    rec = tp[d] / tp[-1]
    return float(np.sum((rec - np.r_[0.0, rec[:-1]]) * prec))


def recall_at_frac(sc, y, frac=0.10):
    n = len(sc)
    k = int(math.ceil(frac * n))
    order = np.argsort(-sc, kind="mergesort")
    s_, y_ = sc[order], y[order]
    thr = s_[k - 1]
    above, tie = s_ > thr, s_ == thr
    need = k - int(above.sum())
    return float((y_[above].sum() + y_[tie].sum() * need / tie.sum()) / y.sum())


def local_counts(s, y):
    """pooled local pairs of one protein: observed bond against non-observed bonds at +-1..3 (ties 0.5); returns (concordance, pairs)."""
    n = len(y)
    pos = np.flatnonzero(y == 1)
    num = den = 0.0
    for d in (-3, -2, -1, 1, 2, 3):
        j = pos + d
        v = (j >= 0) & (j < n)
        a, j = pos[v], j[v]
        o = y[j] == 0
        a, j = a[o], j[o]
        num += float((s[a] > s[j]).sum() + 0.5 * (s[a] == s[j]).sum())
        den += float(len(a))
    return num, den


def local_counts_by_side(s, y, T):
    """The local pairs split by the side of the non-observed bond: 'interior' = on the interior-facing side
    (N-bond: higher bond index; C-bond: lower bond index), 'outward' = on the outward-facing side. Returns (num_in, den_in, num_out, den_out)."""
    n = len(y)
    pos = np.flatnonzero(y == 1)
    num = {"in": 0.0, "out": 0.0}
    den = {"in": 0.0, "out": 0.0}
    for d in (-3, -2, -1, 1, 2, 3):
        side = "in" if ((T == "N" and d > 0) or (T == "C" and d < 0)) else "out"
        j = pos + d
        v = (j >= 0) & (j < n)
        a, j = pos[v], j[v]
        o = y[j] == 0
        a, j = a[o], j[o]
        num[side] += float((s[a] > s[j]).sum() + 0.5 * (s[a] == s[j]).sum())
        den[side] += float(len(a))
    return num["in"], den["in"], num["out"], den["out"]


def per_protein(scores, lab, ebase, nb, keep=None, T=None):
    """Per-protein metrics of the evaluation proteins; with T also the side-split local pairs (arrays local_interior_n/d, local_outward_n/d)."""
    n = len(nb)
    out = dict(bond_auroc=np.full(n, np.nan), bond_auprc=np.full(n, np.nan), bond_recall_at_10pct=np.full(n, np.nan),
               local_n=np.zeros(n), local_d=np.zeros(n), npos=np.zeros(n, int))
    for k in range(n):
        if keep is not None and not keep[k]:
            continue
        s = scores[ebase[k]:ebase[k] + nb[k]]
        y = lab[ebase[k]:ebase[k] + nb[k]]
        m = y >= 0
        ss, yy = s[m], y[m]
        n1 = int(yy.sum())
        if n1 == 0 or n1 == len(yy):
            continue
        out["bond_auroc"][k] = auroc(ss, yy)
        out["bond_auprc"][k] = ap_ties(ss, yy)
        out["bond_recall_at_10pct"][k] = recall_at_frac(ss, yy)
        out["npos"][k] = n1
        out["local_n"][k], out["local_d"][k] = local_counts(s, y)
    if T is not None:
        for key in ("local_interior_n", "local_interior_d", "local_outward_n", "local_outward_d"):
            out[key] = np.zeros(n)
        for i in range(n):
            if np.isnan(out["bond_auroc"][i]):
                continue
            s = scores[ebase[i]:ebase[i] + nb[i]]
            y = lab[ebase[i]:ebase[i] + nb[i]]
            out["local_interior_n"][i], out["local_interior_d"][i], out["local_outward_n"][i], out["local_outward_d"][i] = local_counts_by_side(s, y, T)
    return out


def boot(pp, gid, rng, B=2000):
    """Macro means with 95 % percentile intervals from a bootstrap over sequence groups (conditional on the study).
    With side-split arrays in pp the side-split Local-AUC and its number of pairs are added, with the same resampled groups."""
    ok = ~np.isnan(pp["bond_auroc"])
    g = gid[ok]
    ug, inv = np.unique(g, return_inverse=True)
    G = len(ug)
    draw = rng.integers(0, G, size=(B, G))
    cnt = np.bincount(inv, minlength=G).astype(float)
    res = {}
    for key in ("bond_auroc", "bond_auprc", "bond_recall_at_10pct"):
        v = pp[key][ok]
        sm = np.bincount(inv, weights=v, minlength=G)
        est = v.mean()
        bs = sm[draw].sum(1) / cnt[draw].sum(1)
        res[key] = (float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    nn = np.bincount(inv, weights=pp["local_n"][ok], minlength=G)
    dd = np.bincount(inv, weights=pp["local_d"][ok], minlength=G)
    if dd.sum() > 0:
        est = nn.sum() / dd.sum()
        bs = nn[draw].sum(1) / np.maximum(dd[draw].sum(1), 1)
        res["local_auc"] = (float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    else:
        res["local_auc"] = (float("nan"),) * 3
    res["n_proteins"] = int(ok.sum())
    res["n_groups"] = int(G)
    res["n_pos"] = int(pp["npos"][ok].sum())
    if "local_interior_n" in pp:
        res["local_auc_interior"] = boot_ratio(pp["local_interior_n"], pp["local_interior_d"], gid, ok, rng, B)
        res["local_auc_outward"] = boot_ratio(pp["local_outward_n"], pp["local_outward_d"], gid, ok, rng, B)
        res["local_pairs_interior"] = float(pp["local_interior_d"][ok].sum())
        res["local_pairs_outward"] = float(pp["local_outward_d"][ok].sum())
    return res


def boot_ratio(num, den, gid, ok, rng, B=2000):
    """Pooled ratio sum(num)/sum(den) over the proteins with ok, with a 95 % percentile interval from a bootstrap over groups
    (the same resampling as boot)."""
    g = gid[ok]
    ug, inv = np.unique(g, return_inverse=True)
    G = len(ug)
    draw = rng.integers(0, G, size=(B, G))
    nn = np.bincount(inv, weights=num[ok], minlength=G)
    dd = np.bincount(inv, weights=den[ok], minlength=G)
    if dd.sum() == 0:
        return (float("nan"),) * 3
    est = nn.sum() / dd.sum()
    bs = nn[draw].sum(1) / np.maximum(dd[draw].sum(1), 1)
    return float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_diff(ppA, ppB, gid, rng, B=2000):
    """Paired difference of the macro Bond-AUROC of two models on the same proteins; 95 % percentile interval from a group bootstrap."""
    ok = ~np.isnan(ppA["bond_auroc"]) & ~np.isnan(ppB["bond_auroc"])
    v = (ppA["bond_auroc"] - ppB["bond_auroc"])[ok]
    ug, inv = np.unique(gid[ok], return_inverse=True)
    G = len(ug)
    draw = rng.integers(0, G, size=(B, G))
    cnt = np.bincount(inv, minlength=G).astype(float)
    sm = np.bincount(inv, weights=v, minlength=G)
    bs = sm[draw].sum(1) / cnt[draw].sum(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def shuffle_control(scores, lab, ebase, nb, rng, nshuf=20):
    """Shuffled-label control: permute the labels of the non-excluded bonds inside every protein; the macro AUROC should be 0.5."""
    vals = []
    for _ in range(nshuf):
        a = []
        for k in range(len(nb)):
            y = lab[ebase[k]:ebase[k] + nb[k]]
            m = np.flatnonzero(y >= 0)
            yy = y[m]
            n1 = int(yy.sum())
            if n1 == 0 or n1 == len(yy):
                continue
            yp = rng.permutation(yy)
            a.append(auroc(scores[ebase[k]:ebase[k] + nb[k]][m], yp))
        vals.append(float(np.mean(a)))
    return float(np.mean(vals)), float(np.std(vals))


def valid_proteins(lab, ebase, nb):
    """Which evaluation proteins have a defined Bond-AUROC under a label set (at least one observed and one other bond). It depends on the labels only, not on a score."""
    ok = np.zeros(len(nb), bool)
    for k in range(len(nb)):
        y = lab[ebase[k]:ebase[k] + nb[k]]
        yy = y[y >= 0]
        n1 = int(yy.sum())
        ok[k] = not (n1 == 0 or n1 == len(yy))
    return ok


def skip_boot(ok, gid, rng, B=2000, side_split=False):
    """Advance the bootstrap stream exactly as boot() (3 draws with side-split arrays, 1 without) would do for the proteins `ok`, without a score.
    Used by an entry point that runs only a part of the models of a study, so that the intervals it writes are the ones of the full run."""
    G = len(np.unique(gid[ok]))
    for _ in range(3 if side_split else 1):
        rng.integers(0, G, size=(B, G))


def skip_boot_diff(ok, gid, rng, B=2000):
    G = len(np.unique(gid[ok]))
    rng.integers(0, G, size=(B, G))
