"""Independent brute-force reference of the peptide metrics of `ligterm_repro.peptide_ranking` for ONE protein, and the deterministic choice of the proteins on which the development runner checks the fast implementation against it.
It shares no code with peptide_eval: it enumerates every candidate and loops over every pair. The unit tests compare it with a second reference written in the test file."""
import numpy as np

VARIANTS = ("LEN", "N", "C", "NC", "NCL")


def brute(sN, sC, L, positives, excluded, g, variant, lo=9, hi=30, delta=3, labN=None, labC=None):
    """-> dict(F, M_num, M_den, X_num, X_den, n_pos, n_pos_dropped, n_neg) for one protein and one variant; labN / labC given = CLEAN negatives (a candidate is a negative only if both of its bonds carry label 0; a near-miss needs label 0 at each SHIFTED boundary)"""
    clean = labN is not None
    cand = [(s, e) for s in range(2, L) for e in range(s, L) if lo <= e - s + 1 <= hi]
    cset = set(cand)
    posl = list(positives)
    pos = sorted({p for p in posl if p in cset})
    ps = set(pos)
    ex = set(excluded) - ps
    base = [q for q in cand if q not in ps and q not in ex]
    neg = [q for q in base if (not clean) or (labN[q[0] - 2] == 0 and labC[q[1] - 1] == 0)]
    sc = {}
    for s, e in cand:
        n, c, gl = sN[s - 2], sC[e - 1], g[e - s + 1 - lo]
        sc[(s, e)] = {"LEN": gl, "N": n, "C": c, "NC": n + c, "NCL": n + c + gl}[variant]

    def win(a, b):
        return 1.0 if a > b else (0.5 if a == b else 0.0)
    F = (sum(win(sc[p], sc[q]) for p in pos for q in neg) / (len(pos) * len(neg))) if pos and neg else float("nan")
    mn = md = xn = xd = 0.0
    for p in pos:
        for q in neg:
            if q[1] - q[0] == p[1] - p[0]:
                mn += win(sc[p], sc[q])
                md += 1.0
        for q in base:
            di, dj = q[0] - p[0], q[1] - p[1]
            if abs(di) <= delta and abs(dj) <= delta and (di, dj) != (0, 0):
                if clean and ((di != 0 and labN[q[0] - 2] != 0) or (dj != 0 and labC[q[1] - 1] != 0)):
                    continue
                xn += win(sc[p], sc[q])
                xd += 1.0
    return dict(F=F, M_num=mn, M_den=md, X_num=xn, X_den=xd, n_pos=len(pos), n_pos_dropped=len(set(posl)) - len(pos), n_neg=len(neg))


def select_guard_proteins(L, npos, short_n=2, dense_n=4, maxlen=200, extra=6):
    """the order in which the evaluation proteins are checked: the `short_n` shortest proteins first, then the proteins of at most `maxlen` residues with the MOST positive candidates (ties: the shorter, then the earlier), `dense_n` of them as the base set;
    `extra` more of the same ordering are the reserve from which the runner draws until the clean length-matched and near-miss populations have been compared (non-vacuously). Returns (base, reserve): lists of indices into the arrays L and npos."""
    L, npos = np.asarray(L), np.asarray(npos)
    short = np.argsort(L, kind="stable")[:short_n].tolist()
    elig = np.flatnonzero((L <= maxlen) & (npos > 0))
    order = elig[np.lexsort((elig, L[elig], -npos[elig]))].tolist()
    base = list(short)
    for i in order:
        if len([b for b in base if b not in short]) >= dense_n:
            break
        if i not in base:
            base.append(i)
    reserve = [i for i in order if i not in base][:extra]
    return base, reserve
