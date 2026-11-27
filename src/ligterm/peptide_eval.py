"""Peptide-level evaluation of one protein (an evaluation tool of the development protocol, not needed for scoring): a pure function of the bond scores, the observed peptides, the excluded peptides and (optionally) the primary bond labels; no study or file knowledge, so it is tested against brute force.
Conventions (protocol v1): residues 1-based; bond i between residues i and i+1; a peptide (start, end) has its N-terminal bond at start-1 and its C-terminal bond at end, so both bonds lie inside the protein only for start >= 2 and end <= L-1; scores s_N, s_C and labels are arrays over the bonds 1..L-1 (index k = bond k+1).
Candidates: every (start, end) with lo <= end-start+1 <= hi and both bonds inside the protein. Positives: the observed peptides of the evaluated study that are candidates. Excluded (never negatives): peptides equal to an observed unit of another study or to a repeat of a positive.
Negatives, ALL mode (labN = labC = None): every other candidate. Negatives, CLEAN mode (the v1 label semantics applied to both boundaries; labN, labC = the primary bond labels of the protein for the N and the C terminus, 1 POS, 0 NEG, -1 excluded): a candidate is a negative only if
its N-bond has labN 0 and its C-bond has labC 0; a near-miss negative needs label 0 at each SHIFTED boundary (the boundary it shares with the positive is the positive's own and is not checked).
Score variants of a candidate: LEN = g(length); N = s_N(start-1); C = s_C(end); NC = N + C; NCL = N + C + g(length).
Populations: F = positives against all negatives (protein AUROC, ties 0.5); M = each positive against the negatives of its own length (share of pairs won, ties 0.5); X = each positive against its near-miss negatives (|start shift| <= delta and |end shift| <= delta, not both zero; pooled pairs, ties 0.5).
`prepare` computes everything that does not depend on the scores (the positives, the exclusions, the negative masks); it is computed once per protein and mode and shared by every model of the stage (`prepared=`); the result is identical to computing it inside the call."""
import numpy as np

VARIANTS = ("LEN", "N", "C", "NC", "NCL")


def _score_matrix(sN, sC, L, lo, hi, g, variant):
    """dict length -> (starts, scores) for every candidate length"""
    out = {}
    for ell in range(lo, hi + 1):
        s = np.arange(2, L - ell + 1)
        if len(s) == 0:
            continue
        e = s + ell - 1
        n, c = sN[s - 2], sC[e - 1]
        gl = g[ell - lo]
        sc = {"LEN": np.full(len(s), gl), "N": n, "C": c, "NC": n + c, "NCL": n + c + gl}[variant]
        out[ell] = (s, sc)
    return out


def prepare(L, positives, excluded, lo=9, hi=30, labN=None, labC=None):
    """the score-independent part of protein_metrics for one protein and one mode: the positives that are candidates (and the number dropped), the excluded peptides, the negative mask of every candidate length (True = negative);
    labN, labC (both or none): the primary bond labels of the protein, which make the negatives CLEAN."""
    clean = labN is not None or labC is not None
    if clean:
        assert labN is not None and labC is not None, "clean mode needs the labels of both termini"
        labN, labC = np.asarray(labN), np.asarray(labC)
        assert len(labN) == len(labC) == max(L - 1, 0), "the label arrays must cover the bonds 1..L-1"
    positives = list(positives)
    pos = sorted({(s, e) for s, e in positives if s >= 2 and e <= L - 1 and lo <= e - s + 1 <= hi})
    dropped = len(set(positives)) - len(pos)
    posset = set(pos)
    excl = {(s, e) for s, e in excluded} - posset
    byl = {}
    for ps, pe in posset | excl:
        byl.setdefault(pe - ps + 1, []).append(ps)
    masks = {}
    for ell in range(lo, hi + 1):
        s = np.arange(2, L - ell + 1)
        if len(s) == 0:
            continue
        m = np.ones(len(s), bool)
        for ps in byl.get(ell, ()):
            if 2 <= ps <= L - ell:                                   # the observed or excluded peptide of this length is a candidate start
                m[ps - 2] = False
        if clean:
            e = s + ell - 1
            m &= (labN[s - 2] == 0) & (labC[e - 1] == 0)
        masks[ell] = m
    return dict(pos=pos, posset=posset, excl=excl, dropped=dropped, masks=masks, clean=clean, labN=labN if clean else None, labC=labC if clean else None)


def protein_metrics(sN, sC, L, positives, excluded, g, lo=9, hi=30, delta=3, labN=None, labC=None, prepared=None):
    """returns {variant: dict(F=auroc or nan, M_num, M_den, X_num, X_den, n_pos, n_pos_dropped)} for one protein.
    positives, excluded: iterables of (start, end); positives outside the candidate set (a bond outside the protein, or a length outside lo..hi) are dropped (and counted in 'n_pos_dropped').
    labN, labC: optional primary bond labels (see the module docstring); given, the negatives are CLEAN. prepared: the result of `prepare` for the same protein and mode (then positives, excluded, labN and labC are not read)."""
    pr = prepared if prepared is not None else prepare(L, positives, excluded, lo, hi, labN, labC)
    pos, posset, excl, dropped, masks, clean = pr["pos"], pr["posset"], pr["excl"], pr["dropped"], pr["masks"], pr["clean"]
    labN, labC = pr["labN"], pr["labC"]
    res = {}
    for v in VARIANTS:
        sm = _score_matrix(sN, sC, L, lo, hi, g, v)
        # F: positives against all negatives
        neg_all = np.concatenate([sm[ell][1][masks[ell]] for ell in sm]) if sm else np.empty(0)
        pos_sc = np.array([sm[e - s + 1][1][s - 2] for s, e in pos])
        F = np.nan
        if len(pos) and len(neg_all):
            srt = np.sort(neg_all)                                   # exact, tie-aware: wins = (negatives below) + 0.5 (ties)
            below = np.searchsorted(srt, pos_sc, side="left")
            upto = np.searchsorted(srt, pos_sc, side="right")
            F = float((below + 0.5 * (upto - below)).sum() / (len(pos_sc) * len(srt)))
        # M: each positive against the negatives of its own length
        mn = md = 0.0
        srt_len = {}
        for (s, e), ps in zip(pos, pos_sc):
            ell = e - s + 1
            if ell not in srt_len:
                srt_len[ell] = np.sort(sm[ell][1][masks[ell]])
            neg = srt_len[ell]
            if len(neg) == 0:
                continue
            below = np.searchsorted(neg, ps, side="left")
            upto = np.searchsorted(neg, ps, side="right")
            mn += below + 0.5 * (upto - below)
            md += len(neg)
        # X: near-miss negatives (never an observed or excluded peptide; in clean mode each shifted boundary must carry label 0)
        xn = xd = 0.0
        for (s, e), ps in zip(pos, pos_sc):
            for di in range(-delta, delta + 1):
                for dj in range(-delta, delta + 1):
                    if di == 0 and dj == 0:
                        continue
                    s2, e2 = s + di, e + dj
                    ell = e2 - s2 + 1
                    if s2 < 2 or e2 > L - 1 or not (lo <= ell <= hi):
                        continue
                    if (s2, e2) in posset or (s2, e2) in excl:
                        continue
                    if clean and ((di != 0 and labN[s2 - 2] != 0) or (dj != 0 and labC[e2 - 1] != 0)):
                        continue
                    sc2 = sm[ell][1][s2 - 2]
                    xn += 1.0 if ps > sc2 else (0.5 if ps == sc2 else 0.0)
                    xd += 1.0
        res[v] = dict(F=float(F), M_num=float(mn), M_den=float(md), X_num=float(xn), X_den=float(xd), n_pos=len(pos), n_pos_dropped=dropped)
    return res
