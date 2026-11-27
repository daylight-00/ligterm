"""Synthetic tests of the peptide-level ranking metrics: protein_metrics (ALL and CLEAN negatives) against a brute-force reference written independently (loops over all peptides and pairs, no shared code). No real data, no real model.
usage: python3 -B test_peptide_ranking.py        (numpy only; prints 'n/N checks passed'; exit 1 on any failure)"""
import os, sys, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from ligterm_repro import peptide_ranking as PE                                              # noqa: E402

N_OK = N_ALL = 0


def ok(cond, msg):
    global N_OK, N_ALL
    N_ALL += 1
    if cond:
        N_OK += 1
    else:
        print("FAIL:", msg)


def brute(sN, sC, L, positives, excluded, g, lo, hi, delta, labN=None, labC=None):
    def score(v, s, e):
        ell = e - s + 1
        n, c, gl = sN[s - 2], sC[e - 1], g[ell - lo]
        return {"LEN": gl, "N": n, "C": c, "NC": n + c, "NCL": n + c + gl}[v]

    def win(a, b):
        return 1.0 if a > b else (0.5 if a == b else 0.0)

    clean = labN is not None
    cand = [(s, e) for s in range(2, L) for e in range(s, L) if lo <= e - s + 1 <= hi]
    cset = set(cand)
    pos = sorted({(s, e) for s, e in positives if (s, e) in cset})
    dropped = len(set(positives)) - len(pos)
    posset = set(pos)
    excl = set(excluded) - posset
    base = [p for p in cand if p not in posset and p not in excl]                    # not observed, not excluded
    neg = [q for q in base if (not clean) or (labN[q[0] - 2] == 0 and labC[q[1] - 1] == 0)]
    out = {}
    for v in PE.VARIANTS:
        if pos and neg:
            F = sum(win(score(v, *p), score(v, *q)) for p in pos for q in neg) / (len(pos) * len(neg))
        else:
            F = float("nan")
        mn = md = xn = xd = 0.0
        for p in pos:
            for q in neg:
                if q[1] - q[0] == p[1] - p[0]:
                    mn += win(score(v, *p), score(v, *q))
                    md += 1.0
            for q in base:                                                           # near-misses: their own rule, not the both-boundaries-clean rule
                di, dj = q[0] - p[0], q[1] - p[1]
                if abs(di) <= delta and abs(dj) <= delta and (di, dj) != (0, 0):
                    if clean and ((di != 0 and labN[q[0] - 2] != 0) or (dj != 0 and labC[q[1] - 1] != 0)):
                        continue
                    xn += win(score(v, *p), score(v, *q))
                    xd += 1.0
        out[v] = dict(F=F, M_num=mn, M_den=md, X_num=xn, X_den=xd, n_pos=len(pos), n_pos_dropped=dropped)
    return out


def same(a, b):
    for v in PE.VARIANTS:
        for k in ("M_num", "M_den", "X_num", "X_den", "n_pos", "n_pos_dropped"):
            if abs(a[v][k] - b[v][k]) > 1e-9:
                return False, f"{v}.{k}: {a[v][k]} vs {b[v][k]}"
        fa, fb = a[v]["F"], b[v]["F"]
        if not ((math.isnan(fa) and math.isnan(fb)) or abs(fa - fb) < 1e-12):
            return False, f"{v}.F: {fa} vs {fb}"
    return True, ""


rng = np.random.default_rng(20261116)
for trial in range(90):
    L = int(rng.choice([6, 11, 12, 15, 24, 33, 47]))
    lo, hi, delta = 9, int(rng.choice([12, 20, 30])), int(rng.choice([1, 2, 3]))
    tie_world = trial % 3 == 0                                     # integer scores: many ties
    nb = max(L - 1, 0)
    sN = rng.integers(0, 3, size=nb).astype(float) if tie_world else rng.normal(size=nb)
    sC = rng.integers(0, 3, size=nb).astype(float) if tie_world else rng.normal(size=nb)
    g = rng.integers(0, 2, size=hi - lo + 1).astype(float) if tie_world else rng.normal(size=hi - lo + 1)
    allp = [(s, e) for s in range(1, L + 1) for e in range(s, L + 1)]                 # including peptides at the protein ends and of any length
    pos = [allp[i] for i in rng.choice(len(allp), size=min(len(allp), int(rng.integers(0, 6))), replace=False)] if allp else []
    exc = [allp[i] for i in rng.choice(len(allp), size=min(len(allp), int(rng.integers(0, 12))), replace=False)] if allp else []
    labN = rng.choice([-1, 0, 0, 0, 1], size=nb)
    labC = rng.choice([-1, 0, 0, 0, 1], size=nb)
    got = PE.protein_metrics(sN, sC, L, pos, exc, g, lo, hi, delta)
    ref = brute(sN, sC, L, pos, exc, g, lo, hi, delta)
    good, why = same(got, ref)
    ok(good, f"ALL trial {trial} (L={L}, hi={hi}, delta={delta}, ties={tie_world}): {why}")
    got = PE.protein_metrics(sN, sC, L, pos, exc, g, lo, hi, delta, labN, labC)
    ref = brute(sN, sC, L, pos, exc, g, lo, hi, delta, labN, labC)
    good, why = same(got, ref)
    ok(good, f"CLEAN trial {trial} (L={L}, hi={hi}, delta={delta}, ties={tie_world}): {why}")
    # shared preparation: the result with prepared masks equals the result computed inside the call, in both modes (the runner shares one preparation across all models)
    prA, prC = PE.prepare(L, pos, exc, lo, hi), PE.prepare(L, pos, exc, lo, hi, labN, labC)
    g1, w1 = same(PE.protein_metrics(sN, sC, L, None, None, g, lo, hi, delta, prepared=prA), ref_all := brute(sN, sC, L, pos, exc, g, lo, hi, delta))
    g2, w2 = same(PE.protein_metrics(sN, sC, L, None, None, g, lo, hi, delta, prepared=prC), ref)
    ok(g1 and g2, f"prepared masks give the brute-force result in both modes (trial {trial}): {w1} {w2}")

# hand-checkable cases
L = 30
sN = np.zeros(L - 1); sC = np.zeros(L - 1); g = np.zeros(22)
sN[5 - 2] = 5.0; sC[18 - 1] = 5.0                                   # the peptide (5, 18) has the only high N and C scores
r = PE.protein_metrics(sN, sC, L, [(5, 18)], [], g, 9, 30, 3)
ok(abs(r["NC"]["F"] - 1.0) < 1e-12 and r["NC"]["M_num"] == r["NC"]["M_den"] > 0 and r["NC"]["X_num"] == r["NC"]["X_den"] > 0, "a planted high-scoring peptide wins every comparison")
ok(abs(r["LEN"]["F"] - 0.5) < 1e-12 and r["LEN"]["M_num"] == 0.5 * r["LEN"]["M_den"], "the length-only score is a tie everywhere when the length prior is flat: F = 0.5, M = 0.5")
ok(r["NCL"]["X_den"] == 48 - sum(1 for di in range(-3, 4) for dj in range(-3, 4) if (di, dj) != (0, 0) and not (2 <= 5 + di and 18 + dj <= L - 1 and 9 <= (18 + dj) - (5 + di) + 1 <= 30)), "near-miss count = 48 minus the shifted peptides outside the candidate set")
gg = np.log(np.arange(1, 23, dtype=float))                        # a non-flat length prior: LEN differs between lengths
r = PE.protein_metrics(sN, sC, L, [(5, 18)], [], gg, 9, 30, 3)
ok(abs(r["LEN"]["M_num"] - 0.5 * r["LEN"]["M_den"]) < 1e-12, "in M the length-only score is exactly 0.5 whatever the prior (g is constant within a length)")
ok(all(abs(r["NC"][k] - r["NCL"][k]) < 1e-12 for k in ("M_num", "M_den")), "in M, NC equals NCL (g is constant within a length)")
z = np.zeros(L - 1)
r = PE.protein_metrics(z, z, L, [(5, 18)], [], gg, 9, 30, 3)
ok(abs(r["NC"]["F"] - 0.5) < 1e-12 and r["NCL"]["F"] < 0.5 - 1e-6 and abs(r["NC"]["M_num"] - 0.5 * r["NC"]["M_den"]) < 1e-12, "all scores zero, non-flat prior: NC is a tie everywhere (0.5); NCL ranks the positive below the lengths with a higher prior in F, and still ties in M")
r = PE.protein_metrics(sN, sC, L, [(1, 12), (5, 31), (5, 10), (5, 18)], [], g, 9, 30, 3)
ok(r["NC"]["n_pos"] == 1 and r["NC"]["n_pos_dropped"] == 3, "positives at the protein end, beyond the protein or outside the length range are dropped and counted")
r = PE.protein_metrics(sN, sC, L, [], [], g, 9, 30, 3)
ok(math.isnan(r["NC"]["F"]) and r["NC"]["M_den"] == 0 and r["NC"]["X_den"] == 0, "no positive: F undefined, no pairs")
every = [(s, e) for s in range(2, L) for e in range(s, L) if 9 <= e - s + 1 <= 30]
r = PE.protein_metrics(sN, sC, L, [(5, 18)], [p for p in every if p != (5, 18)], g, 9, 30, 3)
ok(math.isnan(r["NC"]["F"]) and r["NC"]["M_den"] == 0, "all other candidates excluded: no negative, F undefined")
r1 = PE.protein_metrics(sN, sC, L, [(5, 18)], [(5, 19)], g, 9, 30, 3)
r0 = PE.protein_metrics(sN, sC, L, [(5, 18)], [], g, 9, 30, 3)
ok(r1["NC"]["X_den"] == r0["NC"]["X_den"] - 1, "an excluded near-miss peptide leaves the near-miss pairs")
# clean mode by hand: the positive (5, 18); all bonds label 0 except the positive's own (N bond 4, C bond 18 = label 1) and one observed C terminus of another unit at bond 19
labN = np.zeros(L - 1, int); labC = np.zeros(L - 1, int)
labN[4 - 1] = 1; labC[18 - 1] = 1
r_all = PE.protein_metrics(sN, sC, L, [(5, 18)], [], g, 9, 30, 3)
r_cl = PE.protein_metrics(sN, sC, L, [(5, 18)], [], g, 9, 30, 3, labN, labC)
ok(r_cl["NC"]["X_den"] == r_all["NC"]["X_den"], "clean mode, only the positive's own bonds labelled 1: every near-miss keeps its unshifted boundary unchecked and its shifted boundaries are label 0, so X is unchanged")
labC2 = labC.copy(); labC2[19 - 1] = 1                                # an observed C terminus at bond 19
r_cl2 = PE.protein_metrics(sN, sC, L, [(5, 18)], [], g, 9, 30, 3, labN, labC2)
n_c19 = sum(1 for di in range(-3, 4) for dj in range(-3, 4) if (di, dj) != (0, 0) and dj == 1 and 2 <= 5 + di and 18 + dj <= L - 1 and 9 <= (18 + dj) - (5 + di) + 1 <= 30)
ok(r_cl2["NC"]["X_den"] == r_cl["NC"]["X_den"] - n_c19 and n_c19 == 7, "an observed terminus at the shifted C bond 19 removes every near-miss that ends there (7 of them), and only those")
labN3 = labN.copy(); labN3[3 - 1] = -1                               # an excluded N bond at bond 3 (start 4)
r_cl3 = PE.protein_metrics(sN, sC, L, [(5, 18)], [], g, 9, 30, 3, labN3, labC)
ok(r_cl3["NC"]["X_den"] < r_cl["NC"]["X_den"], "an excluded (label -1) shifted boundary is not a negative")
ok(r_cl["NC"]["M_den"] == r_all["NC"]["M_den"] == len(range(2, L - 14 + 1)) - 1, "same length: a candidate that shares the positive's N bond also shares its C bond, i.e. it is the positive itself; clean and all negatives of the length coincide (the length-14 candidates minus the positive)")
labN4 = labN.copy(); labN4[7 - 1] = -1                               # another unit's N terminus (excluded label) at bond 7: the length-14 candidate (8, 21) is no clean negative
r_cl4 = PE.protein_metrics(sN, sC, L, [(5, 18)], [], g, 9, 30, 3, labN4, labC)
ok(r_cl4["NC"]["M_den"] == r_all["NC"]["M_den"] - 1, "clean mode removes exactly the length-14 candidate whose N bond carries another label than 0")
labC5 = labC.copy(); labC5[24 - 1] = 1                               # an observed C terminus at bond 24: the length-14 candidate (11, 24) is no clean negative
r_cl5 = PE.protein_metrics(sN, sC, L, [(5, 18)], [], g, 9, 30, 3, labN, labC5)
ok(r_cl5["NC"]["M_den"] == r_all["NC"]["M_den"] - 1, "clean mode removes exactly the length-14 candidate whose C bond carries another label than 0")
bad_ok = False
try:
    PE.protein_metrics(sN, sC, L, [(5, 18)], [], g, 9, 30, 3, labN, None)
except AssertionError:
    bad_ok = True
ok(bad_ok, "clean mode needs the labels of both termini")

# ---- the reference of the runner's brute-force guard (ligterm_repro.peptide_reference): against this file's own brute force (a separate write-up), against the fast implementation, in both modes, with ties, with float32 scores (the real-data dtype) and with duplicated positives
from ligterm_repro import peptide_reference as PR                                            # noqa: E402

KEYS_R = ("M_num", "M_den", "X_num", "X_den", "n_pos", "n_pos_dropped")


def ref_same(r, t, label):
    """PR.brute result r (one variant) against another result t (same keys) -> (bool, why)"""
    for k in KEYS_R:
        if abs(r[k] - t[k]) > 1e-9:
            return False, f"{label} {k}: {r[k]} vs {t[k]}"
    fr, ft = r["F"], t["F"]
    if not ((math.isnan(fr) and math.isnan(ft)) or abs(fr - ft) < 1e-12):
        return False, f"{label} F: {fr} vs {ft}"
    return True, ""


rng = np.random.default_rng(20261117)
n_neg_seen = 0
for trial in range(72):
    L = int(rng.choice([6, 11, 12, 15, 24, 33, 47]))
    lo, hi, delta = 9, int(rng.choice([12, 20, 30])), int(rng.choice([1, 2, 3]))
    tie_world = trial % 3 == 0
    f32 = trial % 2 == 1                                                  # the real scores are float32 (the window-model families) while the length prior g is float64
    nb = max(L - 1, 0)
    sN = rng.integers(0, 3, size=nb).astype(float) if tie_world else rng.normal(size=nb)
    sC = rng.integers(0, 3, size=nb).astype(float) if tie_world else rng.normal(size=nb)
    if f32:
        sN, sC = sN.astype(np.float32), sC.astype(np.float32)
    g = rng.integers(0, 2, size=hi - lo + 1).astype(float) if tie_world else rng.normal(size=hi - lo + 1)
    allp = [(s, e) for s in range(1, L + 1) for e in range(s, L + 1)]
    pos = [allp[i] for i in rng.choice(len(allp), size=min(len(allp), int(rng.integers(0, 6))), replace=False)]
    if trial % 5 == 0 and pos:
        pos = pos + [pos[0]]                                              # a duplicated positive is one positive
    exc = [allp[i] for i in rng.choice(len(allp), size=min(len(allp), int(rng.integers(0, 12))), replace=False)]
    labN = rng.choice([-1, 0, 0, 0, 1], size=nb)
    labC = rng.choice([-1, 0, 0, 0, 1], size=nb)
    for clean in (False, True):
        kw = dict(labN=labN, labC=labC) if clean else {}
        t_ref = brute(sN, sC, L, pos, exc, g, lo, hi, delta, **kw)
        fast = PE.protein_metrics(sN, sC, L, pos, exc, g, lo, hi, delta, **kw)
        pr = PE.prepare(L, pos, exc, lo, hi, **kw)
        n_neg_fast = int(sum(int(m.sum()) for m in pr["masks"].values()))
        for v in PE.VARIANTS:
            r = PR.brute(sN, sC, L, pos, exc, g, v, lo, hi, delta, **kw)
            good, why = ref_same(r, t_ref[v], "test brute")
            ok(good, f"PR.brute against the test's brute force: trial {trial} {v} clean={clean}: {why}")
            good, why = ref_same(r, fast[v], "fast")
            ok(good, f"PR.brute against the fast implementation: trial {trial} {v} clean={clean}: {why}")
            ok(r["n_neg"] == n_neg_fast, f"PR.brute's negative count equals the prepared negative masks: trial {trial} {v} clean={clean}: {r['n_neg']} vs {n_neg_fast}")
        n_neg_seen += n_neg_fast

ok(n_neg_seen > 0, "the random worlds contain negatives (the comparison is not vacuous)")

# hand cases of the reference itself
L = 30
sN = np.zeros(L - 1); sC = np.zeros(L - 1); g = np.zeros(22)
sN[5 - 2] = 5.0; sC[18 - 1] = 5.0
r = PR.brute(sN, sC, L, [(5, 18)], [], g, "NC")
ok(abs(r["F"] - 1.0) < 1e-12 and r["M_num"] == r["M_den"] > 0 and r["X_num"] == r["X_den"] > 0 and r["n_pos"] == 1 and r["n_pos_dropped"] == 0, "reference: a planted high-scoring peptide wins every comparison")
r = PR.brute(sN, sC, L, [(1, 12), (5, 31), (5, 10), (5, 18)], [], g, "NC")
ok(r["n_pos"] == 1 and r["n_pos_dropped"] == 3, "reference: positives at the protein end, beyond the protein or outside the length range are dropped and counted")
r = PR.brute(sN, sC, L, [(5, 18), (5, 18)], [], g, "NC")
ok(r["n_pos"] == 1 and r["n_pos_dropped"] == 0, "reference: a duplicated positive is one positive")
every = [(s, e) for s in range(2, L) for e in range(s, L) if 9 <= e - s + 1 <= 30]
r = PR.brute(sN, sC, L, [(5, 18)], [p for p in every if p != (5, 18)], g, "NC")
ok(math.isnan(r["F"]) and r["M_den"] == 0 and r["n_neg"] == 0, "reference: all other candidates excluded: no negative, F undefined")
r = PR.brute(sN, sC, L, [], [], g, "NC")
ok(math.isnan(r["F"]) and r["M_den"] == 0 and r["X_den"] == 0 and r["n_pos"] == 0, "reference: no positive: F undefined, no pairs")
labN = np.zeros(L - 1, int); labC = np.zeros(L - 1, int)
labN[4 - 1] = 1; labC[18 - 1] = 1
r_all = PR.brute(sN, sC, L, [(5, 18)], [], g, "NC")
r_cl = PR.brute(sN, sC, L, [(5, 18)], [], g, "NC", labN=labN, labC=labC)
labC2 = labC.copy(); labC2[19 - 1] = 1
r_cl2 = PR.brute(sN, sC, L, [(5, 18)], [], g, "NC", labN=labN, labC=labC2)
ok(r_cl["X_den"] == r_all["X_den"] and r_cl2["X_den"] == r_cl["X_den"] - 7, "reference: an observed terminus at the shifted C bond 19 removes the 7 near-misses that end there; the positive's own bonds do not")
labN4 = labN.copy(); labN4[7 - 1] = -1
r_cl4 = PR.brute(sN, sC, L, [(5, 18)], [], g, "NC", labN=labN4, labC=labC)
ok(r_cl4["M_den"] == r_all["M_den"] - 1, "reference: an excluded N bond removes exactly one length-matched clean negative (the length-14 candidate starting there)")

# the choice of the proteins of the guard: the two shortest first, then the proteins of at most 200 residues with the most positives (ties: the shorter, then the earlier), the reserve the next in the same order
Lg = np.array([500, 30, 25, 25, 150, 180, 200, 201, 90, 60, 60, 300])
ng = np.array([5, 1, 0, 2, 3, 7, 2, 9, 7, 1, 1, 4])
base, reserve = PR.select_guard_proteins(Lg, ng)
ok(base == [2, 3, 8, 5, 4, 6] and reserve == [1, 9, 10], f"guard proteins: the two shortest (2, 3), then the most positives among those of at most 200 residues (8 before 5: same count, shorter first; then 4, 6; protein 3 counted once; the 201-residue protein 7 never), reserve 1, 9, 10: {base} {reserve}")
ok(len(set(base)) == len(base) and not (set(base) & set(reserve)), "guard proteins: no index twice, the reserve is disjoint from the base")
base2, reserve2 = PR.select_guard_proteins(Lg.copy(), ng.copy())
ok(base2 == base and reserve2 == reserve, "guard proteins: deterministic")
base3, reserve3 = PR.select_guard_proteins(np.array([40, 40, 40, 40, 40, 40, 40, 40]), np.array([3, 3, 3, 3, 3, 3, 3, 3]))
ok(base3 == [0, 1, 2, 3, 4, 5] and reserve3 == [6, 7], f"guard proteins: all equal -> the earlier first: {base3} {reserve3}")
base4, reserve4 = PR.select_guard_proteins(np.array([900, 800, 700]), np.array([5, 5, 5]))
ok(base4 == [2, 1] and reserve4 == [], f"guard proteins: none of at most 200 residues -> the two shortest only, no reserve: {base4} {reserve4}")
base7, reserve7 = PR.select_guard_proteins(np.array([100, 100, 100, 100]), np.array([2, 0, 0, 0]))
ok(base7 == [0, 1] and reserve7 == [], f"guard proteins: a protein without a positive that is not among the two shortest is never a dense pick (and is no reserve): {base7} {reserve7}")
base5, reserve5 = PR.select_guard_proteins(np.array([120]), np.array([4]))
ok(base5 == [0] and reserve5 == [], f"guard proteins: a single protein: {base5} {reserve5}")
base6, reserve6 = PR.select_guard_proteins(Lg, ng, short_n=1, dense_n=2, extra=1)
ok(base6 == [2, 8, 5] and reserve6 == [4], f"guard proteins: the counts are parameters (one shortest, two dense, one reserve): {base6} {reserve6}")

print(f"{N_OK}/{N_ALL} checks passed")
sys.exit(0 if N_OK == N_ALL else 1)
