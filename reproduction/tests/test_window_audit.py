"""Planted-oracle tests of ligterm_repro.window_audit: the window geometry against the scorer's own window code, the exact scan against a naive brute force, a planted copy with known substitutions, the boundaries between training proteins, the masks and the merge of partial scans.
usage: python3 -B test_window_audit.py   (the geometry check needs the `ligterm` package next to reproduction/ or on PYTHONPATH; without it that one check is skipped)"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))           # src/ of the repository: the ligterm package
from ligterm_repro.window_audit import AA, CENTRAL, QUERY_OTHER, SEP, W, brute_force, build_training, encode, locate, merge, scan, window_start   # noqa: E402
try:
    from ligterm.features import window_tokens
except ImportError:
    window_tokens = None

rng = np.random.default_rng(20261120)


def rnd(n):
    return "".join(rng.choice(list(AA), n))


# 1. the geometry of the window against the window code of the scorer (whole proteins, interior bonds)
for L in ((40, 75, 130) if window_tokens is not None else ()):
    s = rnd(L)
    tok = window_tokens(s, 16)                                            # row b - 1 is bond b
    enc = encode(s, 20)
    for b in range(1, L):
        st = window_start(b, L)
        if st is None:
            assert b < 16 or b > L - 16
            continue
        assert 16 <= b <= L - 16 and np.array_equal(tok[b - 1], enc[st:st + W]), (L, b)
print("1. window_start and the 32 residues equal the window of the scorer for every interior bond; the bonds within 15 residues of an end have no full window" if window_tokens is not None else "1. skipped: the ligterm package is not available")

# 2. the exact scan equals the naive brute force on random data (with non-standard residues on both sides)
accs = [f"P{i}" for i in range(12)]
seqs = {a: rnd(int(rng.integers(34, 90))) for a in accs}
seqs["P3"] = seqs["P3"][:20] + "XU" + seqs["P3"][22:]                          # non-standard residues in the training side
concat, offsets, valid = build_training(seqs, accs)
nstart = len(valid)
Q = np.stack([encode(rnd(W), QUERY_OTHER) for _ in range(30)])
Q[0, 5] = QUERY_OTHER                                                          # a non-standard residue in a query
for lo, hi in ((0, nstart), (50, 400)):
    r = scan(Q, concat, valid, lo, hi, chunk=97)
    b32, b16 = brute_force(Q, concat, valid, lo, hi)
    assert np.array_equal(r["m32"], b32) and np.array_equal(r["m16"], b16), (lo, hi)
print("2. the scan equals the brute force on 30 random queries, with chunks that do not divide the range and with non-standard residues")

# 3. a planted copy with known substitutions: 5 of 32 positions changed -> exactly 27 identical, found at the planted window; 3 of the central 16 changed -> 13
a = "P7"
st = 11
w = concat[offsets[a] + st: offsets[a] + st + W].copy()
q = w.copy()
for j in (0, 3, 9, 20, 30):
    q[j] = (q[j] + 7) % 20 if q[j] < 20 else 5
q16 = int(sum(q[CENTRAL[0]:CENTRAL[1]] == w[CENTRAL[0]:CENTRAL[1]]))
r = scan(np.stack([q, w]), concat, valid, 0, nstart, chunk=61)
assert r["m32"][1] == 32 and r["a32"][1] == offsets[a] + st, "an exact copy must give 32 identical positions at its own start"
assert r["m32"][0] >= 27 and r["m32"][0] <= 31, r["m32"][0]
b32, _ = brute_force(np.stack([q]), concat, valid, 0, nstart)
assert r["m32"][0] == b32[0]
acc_found, pos_found = locate(int(r["a32"][1]), np.array([offsets[x] for x in accs]), accs)
assert (acc_found, pos_found) == (a, st)
print(f"3. a planted exact copy gives 32 at its own window ({acc_found}, start {pos_found}); a copy with 5 substitutions gives {r['m32'][0]}, equal to the brute force; central 16: {q16} identical")

# 4. no window straddles two training proteins: a query that is the end of one protein followed by the start of the next cannot reach 32
end = concat[offsets["P4"] + len(seqs["P4"]) - 10: offsets["P4"] + len(seqs["P4"])]
nxt = concat[offsets["P5"]: offsets["P5"] + 22]
q = np.concatenate([end, nxt]).astype(np.int8)
r = scan(np.stack([q]), concat, valid, 0, nstart, chunk=128)
assert r["m32"][0] < 32 and r["m32"][0] == brute_force(np.stack([q]), concat, valid, 0, nstart)[0][0]
for s in range(nstart):
    if valid[s]:
        assert not (concat[s:s + W] == SEP).any()
print(f"4. a query made of the end of one protein and the start of the next reaches only {r['m32'][0]} (windows never cross a boundary); every valid window is free of separators")

# 5. masks: the maximum over a masked set of windows, and the merge of partial scans over disjoint ranges equals the scan over the whole range
mask = np.zeros(nstart, bool)
mask[offsets["P2"] + 3] = True
mask[offsets["P9"] + 1] = True
Qm = np.stack([concat[offsets["P2"] + 3: offsets["P2"] + 3 + W].copy(), concat[offsets["P6"] + 2: offsets["P6"] + 2 + W].copy()])
full = scan(Qm, concat, valid, 0, nstart, masks=(mask,), chunk=89)
assert full["mk32"][0][0] == 32 and full["mk32"][0][1] < 32, "the first query is a masked window; the second is an unmasked one"
exp = [max(int((Qm[i] == concat[s:s + W]).sum()) for s in np.flatnonzero(mask & valid)) for i in range(2)]
assert list(full["mk32"][0]) == exp
cut = [0, 130, 333, nstart]
parts = [scan(Qm, concat, valid, cut[i], cut[i + 1], masks=(mask,), chunk=50) for i in range(3)]
mg = merge(parts)
assert np.array_equal(mg["m32"], full["m32"]) and np.array_equal(mg["m16"], full["m16"]) and np.array_equal(mg["mk32"][0], full["mk32"][0]) and np.array_equal(mg["mk16"][0], full["mk16"][0])
assert np.array_equal(mg["a32"], full["a32"])
print("5. the masked maxima equal the naive maxima over the masked windows; the merge of three partial scans equals the scan of the whole range (maxima and the located windows)")
print("WINDOW AUDIT LIB TESTS: passed")
