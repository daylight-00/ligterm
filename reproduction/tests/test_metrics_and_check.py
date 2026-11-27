"""Tests of the metric functions against naive definitions, of the study-level values of the comparison with the published tool, and of the checker of regenerated results (its negative controls).
usage: python3 -B test_metrics_and_check.py        (numpy and scipy only; prints 'n/N checks passed'; exit 1 on any failure)"""
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from ligterm_repro import check, metrics, np_comparison, records  # noqa: E402

N_OK = N_ALL = 0


def ok(cond, msg):
    global N_OK, N_ALL
    N_ALL += 1
    if cond:
        N_OK += 1
    else:
        print("FAIL:", msg)


rng = np.random.default_rng(20261127)

# 1. AUROC, average precision, recall at the top 10 % and the local pairs against their definitions written as loops (scores with many ties)
for trial in range(60):
    n = int(rng.integers(8, 60))
    s = rng.integers(0, 6, size=n).astype(float) if trial % 2 else rng.normal(size=n)
    y = (rng.random(n) < 0.3).astype(int)
    if y.sum() == 0 or y.sum() == n:
        y[0], y[1] = 1, 0
    pos, neg = s[y == 1], s[y == 0]
    ref = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))
    ok(abs(metrics.auroc(s, y) - ref) < 1e-12, f"auroc trial {trial}")
    # average precision with ties: thresholds at the distinct scores, precision and recall of everything at or above the threshold
    ap = 0.0
    prev_rec = 0.0
    for t in sorted(set(s.tolist()), reverse=True):
        sel = s >= t
        rec = y[sel].sum() / y.sum()
        ap += (rec - prev_rec) * (y[sel].sum() / sel.sum())
        prev_rec = rec
    ok(abs(metrics.ap_ties(s, y) - ap) < 1e-12, f"ap_ties trial {trial}")
    k = int(math.ceil(0.10 * n))
    thr = np.sort(s)[::-1][k - 1]
    above, tie = s > thr, s == thr
    ref_r = (y[above].sum() + y[tie].sum() * (k - above.sum()) / tie.sum()) / y.sum()
    ok(abs(metrics.recall_at_frac(s, y) - ref_r) < 1e-12, f"recall_at_frac trial {trial}")
    num = den = 0.0
    for i in np.flatnonzero(y == 1):
        for d in (-3, -2, -1, 1, 2, 3):
            j = i + d
            if 0 <= j < n and y[j] == 0:
                num += (s[i] > s[j]) + 0.5 * (s[i] == s[j])
                den += 1
    ok(metrics.local_counts(s, y) == (num, den), f"local_counts trial {trial}")

# 2. the study-level values of the comparison with the published tool: macro mean, macro share and pooled ratio, with undefined proteins left out
a = dict(peptide_auroc=np.array([0.5, np.nan, 0.9, 0.7]), length_matched_num=np.array([1.0, 2.0, 0.0, 3.0]), length_matched_den=np.array([2.0, 0.0, 0.0, 4.0]),
         near_miss_num=np.array([1.0, 2.0, 3.0, 4.0]), near_miss_den=np.array([2.0, 2.0, 4.0, 4.0]), recall_at_10=np.array([0.0, 1.0, np.nan, 0.5]))
idx = np.arange(4)
ok(abs(np_comparison.study_value(a, idx, "peptide_auroc") - np.mean([0.5, 0.9, 0.7])) < 1e-15, "macro mean leaves out the undefined protein")
ok(abs(np_comparison.study_value(a, idx, "length_matched_auc") - np.mean([0.5, 0.75])) < 1e-15, "macro share leaves out the proteins without pairs")
ok(abs(np_comparison.study_value(a, idx, "near_miss_auc") - 10 / 12) < 1e-15, "pooled ratio")
ok(abs(np_comparison.study_value(a, idx, "recall_at_10") - np.mean([0.0, 1.0, 0.5])) < 1e-15, "retrieval measure: macro mean")
bs = np.array([[0, 0, 2], [1, 3, 3]])
ok(np.allclose(np_comparison.study_value(a, bs, "peptide_auroc"), [np.mean([0.5, 0.5, 0.9]), np.mean([0.7, 0.7])]), "resampled indices: one value per resample")

# 3. the stream of a (study, terminus) is [seed, study, 0 or 1]
ok(records.terminus_stream(7, "123", "N") == [7, 123, 0] and records.terminus_stream(7, "123", "C") == [7, 123, 1], "terminus_stream")

# 4. the checker: table round trip with exact float text, and the negative controls of compare_cells


def cell(study="1", T="N", model="LR-Both8", fit=None, auroc=0.7123456789012345, **kw):
    d = {k_: None for k_ in check.COLUMNS}
    d.update(study=study, terminus=T, analysis="primary", kind="model", model=model, fit=fit, n_proteins=10, n_groups=5, n_pos=40, bond_auroc=auroc, bond_auroc_lo=auroc - 0.01, bond_auroc_hi=auroc + 0.01)
    d.update(kw)
    return d


exp = [cell(), cell(T="C"), cell(model="NetCleave-style (released decoys)", T="C", auroc=0.6), cell(study="2", fit="cv", model="GB-Both8", auroc=float("nan"))]
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "t.tsv.gz")
    check.write_table(exp, p)
    back = check.read_table(p)
    ok(all((r["bond_auroc"] == e["bond_auroc"]) or (math.isnan(r["bond_auroc"]) and math.isnan(e["bond_auroc"])) for r, e in zip(back, exp)) and [r["fit"] for r in back] == [e["fit"] for e in exp], "table round trip is exact")
    q = os.path.join(d, "u.tsv.gz")
    check.write_table(exp, q)
    ok(open(p, "rb").read() == open(q, "rb").read(), "the same table gives the same bytes")


def run(new, subset=False):
    return check.compare_cells(new, exp, subset=subset)["problems"]


ok(run([dict(r) for r in exp]) == [], "identical cells: no problem")
bad = [dict(r) for r in exp]
bad[0]["bond_auroc"] += 1e-12
ok(any("bond_auroc" in p for p in run(bad)), "a non-network cell differing by 1e-12 is refused")
bad = [dict(r) for r in exp]
bad[2]["bond_auroc"] += 5e-5
ok(run(bad) == [], "a network cell differing by 5e-5 is accepted")
bad[2]["bond_auroc"] += 1.5e-4
ok(run(bad) != [], "a network cell differing by 2e-4 is refused")
bad = [dict(r) for r in exp]
bad[1]["n_pos"] = 41
ok(run(bad) != [], "a changed count is refused")
bad = [dict(r) for r in exp]
bad[0]["n_pos"] = 40.0
ok(run(bad) != [], "a value of another type is refused")
ok(run([dict(r) for r in exp][1:]) != [], "a missing cell is refused")
ok(run([dict(r) for r in exp] + [dict(exp[0])]) != [], "a repeated cell is refused")
bad = [dict(r) for r in exp]
bad[3]["bond_auroc"] = 0.5
ok(run(bad) != [], "a non-finite value that became finite (or the reverse) is refused")
ok(run([dict(r) for r in exp if r["study"] == "1"]) != [] and run([dict(r) for r in exp if r["study"] == "1"], subset=True) == [], "a partial re-run is refused unless --subset is given, and then compared on its studies")
ok(run([dict(r) for r in exp] + [cell(study="3")]) != [], "a study that is not expected is refused")

print(f"{N_OK}/{N_ALL} checks passed")
sys.exit(0 if N_OK == N_ALL else 1)
