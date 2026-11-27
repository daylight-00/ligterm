#!/usr/bin/env python3
"""Audit of the external test cohort: (1) whether any study of the cohort is a source of a training unit, (2) how similar the 32-residue windows around its observed terminal bonds are to the windows of the
training proteins. Neither part loads a model or reads a score, except the descriptive sensitivity (rule 5), which reads the per-item AUROCs written by 50_external_evaluation.py.

(1) Independence check. The training table holds 447,644 candidate units, 417,415 of them verified against UniProt (the rows of the release fit). Only the PubMed-identifier lists of the units are read:
is any candidate study, and in particular any of the four studies of the cohort, listed as a source of a training unit?

(2) Window audit. The cohort was screened at the level of whole proteins (no identical 9-mer with any training protein; no hit of identity >= 0.40 over >= 0.50 of the query; phmmer, E <= 1e-3). That screen is
the authors' own criterion, and 4 of the cohort proteins have a significant best hit of identity >= 0.40 over less than half of the protein (class H1). The scorers read a 32-residue window around a bond, so the
direct question is how similar the windows of the cohort are to the windows of the training proteins. The audit describes this; it does not change the cohort, its composition or its performance, and it does not
redefine independence. Rules, fixed before the first run:
 1. Population: the unique (protein, bond, terminus) windows of the observed terminal bonds of the cohort (the positives of the 152 items; labels as in 50_external_evaluation.py), restricted to windows fully
    inside the protein (a bond within 15 residues of an end has no full window; their number is reported). For comparison: up to 5 random negative windows of the same protein and terminus per observed window
    (seed [20261120, 51]; excluded bonds are not drawn), and one shuffled copy of every observed window (a random permutation of its 32 residues; seed [20261120, 52]) as the chance reference. Five exact copies of
    random training windows serve as canaries (they must give 32 of 32).
 2. Training windows: every window that lies fully inside one of the 16,873 training proteins, for every bond. Separately: the windows at the training positive bonds of the same terminus (N bond = start - 1,
    C bond = end of the verified units).
 3. Statistic: the maximum, over the training windows, of the ungapped identity (identical residues at the 32 aligned positions; also over the central 16 positions P8..P8'). A non-standard residue never
    matches. The scan is exact.
 4. Report (descriptive; no gate): per group and terminus the number of windows, the median, the 90th and 99th percentile and the maximum of the identity, the numbers at or above 0.5, 0.6, 0.7, 0.8 and 0.9
    (32 positions) and at or above 0.625, 0.75, 0.875 and 1.0 (central 16); the same against the training-positive windows; the 99th percentile of the shuffled identity is the background; the share of observed
    windows above it; the probability that an observed window has a higher identity than a random negative one (ties 0.5); every observed window at or above 0.7 with the training window that gives it; the
    observed windows by homology class (H0, H1, H2).
 5. Trigger of a descriptive sensitivity: if any observed window has an identity of 0.70 or more over the 32 positions, the macro AUROC of LigTerm and LigTerm-GB on the cohort is computed once, from the
    per-item AUROCs (no scoring), without the items that hold such a window at that terminus, and printed next to the primary values. No interval; the primary result stays.
 6. Guards: the items are the 152 of the cohort and their positives the 191 (N) and 181 (C) of the evaluation; the training proteins are 16,873; the canaries give 32 of 32.

usage: 51_external_cohort_audit.py --data <datasets folder> --external <datasets/external folder> --per-item <external per-item csv of 50_external_evaluation.py> --out <folder> [--nproc 8] [--prepare-only]
"""
import argparse
import collections
import csv
import gzip
import json
import math
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "lib"))
from ligterm_repro.external import UNSEEN_PROTEIN, bond_of, build_labels  # noqa: E402
from ligterm_repro.window_audit import QUERY_OTHER, W, build_training, encode, locate, merge, scan, window_start  # noqa: E402

STUDIES = ["42797554", "29557506", "41191777", "29632711"]
CLEAN = ("H0", "H1", "H2")
EXPECTED_POSITIVES = {"N": 191, "C": 181}
TH32, TH16 = (0.5, 0.6, 0.7, 0.8, 0.9), (0.625, 0.75, 0.875, 1.0)
COHORT = "external_test_cohort"
MODEL_COLUMN = {"LigTerm": "LigTerm", "LigTerm-GB": "LigTerm-GB"}


_SCAN = {}


def _work(i):
    c = _SCAN["cuts"]
    return scan(_SCAN["Q"], _SCAN["concat"], _SCAN["valid"], c[i], c[i + 1], masks=_SCAN["masks"])


def jclean(o):
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, dict):
        return {k: jclean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jclean(v) for v in o]
    return o


def independence_check(data, external):
    cand = [s["pmid"] for s in json.load(open(os.path.join(external, "candidate_strata_counts.json")))["studies"]]
    seqs = json.load(open(os.path.join(data, "protein_sequences.json")))
    csv.field_size_limit(10 ** 8)
    n_units = n_verified = 0
    hit = {p: 0 for p in cand}
    all_pm, ver_pm, prots = set(), set(), set()
    with gzip.open(os.path.join(data, "units.tsv.gz"), "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            n_units += 1
            pms = r["pmids"].split(",")
            all_pm.update(pms)
            ok = bool(seqs.get(r["acc"])) and seqs[r["acc"]][int(r["start"]) - 1:int(r["end"])] == r["seq"]
            if ok:
                n_verified += 1
                ver_pm.update(pms)
                prots.add(r["acc"])
            for p in pms:
                if p in hit:
                    hit[p] += 1
    return dict(units_in_table=n_units, verified_units=n_verified, proteins_with_verified_unit=len(prots), distinct_pubmed_ids_any_unit=len(all_pm), distinct_pubmed_ids_verified_units=len(ver_pm),
                cohort_studies_listed_as_training_source={p: hit[p] for p in STUDIES}, candidate_studies=len(cand), units_listing_a_candidate_study=sum(hit.values()),
                candidate_studies_in_training_table=sorted(p for p in cand if p in all_pm))


def independence_lines(d):
    return [f"units in the table: {d['units_in_table']} | verified: {d['verified_units']} | proteins with a verified unit: {d['proteins_with_verified_unit']}",
            f"distinct PubMed identifiers listed by any unit: {d['distinct_pubmed_ids_any_unit']} | by at least one verified unit: {d['distinct_pubmed_ids_verified_units']}",
            "the four cohort studies listed as a source of any training unit: " + ", ".join(f"{p} {n}" for p, n in d["cohort_studies_listed_as_training_source"].items()),
            f"all {d['candidate_studies']} candidate studies: units that list any of them: {d['units_listing_a_candidate_study']}; candidate studies that occur in the training table at all: {d['candidate_studies_in_training_table']}"]


def summ(x, th):
    x = np.asarray(x, float)
    if len(x) == 0:
        return dict(n=0)
    d = dict(n=int(len(x)), median=float(np.median(x)), p90=float(np.percentile(x, 90)), p99=float(np.percentile(x, 99)), max=float(x.max()))
    d["at_or_above"] = {f"{t:g}": int((x >= t - 1e-12).sum()) for t in th}
    return d


def auc(a, b):
    a, b = np.sort(np.asarray(a, float)), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    lo = np.searchsorted(a, b, side="left")
    hi = np.searchsorted(a, b, side="right")
    return float(((len(a) - hi) + 0.5 * (hi - lo)).sum() / (len(a) * len(b)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--external", required=True)
    ap.add_argument("--per-item", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--prepare-only", action="store_true", help="the data, the guards and the query windows only; no similarity is computed")
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out, "window_audit.json")):
        sys.exit(f"refusing to overwrite {a.out}/window_audit.json")
    if not a.prepare_only and not os.path.exists(a.per_item):
        sys.exit(f"{a.per_item} does not exist: run 50_external_evaluation.py first (it writes <prefix>_per_item.csv)")
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    def log(*x):
        print(f"[{time.time() - t0:5.0f}s]", *x, flush=True)

    # ---------------------------------------------------------------- (1) independence check
    ind = independence_check(a.data, a.external)
    with open(os.path.join(a.out, "independence_check.json"), "w") as fh:
        json.dump(ind, fh, indent=1)
    for ln in independence_lines(ind):
        log(ln)
    # ---------------------------------------------------------------- (2) data
    csv.field_size_limit(10 ** 8)
    seqs = json.load(open(os.path.join(a.data, "protein_sequences.json")))
    SEQ = dict(json.load(open(os.path.join(a.external, "sequences.json"))))
    SEQ.update(seqs)
    cls = {r["acc"]: r["class"] for r in csv.DictReader(open(os.path.join(a.external, "homology_classes.tsv")), delimiter="\t")}
    units_by = collections.defaultdict(list)
    cand_by_acc = collections.defaultdict(lambda: collections.defaultdict(list))
    with gzip.open(os.path.join(a.external, "candidate_units.tsv.gz"), "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            s, e = int(r["start"]), int(r["end"])
            units_by[(r["pmid"], r["acc"])].append((s, e, r["seq"], r["stratum"]))
            cand_by_acc[r["acc"]][r["pmid"]].append((s, e))
    train_accs, nb, cb = set(), collections.defaultdict(set), collections.defaultdict(set)
    with gzip.open(os.path.join(a.data, "units.tsv.gz"), "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            acc, s, e = r["acc"], int(r["start"]), int(r["end"])
            sq = seqs.get(acc)
            if sq and sq[s - 1:e] == r["seq"]:
                train_accs.add(acc)
                L = len(sq)
                if s >= 2:
                    nb[acc].add(s - 1)
                if e <= L - 1:
                    cb[acc].add(e)
    train_accs = sorted(train_accs)
    assert len(train_accs) == 16873, f"{len(train_accs)} training proteins, not 16,873"
    log(f"{len(train_accs)} training proteins; units of {len({p for (p, _) in units_by})} candidate studies")

    # ---------------------------------------------------------------- the cohort items and their labels (as in the external evaluation)
    items = []
    for pm in STUDIES:
        for (p_, acc), us in sorted(units_by.items()):
            if p_ != pm:
                continue
            if cls.get(acc) in CLEAN and any(st == UNSEEN_PROTEIN for (_, _, _, st) in us):
                items.append((pm, acc))
    assert len(items) == 152, f"{len(items)} items, not 152"
    labs, pos_n = {}, {"N": 0, "C": 0}
    for (pm, acc) in items:
        L = len(SEQ[acc])
        for T in "NC":
            ob = set()
            for p2, uu in cand_by_acc[acc].items():
                if p2 != pm:
                    ob |= {bond_of(s, e, L, T) for (s, e) in uu}
            ob.discard(None)
            lab = build_labels(SEQ[acc], units_by[(pm, acc)], lambda st: st == UNSEEN_PROTEIN, ob, T)
            labs[(pm, acc, T)] = lab
            pos_n[T] += int((lab == 1).sum())
    assert pos_n == EXPECTED_POSITIVES, pos_n
    assert not (set(acc for (_, acc) in items) & set(train_accs)), "a cohort protein is a training protein"
    log(f"{len(items)} items; positives N {pos_n['N']}, C {pos_n['C']} (equal to the external evaluation)")

    # ---------------------------------------------------------------- the query windows
    rng_neg = np.random.default_rng([20261120, 51])
    rng_shuf = np.random.default_rng([20261120, 52])
    pos = {}                                                                 # (acc, b, T) -> studies
    for (pm, acc) in items:
        for T in "NC":
            for i in np.flatnonzero(labs[(pm, acc, T)] == 1).tolist():
                pos.setdefault((acc, i + 1, T), set()).add(pm)
    pos_keys = sorted(pos)
    partial = {T: 0 for T in "NC"}
    Q, meta = [], []                                                         # meta: (kind, T, acc, b, class, studies)
    for (acc, b, T) in pos_keys:
        L = len(SEQ[acc])
        st = window_start(b, L)
        if st is None:
            partial[T] += 1
            continue
        Q.append(encode(SEQ[acc], QUERY_OTHER)[st:st + W])
        meta.append(("observed", T, acc, b, cls[acc], ",".join(sorted(pos[(acc, b, T)]))))
    n_pos_windows = {T: sum(1 for m in meta if m[0] == "observed" and m[1] == T) for T in "NC"}
    used = {(m[2], m[3], m[1]) for m in meta}
    for (acc, b, T) in [k for k in pos_keys if window_start(k[1], len(SEQ[k[0]])) is not None]:
        L = len(SEQ[acc])
        pm0 = sorted(pos[(acc, b, T)])[0]
        lab = labs[(pm0, acc, T)]
        cand = [i + 1 for i in np.flatnonzero(lab == 0).tolist() if window_start(i + 1, L) is not None and (acc, i + 1, T) not in used]
        take = sorted(rng_neg.choice(cand, size=min(5, len(cand)), replace=False).tolist()) if cand else []
        for bb in take:
            used.add((acc, bb, T))
            Q.append(encode(SEQ[acc], QUERY_OTHER)[window_start(bb, L):window_start(bb, L) + W])
            meta.append(("random_negative", T, acc, bb, cls[acc], ""))
    for i in [i for i, m in enumerate(meta) if m[0] == "observed"]:          # one shuffled copy of every observed window
        m = meta[i]
        Q.append(Q[i][rng_shuf.permutation(W)])
        meta.append(("shuffled", m[1], m[2], m[3], m[4], m[5]))
    rng_can = np.random.default_rng([20261120, 53])
    for _ in range(5):                                                       # exact copies of random training windows
        acc = train_accs[int(rng_can.integers(0, len(train_accs)))]
        L = len(SEQ[acc])
        while L < 40:
            acc = train_accs[int(rng_can.integers(0, len(train_accs)))]
            L = len(SEQ[acc])
        s0 = int(rng_can.integers(0, L - W + 1))
        code = encode(SEQ[acc], QUERY_OTHER)[s0:s0 + W]
        if (code >= 20).any():
            continue
        Q.append(code)
        meta.append(("canary", "-", acc, s0 + 16, "-", ""))
    Q = np.stack(Q).astype(np.int8)
    log(f"{len(Q)} query windows: observed N {n_pos_windows['N']}, C {n_pos_windows['C']} (windows within 15 residues of an end, not audited: N {partial['N']}, C {partial['C']}); "
        f"random negatives {sum(1 for m in meta if m[0] == 'random_negative')}; shuffled {sum(1 for m in meta if m[0] == 'shuffled')}; canaries {sum(1 for m in meta if m[0] == 'canary')}")

    # ---------------------------------------------------------------- the exact scan over every training window
    concat, offsets, valid = build_training(SEQ, train_accs)
    nstart = len(valid)
    maskN = np.zeros(nstart, bool)
    maskC = np.zeros(nstart, bool)
    for acc in train_accs:
        L = len(SEQ[acc])
        for b in nb[acc]:
            st = window_start(b, L)
            if st is not None:
                maskN[offsets[acc] + st] = True
        for b in cb[acc]:
            st = window_start(b, L)
            if st is not None:
                maskC[offsets[acc] + st] = True
    log(f"training windows: {int(valid.sum()):,} (every bond with a full window); at a training positive bond: N {int((maskN & valid).sum()):,}, C {int((maskC & valid).sum()):,}")
    if a.prepare_only:
        log("prepare only: stopping before the scan")
        return
    npart = a.nproc * 4
    _SCAN.update(Q=Q, concat=concat, valid=valid, masks=(maskN, maskC), cuts=[int(round(i * nstart / npart)) for i in range(npart + 1)])
    with get_context("fork").Pool(a.nproc) as pool:                              # the workers inherit _SCAN by fork
        parts = pool.map(_work, range(npart))
    R = merge(parts)
    log("exact scan done")
    kind = np.array([m[0] for m in meta])
    term = np.array([m[1] for m in meta])
    can = np.flatnonzero(kind == "canary")
    assert (R["m32"][can] == 32).all(), f"a canary did not find itself: {R['m32'][can]}"
    log(f"guard passed: the {len(can)} canaries give 32 of 32")
    offs_sorted = np.array([offsets[x] for x in train_accs])
    id32, id16 = R["m32"] / 32.0, R["m16"] / 16.0
    mk32 = np.where(term == "C", R["mk32"][1], R["mk32"][0]) / 32.0
    mk16 = np.where(term == "C", R["mk16"][1], R["mk16"][0]) / 16.0

    # ---------------------------------------------------------------- the report
    out = dict(label="external test cohort, local-window similarity audit (model-free)", n_training_proteins=len(train_accs), n_training_windows=int(valid.sum()), windows_not_audited=partial, groups={})
    for k in ("observed", "random_negative", "shuffled"):
        for T in ("N", "C"):
            sel = (kind == k) & (term == T)
            out["groups"][f"{k}_{T}"] = dict(id32=summ(id32[sel], TH32), id16=summ(id16[sel], TH16), id32_vs_training_positive_windows=summ(mk32[sel], TH32), id16_vs_training_positive_windows=summ(mk16[sel], TH16))
    posm, shufm, negm = kind == "observed", kind == "shuffled", kind == "random_negative"
    bg32, bg16 = float(np.percentile(id32[shufm], 99)), float(np.percentile(id16[shufm], 99))
    out["background"] = dict(shuffled_p99_id32=bg32, shuffled_p99_id16=bg16)
    out["share_of_observed_windows_above_background_p99"] = dict(id32=float((id32[posm] > bg32).mean()), id16=float((id16[posm] > bg16).mean()))
    out["probability_observed_window_above_negative_window"] = {T: dict(id32=auc(id32[negm & (term == T)], id32[posm & (term == T)]),
                                                                        id32_vs_training_positive_windows=auc(mk32[negm & (term == T)], mk32[posm & (term == T)])) for T in ("N", "C")}
    out["observed_windows_by_class"] = {c: summ(id32[posm & np.array([m[4] == c for m in meta])], TH32) for c in ("H0", "H1", "H2")}
    rows = []
    for i, m in enumerate(meta):
        if m[0] == "canary":
            continue
        ta, tp = locate(int(R["a32"][i]), offs_sorted, train_accs) if R["a32"][i] >= 0 else ("", -1)
        rows.append([m[0], m[1], m[2], m[3], m[4], m[5], f"{id32[i]:.4f}", f"{id16[i]:.4f}", f"{mk32[i]:.4f}", f"{mk16[i]:.4f}", ta, tp + 1, tp + 16])
    top = sorted((r for r in rows if r[0] == "observed" and float(r[6]) >= 0.7), key=lambda r: -float(r[6]))
    out["observed_windows_at_or_above_0.70"] = [dict(terminus=r[1], acc=r[2], bond=r[3], cls=r[4], studies=r[5], id32=float(r[6]), id16=float(r[7]), training_acc=r[10], training_window_start=r[11]) for r in top]
    out["top10_observed_windows"] = [dict(terminus=r[1], acc=r[2], bond=r[3], cls=r[4], id32=float(r[6]), id16=float(r[7]), training_acc=r[10], training_window_start=r[11])
                                     for r in sorted((r for r in rows if r[0] == "observed"), key=lambda r: -float(r[6]))[:10]]
    trig = len(top) > 0                                                      # the descriptive sensitivity, only if the trigger of rule 5 is met
    out["trigger_met"] = trig
    if trig:
        held = {T: {(pm, acc) for r in top if r[1] == T for acc in [r[2]] for pm in r[5].split(",")} for T in "NC"}
        it = [r for r in csv.DictReader(open(a.per_item)) if r["stratum"] == COHORT]
        sens = {}
        for T in "NC":
            for model in MODEL_COLUMN:
                col = f"auc_{T}_{MODEL_COLUMN[model]}"
                vals_all = [float(r[col]) for r in it if r[col] != "nan"]
                vals_ex = [float(r[col]) for r in it if r[col] != "nan" and (r["study"], r["acc"]) not in held[T]]
                sens[f"{T}_{model}"] = dict(primary=float(np.mean(vals_all)), without_items=float(np.mean(vals_ex)), n_items_all=len(vals_all), n_items_without=len(vals_ex))
        out["sensitivity"] = sens
    out["guards"] = dict(items=len(items), positives=pos_n, training_proteins=len(train_accs), canaries=int(len(can)))
    json.dump(jclean(out), open(os.path.join(a.out, "window_audit.json"), "w"), indent=1)
    with open(os.path.join(a.out, "window_audit_windows.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "terminus", "acc", "bond", "homology_class", "studies", "identity_32", "identity_central16", "identity_32_vs_training_positive_windows",
                    "identity_central16_vs_training_positive_windows", "best_training_acc", "best_training_window_start", "best_training_bond"])
        w.writerows(rows)
    f2 = lambda d, k: f"{d[k]:.3f}"                                              # noqa: E731
    lines = ["Local-window similarity audit of the external test cohort (model-free; rules in the header of 51_external_cohort_audit.py, fixed before the run)",
             f"training windows {out['n_training_windows']:,} in {len(train_accs):,} proteins; cohort windows not audited (within 15 residues of an end): N {partial['N']}, C {partial['C']}",
             f"background (99th percentile of the shuffled observed windows): identity over 32 positions {bg32:.3f}, over the central 16 {bg16:.3f}"]
    for k in ("observed", "random_negative", "shuffled"):
        for T in ("N", "C"):
            g = out["groups"][f"{k}_{T}"]
            x, y = g["id32"], g["id16"]
            lines.append(f"{k:15s} {T}: n {x['n']:4d} | id32 median {f2(x, 'median')} p90 {f2(x, 'p90')} p99 {f2(x, 'p99')} max {f2(x, 'max')} | at or above " + ", ".join(f"{t} {c}" for t, c in x["at_or_above"].items()) +
                         f" | central16 median {f2(y, 'median')} max {f2(y, 'max')} | at or above " + ", ".join(f"{t} {c}" for t, c in y["at_or_above"].items()))
            g2 = g["id32_vs_training_positive_windows"]
            lines.append(f"{'':15s}    against the training-positive windows only: id32 median {f2(g2, 'median')} p99 {f2(g2, 'p99')} max {f2(g2, 'max')}")
    sh = out["share_of_observed_windows_above_background_p99"]
    lines.append(f"share of observed windows above the background p99: id32 {sh['id32']:.3f}, central16 {sh['id16']:.3f}")
    lines.append("probability that an observed window has a higher identity than a random negative one (ties 0.5): " +
                 "; ".join(f"{T}: id32 {v['id32']:.3f}, against training positives {v['id32_vs_training_positive_windows']:.3f}" for T, v in out["probability_observed_window_above_negative_window"].items()))
    lines.append("observed windows by homology class (id32): " + "; ".join(f"{c}: n {v['n']}, median {v['median']:.3f}, max {v['max']:.3f}" for c, v in out["observed_windows_by_class"].items() if v["n"]))
    lines.append(f"observed windows at or above 0.70: {len(top)}" + ("" if not top else "\n" + "\n".join(f"   {r[1]} {r[2]} bond {r[3]} ({r[4]}, {r[5]}): id32 {r[6]}, central16 {r[7]} to {r[10]} window start {r[11]}" for r in top)))
    lines.append("top 5 observed windows: " + "; ".join(f"{d['terminus']} {d['acc']} b{d['bond']} {d['cls']} id32 {d['id32']:.3f} to {d['training_acc']}" for d in out["top10_observed_windows"][:5]))
    lines.append("trigger of rule 5 (an observed window at or above 0.70): " + ("MET" if trig else "not met; no sensitivity is computed"))
    if trig:
        for k, v in out["sensitivity"].items():
            lines.append(f"   {k}: primary macro AUROC {v['primary']:.3f} (items {v['n_items_all']}); without the items holding such a window {v['without_items']:.3f} (items {v['n_items_without']})")
    lines += [""] + independence_lines(ind)
    open(os.path.join(a.out, "window_audit.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
