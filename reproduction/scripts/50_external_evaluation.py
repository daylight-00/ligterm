#!/usr/bin/env python3
"""External evaluation: the release checkpoints LigTerm and LigTerm-GB scored, once, on candidate units of studies that are not in the training units.

Four studies outside the unit table (42797554, 29557506, 41191777, 29632711) contribute candidate units, each with a stratum computed against the training population: new_peptide_known_protein = a new
peptide on a protein that has training units; unseen_protein_no_shared_9mer = a protein without training units that shares no 9-mer with the training peptides; unseen_protein_shared_9mer = an unseen protein
that shares a 9-mer. The unseen proteins were searched for homologs in the training proteins and put in homology classes H0 to H3. Strata of the evaluation: the external test cohort = the units of unseen
proteins without a shared 9-mer, of class H0, H1 or H2 (its host-source and pathogen-source parts separately); new peptides on training proteins = the units of new_peptide_known_protein; the rest (unseen
proteins that share a 9-mer, and unseen proteins without a shared 9-mer of class H3) is described, not read. The criteria are the authors' own: the cohort is small, not blind and not homology-free.
Labels, per protein and terminus: the terminal bond of the study's units in the stratum is positive; bonds of the same terminus of other units (training units, other studies), of the study's units in other
strata and other occurrences of the same peptide are excluded; the other bonds are negative (ligterm_repro.external). Metrics: macro Bond-AUROC over proteins (ties 0.5), pooled Local-AUC, 95 % intervals
by a bootstrap over protein accessions (2,000 resamples), a label-permutation control, and the paired difference of the two checkpoints.
The checkpoints are scored exactly as released (numpy scorers; their files are checked against their hash lists). Guards: the data equal the release card (the training rows), every unit peptide equals its
sequence at its position, the strata equal the stratum counts of the audit table, the cohort equals the table of the cohort (56 / 91 / 53 / 11 units, 211), all scores are finite, and the library AUROC equals an
independent brute force on 25 items. The selection of the cohort is stated by its stored tables; the searches that produced them are not repeated.

usage: 50_external_evaluation.py --data <datasets folder> --external <datasets/external folder> --ens <models/ens> --gb <models/gb> --out <prefix> [--nproc 4] [--ligterm-root DIR]
"""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "lib"))
from ligterm_repro import data, metrics  # noqa: E402
from ligterm_repro.external import (NEW_PEPTIDE_KNOWN_PROTEIN, UNSEEN_PROTEIN, UNSEEN_PROTEIN_SHARED_9MER, auroc_bruteforce, bond_of, build_labels)  # noqa: E402

STUDIES = ["42797554", "29557506", "41191777", "29632711"]
STRATA = ["external_test_cohort", "external_test_cohort_host", "external_test_cohort_pathogen", "new_peptides_on_training_proteins", "overlap_or_homology_descriptive"]
COHORT, COHORT_HOST, COHORT_PATHOGEN, NEW_PEPTIDES, DESCRIPTIVE = STRATA
SID = {x: i for i, x in enumerate(STRATA)}
DEV_RANGE = {"N": (0.6591, 0.8706), "C": (0.6588, 0.8743)}                  # the per-study development range of LigTerm (19 studies)
DEV_MEDIAN = {"N": 0.7955, "C": 0.7752}
MIN_PROT = 30
NAN3 = (float("nan"),) * 3


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def check_hashes(model_dir):
    n = 0
    for ln in open(os.path.join(model_dir, "sha256.txt")):
        if ln.strip():
            h, rel = ln.split(None, 1)
            assert sha256_file(os.path.join(model_dir, rel.strip())) == h, f"{rel.strip()} of {model_dir} differs from its hash list"
            n += 1
    return n


_M = {}
_SEQ = {}


def _init(ens_dir, gb_dir, ligterm_root):
    sys.path.insert(0, ligterm_root)
    from ligterm.gb import load_model
    _M["ens"], _M["gb"] = load_model(ens_dir), load_model(gb_dir)


def _score(chunk):
    ss = [_SEQ[a] for a in chunk]
    e, g = _M["ens"].bond_scores_many(ss), _M["gb"].bond_scores_many(ss)
    return [(a, e[i], g[i]) for i, a in enumerate(chunk)]


def jclean(o):
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, dict):
        return {k: jclean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jclean(v) for v in o]
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--external", required=True)
    ap.add_argument("--ens", required=True)
    ap.add_argument("--gb", required=True)
    ap.add_argument("--out", required=True, help="output prefix: <prefix>.json, <prefix>_per_item.csv, <prefix>.txt")
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--ligterm-root", default=os.path.join(here, "..", "..", "src"), help="the folder that contains the ligterm package (default: src/ of the repository)")
    a = ap.parse_args()
    if os.path.exists(a.out + ".json"):
        sys.exit(f"refusing to overwrite {a.out}.json")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)       # the folder of the output prefix, before the long computation
    t0 = time.time()

    def log(*x):
        print(f"[{time.time() - t0:5.0f}s]", *x, flush=True)

    nh = check_hashes(a.ens) + check_hashes(a.gb)
    log(f"model files equal their hash lists ({nh} files)")
    # ---------------------------------------------------------------- data
    seqs = json.load(open(os.path.join(a.data, "protein_sequences.json")))
    SEQ = dict(json.load(open(os.path.join(a.external, "sequences.json"))))
    SEQ.update(seqs)                                                          # the sequences of the training context win
    units = []                                                                # (pmid, acc, start, end, peptide, stratum, organism)
    with gzip.open(os.path.join(a.external, "candidate_units.tsv.gz"), "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            units.append((r["pmid"], r["acc"], int(r["start"]), int(r["end"]), r["seq"], r["stratum"], r["organism"]))
    cls = {r["acc"]: r["class"] for r in csv.DictReader(open(os.path.join(a.external, "homology_classes.tsv")), delimiter="\t")}
    CS = json.load(open(os.path.join(a.external, "cohort_counts.json")))
    AT = {s["pmid"]: s["strata"] for s in json.load(open(os.path.join(a.external, "candidate_strata_counts.json")))["studies"]}
    for (pm, acc, s, e, q, st, org) in units:
        if pm in STUDIES:
            assert SEQ[acc][s - 1:e] == q, f"{pm} {acc}: the peptide differs from the sequence at its position"
            assert org, f"{pm} {acc}: no source organism"
    cnt = collections.defaultdict(collections.Counter)
    for (pm, acc, s, e, q, st, org) in units:
        cnt[pm][st] += 1
    for pm, c_ in cnt.items():
        assert dict(c_) == {k: v for k, v in AT[pm].items() if v}, f"{pm}: the strata of the candidate table {dict(c_)} differ from the audit table {AT[pm]}"
    org_of, by_sp, cand_by_acc = {}, {}, {}
    for (pm, acc, s, e, q, st, org) in units:
        org_of.setdefault(acc, org)
        by_sp.setdefault((pm, acc), []).append((s, e, q, st))
        cand_by_acc.setdefault(acc, {}).setdefault(pm, []).append((s, e))
    log(f"{len(units)} candidate units of {len(cnt)} studies equal the stratum counts of the audit table; homology classes of {len(cls)} proteins")
    # ---------------------------------------------------------------- strata -> items
    clean = ("H0", "H1", "H2")
    host = lambda acc: org_of[acc].lower().startswith("homo sapiens")         # noqa: E731

    def positive_strata(X, acc):
        c = cls.get(acc)
        if X in (COHORT, COHORT_HOST, COHORT_PATHOGEN):
            if c not in clean or (X == COHORT_HOST and not host(acc)) or (X == COHORT_PATHOGEN and host(acc)):
                return None
            return {UNSEEN_PROTEIN}
        if X == NEW_PEPTIDES:
            return {NEW_PEPTIDE_KNOWN_PROTEIN}
        return {UNSEEN_PROTEIN, UNSEEN_PROTEIN_SHARED_9MER} if c == "H3" else {UNSEEN_PROTEIN_SHARED_9MER}   # the units of unseen proteins that share a 9-mer, and those of H3 proteins without one

    ITEMS = {X: [] for X in STRATA}
    for pm in STUDIES:
        accs = sorted({acc for (p_, acc) in by_sp if p_ == pm})
        for X in STRATA:
            for acc in accs:
                ps = positive_strata(X, acc)
                if ps is None:
                    continue
                us = by_sp[(pm, acc)]
                if not any(st in ps for (_, _, _, st) in us):
                    continue
                ITEMS[X].append(dict(study=pm, acc=acc, units=us, pos_strata=ps))

    def n_units(X, pm):
        return sum(1 for it in ITEMS[X] if it["study"] == pm for (_, _, _, st) in it["units"] if st in it["pos_strata"])

    for pm in STUDIES:
        its = [it for it in ITEMS[COHORT] if it["study"] == pm]
        assert (n_units(COHORT, pm), len(its)) == (CS[pm]["units"], CS[pm]["proteins"]), f"{pm}: the cohort has {n_units(COHORT, pm)} units in {len(its)} proteins, the table says {CS[pm]['units']} and {CS[pm]['proteins']}"
        assert n_units(NEW_PEPTIDES, pm) == AT[pm][NEW_PEPTIDE_KNOWN_PROTEIN], f"{pm}: {n_units(NEW_PEPTIDES, pm)} units against {AT[pm][NEW_PEPTIDE_KNOWN_PROTEIN]} units of new peptides on known proteins"
        assert n_units(COHORT, pm) + n_units(DESCRIPTIVE, pm) == AT[pm].get(UNSEEN_PROTEIN, 0) + AT[pm].get(UNSEEN_PROTEIN_SHARED_9MER, 0), f"{pm}: the cohort and the descriptive stratum do not add up to the units of unseen proteins"
        assert n_units(COHORT_HOST, pm) + n_units(COHORT_PATHOGEN, pm) == n_units(COHORT, pm) and len([i for i in ITEMS[COHORT_HOST] if i["study"] == pm]) + len([i for i in ITEMS[COHORT_PATHOGEN] if i["study"] == pm]) == len(its), f"{pm}: the host and pathogen parts do not partition the cohort"
    assert sum(n_units(COHORT, pm) for pm in STUDIES) == CS["all_studies"]["units"] == 211
    DESIGN = {X: {pm: dict(items=len([i for i in ITEMS[X] if i["study"] == pm]), units=n_units(X, pm)) for pm in STUDIES} for X in STRATA}
    log("items (study-protein pairs / units in the stratum): " + "; ".join(f"{X} {len(ITEMS[X])} / {sum(n_units(X, pm) for pm in STUDIES)}" for X in STRATA))
    # ---------------------------------------------------------------- training units on the proteins of interest, checked against the release card
    interest = {it["acc"] for X in STRATA for it in ITEMS[X]}
    card = json.load(open(os.path.join(a.gb, "meta.json"), encoding="utf-8"))
    ens_card = json.load(open(os.path.join(a.ens, "meta.json"), encoding="utf-8"))
    assert sha256_file(os.path.join(a.data, "protein_sequences.json")) == card["seqs_sha256"], "the sequence file is not that of the release"
    c = data.load_ctx(os.path.join(a.data, "units.tsv.gz"), os.path.join(a.data, "protein_sequences.json"))
    h = hashlib.sha256()
    for arr in (c.vp, c.vs, c.ve):
        h.update(np.ascontiguousarray(arr, dtype=np.int64).tobytes())
    assert h.hexdigest() == card["training_rows_sha256"] == ens_card["training_rows_sha256"] and c.nv == card["training_units"] == ens_card["training_units"] == 417415, "the verified rows are not the training rows of the release"
    train = {}
    for p, s, e in zip(c.vp.tolist(), c.vs.tolist(), c.ve.tolist()):
        acc = c.names[p]
        if acc in interest:
            train.setdefault(acc, set()).add((s, e))
    log(f"the verified rows are the 417,415 training units of both releases (row hash {h.hexdigest()[:8]}); {len(train)} of the {len(interest)} proteins of interest have training units")
    del c
    # ---------------------------------------------------------------- scoring: the released checkpoints, once per distinct protein
    _SEQ.update(SEQ)
    accs = sorted(interest, key=lambda x: -len(SEQ[x]))
    chunks = [accs[i::a.nproc * 6] for i in range(a.nproc * 6)]
    with get_context("fork").Pool(a.nproc, initializer=_init, initargs=(a.ens, a.gb, os.path.abspath(a.ligterm_root))) as pool:
        done = pool.map(_score, chunks)
    SC = {x: dict(ens=e, gb=g) for part in done for (x, e, g) in part}
    assert len(SC) == len(interest)
    for x, v in SC.items():
        L = len(SEQ[x])
        assert all(len(v[m][i]) == L - 1 and np.isfinite(v[m][i]).all() for m in ("ens", "gb") for i in (0, 1)), f"{x}: the scores do not cover its bonds or are not finite"
    log(f"scored {len(SC)} proteins ({sum(len(SEQ[x]) - 1 for x in SC):,} bonds) with LigTerm and LigTerm-GB")
    # ---------------------------------------------------------------- labels and the per-protein metrics
    rng_u = np.random.default_rng([20261120, 34])
    MODELS = [("LigTerm", "ens"), ("LigTerm-GB", "gb")]
    out = dict(studies=STUDIES, strata=STRATA, dev_range=DEV_RANGE, dev_median=DEV_MEDIAN, min_proteins=MIN_PROT, models=[m for m, _ in MODELS], design=DESIGN, results={}, controls={})
    PP, FLAT, GID, STUDY_OF = {}, {}, {}, {}
    for X in STRATA:
        items = ITEMS[X]
        uacc = {x: i for i, x in enumerate(sorted({it["acc"] for it in items}))}
        GID[X] = np.array([uacc[it["acc"]] for it in items], np.int64)
        STUDY_OF[X] = np.array([it["study"] for it in items])
        for ti, T in enumerate("NC"):
            labs, nb = [], []
            for it in items:
                L = len(SEQ[it["acc"]])
                ob = {bond_of(s, e, L, T) for (s, e) in train.get(it["acc"], ())}
                for pm, uu in cand_by_acc.get(it["acc"], {}).items():
                    if pm != it["study"]:
                        ob |= {bond_of(s, e, L, T) for (s, e) in uu}
                ob.discard(None)
                labs.append(build_labels(SEQ[it["acc"]], it["units"], lambda st, ps=it["pos_strata"]: st in ps, ob, T))
                nb.append(L - 1)
            lab = np.concatenate(labs) if labs else np.zeros(0, np.int8)
            nb = np.array(nb, np.int64)
            ebase = np.concatenate([[0], np.cumsum(nb)[:-1]]) if len(nb) else nb
            FLAT[(X, T)] = (lab, nb, ebase)
            for mname, key in MODELS:
                sc = np.concatenate([SC[it["acc"]][key][ti] for it in items]) if items else np.zeros(0)
                FLAT[(X, T, mname)] = sc
                PP[(X, T, mname)] = metrics.per_protein(sc, lab, ebase, nb)
    log("per-protein metrics computed for all strata, termini and models")
    pick = [(X, int(i)) for X in (COHORT, NEW_PEPTIDES, DESCRIPTIVE) for i in rng_u.choice(len(ITEMS[X]), size=min(9, len(ITEMS[X])), replace=False)][:25]
    nchk = 0
    for (X, i) in pick:
        for T in "NC":
            lab, nb, ebase = FLAT[(X, T)]
            for mname, _ in MODELS:
                ref = auroc_bruteforce(FLAT[(X, T, mname)][ebase[i]:ebase[i] + nb[i]], lab[ebase[i]:ebase[i] + nb[i]])
                got = PP[(X, T, mname)]["bond_auroc"][i]
                assert (math.isnan(ref) and math.isnan(got)) or abs(ref - got) < 1e-12, f"{X} item {i} {T} {mname}: the library AUROC {got} differs from the brute force {ref}"
                nchk += 1
    log(f"guard passed: the library AUROC equals the brute force on {nchk} item-terminus-model cells")

    # ---------------------------------------------------------------- intervals and the report
    def slice_pp(pp, mask):
        return {k: v[mask] for k, v in pp.items()}

    def safe_boot(pp, gid, rng):
        if int((~np.isnan(pp["bond_auroc"])).sum()) == 0:
            return dict(bond_auroc=NAN3, bond_auprc=NAN3, bond_recall_at_10pct=NAN3, local_auc=NAN3, n_proteins=0, n_groups=0, n_pos=0)
        return metrics.boot(pp, gid, rng)

    def safe_boot_diff(x, y, gid, rng):
        if int((~np.isnan(x["bond_auroc"]) & ~np.isnan(y["bond_auroc"])).sum()) == 0:
            return NAN3
        return metrics.boot_diff(x, y, gid, rng)

    def summarise(res):
        return dict(bond_auroc=list(res["bond_auroc"]), local_auc=list(res["local_auc"]), n_proteins=res["n_proteins"], n_clusters=res["n_groups"], n_pos=res["n_pos"], too_small=res["n_proteins"] < MIN_PROT)

    for X in STRATA:
        out["results"][X] = {}
        for ti, T in enumerate("NC"):
            sid = SID[X]
            cell = dict(pooled={}, per_study={})
            for k, (mname, _) in enumerate(MODELS):
                cell["pooled"][mname] = summarise(safe_boot(PP[(X, T, mname)], GID[X], np.random.default_rng([20261120, 31, sid, ti, k])))
            d = safe_boot_diff(PP[(X, T, "LigTerm")], PP[(X, T, "LigTerm-GB")], GID[X], np.random.default_rng([20261120, 31, sid, ti, 2]))
            cell["pooled"]["LigTerm - LigTerm-GB"] = dict(bond_auroc=list(d))
            for si, pm in enumerate(STUDIES):
                mk = STUDY_OF[X] == pm
                if not mk.any():
                    continue
                cell["per_study"][pm] = {}
                for k, (mname, _) in enumerate(MODELS):
                    cell["per_study"][pm][mname] = summarise(safe_boot(slice_pp(PP[(X, T, mname)], mk), GID[X][mk], np.random.default_rng([20261120, 32, si, sid, ti, k])))
            out["results"][X][T] = cell
    for ti, T in enumerate("NC"):                                             # the label-permutation control on the cohort (macro AUROC must be 0.5)
        lab, nb, ebase = FLAT[(COHORT, T)]
        m_, s_ = metrics.shuffle_control(FLAT[(COHORT, T, "LigTerm")], lab, ebase, nb, np.random.default_rng([20261120, 33, ti]))
        out["controls"][f"label_permutation_{COHORT}_{T}"] = dict(mean=m_, sd=s_)
    r1 = out["results"][COHORT]
    est = {T: r1[T]["pooled"]["LigTerm"]["bond_auroc"] for T in "NC"}
    above = {T: bool(est[T][1] > 0.5) for T in "NC"}
    where = {T: ("within" if DEV_RANGE[T][0] <= est[T][0] <= DEV_RANGE[T][1] else ("below" if est[T][0] < DEV_RANGE[T][0] else "above")) for T in "NC"}
    if above["N"] and above["C"]:
        phrase = "above chance (lower 95 % bound > 0.5) at both termini"
    elif above["N"] or above["C"]:
        phrase = f"above chance (lower 95 % bound > 0.5) at the {'N' if above['N'] else 'C'} terminus only"
    else:
        phrase = "not clearly above chance at either terminus"
    npro = {T: r1[T]["pooled"]["LigTerm"]["n_proteins"] for T in "NC"}
    out["reading"] = dict(above_chance=above, versus_dev_range=where, n_proteins=npro,
                          sentence=(f"on proteins absent from the training population with no detected homolog under the stated search criteria (no shared 9-mer; no hit of identity >= 0.40 over >= 0.50 of the query at E <= 1e-3), the released LigTerm ranked the observed terminal bonds {phrase} "
                                    f"(macro AUROC N {est['N'][0]:.3f} [{est['N'][1]:.3f}, {est['N'][2]:.3f}], C {est['C'][0]:.3f} [{est['C'][1]:.3f}, {est['C'][2]:.3f}]; {npro['N']} and {npro['C']} proteins)"))
    out["guards"] = dict(strata_equal_audit_table=True, e1_equals_clean_subset_table=True, training_rows_equal_release=True, sequences_verified=True, brute_force_cells=nchk, scored_proteins=len(SC))
    json.dump(jclean(out), open(f"{a.out}.json", "w"), indent=1)
    with open(f"{a.out}_per_item.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stratum", "study", "acc", "organism", "class", "length", "units_in_stratum", "positive_bonds_N", "positive_bonds_C", "auc_N_LigTerm", "auc_N_LigTerm-GB", "auc_C_LigTerm", "auc_C_LigTerm-GB"])
        for X in STRATA:
            for i, it in enumerate(ITEMS[X]):
                row = [X, it["study"], it["acc"], org_of[it["acc"]], cls.get(it["acc"], ""), len(SEQ[it["acc"]]), sum(1 for (_, _, _, st) in it["units"] if st in it["pos_strata"]),
                       int(PP[(X, "N", "LigTerm")]["npos"][i]), int(PP[(X, "C", "LigTerm")]["npos"][i])]
                row += [f"{PP[(X, T, m)]['bond_auroc'][i]:.6f}" for T in "NC" for m in ("LigTerm", "LigTerm-GB")]
                w.writerow(row)
    fmt = lambda b: f"{b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}]"                     # noqa: E731
    lines = ["EXTERNAL EVALUATION of the release checkpoints (run once): macro Bond-AUROC over proteins [95 % interval, cluster bootstrap over protein accessions], pooled over the four studies",
             f"development range of LigTerm (per study, 19 studies): N {DEV_RANGE['N'][0]} to {DEV_RANGE['N'][1]} (median {DEV_MEDIAN['N']}); C {DEV_RANGE['C'][0]} to {DEV_RANGE['C'][1]} (median {DEV_MEDIAN['C']})"]
    for X in STRATA:
        for T in "NC":
            cc = out["results"][X][T]["pooled"]
            e, g = cc["LigTerm"], cc["LigTerm-GB"]
            lines.append(f"  {X} {T}: LigTerm {fmt(e['bond_auroc'])}  Local-AUC {e['local_auc'][0]:.3f} | LigTerm-GB {fmt(g['bond_auroc'])} | LigTerm - LigTerm-GB {fmt(cc['LigTerm - LigTerm-GB']['bond_auroc'])} | proteins {e['n_proteins']}, clusters {e['n_clusters']}, positive bonds {e['n_pos']}" + ("  TOO SMALL TO READ" if e["too_small"] else ""))
    lines.append("control: label permutation on the cohort (macro AUROC of LigTerm with permuted labels; must be 0.5): " + ", ".join(f"{k[-1]} {v['mean']:.3f} (sd {v['sd']:.3f})" for k, v in out["controls"].items()))
    lines.append(f"READING: above chance {out['reading']['above_chance']}; the cohort point estimate against the development range of LigTerm: {out['reading']['versus_dev_range']}")
    lines.append("SENTENCE (fixed in advance): " + out["reading"]["sentence"])
    open(f"{a.out}.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
