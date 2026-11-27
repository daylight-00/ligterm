"""Protein-representation comparison (ESMC frozen and LoRA): the aggregation of the stored per-study records.

Reads the stored records of the comparison (one file per study for the window MLP and the frozen-ESMC arm, one for the LoRA-adapted arm), checks that they are consistent with the specification and with one another
(`check_runs`: any failure stops the aggregation) and writes the per-study tables, the medians over the 19 unseen development studies, the paired contrasts with their group-bootstrap intervals, the gates and the
pre-specified readings, the extension rule, the window-MLP floor diagnostic, the strata by protein length, the selection summary (epochs of the heads and the models at the cap) and the spread scale of the ensembles.
Models: LR-Both16-CV (the one-hot window-16 logistic model with the penalty chosen on the training folds, copied from the earlier record), LigTerm-GB (its refit), Window MLP, ESMC MLP (Frozen), ESMC MLP (LoRA) and the
three ensembles of LigTerm-GB with one arm each (Scale-matched ensemble, GB + frozen ESMC MLP, GB + LoRA ESMC MLP).
Pre-specified rules (the specification of the comparison): the development gate of a contrast: the median over the 19 unseen studies of the per-study Bond-AUROC difference >= +0.01 and a positive difference in at
least 13 of the 19, at both termini; the subset screen (LoRA arm on the five studies): median >= +0.01 and a positive difference in at least 4 of 5 at both termini; competence: the median difference to LigTerm-GB >= -0.01 at
both termini (repeated on the Length-matched AUC of both_bonds; a disagreement is stated); the extension rule: over the five subset studies the median of (ESMC MLP (LoRA) - ESMC MLP (Frozen)) >= +0.005 at both termini and
positive in at least 4 of 5. The floor of the window MLP against LR-Both16-CV is reported as a diagnostic. This is model selection on exposed development studies, not a test.
The arithmetic, the order of the bootstrap draws and the rounding are those of the stored aggregate. Whether the LoRA arm was run on all studies or on the five subset studies is read from the records present; the extension rule is computed from the subset studies either way.
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

from . import development_records as DR
from .development_records import bvec, mkey_of, pvec, printed

PEP_KEYS = DR.PEP_KEYS
K = mkey_of
LR16_CV, LIGTERM_GB = K("LR-Both16", "cv"), K("LigTerm-GB")
WINDOW_MLP, FROZEN, LORA = K("Window MLP"), K("ESMC MLP (Frozen)"), K("ESMC MLP (LoRA)")
ENSEMBLE_SCALE, ENSEMBLE_FROZEN, ENSEMBLE_LORA = K("Scale-matched ensemble"), K("GB + frozen ESMC MLP"), K("GB + LoRA ESMC MLP")
FIRST_PART_MODELS = [LR16_CV, LIGTERM_GB, WINDOW_MLP, FROZEN, ENSEMBLE_SCALE, ENSEMBLE_FROZEN]
SECOND_PART_MODELS = [LR16_CV, LIGTERM_GB, LORA, ENSEMBLE_LORA]
PART_OF = {"window_mlp_and_frozen": FIRST_PART_MODELS, "lora": SECOND_PART_MODELS}
ARMS = (WINDOW_MLP, FROZEN, LORA)
MEDIAN_ORDER = (ENSEMBLE_SCALE, ENSEMBLE_FROZEN, ENSEMBLE_LORA, LR16_CV, LIGTERM_GB, WINDOW_MLP, FROZEN, LORA)
# (contrast name, model A, model B, bootstrapped): the contrast is A minus B
CONTRASTS = (("frozen_vs_window_mlp", FROZEN, WINDOW_MLP, True), ("lora_vs_frozen", LORA, FROZEN, True), ("frozen_vs_ligterm_gb", FROZEN, LIGTERM_GB, True), ("lora_vs_ligterm_gb", LORA, LIGTERM_GB, True),
             ("ensemble_frozen_vs_scale_matched", ENSEMBLE_FROZEN, ENSEMBLE_SCALE, True), ("ensemble_lora_vs_scale_matched", ENSEMBLE_LORA, ENSEMBLE_SCALE, True), ("scale_matched_vs_ligterm_gb", ENSEMBLE_SCALE, LIGTERM_GB, True),
             ("ensemble_frozen_vs_ligterm_gb", ENSEMBLE_FROZEN, LIGTERM_GB, True), ("ensemble_lora_vs_ligterm_gb", ENSEMBLE_LORA, LIGTERM_GB, True),
             ("window_mlp_vs_lr_both16_cv", WINDOW_MLP, LR16_CV, False), ("window_mlp_vs_ligterm_gb", WINDOW_MLP, LIGTERM_GB, False), ("frozen_vs_lr_both16_cv", FROZEN, LR16_CV, False),
             ("lora_vs_lr_both16_cv", LORA, LR16_CV, False), ("ensemble_frozen_vs_frozen", ENSEMBLE_FROZEN, FROZEN, False), ("ensemble_lora_vs_lora", ENSEMBLE_LORA, LORA, False),
             ("scale_matched_vs_window_mlp", ENSEMBLE_SCALE, WINDOW_MLP, False))
BOOT_CONTRASTS = tuple((n, a, b) for n, a, b, boot in CONTRASTS if boot)
CONTRAST_MODELS = {n: (a, b) for n, a, b, _ in CONTRASTS}
BOTH, BOTH_L = "both_bonds", "both_bonds_and_length"
METRICS = (("N", "bond_auroc"), ("C", "bond_auroc"), ("N", "local_auc"), ("C", "local_auc"), (BOTH, "peptide_auroc"), (BOTH, "length_matched_auc"), (BOTH, "near_miss_auc"),
           (BOTH_L, "peptide_auroc"), (BOTH_L, "length_matched_auc"), (BOTH_L, "near_miss_auc"))
# the pre-specified questions: the reading name -> the contrast(s) it reads
Q_FROZEN, Q_LORA, Q_COMPETENCE, Q_INCREMENTAL = "frozen_representation", "lora_adaptation", "competence", "incremental_value"
WORDING = {Q_FROZEN: ("the frozen ESMC representation carries terminal information beyond residue identity for this head", "the frozen ESMC representation was not found to carry terminal information beyond residue identity for this head"),
           Q_LORA: ("the LoRA-adapted training procedure improved on the tested frozen-encoder procedure", "no consistent improvement of the LoRA-adapted training procedure over the tested frozen-encoder procedure was found at this power"),
           Q_INCREMENTAL: ("the pretrained representation adds incremental value to LigTerm-GB beyond a sequence-only head of the same form (the ensembles differ in the whole arm pipeline, not only in the input of the head)",
                           "no incremental value of the pretrained representation was found at this power")}
SUBSET_SUFFIX = " (a screen on the five subset studies only: no statement at the level of the development gate)"        # under the subset screen the development-gate sentence is never printed alone
FLOOR_SENTENCE = ("**The window MLP did not reach the linear one-hot baseline** (an unfavourable result for this head, not evidence of an implementation defect): the questions are read for THIS head only, "
                  "nothing is said about the value of protein representations in general, and a failed competence is not evidence about ESMC.")


# ---------------------------------------------------------------------------------------------------------------------------------- the records
def merged_view(r1, r2):
    """one view of the two records of a study: the union of the models"""
    base = r1 if r1 is not None else r2
    m = {k: base[k] for k in ("study", "seen", "units", "rows", "bond_proteins", "peptide_proteins", "variants")}
    m.update(bond={}, peptide={}, bond_summary={}, peptide_summary={}, scale={T: {} for T in "NC"}, fits={}, device={}, gpu_seconds={}, floor=None)
    for r in (r1, r2):
        if r is None:
            continue
        for k in ("bond", "peptide", "bond_summary", "peptide_summary", "fits", "device", "gpu_seconds"):
            m[k].update(r[k])
        for T in "NC":
            m["scale"][T].update(r["scale"][T])
        if r.get("floor"):
            m["floor"] = r["floor"]
    return m


def check_runs(recs, studies, lora_studies, software, subset, tol=1e-9, seen=None, floor_study=0.05):
    """consistency of the records with the specification and with one another -> (fatal, notes). recs: {study: (first-part record or None, LoRA-part record or None)}. Fatal: the study sets (first part for every study, LoRA part for exactly
    lora_studies); the software of every record equal; the model sets; identical units, rows and protein lists in the two parts; vector lengths and bounds; non-finite pair counts; the quantities that depend on the labels only
    (pair counts and the pattern of the defined AUROC) equal across models, variants and parts; the seen flag equal to the configured list (when given); the LigTerm-GB and LR-Both16-CV vectors equal in the two parts; the summaries of the
    records equal to the recomputed ones. Notes: models at the epoch cap, violated floor margins."""
    fatal, notes = [], []
    first_part, second_part = "window_mlp_and_frozen", "lora"

    def bad(s, msg):
        fatal.append(f"{s}: {msg}")
    if sorted(s for s, (a, b) in recs.items() if a is not None) != sorted(studies):
        fatal.append(f"first-part records exist for {sorted(s for s, (a, b) in recs.items() if a is not None)}, expected {sorted(studies)}")
    if sorted(s for s, (a, b) in recs.items() if b is not None) != sorted(lora_studies):
        fatal.append(f"LoRA-part records exist for {sorted(s for s, (a, b) in recs.items() if b is not None)}, expected {sorted(lora_studies)}")
    first_env = None
    for s in sorted(recs):
        r1, r2 = recs[s]
        for tag, r, models, part in ((first_part, r1, FIRST_PART_MODELS, first_part), (second_part, r2, SECOND_PART_MODELS, second_part)):
            if r is None:
                continue
            if str(r.get("study")) != str(s) or r.get("part") != part or list(r["models"]) != models:
                bad(f"{s} {tag}", "study, part or model list differ from the expectation")
            if seen is not None and bool(r.get("seen")) != (s in seen):
                bad(f"{s} {tag}", f"the seen flag {r.get('seen')} differs from the configured list")
            env = r.get("env", {})
            key = {k: env.get(k) for k in ("python", "numpy", "scipy", "sklearn")}
            if first_env is None:
                first_env = key
            if key != first_env:
                bad(f"{s} {tag}", f"software differs from the other records: {key} against {first_env}")
            if software is not None and key != {k: software.get(k) for k in key}:
                bad(f"{s} {tag}", f"software differs from the specified one: {key} against {software}")
            u = r["units"]
            if u["total"] != u["candidate"] + u["dropped_bond_outside"] + u["dropped_length_only"]:
                bad(f"{s} {tag}", "the units are not candidate + dropped (bond outside) + dropped (length)")
            pp = r["peptide_proteins"]
            npro = len(pp["group"])
            if any(len(pp[k]) != npro for k in ("acc", "length", "n_pos")) or npro != u["evaluation_proteins"] or sum(pp["n_pos"]) != u["candidate"]:
                bad(f"{s} {tag}", "the peptide-level protein list does not match the evaluation proteins or the candidate units")
            ref_b, ref_p = {}, None
            for T in "NC":
                bp = r["bond_proteins"][T]
                nb = len(bp["group"])
                if len(bp["acc"]) != nb or len(set(bp["acc"])) != nb or len(bp["length"]) != nb:
                    bad(f"{s} {tag}", f"{T}: the bond-level protein list is inconsistent")
                for m in models:
                    vs = {k: bvec(r, m, T, k) for k in DR.BOND_KEYS}
                    if any(len(a) != nb for a in vs.values()):
                        bad(f"{s} {tag}", f"{T} {printed(m)}: a vector does not have {nb} entries")
                        continue
                    ok = np.isfinite(vs["bond_auroc"])
                    if (vs["bond_auroc"][ok] < -tol).any() or (vs["bond_auroc"][ok] > 1 + tol).any() or (vs["local_pairs_won"] < -tol).any() or (vs["local_pairs_won"] > vs["local_pairs"] + tol).any() or (vs["local_pairs"] < 0).any():
                        bad(f"{s} {tag}", f"{T} {printed(m)}: a metric is outside its bounds")
                    if T not in ref_b:
                        ref_b[T] = (vs["local_pairs"], vs["n_positive_bonds"], ok)
                    elif not (np.array_equal(vs["local_pairs"], ref_b[T][0]) and np.array_equal(vs["n_positive_bonds"], ref_b[T][1]) and np.array_equal(ok, ref_b[T][2])):
                        bad(f"{s} {tag}", f"{T} {printed(m)}: the local pair counts, the positives or the pattern of the defined AUROC differ from the first model (they depend on the labels only)")
            for m in models:
                for v in r["variants"]:
                    mv = {k: pvec(r, m, v, k) for k in PEP_KEYS}
                    if any(len(a) != npro for a in mv.values()):
                        bad(f"{s} {tag}", f"{printed(m)} {v}: a peptide vector does not have {npro} entries")
                        continue
                    F, Mn, Md, Xn, Xd = (mv[k] for k in PEP_KEYS)
                    okF = np.isfinite(F)
                    if not all(np.isfinite(a).all() for a in (Mn, Md, Xn, Xd)):
                        bad(f"{s} {tag}", f"{printed(m)} {v}: a non-finite pair count")
                        continue
                    if (F[okF] < -tol).any() or (F[okF] > 1 + tol).any() or (Mn < -tol).any() or (Mn > Md + tol).any() or (Xn < -tol).any() or (Xn > Xd + tol).any() or (Md < 0).any() or (Xd < 0).any():
                        bad(f"{s} {tag}", f"{printed(m)} {v}: a peptide-level quantity is outside its bounds")
                    if ref_p is None:
                        ref_p = (Md, Xd, okF)
                    elif not (np.array_equal(Md, ref_p[0]) and np.array_equal(Xd, ref_p[1]) and np.array_equal(okF, ref_p[2])):
                        bad(f"{s} {tag}", f"{printed(m)} {v}: the length-matched and near-miss pair counts or the pattern of the defined Peptide-AUROC differ from the first model and variant (they depend on the labels only)")
            st = DR.study_stats(r, models, r["variants"])
            for m in models:
                for T in "NC":
                    for q in ("bond_auroc", "local_auc"):
                        x, y = r["bond_summary"][m][T][q], st[m][T][q]
                        if not ((math.isnan(x) and math.isnan(y)) or abs(x - y) <= tol):
                            bad(f"{s} {tag}", f"{printed(m)} {T}: the stored {q} {x} differs from the recomputed {y}")
                for v in r["variants"]:
                    for q in ("peptide_auroc", "length_matched_auc", "near_miss_auc"):
                        x, y = r["peptide_summary"][m][v][q], st[m][v][q]
                        if not ((math.isnan(x) and math.isnan(y)) or abs(x - y) <= tol):
                            bad(f"{s} {tag}", f"{printed(m)} {v}: the stored {q} {x} differs from the recomputed {y}")
            for a, fits in r["fits"].items():
                for T in "NC":
                    f = fits[T]
                    if not (1 <= f["e_star"] <= f["e_cap"]) or f["e_star"] != f["select"]["best_epoch"] or bool(f["at_cap"]) != (f["select"]["best_epoch"] >= f["e_cap"]):
                        bad(f"{s} {tag}", f"{T} {printed(a)}: the selected epoch {f['e_star']} or the at-cap flag is inconsistent with the cap {f['e_cap']}")
                    if f["at_cap"]:
                        notes.append(f"{s} {T} {printed(a)}: the selected epoch {f['e_star']} is at the epoch cap")
        if r1 is not None and r2 is not None:
            for k in ("units", "rows", "bond_proteins", "peptide_proteins"):
                if r1[k] != r2[k]:
                    bad(s, f"the two parts differ in {k}")
            for m in (LR16_CV, LIGTERM_GB):
                for T in "NC":
                    for k in DR.BOND_KEYS:
                        if r1["bond"][m][T][k] != r2["bond"][m][T][k]:
                            bad(s, f"the two parts differ in the {printed(m)} vector {T} {k}")
                for v in r1["variants"]:
                    for k in PEP_KEYS:
                        if r1["peptide"][m][v][k] != r2["peptide"][m][v][k]:
                            bad(s, f"the two parts differ in the {printed(m)} peptide vector {v} {k}")
        r = r1 if r1 is not None else r2
        if r.get("floor"):
            for T in "NC":
                if r["floor"][T]["violated"]:
                    notes.append(f"{s} {T}: the window-MLP margin of {floor_study} below LR-Both16-CV is violated (a recorded diagnostic)")
    return fatal, notes


# ------------------------------------------------------------------------------------------------------------------------ the bootstrap and the tables
def _ratio_boot(inv, G, num, den, draw):
    sn = np.bincount(inv, weights=num, minlength=G)
    sd = np.bincount(inv, weights=den, minlength=G)
    return sn[draw].sum(1) / np.maximum(sd[draw].sum(1), 1e-300)


def _ci(a):
    return np.percentile(a, [2.5, 97.5]).tolist()


class Boot:
    """the paired group-bootstrap draws of one study (the same resamples for every contrast and quantity of a family; seed [seed, stage, study, 0 / 1 / 2 for N / C / peptide])"""

    def __init__(self, rec, B, seed, variants):
        self.rec, self.B, self.variants = rec, B, variants
        self.d = {}
        for ti, T in enumerate("NC"):
            ug, inv = np.unique(np.array(rec["bond_proteins"][T]["group"]), return_inverse=True)
            self.d[T] = (inv, len(ug), np.random.default_rng(seed + [int(rec["study"]), ti]).integers(0, len(ug), size=(B, len(ug))))
        ug, inv = np.unique(np.array(rec["peptide_proteins"]["group"]), return_inverse=True)
        self.d["pep"] = (inv, len(ug), np.random.default_rng(seed + [int(rec["study"]), 2]).integers(0, len(ug), size=(B, len(ug))))

    def contrast(self, A, Bm):
        rec, out = self.rec, {}
        for T in "NC":
            inv, G, draw = self.d[T]
            aA, aB = bvec(rec, A, T, "bond_auroc"), bvec(rec, Bm, T, "bond_auroc")
            ok = np.isfinite(aA) & np.isfinite(aB)
            out[(T, "bond_auroc")] = _ci(_ratio_boot(inv, G, np.where(ok, aA - aB, 0.0), ok.astype(float), draw))
            out[(T, "local_auc")] = _ci(_ratio_boot(inv, G, bvec(rec, A, T, "local_pairs_won") - bvec(rec, Bm, T, "local_pairs_won"), bvec(rec, A, T, "local_pairs"), draw))
        inv, G, draw = self.d["pep"]
        for v in self.variants:
            FA, FB = pvec(rec, A, v, "peptide_auroc"), pvec(rec, Bm, v, "peptide_auroc")
            okF = np.isfinite(FA) & np.isfinite(FB)
            out[(v, "peptide_auroc")] = _ci(_ratio_boot(inv, G, np.where(okF, FA - FB, 0.0), okF.astype(float), draw))
            Md = pvec(rec, A, v, "length_matched_den")
            okM = Md > 0
            rA = pvec(rec, A, v, "length_matched_num") / np.where(okM, Md, 1.0)
            rB = pvec(rec, Bm, v, "length_matched_num") / np.where(okM, Md, 1.0)
            out[(v, "length_matched_auc")] = _ci(_ratio_boot(inv, G, np.where(okM, rA - rB, 0.0), okM.astype(float), draw))
            out[(v, "near_miss_auc")] = _ci(_ratio_boot(inv, G, pvec(rec, A, v, "near_miss_num") - pvec(rec, Bm, v, "near_miss_num"), pvec(rec, A, v, "near_miss_den"), draw))
        return out


def strata_stats(rec, m, max_residues):
    out = {}
    for T in "NC":
        Lp = np.array(rec["bond_proteins"][T]["length"])
        auc, e3n, e3d = bvec(rec, m, T, "bond_auroc"), bvec(rec, m, T, "local_pairs_won"), bvec(rec, m, T, "local_pairs")
        for name, msk in (("short", Lp <= max_residues), ("long", Lp > max_residues)):
            ok = np.isfinite(auc) & msk
            out.setdefault(T, {})[name] = dict(n=int(msk.sum()), bond_auroc=float(auc[ok].mean()) if ok.any() else math.nan,
                                               local_auc=float(e3n[msk].sum() / e3d[msk].sum()) if e3d[msk].sum() > 0 else math.nan)
    return out


def contrast_table(ps, A, Bm, studies):
    """per metric: median over `studies`, the number of positive differences, per-study values; only the studies where both models exist"""
    have = [s for s in studies if A in ps[s] and Bm in ps[s]]
    c = {}
    for a, q in METRICS:
        d = [ps[s][A][a][q] - ps[s][Bm][a][q] for s in have]
        c[(a, q)] = dict(median=DR.med(d), n_positive=int(sum(1 for x in d if x > 0)), n=len(d), per_study={s: x for s, x in zip(have, d)})
    return c


def gate_of(c, n_expected, med_thr, n_thr):
    """the gate on the Bond-AUROC at both termini: -> dict(N, C, passed, evaluable)"""
    g = {T: dict(median=c[(T, "bond_auroc")]["median"], n_positive=c[(T, "bond_auroc")]["n_positive"], n=c[(T, "bond_auroc")]["n"],
                 passed=bool(c[(T, "bond_auroc")]["n"] == n_expected and c[(T, "bond_auroc")]["median"] >= med_thr and c[(T, "bond_auroc")]["n_positive"] >= n_thr)) for T in "NC"}
    return dict(N=g["N"], C=g["C"], passed=bool(g["N"]["passed"] and g["C"]["passed"]), evaluable=bool(g["N"]["n"] == n_expected))


def competence_of(c, EV):
    """competence: the median difference to LigTerm-GB >= -0.01 at both termini (Bond-AUROC), repeated on the Length-matched AUC of both_bonds"""
    thr = EV["competence"]
    auc = bool(all(c[(T, "bond_auroc")]["median"] >= thr for T in "NC"))
    mm = bool(c[(BOTH, "length_matched_auc")]["median"] >= thr)
    return dict(competent=auc, competent_on_length_matched_auc=mm, agree=bool(auc == mm), median_N=c[("N", "bond_auroc")]["median"], median_C=c[("C", "bond_auroc")]["median"],
                median_length_matched_auc=c[(BOTH, "length_matched_auc")]["median"], n=c[("N", "bond_auroc")]["n"])


def extension_rule(ps, subset, EV):
    """the extension rule on the five subset studies: the median of (LoRA - frozen) >= +0.005 at both termini and positive in at least 4 of the 5"""
    c = contrast_table(ps, LORA, FROZEN, subset)
    t = {T: dict(median=c[(T, "bond_auroc")]["median"], n_positive=c[(T, "bond_auroc")]["n_positive"], n=c[(T, "bond_auroc")]["n"],
                 passed=bool(c[(T, "bond_auroc")]["n"] == len(subset) and c[(T, "bond_auroc")]["median"] >= EV["extension_median"] and c[(T, "bond_auroc")]["n_positive"] >= EV["extension_min"])) for T in "NC"}
    return dict(extension=bool(t["N"]["passed"] and t["C"]["passed"]), termini=t, subset=list(subset), per_study={s: {T: c[(T, "bond_auroc")]["per_study"].get(s) for T in "NC"} for s in subset})


def aggregate(merged, studies, unseen, seen, lora_studies, lora_all, cfg):
    """merged: {study: merged view}; returns the result dict"""
    EV = cfg["levels"]
    B, SEED, SUBSET = cfg["bootstrap"], list(cfg["bootstrap_seed"]), list(cfg["lora_subset"])
    res = dict(studies=sorted(studies), unseen=list(unseen), seen=list(seen), lora_studies=sorted(lora_studies), lora_extended=bool(lora_all), per_study={}, strata={}, boot={}, contrasts={}, medians={}, gates={}, readings={}, floor={},
               selection={}, scale={}, timing={}, device={})
    ps = res["per_study"]
    for s, r in merged.items():
        models = list(r["bond"].keys())
        ps[s] = DR.study_stats(r, models, r["variants"])
        res["strata"][s] = {m: strata_stats(r, m, cfg["max_residues"]) for m in models}
        bt = Boot(r, B, SEED, r["variants"])
        res["boot"][s] = {name: bt.contrast(A, Bm) for name, A, Bm in BOOT_CONTRASTS if A in r["bond"] and Bm in r["bond"]}
        res["selection"][s] = {a: {T: dict(e_star=f[T]["e_star"], e_cap=f[T]["e_cap"], at_cap=f[T]["at_cap"], seconds=f[T]["seconds"]) for T in "NC"} for a, f in r["fits"].items()}
        res["scale"][s] = r["scale"]
        res["timing"][s] = r["gpu_seconds"]
        res["device"][s] = r["device"]
    all_models = [m for m in MEDIAN_ORDER if any(m in ps[s] for s in ps)]               # the order of the tables of the original aggregation (its internal model ids, sorted)
    for m in all_models:
        sel = [s for s in unseen if m in ps[s]]
        res["medians"][m] = dict(n=len(sel), values={(a, q): DR.med([ps[s][m][a][q] for s in sel]) for a, q in METRICS})
    for name, A, Bm, _ in CONTRASTS:
        c = contrast_table(ps, A, Bm, list(unseen))
        res["contrasts"][name] = dict(model_a=A, model_b=Bm, table=c)
    n_all, n_sub = EV["n_unseen"], EV["subset_n"]

    def gate_for(name):
        c = res["contrasts"][name]["table"]
        n = c[("N", "bond_auroc")]["n"]
        if n == n_all:
            return dict(kind="development_gate", **gate_of(c, n_all, EV["gate_median"], EV["gate_min"]))
        if n == n_sub:
            return dict(kind="subset_screen", **gate_of(c, n_sub, EV["gate_median"], EV["subset_min"]))
        return dict(kind="not_evaluable", passed=False, evaluable=False, n=n)
    for name in ("frozen_vs_window_mlp", "lora_vs_frozen", "ensemble_frozen_vs_scale_matched", "ensemble_lora_vs_scale_matched"):
        res["gates"][name] = gate_for(name)
    comp = {}
    for name in ("frozen_vs_ligterm_gb", "lora_vs_ligterm_gb"):
        comp[name] = competence_of(res["contrasts"][name]["table"], EV)
        comp[name]["exceeds_gate"] = gate_for(name)
    res["readings"][Q_COMPETENCE] = comp

    def word(key, gt):
        w = WORDING[key][0] if gt["passed"] else WORDING[key][1]
        return w + (SUBSET_SUFFIX if gt.get("kind") == "subset_screen" else "")
    for name, key in (("frozen_vs_window_mlp", Q_FROZEN), ("lora_vs_frozen", Q_LORA)):
        gt = res["gates"][name]
        res["readings"][key] = dict(gate=gt, wording=word(key, gt))
    res["readings"][Q_INCREMENTAL] = {name: dict(gate=res["gates"][name], wording=word(Q_INCREMENTAL, res["gates"][name])) for name in ("ensemble_frozen_vs_scale_matched", "ensemble_lora_vs_scale_matched")}
    sub_in = [s for s in SUBSET if s in ps and LORA in ps[s] and FROZEN in ps[s]]
    res["extension_rule"] = extension_rule(ps, sub_in, EV) if len(sub_in) == len(SUBSET) else None
    window_mlp_auc = {s: {T: ps[s][WINDOW_MLP][T]["bond_auroc"] for T in "NC"} for s in studies if WINDOW_MLP in ps[s]}
    reference_auc = {s: {T: ps[s][LR16_CV][T]["bond_auroc"] for T in "NC"} for s in studies}
    if len(window_mlp_auc) == len(studies):
        res["floor"] = DR.floor_diagnostic(window_mlp_auc, reference_auc, EV["floor_study"], EV["floor_panel"], list(unseen))
    return res


# ----------------------------------------------------------------------------------------------------------------------------------- the output
def fmt(x, d=4):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def render_report(res):
    un, seen = res["unseen"], res["seen"]
    L_ = ["# Protein-representation comparison (ESMC frozen and LoRA): a learned window head on a one-hot control, the frozen ESMC 300M store and a LoRA-adapted ESMC 300M\n"]
    L_.append(f"Studies: {len(res['studies'])} run, {len(un)} unseen (primary denominator), shown beside them: {', '.join(seen) if seen else 'none'}; the LoRA arm on {len(res['lora_studies'])} studies "
              f"({'all, by the extension rule' if res['lora_extended'] else 'the five-study subset'}). Medians are over the unseen studies that have the model. Every model of a study is fitted on the same rows; "
              "LR-Both16-CV is copied from the earlier record, LigTerm-GB is the refit that reproduces it.\n")
    L_.append("## Medians over the unseen studies\n")
    L_.append("| model | n | Bond-AUROC N | Bond-AUROC C | Local-AUC N | Local-AUC C | Peptide-AUROC (both_bonds) | Length-matched AUC (both_bonds) | Near-miss AUC (both_bonds) | Peptide-AUROC (+length) | Length-matched AUC (+length) | Near-miss AUC (+length) |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in (LR16_CV, LIGTERM_GB, WINDOW_MLP, FROZEN, LORA, ENSEMBLE_SCALE, ENSEMBLE_FROZEN, ENSEMBLE_LORA):
        if m in res["medians"]:
            r = res["medians"][m]
            L_.append(f"| {printed(m)} | {r['n']} | " + " | ".join(fmt(r["values"][k], 3) for k in METRICS) + " |")
    L_.append("\n## Contrasts (median per-study difference over the unseen studies that have both models; positive studies in brackets)\n")
    L_.append("| contrast | n | Bond-AUROC N | Bond-AUROC C | Local-AUC N | Local-AUC C | Length-matched AUC (both_bonds) | Near-miss AUC (both_bonds) |\n|---|---|---|---|---|---|---|---|")
    for name, A, Bm, _ in CONTRASTS:
        c = res["contrasts"][name]["table"]
        if c[("N", "bond_auroc")]["n"]:
            L_.append(f"| {name}: {printed(A)} - {printed(Bm)} | {c[('N', 'bond_auroc')]['n']} | " +
                      " | ".join(f"{fmt(c[k]['median'])} ({c[k]['n_positive']})" for k in (("N", "bond_auroc"), ("C", "bond_auroc"), ("N", "local_auc"), ("C", "local_auc"), (BOTH, "length_matched_auc"), (BOTH, "near_miss_auc"))) + " |")
    L_.append("\n## Pre-specified readings\n")
    for key in (Q_FROZEN, Q_LORA):
        r = res["readings"][key]
        g = r["gate"]
        L_.append(f"- **{key}** ({g.get('kind')}; N: median {fmt(g.get('N', {}).get('median'))}, {g.get('N', {}).get('n_positive')} positive; C: median {fmt(g.get('C', {}).get('median'))}, {g.get('C', {}).get('n_positive')} positive): passed = {g.get('passed')}: {r['wording']}")
    floor_hit = bool(res.get("floor") and (res["floor"]["panel_violated"] or res["floor"]["study_violations"]))
    for name, r in res["readings"][Q_COMPETENCE].items():
        L_.append(f"- **{Q_COMPETENCE}, {name}** (median difference to LigTerm-GB >= -0.01 at both termini): competent = {r['competent']} (N {fmt(r['median_N'])}, C {fmt(r['median_C'])}; on the Length-matched AUC {fmt(r['median_length_matched_auc'])}: "
                  f"competent = {r['competent_on_length_matched_auc']}; the two {'agree' if r['agree'] else 'DISAGREE'}); exceeds (the gate) = {r['exceeds_gate'].get('passed')}"
                  + (" (the window-MLP floor was violated: a failed competence of this head is not evidence about ESMC)" if floor_hit and not r["competent"] else ""))
    if floor_hit:
        L_.append(f"\n{FLOOR_SENTENCE}\n")
    for name, r in res["readings"][Q_INCREMENTAL].items():
        L_.append(f"- **{Q_INCREMENTAL}, {name}** ({r['gate'].get('kind')}): passed = {r['gate'].get('passed')}: {r['wording']}")
    if res.get("extension_rule"):
        e = res["extension_rule"]
        L_.append(f"- **extension rule** (ESMC MLP (LoRA) - ESMC MLP (Frozen) over the five subset studies >= +0.005 at both termini and positive in at least 4 of 5): N median {fmt(e['termini']['N']['median'])} ({e['termini']['N']['n_positive']} of 5), "
                  f"C median {fmt(e['termini']['C']['median'])} ({e['termini']['C']['n_positive']} of 5): extension = {e['extension']}")
    fl = res.get("floor")
    if fl:
        L_.append(f"\n## Window-MLP floor (diagnostic)\nMedian of (Window MLP - LR-Both16-CV) over the unseen studies: N {fmt(fl['median_unseen']['N'])}, C {fmt(fl['median_unseen']['C'])}; panel margin 0.02 violated: {fl['panel_violated']}; "
                  f"studies with a violated 0.05 margin: {', '.join(fl['study_violations']) or 'none'}.")
        if fl["panel_violated"] or fl["study_violations"]:
            L_.append(FLOOR_SENTENCE)
    L_.append("\n## Per-study differences of the questions' contrasts with their 95% group-bootstrap intervals (Bond-AUROC; N / C)\n")
    qn = [n for n, A, Bm in BOOT_CONTRASTS if any(n in res["boot"].get(s, {}) for s in res["studies"])]
    L_.append("| study | " + " | ".join(qn) + " |\n|---|" + "---|" * len(qn))
    cmap = {n: (A, Bm) for n, A, Bm in BOOT_CONTRASTS}
    for s in res["studies"]:
        cells = []
        for n in qn:
            A, Bm = cmap[n]
            if n in res["boot"].get(s, {}) and A in res["per_study"][s] and Bm in res["per_study"][s]:
                parts = []
                for T in "NC":
                    dlt = res["per_study"][s][A][T]["bond_auroc"] - res["per_study"][s][Bm][T]["bond_auroc"]
                    lo, hi = res["boot"][s][n][(T, "bond_auroc")]
                    parts.append(f"{dlt:+.3f} [{lo:+.3f}, {hi:+.3f}]")
                cells.append(" / ".join(parts))
            else:
                cells.append("-")
        L_.append(f"| {s}{' (seen)' if s in seen else ''} | " + " | ".join(cells) + " |")
    L_.append("\n## Per-study Bond-AUROC (N / C)\n")
    ms = [m for m in (LR16_CV, LIGTERM_GB, WINDOW_MLP, FROZEN, LORA, ENSEMBLE_FROZEN, ENSEMBLE_LORA) if m in res["medians"]]
    L_.append("| study | " + " | ".join(printed(m) for m in ms) + " |\n|---|" + "---|" * len(ms))
    for s in res["studies"]:
        L_.append(f"| {s}{' (seen)' if s in seen else ''} | " + " | ".join((f"{fmt(res['per_study'][s][m]['N']['bond_auroc'], 3)} / {fmt(res['per_study'][s][m]['C']['bond_auroc'], 3)}" if m in res["per_study"][s] else "-") for m in ms) + " |")
    L_.append("\n## Selection (the epoch of the one selection rule; at the cap)\n")
    for a in ARMS:
        es = [(s, T, res["selection"][s][a][T]) for s in res["studies"] if a in res["selection"][s] for T in "NC"]
        if es:
            L_.append(f"- {printed(a)}: {len(es)} models; selected epoch median {DR.med([e[2]['e_star'] for e in es]):.0f}, range {min(e[2]['e_star'] for e in es)} to {max(e[2]['e_star'] for e in es)}; at the cap "
                      f"{sum(1 for e in es if e[2]['at_cap'])}; seconds per model median {DR.med([e[2]['seconds'] for e in es]):.0f}")
    L_.append("\n## What the result does not show\nIt is an exploratory development result on exposed development studies for this head, this optimiser setting and this selection rule; it is not a blind validation, names no cause, does not bear on proteolysis, "
              "antigen processing or presentation, and a null is weak evidence against a small gain (the gate has little power below +0.02; two termini, two gates). The arms differ in pipeline as well as in input: the contrast of the "
              "LoRA arm with the frozen arm is a contrast of two specified procedures.")
    return "\n".join(L_) + "\n"


def _mf(m):
    return DR.mrow(m)


def _level(a, q):
    return {**DR.level_fields(a), "metric": q}


def summary_json(res):
    """the result as rows (model, fit, terminus, variant, metric are fields of a row); the order of every list follows the computation"""
    out = dict(studies=res["studies"], unseen=res["unseen"], seen=res["seen"], lora_studies=res["lora_studies"], lora_extended=res["lora_extended"])
    out["per_study_bond"] = [dict(study=s, **_mf(m), terminus=T, bond_auroc=st[T]["bond_auroc"], local_auc=st[T]["local_auc"]) for s in res["studies"] for m, st in res["per_study"][s].items() for T in "NC"]
    variants = sorted({v for s in res["studies"] for st in res["per_study"][s].values() for v in st if v not in ("N", "C")}, key=lambda v: (v != BOTH, v))
    out["per_study_peptide"] = [dict(study=s, **_mf(m), variant=v, **st[v]) for s in res["studies"] for m, st in res["per_study"][s].items() for v in variants]
    out["strata"] = [dict(study=s, **_mf(m), terminus=T, stratum=name, n_proteins=d["n"], bond_auroc=d["bond_auroc"], local_auc=d["local_auc"])
                     for s in res["studies"] for m, tt in res["strata"][s].items() for T, ss in tt.items() for name, d in ss.items()]
    out["bootstrap"] = [dict(study=s, contrast=n, **_level(a, q), lo=lo, hi=hi) for s in res["studies"] for n, qs in res["boot"].get(s, {}).items() for (a, q), (lo, hi) in qs.items()]
    out["contrasts"] = [dict(contrast=n, **_contrast_head(c), **_level(a, q), median=t["median"], n_positive=t["n_positive"], n=t["n"])
                        for n, c in res["contrasts"].items() for (a, q), t in c["table"].items()]
    out["contrast_studies"] = [dict(contrast=n, **_level(a, q), study=s, difference=x) for n, c in res["contrasts"].items() for (a, q), t in c["table"].items() for s, x in t["per_study"].items()]
    out["medians"] = [dict(**_mf(m), n=d["n"], values=[dict(**_level(a, q), median=v) for (a, q), v in d["values"].items()]) for m, d in res["medians"].items()]
    out["gates"] = [dict(contrast=n, **_gate(g)) for n, g in res["gates"].items()]
    rd = {}
    for key in (Q_FROZEN, Q_LORA):
        rd[key] = dict(gate=_gate(res["readings"][key]["gate"]), wording=res["readings"][key]["wording"])
    rd[Q_COMPETENCE] = [dict(contrast=n, **{k: v for k, v in r.items() if k != "exceeds_gate"}, exceeds_gate=_gate(r["exceeds_gate"])) for n, r in res["readings"][Q_COMPETENCE].items()]
    rd[Q_INCREMENTAL] = [dict(contrast=n, gate=_gate(r["gate"]), wording=r["wording"]) for n, r in res["readings"][Q_INCREMENTAL].items()]
    out["readings"] = rd
    out["floor"] = res["floor"] and dict(per_study=[dict(study=s, terminus=T, difference=x) for s, d in res["floor"]["per_study"].items() for T, x in d.items()], median_unseen=[dict(terminus=T, median=x) for T, x in res["floor"]["median_unseen"].items()],
                                          study_violations=res["floor"]["study_violations"], panel_violated=res["floor"]["panel_violated"])
    out["selection"] = [dict(study=s, **_mf(a), terminus=T, **d) for s in res["studies"] for a, tt in res["selection"][s].items() for T, d in tt.items()]
    out["score_scaling"] = [dict(study=s, terminus=T, **_mf(m), **d) for s in res["studies"] for T, mm in res["scale"][s].items() for m, d in mm.items()]
    out["gpu_seconds"] = [dict(study=s, **_mf(a), seconds=x) for s in res["studies"] for a, x in res["timing"][s].items()]
    out["device"] = [dict(study=s, **_mf(a), **d) for s in res["studies"] for a, d in res["device"][s].items()]
    e = res["extension_rule"]
    out["extension_rule"] = e and dict(extension=e["extension"], termini=[dict(terminus=T, **d) for T, d in e["termini"].items()], subset=e["subset"],
                                       per_study=[dict(study=s, terminus=T, difference=x) for s, d in e["per_study"].items() for T, x in d.items()])
    out["notes"] = res["notes"]
    return out


def _contrast_head(c):
    a, b = _mf(c["model_a"]), _mf(c["model_b"])
    d = {"model_a": a["model"]}
    if "fit" in a:
        d["fit_a"] = a["fit"]
    d["model_b"] = b["model"]
    if "fit" in b:
        d["fit_b"] = b["fit"]
    return d


def _gate(g):
    if "N" not in g:
        return dict(g)
    return dict(kind=g["kind"], termini=[dict(terminus=T, **g[T]) for T in "NC"], passed=g["passed"], evaluable=g["evaluable"])


def _csv_rows_study(res):
    for s in res["studies"]:
        for m, st in res["per_study"][s].items():
            for a, q in METRICS:
                yield (s, s in res["seen"], m, a, q, st[a][q])


def write_outputs(res, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "representation_comparison_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary_json(res), fh, indent=1, default=float)
        fh.write("\n")
    DR.write_per_study_csv(os.path.join(out_dir, "representation_comparison_per_study.csv"), _csv_rows_study(res))
    with open(os.path.join(out_dir, "representation_comparison_bootstrap.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["study", "contrast", "terminus", "variant", "metric", "ci_low", "ci_high"])
        for s in res["studies"]:
            for name, qs in res["boot"].get(s, {}).items():
                for (a, q), (lo, hi) in qs.items():
                    w.writerow([s, name, a if a in "NC" else "", a if a not in "NC" else "", q, lo, hi])
    with open(os.path.join(out_dir, "representation_comparison_report.md"), "w", encoding="utf-8") as fh:
        fh.write(render_report(res))


# ------------------------------------------------------------------------------------------------------------------------------------ the command
def load_records(rdir, studies):
    recs = {}
    for s in studies:
        p1, p2 = (os.path.join(rdir, f"study_{s}_{part}.json") for part in ("window_mlp_and_frozen", "lora"))
        recs[s] = (DR.load_record(p1) if os.path.exists(p1) else None, DR.load_record(p2) if os.path.exists(p2) else None)
    return recs


def run(records_dir, out_dir, cfg):
    studies = list(cfg["studies"])
    seen = [s for s in cfg["seen"] if s in studies]
    unseen = [s for s in studies if s not in seen]
    subset = list(cfg["lora_subset"])
    EV = cfg["levels"]
    assert len(unseen) == EV["n_unseen"] and all(s in unseen for s in subset), "the configuration does not hold 19 unseen studies and the subset of the specification"
    recs = load_records(records_dir, studies)
    have_lora = sorted(s for s, (a, b) in recs.items() if b is not None)
    extended = sorted(have_lora) == sorted(studies)
    lora_studies = studies if extended else subset
    stray = sorted(f for f in os.listdir(records_dir) if f.startswith("study_") and f.endswith(".json") and f.split("_", 2)[1] not in set(studies))
    fatal, notes = check_runs(recs, studies, lora_studies, cfg.get("software"), subset, tol=EV["tolerance"], seen=seen, floor_study=EV["floor_study"])
    if stray:
        fatal.append(f"the records folder holds records of studies that are not in the configuration: {stray}")
    if fatal:
        sys.exit("AGGREGATION STOPPED: the records are inconsistent:\n  " + "\n  ".join(fatal))
    merged = {s: merged_view(*recs[s]) for s in studies}
    res = aggregate(merged, studies, unseen, seen, lora_studies, extended, cfg)
    res["notes"] = notes
    write_outputs(res, out_dir)
    print("wrote representation_comparison_summary.json, _per_study.csv, _bootstrap.csv and _report.md")
    return res


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(prog="44_development_benchmark.py representation-comparison", description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", required=True, help="folder with the stored per-study records (study_<pmid>_window_mlp_and_frozen.json, study_<pmid>_lora.json)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--config", default=os.path.join(here, "..", "..", "configs", "representation_comparison.json"))
    a = ap.parse_args(argv)
    for f in ("representation_comparison_summary.json", "representation_comparison_per_study.csv", "representation_comparison_bootstrap.csv", "representation_comparison_report.md"):
        if os.path.exists(os.path.join(a.out, f)):
            sys.exit(f"refusing to overwrite {os.path.join(a.out, f)}")
    run(a.records, a.out, json.load(open(a.config)))
