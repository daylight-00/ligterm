#!/usr/bin/env python3
"""Comparison with the published tool MHCII-NP: LigTerm-1S, LigTerm-GB and the Paul-style motif against MHCII-NP on 19 development studies, from stored per-protein values.

The comparison was pre-specified: its rules (the population, the pairing of scores, the reading) were written before any score of the tool for a benchmark protein existed, on an exposed development subset;
it is pre-specified, not confirmatory. Population: peptide lengths 13 to 23. Pairing of the scores of a LigTerm model with the raw score of the tool: Peptide-AUROC (all candidates) with the combined
score of the model, length-matched AUC (each observed peptide against the candidates of its own length) with the bond scores, near-miss AUC (start and end each shifted by at most 3) with the combined score,
near-miss AUC of the same length with the bond scores; and a length-free pair of the bond scores against the score of the tool divided by its length factor. Per study: Peptide-AUROC and the length-matched AUC
are macro means over proteins, the near-miss values pooled pairs; ties count 0.5. Uncertainty: the PROTEIN is the unit; a per-study bootstrap over proteins (2,000 resamples; random stream [20261120, 14, study
index]) and a two-stage bootstrap of the median over studies. Reading (practical, not significance): 'exceeds' = the median over the 19 studies of the per-study paired difference >= +0.01 and positive in
at least 13 of 19; 'reversed' = <= -0.01 and negative in at least 13; otherwise 'no consistent difference'. Headline: LigTerm-1S exceeds the tool iff Peptide-AUROC and the length-matched AUC are both 'exceeds'.

--supplement: the post hoc supplement on the same proteins. LigTerm (the release ensemble) against the tool and against the other models, and retrieval measures: the share of the observed peptides among the
top 10 and top 50 candidates of a protein, and the tolerance hit (a top candidate with start and end each within 3 residues of an observed peptide); these measures have no reading rule (descriptive); the expected
recall of a random ranking is given per study and by protein length. It never replaces the pre-specified comparison.

The raw responses of the web service are not redistributed, so this script starts from the stored per-protein values (written with 10 significant digits); the results agree with the stored ones to that
precision. usage: 45_mhcii_np_comparison.py --per-protein <per_protein.csv> --manifest <manifest.tsv> --population <population.tsv> --out <folder> [--supplement]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from ligterm_repro.np_comparison import study_value  # noqa: E402

B_BOOT, SEEDB = 2000, [20261120, 14]
ORIG = ["Paul-style motif", "LigTerm-GB", "LigTerm-1S"]
TOOL, TOOL_ADJ = "MHCII-NP", "MHCII-NP (length-adjusted)"
METR = ("peptide_auroc", "length_matched_auc", "near_miss_auc", "near_miss_auc_same_length")
METRN = ("peptide_auroc_length_free", "near_miss_auc_length_free")
RET = ("recall_at_10", "recall_at_50", "tolerance_hit_at_10", "tolerance_hit_at_50")
GATE_MED, GATE_MIN = 0.01, 13
CON_ORIG = [("LigTerm-1S", TOOL), ("LigTerm-GB", TOOL), ("Paul-style motif", TOOL), ("LigTerm-1S", "LigTerm-GB"), ("LigTerm-1S", "Paul-style motif"), ("LigTerm-GB", "Paul-style motif")]
CON_N_ORIG = [(m, TOOL_ADJ) for m in ORIG]
CON_NEW = [("LigTerm", TOOL), ("LigTerm", "LigTerm-1S"), ("LigTerm", "LigTerm-GB"), ("LigTerm", "Paul-style motif")]
CON_N_NEW = [("LigTerm", TOOL_ADJ)]
KEYS = ("peptide_auroc", "length_matched_num", "length_matched_den", "near_miss_num", "near_miss_den", "near_miss_same_length_num", "near_miss_same_length_den",
        "peptide_auroc_length_free", "near_miss_length_free_num", "near_miss_length_free_den")
KEYS_SUPP = KEYS + RET + ("n_candidates",)
HEADLINE = {"exceeds_in_both": "LigTerm-1S exceeds the published tool in Peptide-AUROC and in Length-matched AUC",
            "tool_exceeds": "the published tool exceeds LigTerm-1S in Peptide-AUROC or in Length-matched AUC",
            "exceeds_in_one": "LigTerm-1S exceeds the published tool in one of Peptide-AUROC and Length-matched AUC only",
            "no_consistent_difference": "no consistent difference between LigTerm-1S and the published tool in Peptide-AUROC and Length-matched AUC"}
LENGTH_CLASSES = (("L<=300", 0, 300), ("301-600", 301, 600), (">600", 601, 10 ** 9))


def reading(med, npos, nneg):
    return "exceeds" if (med >= GATE_MED and npos >= GATE_MIN) else ("reversed" if (med <= -GATE_MED and nneg >= GATE_MIN) else "no consistent difference")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--per-protein", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--supplement", action="store_true", help="the post hoc supplement (needs the per-protein table with the retrieval measures and the release ensemble)")
    a = ap.parse_args()
    SUPP = a.supplement
    MODELS = ORIG + (["LigTerm"] if SUPP else [])
    KEYSX = KEYS_SUPP if SUPP else KEYS
    os.makedirs(a.out, exist_ok=True)
    man = list(csv.DictReader(open(a.manifest), delimiter="\t"))
    studies = sorted({r["study"] for r in man}, key=lambda s: int([r for r in man if r["study"] == s][0]["study_index"]))
    assert len(studies) == 19
    pop = {r["study"]: r for r in csv.DictReader(open(a.population), delimiter="\t")}
    table = list(csv.DictReader(open(a.per_protein)))
    allm = MODELS + [TOOL, TOOL_ADJ]
    NEEDED = {m: ((METR + METRN + RET) if m in MODELS else (METR + RET) if m == TOOL else METRN) if SUPP else ((METR + METRN) if m in MODELS else METR if m == TOOL else METRN) for m in allm}
    # ---- the proteins of every study, in the order of the manifest; the proteins of the first K positions that are not in the table were dropped for all models
    bypos = {s: {int(r["pos"]): r for r in man if r["study"] == s} for s in studies}
    rows_of = {s: [r for r in table if r["study"] == s] for s in studies}
    have = {s: {r["acc"] for r in rows_of[s]} for s in studies}
    K = max(int(r["pos"]) for r in man if r["acc"] in have[r["study"]])
    PER = {s: {m: {k: [] for k in KEYSX} for m in allm} for s in studies}
    meta = {s: [] for s in studies}
    DROPS = []
    for s in studies:
        order = {bypos[s][pos]["acc"]: pos for pos in range(1, K + 1)}
        by_acc = {r["acc"]: r for r in rows_of[s]}
        assert set(by_acc) <= set(order), f"{s}: proteins of the table are not among the first {K} of the manifest"
        for pos in range(1, K + 1):
            acc = bypos[s][pos]["acc"]
            if acc not in by_acc:
                DROPS.append(dict(study=s, pos=pos, acc=acc, L=int(bypos[s][pos]["L"]), n_pos=int(bypos[s][pos]["n_pos"]), windows=int(bypos[s][pos]["windows"])))
                continue
            r = by_acc[acc]
            for m in allm:
                for k in KEYSX:
                    PER[s][m][k].append(float(r[f"{m}:{k}"]))
            meta[s].append(dict(acc=acc, L=int(r["L"]), n_pos=int(r["n_pos"]), windows=int(r["windows"])))
        for m in allm:
            PER[s][m] = {k: np.array(v, np.float64) for k, v in PER[s][m].items()}
    print(f"studies {len(studies)}; proteins per study in the analysis: {K} (minus the dropped ones); {sum(len(v) for v in meta.values())} proteins, {len(DROPS)} dropped for all models")
    if SUPP:
        for s in studies:                                                       # the candidate sets of the models are the same
            for m in MODELS + [TOOL]:
                assert np.array_equal(PER[s][m]["n_candidates"], PER[s]["LigTerm-1S"]["n_candidates"]), f"{s}: the candidate sets of the models differ"
    # ---- per-study values, paired differences, protein bootstrap
    rngs = {s: np.random.default_rng(SEEDB + [i]) for i, s in enumerate(studies)}
    val, rep, diff, rnd = {}, {}, {}, {}
    allcon = [(x, y, METR) for x, y in CON_ORIG] + [(x, y, METRN) for x, y in CON_N_ORIG]
    if SUPP:
        allcon += [(x, y, METR + RET) for x, y in CON_NEW] + [(x, y, METRN) for x, y in CON_N_NEW]
    for s in studies:
        n = len(meta[s])
        idx_b = rngs[s].integers(0, n, size=(B_BOOT, n))
        full = np.arange(n)
        for m in allm:
            for mt in NEEDED[m]:
                val[(s, m, mt)] = float(study_value(PER[s][m], full, mt))
                rep[(s, m, mt)] = study_value(PER[s][m], idx_b, mt)
        if SUPP:
            nc_ = PER[s]["LigTerm-1S"]["n_candidates"]
            for kk in (10, 50):
                rnd[(s, kk)] = float(np.mean(np.minimum(1.0, kk / nc_)))
        for x, y, mts in allcon:
            for mt in mts:
                diff[(s, x, y, mt)] = val[(s, x, mt)] - val[(s, y, mt)]
                rep[(s, x, y, mt)] = rep[(s, x, mt)] - rep[(s, y, mt)]
    rng2 = np.random.default_rng(SEEDB + [999])
    out = dict(K=K, studies=studies, dropped=DROPS, per_study={}, contrasts={})
    for x, y, mts in allcon:
        for mt in mts:
            d = np.array([diff[(s, x, y, mt)] for s in studies])
            R = np.array([rep[(s, x, y, mt)] for s in studies])
            stat = np.array([np.median(R[rng2.integers(0, 19, 19), rng2.integers(0, B_BOOT, 19)]) for _ in range(B_BOOT)])
            lo, hi = float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))
            npos, nneg = int((d > 0).sum()), int((d < 0).sum())
            med = float(np.median(d))
            out["contrasts"].setdefault(f"{x} - {y}", {})[mt] = dict(median=med, n_positive=npos, n_negative=nneg, ci_lo=lo, ci_hi=hi, half_width=(hi - lo) / 2, min=float(d.min()), max=float(d.max()),
                                                                     reading=(reading(med, npos, nneg) if mt not in RET else "descriptive (no gate)"))
    if SUPP:
        for s in studies:
            n = len(meta[s])
            out["per_study"][s] = dict(n_proteins=n, n_positives=int(sum(x["n_pos"] for x in meta[s])), values={m: {mt: val[(s, m, mt)] for mt in NEEDED[m]} for m in allm},
                                       random_expectation={"random_recall_at_10": rnd[(s, 10)], "random_recall_at_50": rnd[(s, 50)]})
        ln = {}
        for cls, lo_, hi_ in LENGTH_CLASSES:
            sel = {s: np.array([lo_ <= x["L"] <= hi_ for x in meta[s]]) for s in studies}
            cell = {"items": int(sum(sel[s].sum() for s in studies))}
            for m in MODELS + [TOOL]:
                cell[m] = {mt: float(np.nanmean(np.concatenate([PER[s][m][mt][sel[s]] for s in studies]))) for mt in RET}
            cell["random_exact_recall"] = {f"random_recall_at_{kk}": float(np.mean(np.concatenate([np.minimum(1.0, kk / PER[s]["LigTerm-1S"]["n_candidates"][sel[s]]) for s in studies]))) for kk in (10, 50)}
            ln[cls] = cell
        out["by_protein_length_pooled_items"] = ln
        cF, cM = out["contrasts"][f"LigTerm - {TOOL}"]["peptide_auroc"]["reading"], out["contrasts"][f"LigTerm - {TOOL}"]["length_matched_auc"]["reading"]
        verdict = "exceeds" if (cF == "exceeds" and cM == "exceeds") else ("is exceeded by" if (cF == "reversed" and cM == "reversed") else "does not exceed")
        out["sentence"] = f"in a post hoc supplement on the same sample, LigTerm {verdict} the published tool in Peptide-AUROC and Length-matched AUC by the pre-specified rule"
        json.dump(out, open(os.path.join(a.out, "supplement.json"), "w"), indent=1)
        lines = [f"POST HOC SUPPLEMENT: {len(studies)} development studies, {K} proteins per study, peptide lengths 13 to 23, the same {sum(len(v) for v in meta.values())} items; dropped for all models: {len(DROPS)}",
                 "medians over the studies of the per-study values:"]
        for m in allm:
            lines.append(f"  {m:28s} " + "  ".join(f"{mt} {np.median([val[(s, m, mt)] for s in studies]):.4f}" for mt in NEEDED[m]))
        lines.append("  random expectation of the exact recall: " + "  ".join(f"random_recall_at_{kk} {float(np.median([rnd[(s, kk)] for s in studies])):.4f}" for kk in (10, 50)))
        lines.append("paired differences (median over the studies; studies with a positive / negative difference; 95% interval of the median by the two-stage protein bootstrap; the retrieval measures have no reading rule):")
        for x, y, mts in allcon:
            c = out["contrasts"][f"{x} - {y}"]
            lines.append(f"  {x} - {y}: " + " | ".join(f"{mt} {c[mt]['median']:+.4f} ({c[mt]['n_positive']}+/{c[mt]['n_negative']}-) [{c[mt]['ci_lo']:+.4f}, {c[mt]['ci_hi']:+.4f}] {c[mt]['reading'] if mt not in RET else 'descriptive'}" for mt in mts))
        lines.append("SENTENCE: " + out["sentence"])
        open(os.path.join(a.out, "supplement.txt"), "w").write("\n".join(lines) + "\n")
        print("\n".join(lines))
        return
    primary = "LigTerm-1S - " + TOOL
    perstudy_hw = float(np.median([(np.percentile(rep[(s, "LigTerm-1S", TOOL, "peptide_auroc")], 97.5) - np.percentile(rep[(s, "LigTerm-1S", TOOL, "peptide_auroc")], 2.5)) / 2 for s in studies]))
    hwF = out["contrasts"][primary]["peptide_auroc"]["half_width"]
    for s in studies:
        pp, n = pop[s], len(meta[s])
        elp, elpos, ex1000, p1323 = int(pp["eligible_proteins"]), int(pp["eligible_positives"]), int(pp["excluded_over_1000_positives"]), int(pp["positives_13_23"])
        out["per_study"][s] = dict(n_proteins=n, n_positives=int(sum(x["n_pos"] for x in meta[s])), windows=int(sum(x["windows"] for x in meta[s])), eligible_proteins=elp, eligible_positives=elpos,
                                   share_of_eligible_proteins=n / elp, share_of_eligible_positives=sum(x["n_pos"] for x in meta[s]) / elpos, excluded_over_1000_positive_share=ex1000 / max(1, p1323),
                                   values={m: {mt: val[(s, m, mt)] for mt in NEEDED[m]} for m in allm},
                                   differences={f"{x} - {y}": {mt: dict(value=diff[(s, x, y, mt)], lo=float(np.percentile(rep[(s, x, y, mt)], 2.5)), hi=float(np.percentile(rep[(s, x, y, mt)], 97.5))) for mt in mts} for x, y, mts in allcon})
    prim = out["contrasts"][primary]
    f, mm = prim["peptide_auroc"]["reading"], prim["length_matched_auc"]["reading"]
    code = ("exceeds_in_both" if (f == "exceeds" and mm == "exceeds") else "tool_exceeds" if "reversed" in (f, mm) else "exceeds_in_one" if "exceeds" in (f, mm) else "no_consistent_difference")
    out["headline"], out["headline_text"] = code, HEADLINE[code]
    out["precision"] = dict(primary_contrast=f"{primary}, peptide_auroc", criterion_half_width=0.03, half_width_across_study_median=hwF, per_study_half_width_median=perstudy_hw)
    json.dump(out, open(os.path.join(a.out, "comparison.json"), "w"), indent=1)
    lines = [f"Comparison with {TOOL}: {len(studies)} development studies, {K} proteins per study (first {K} of the sample order), peptide lengths 13 to 23",
             f"proteins analysed: {sum(len(v) for v in meta.values())}; dropped for all models: {len(DROPS)}", "medians over the studies of the per-study values:"]
    for m in allm:
        lines.append(f"  {m:28s} " + "  ".join(f"{mt} {np.median([val[(s, m, mt)] for s in studies]):.4f}" for mt in NEEDED[m]))
    lines.append("paired differences (median over the studies; studies with a positive difference; 95% interval of the median by the two-stage protein bootstrap; reading):")
    for x, y, mts in allcon:
        c = out["contrasts"][f"{x} - {y}"]
        lines.append(f"  {x} - {y}: " + " | ".join(f"{mt} {c[mt]['median']:+.4f} ({c[mt]['n_positive']}/19) [{c[mt]['ci_lo']:+.4f}, {c[mt]['ci_hi']:+.4f}] {c[mt]['reading']}" for mt in mts))
    lines.append(f"HEADLINE (pre-specified rule: both 'exceeds'): {out['headline_text']}")
    lines.append(f"precision: half-width of the across-study interval of the primary contrast {hwF:.4f} (criterion 0.03); median per-study half-width {perstudy_hw:.4f}")
    open(os.path.join(a.out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
