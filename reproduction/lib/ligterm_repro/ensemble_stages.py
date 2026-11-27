"""The two ensemble stages of the predictor development: aggregation of their stored per-study records.

Ensemble score scaling (20 records): LigTerm-1S = (LigTerm-GB logit + Window MLP logit) / 2 with a fixed scale of 1 at both termini, recombined from the stored per-bond logits of one development study. Per-study values, the medians over the
19 unseen development studies of the paired differences and the number of studies with a positive difference, and the reading of the pre-specified rule applied mechanically. No bootstrap and no other contrast is formed: the contrasts are the
three that the rule names (LigTerm-1S against LigTerm-GB; the Scale-matched ensemble against LigTerm-GB as the reproduction of the earlier record; LigTerm-1S against the Scale-matched ensemble to show what the fixed scale changes), plus
Window MLP against LigTerm-GB for orientation.
Ensemble seed robustness (19 records): LigTerm-GB; the Scale-matched ensemble (as a guard); LigTerm-1S with the stored seed of the window MLP (seed 0) and with two new seeds (1 and 2); LigTerm = (LigTerm-GB + the mean of the three window-MLP logits) / 2
(the release recipe). The pre-specified reading: LigTerm is supported if (a) at each terminus the median of the per-study Bond-AUROC differences to LigTerm-GB is >= +0.01 and positive in >= 13 of 19 studies, and (b) for each of the
Length-matched AUC and the Near-miss AUC of both_bonds the median is > 0 and positive in >= 13 of 19; the result is seed-robust if, in addition, for each of the three single-seed LigTerm-1S the median over the 19 studies of the Bond-AUROC
difference to LigTerm-GB is positive at both termini and positive in >= 13 of 19 studies at both termini. The reading states whether LigTerm is supported and whether it is seed-robust. The spread of the
three medians over the seeds is reported; the control (the stored seed refitted on the GPU, two studies) has no rule. Development evidence on exposed data, not a test.
The stored records are the outputs of the runs of these stages (the GPU fits are never repeated).
"""
import argparse
import json
import os
import sys

import numpy as np

from . import development_records as DR
from .development_records import mkey_of, printed

K = mkey_of
LIGTERM_GB, WINDOW_MLP, ENSEMBLE_SCALE = K("LigTerm-GB"), K("Window MLP"), K("Scale-matched ensemble")
LIGTERM_1S = K("LigTerm-1S")
SEED_MODELS = [K("LigTerm-1S", seed=0), K("LigTerm-1S", seed=1), K("LigTerm-1S", seed=2)]
LIGTERM = K("LigTerm")
BOTH, BOTH_L = "both_bonds", "both_bonds_and_length"
VARIANTS = (BOTH, BOTH_L)
METRICS = (("N", "bond_auroc"), ("C", "bond_auroc"), ("N", "local_auc"), ("C", "local_auc"), (BOTH, "peptide_auroc"), (BOTH, "length_matched_auc"), (BOTH, "near_miss_auc"),
           (BOTH_L, "peptide_auroc"), (BOTH_L, "length_matched_auc"), (BOTH_L, "near_miss_auc"))
SCALING_MODELS = [LIGTERM_GB, WINDOW_MLP, ENSEMBLE_SCALE, LIGTERM_1S]
SCALING_CONTRASTS = ((LIGTERM_1S, LIGTERM_GB), (ENSEMBLE_SCALE, LIGTERM_GB), (LIGTERM_1S, ENSEMBLE_SCALE), (WINDOW_MLP, LIGTERM_GB))
SEEDS_MODELS = [LIGTERM_GB, ENSEMBLE_SCALE] + SEED_MODELS + [LIGTERM]
SEEDS_CONTRASTS = ((LIGTERM, LIGTERM_GB), (SEED_MODELS[0], LIGTERM_GB), (SEED_MODELS[1], LIGTERM_GB), (SEED_MODELS[2], LIGTERM_GB), (LIGTERM, SEED_MODELS[0]), (ENSEMBLE_SCALE, LIGTERM_GB))
STAGE_SCALING, STAGE_SEEDS = "ensemble_scaling", "ensemble_seed_robustness"
RULE_TEXT = {STAGE_SCALING: "pre-specified rule of the ensemble score scaling (fixed scale 1)", STAGE_SEEDS: "pre-specified rule of the ensemble seed robustness"}
CRITERION = {
    "a": "Bond-AUROC, median of the per-study differences >= +0.01 and positive in >= 13 of 19 studies, at the N and at the C terminus",
    "b": "Length-matched AUC and Near-miss AUC of the both_bonds score, median of the per-study differences > 0 and positive in >= 13 of 19 studies, each",
    "a_seeds": "LigTerm minus LigTerm-GB: Bond-AUROC, median of the per-study differences >= +0.01 and positive in >= 13 of 19 studies, at the N and at the C terminus",
    "b_seeds": "LigTerm minus LigTerm-GB: Length-matched AUC and Near-miss AUC of the both_bonds score, median of the per-study differences > 0 and positive in >= 13 of 19 studies, each",
    "seed": "each single-seed LigTerm-1S minus LigTerm-GB: Bond-AUROC median > 0 and positive in >= 13 of 19 studies, at the N and at the C terminus",
}
CONCLUSION = {True: "LigTerm is supported and seed-robust under this rule", False: "LigTerm is not both supported and seed-robust under this rule"}


def load_stage_records(records_dir, studies):
    recs = {}
    for s in studies:
        recs[s] = DR.load_record(os.path.join(records_dir, f"study_{s}.json"))
    extra = sorted(f for f in os.listdir(records_dir) if f.startswith("study_") and f.endswith(".json") and f[len("study_"):-len(".json")] not in set(studies))
    assert not extra, f"the records folder holds records of studies that are not in the configuration: {extra}"
    return recs


def tables(recs, models, contrasts, unseen):
    ST = {s: DR.study_stats(recs[s], models, VARIANTS) for s in recs}
    per_study = {s: {m: {(t, k): ST[s][m][t][k] for t, k in METRICS} for m in models} for s in sorted(recs)}
    medians = {m: {(t, k): DR.med([ST[s][m][t][k] for s in unseen]) for t, k in METRICS} for m in models}
    cons = {}
    for a, b in contrasts:
        c = {}
        for t, k in METRICS:
            d = np.array([ST[s][a][t][k] - ST[s][b][t][k] for s in unseen], np.float64)
            c[(t, k)] = dict(median=DR.med(d), n_positive=int((d > 0).sum()), n_defined=int(np.isfinite(d).sum()), min=float(np.nanmin(d)), max=float(np.nanmax(d)))
        cons[(a, b)] = c
    return ST, per_study, medians, cons


def reading_a_b(c, gate_med, gate_min):
    a_ok = {T: bool(c[(T, "bond_auroc")]["median"] >= gate_med and c[(T, "bond_auroc")]["n_positive"] >= gate_min) for T in "NC"}
    b_ok = {q: bool(c[(BOTH, q)]["median"] > 0 and c[(BOTH, q)]["n_positive"] >= gate_min) for q in ("length_matched_auc", "near_miss_auc")}
    return a_ok, b_ok


# ------------------------------------------------------------------------------------------------------------------------------ output helpers
def _contrast_rows(cons):
    rows = []
    for (a, b), c in cons.items():
        fa, fb = DR.mrow(a), DR.mrow(b)
        head = {"model_a": fa["model"], **({"seed_a": fa["seed"]} if "seed" in fa else {}), "model_b": fb["model"], **({"seed_b": fb["seed"]} if "seed" in fb else {})}
        for (t, k), v in c.items():
            rows.append(dict(head, **DR.level_fields(t), metric=k, **v))
    return rows


def _model_rows(per_study, models, studies):
    return [dict(study=s, **DR.mrow(m), **DR.level_fields(t), metric=k, value=per_study[s][m][(t, k)]) for s in studies for m in models for t, k in METRICS]


def _median_rows(medians, models):
    return [dict(**DR.mrow(m), **DR.level_fields(t), metric=k, median=medians[m][(t, k)]) for m in models for t, k in METRICS]


def _model_line(m, med_, w):
    return (f"  {printed(m):{w}s} Bond-AUROC N {med_['N', 'bond_auroc']:.4f}  C {med_['C', 'bond_auroc']:.4f}   "
            f"Length-matched AUC {med_[BOTH, 'length_matched_auc']:.4f}  Near-miss AUC {med_[BOTH, 'near_miss_auc']:.4f}")


def _contrast_line(a, b, cc):
    return (f"  {printed(a)} - {printed(b)}: Bond-AUROC N {cc['N', 'bond_auroc']['median']:+.4f} ({cc['N', 'bond_auroc']['n_positive']}) C {cc['C', 'bond_auroc']['median']:+.4f} ({cc['C', 'bond_auroc']['n_positive']}) | "
            f"Local-AUC N {cc['N', 'local_auc']['median']:+.4f} C {cc['C', 'local_auc']['median']:+.4f} | "
            f"Peptide-AUROC {cc[BOTH, 'peptide_auroc']['median']:+.4f} ({cc[BOTH, 'peptide_auroc']['n_positive']}) Length-matched AUC {cc[BOTH, 'length_matched_auc']['median']:+.4f} ({cc[BOTH, 'length_matched_auc']['n_positive']}) "
            f"Near-miss AUC {cc[BOTH, 'near_miss_auc']['median']:+.4f} ({cc[BOTH, 'near_miss_auc']['n_positive']}) | Peptide-AUROC (+length) {cc[BOTH_L, 'peptide_auroc']['median']:+.4f} ({cc[BOTH_L, 'peptide_auroc']['n_positive']})")


def write_outputs(out_dir, stem, summary, csv_rows, text):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{stem}_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, default=float)
        fh.write("\n")
    DR.write_per_study_csv(os.path.join(out_dir, f"{stem}_per_study.csv"), csv_rows, ".6f")
    with open(os.path.join(out_dir, f"{stem}_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)


# --------------------------------------------------------------------------------------------------------------------------- ensemble score scaling
def compute_scaling(recs, cfg):
    gate_med, gate_min, n_unseen = cfg["gate_median"], cfg["gate_min"], cfg["n_unseen"]
    assert len(recs) == len(cfg["studies"]), f"expected {len(cfg['studies'])} records, found {len(recs)}"
    unseen = sorted(s for s, r in recs.items() if not r["seen"])
    assert len(unseen) == n_unseen, f"{len(unseen)} unseen studies, expected {n_unseen}"
    seen = [s for s in sorted(recs) if s not in unseen]
    ST, per_study, medians, cons = tables(recs, SCALING_MODELS, SCALING_CONTRASTS, unseen)
    c = cons[(LIGTERM_1S, LIGTERM_GB)]
    a_ok, b_ok = reading_a_b(c, gate_med, gate_min)
    reading = dict(rule=RULE_TEXT[STAGE_SCALING], criterion_a=CRITERION["a"], criterion_b=CRITERION["b"], a=a_ok, b=b_ok, a_met=all(a_ok.values()), b_met=all(b_ok.values()),
                   keeps_the_improvement=bool(all(a_ok.values()) and all(b_ok.values())))
    return dict(studies=sorted(recs), seen_flags={s: bool(recs[s]["seen"]) for s in recs}, unseen=unseen, seen=seen, per_study=per_study, medians=medians, contrasts=cons, reading=reading)


def outputs_scaling(I):
    """(summary JSON object, per-study table rows, text report) of the ensemble score scaling from its computed (or converted stored) result I"""
    reading, studies, unseen = I["reading"], I["studies"], I["unseen"]
    a_ok, b_ok = reading["a"], reading["b"]
    summary = dict(unseen=unseen, seen=I["seen"], per_study=_model_rows(I["per_study"], SCALING_MODELS, studies), medians=_median_rows(I["medians"], SCALING_MODELS), contrasts=_contrast_rows(I["contrasts"]),
                   reading=dict(rule=reading["rule"], criterion_a=reading["criterion_a"], criterion_b=reading["criterion_b"], a=[dict(terminus=T, met=v) for T, v in a_ok.items()],
                                b=[dict(metric=q, met=v) for q, v in b_ok.items()], a_met=reading["a_met"], b_met=reading["b_met"], keeps_the_improvement=reading["keeps_the_improvement"]))
    csv_rows = [(s, I["seen_flags"][s], m, t, k, I["per_study"][s][m][(t, k)]) for s in studies for m in SCALING_MODELS for t, k in METRICS]
    lines = [f"Ensemble score scaling: {len(unseen)} unseen development studies (+{len(studies) - len(unseen)} seen study listed in the per-study table); medians of per-study values:"]
    for m in SCALING_MODELS:
        lines.append(_model_line(m, I["medians"][m], 24))
    lines.append("paired differences (median over the unseen studies; studies with a positive difference):")
    for a, b in SCALING_CONTRASTS:
        lines.append(_contrast_line(a, b, I["contrasts"][(a, b)]))
    lines.append(f"READING (mechanical): (a) Bond-AUROC gate N {a_ok['N']}, C {a_ok['C']} -> {reading['a_met']}; (b) Length-matched AUC {b_ok['length_matched_auc']}, Near-miss AUC {b_ok['near_miss_auc']} -> {reading['b_met']}; "
                 f"LigTerm-1S keeps the improvement: {reading['keeps_the_improvement']}")
    return summary, csv_rows, "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------------------------------- ensemble seed robustness
def compute_seed_robustness(recs, cfg):
    gate_med, gate_min, n_unseen = cfg["gate_median"], cfg["gate_min"], cfg["n_unseen"]
    assert len(recs) == n_unseen, f"expected {n_unseen} records, found {len(recs)}"
    assert all(not r["seen"] and r["stage"] == STAGE_SEEDS and r["models"] == SEEDS_MODELS for r in recs.values()), "a record is not an unseen-study record of this stage with the expected model list"
    unseen = sorted(recs)
    ST, per_study, medians, cons = tables(recs, SEEDS_MODELS, SEEDS_CONTRASTS, unseen)
    c = cons[(LIGTERM, LIGTERM_GB)]
    a_ok, b_ok = reading_a_b(c, gate_med, gate_min)
    supported = bool(all(a_ok.values()) and all(b_ok.values()))
    seed_ok = {}
    for m in SEED_MODELS:
        cm = cons[(m, LIGTERM_GB)]
        seed_ok[m] = {T: bool(cm[(T, "bond_auroc")]["median"] > 0 and cm[(T, "bond_auroc")]["n_positive"] >= gate_min) for T in "NC"}
    robust = bool(all(all(v.values()) for v in seed_ok.values()))
    spread = {T: dict(medians={m: cons[(m, LIGTERM_GB)][(T, "bond_auroc")]["median"] for m in SEED_MODELS},
                      range=float(max(cons[(m, LIGTERM_GB)][(T, "bond_auroc")]["median"] for m in SEED_MODELS) - min(cons[(m, LIGTERM_GB)][(T, "bond_auroc")]["median"] for m in SEED_MODELS))) for T in "NC"}
    both_met = bool(supported and robust)
    sd = {"N": [], "C": []}
    for s in unseen:
        for k in ("1", "2"):
            for T in "NC":
                sd[T].append(dict(study=s, seed_index=int(k), e_star=recs[s]["seeds"][k][T]["e_star"], at_cap=recs[s]["seeds"][k][T]["at_cap"]))
    fits = {T: dict(n_fits=len(v), n_at_cap=int(sum(x["at_cap"] for x in v)), e_star_median=float(np.median([x["e_star"] for x in v])), e_star_min=int(min(x["e_star"] for x in v)),
                    e_star_max=int(max(x["e_star"] for x in v))) for T, v in sd.items()}
    control = {s: recs[s]["control"] for s in unseen if recs[s].get("control")}
    reading = dict(rule=RULE_TEXT[STAGE_SEEDS], criterion_a=CRITERION["a_seeds"], criterion_b=CRITERION["b_seeds"], criterion_seed=CRITERION["seed"], a=a_ok, b=b_ok, a_met=all(a_ok.values()), b_met=all(b_ok.values()),
                   ligterm_supported=supported, seed=seed_ok, seed_robust=robust, spread_of_seed_medians=spread, supported_and_seed_robust=both_met, conclusion=CONCLUSION[both_met])
    return dict(studies=unseen, unseen=unseen, per_study=per_study, medians=medians, contrasts=cons, reading=reading, new_seed_fits=fits, control=control)


def outputs_seed_robustness(I):
    reading, unseen = I["reading"], I["unseen"]
    a_ok, b_ok, seed_ok, spread, fits, control = reading["a"], reading["b"], reading["seed"], reading["spread_of_seed_medians"], I["new_seed_fits"], I["control"]
    summary = dict(unseen=unseen, per_study=_model_rows(I["per_study"], SEEDS_MODELS, unseen), medians=_median_rows(I["medians"], SEEDS_MODELS), contrasts=_contrast_rows(I["contrasts"]),
                   reading=dict(rule=reading["rule"], criterion_a=reading["criterion_a"], criterion_b=reading["criterion_b"], criterion_seed=reading["criterion_seed"],
                                a=[dict(terminus=T, met=v) for T, v in a_ok.items()], b=[dict(metric=q, met=v) for q, v in b_ok.items()], a_met=reading["a_met"], b_met=reading["b_met"],
                                ligterm_supported=reading["ligterm_supported"], seed=[dict(**DR.mrow(m), terminus=T, met=v) for m, d in seed_ok.items() for T, v in d.items()], seed_robust=reading["seed_robust"],
                                spread_of_seed_medians=[dict(terminus=T, medians=[dict(**DR.mrow(m), median=x) for m, x in d["medians"].items()], range=d["range"]) for T, d in spread.items()],
                                supported_and_seed_robust=reading["supported_and_seed_robust"], conclusion=reading["conclusion"]),
                   new_seed_fits=[dict(terminus=T, **d) for T, d in fits.items()], control=[dict(study=s, terminus=T, **v) for s, cc in control.items() for T, v in cc.items()])
    csv_rows = [(s, False, m, t, k, I["per_study"][s][m][(t, k)]) for s in unseen for m in SEEDS_MODELS for t, k in METRICS]
    lines = [f"Ensemble seed robustness: {len(unseen)} unseen development studies; medians of per-study values:"]
    for m in SEEDS_MODELS:
        lines.append(_model_line(m, I["medians"][m], 24))
    lines.append("paired differences (median over the unseen studies; studies with a positive difference of 19):")
    for a, b in SEEDS_CONTRASTS:
        lines.append(_contrast_line(a, b, I["contrasts"][(a, b)]))
    lines.append(f"READING (mechanical): LigTerm (a) Bond-AUROC gate N {a_ok['N']}, C {a_ok['C']} -> {reading['a_met']}; (b) Length-matched AUC {b_ok['length_matched_auc']}, Near-miss AUC {b_ok['near_miss_auc']} -> {reading['b_met']}; "
                 f"LigTerm supported: {reading['ligterm_supported']}")
    lines.append("  seed robustness (single-seed LigTerm-1S: median Bond-AUROC difference > 0 and >= 13 of 19 positive, at N and C): " +
                 "; ".join(f"{printed(m)}: N {seed_ok[m]['N']} C {seed_ok[m]['C']}" for m in SEED_MODELS) + f" -> seed-robust: {reading['seed_robust']}")
    lines.append("  spread of the three medians over the seeds: " + "; ".join(f"{T}: " + ", ".join(f"{printed(m)} {v:+.4f}" for m, v in spread[T]["medians"].items()) + f" (range {spread[T]['range']:.4f})" for T in "NC"))
    lines.append(f"  => {reading['conclusion']}")
    lines.append(f"new-seed fits: {fits['N']['n_fits']} per terminus; at the epoch cap: N {fits['N']['n_at_cap']}, C {fits['C']['n_at_cap']}; selected epoch median N {fits['N']['e_star_median']:.0f} "
                 f"(range {fits['N']['e_star_min']}-{fits['N']['e_star_max']}), C {fits['C']['e_star_median']:.0f} (range {fits['C']['e_star_min']}-{fits['C']['e_star_max']})")
    for s, cc in control.items():
        lines.append(f"control (stored seed refitted, study {s}): " + "; ".join(
            f"{T}: max|dlogit| {cc[T]['max_abs_logit_diff']:.4f}, mean|dlogit| {cc[T]['mean_abs_logit_diff']:.4f}, r {cc[T]['pearson_r']:.5f}, Bond-AUROC {cc[T]['bond_auroc_recorded']:.4f} -> {cc[T]['bond_auroc_control']:.4f}, "
            f"selected epoch {cc[T]['e_star_recorded']} -> {cc[T]['e_star_control']}" for T in "NC"))
    return summary, csv_rows, "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------------------------------------------ the commands
def main(argv, stage):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(prog=f"44_development_benchmark.py {'ensemble-scaling' if stage == STAGE_SCALING else 'ensemble-seed-robustness'}", description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", required=True, help="folder with the stored per-study records (study_<pmid>.json)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--config", default=os.path.join(here, "..", "..", "configs", "ensemble_stages.json"))
    a = ap.parse_args(argv)
    for suffix in ("summary.json", "per_study.csv", "report.txt"):
        if os.path.exists(os.path.join(a.out, f"{stage}_{suffix}")):
            sys.exit(f"refusing to overwrite {os.path.join(a.out, f'{stage}_{suffix}')}")
    cfg = json.load(open(a.config))
    studies = cfg["studies"] if stage == STAGE_SCALING else [s for s in cfg["studies"] if s not in cfg["seen"]]
    recs = load_stage_records(a.records, studies)
    if stage == STAGE_SCALING:
        summary, csv_rows, text = outputs_scaling(compute_scaling(recs, cfg))
    else:
        summary, csv_rows, text = outputs_seed_robustness(compute_seed_robustness(recs, cfg))
    write_outputs(a.out, stage, summary, csv_rows, text)
    print(text, end="")
