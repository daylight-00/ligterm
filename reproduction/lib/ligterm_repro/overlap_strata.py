"""Overlap-stratified analysis of the development studies (post hoc, exploratory, descriptive; no gate and no reading label).

Question: how much of the Bond-AUROC of the development studies rests on observed ligand-terminal bonds that the training rows already contain? At a terminus, an observed ligand-terminal bond (protein, bond)
of the primary labels of a study is SHARED with the training if it is also a terminal bond of the same terminus of a unit in the strict training rows of that study's refit, and NOVEL to the training if it is not. The
analysis of a stratum excludes (-1) the positives of the other stratum, nothing else changes. No model is fitted and no new score is made: LigTerm-GB and LigTerm are read from the STORED per-bond logits of the
19 unseen development studies, assembled as the ensemble stages assembled them (LigTerm-GB = the stored LigTerm-GB logits; LigTerm = 0.5 * (LigTerm-GB + the mean of the three window-MLP logits of the stored seed and
the two later seeds)).

Result of the stored run (what the script reproduces): the shared stratum is empty in every study at both termini, because the strict training rows of a refit exclude every sequence group that contains a unit of
the evaluated study and every evaluation protein is such a protein (the checks of `training.check_disjoint`). The development bond-level evaluation is therefore protein-group-disjoint from the training rows by
construction; novel equals all, every novel-minus-all difference is 0, and the medians are those of the per-study records. It is not a measurement of how much of the performance rests on easily reproduced bonds.

Two sub-commands:
  study      one per-study record from the stored logits (the per-study part of the analysis). Guards (a failure stops the study, nothing is written): the strict training rows equal the stored information of the
             refit (their number and the filter counts); the evaluation proteins of the stored logits are the study's; every score is finite and covers every evaluation bond; with --records, the per-protein
             Bond-AUROC, local-pair counts and numbers of positive bonds of ALL positives equal the records of the ensemble seed-robustness stage within the tolerance of the configuration for LigTerm-GB and LigTerm at
             both termini; the shared and novel positives partition the positives (inside `split_labels`).
  summarize  the aggregation of the per-study records (medians over the studies, paired differences; a study-terminus with fewer than `min_positive_bonds` positive bonds in a stratum is left out of that stratum's
             median, the numbers left out are reported): the summary JSON, the per-study table and a text report.

usage: 44_development_benchmark.py overlap-strata study --data <datasets folder> --logits <folder with <pmid>.npz> --study PMID [PMID ...] --out <folder> [--records <folder with the ensemble seed-robustness records>] [--config FILE]
       44_development_benchmark.py overlap-strata summarize --records <folder with the per-study records> --out <folder> [--config FILE]
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

from . import evaluation, metrics, training
from .data import KEY, bond_of, load_ctx

LABEL = "Development studies stratified by overlap of the positive bonds with the training units (post hoc, exploratory, descriptive)"
GUARDS = ("the strict rows equal the stored information of the refits; the per-protein values of ALL positives equal the records of the ensemble seed-robustness stage (1e-9) for LigTerm-GB and LigTerm at both termini; "
          "shared + novel = all positives")
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "..", "..", "configs", "overlap_strata.json")
WINDOW_SEEDS = (0, 1, 2)                                   # the stored window-MLP head (seed 0) and the two later seeds
CSV_FIELDS = ["study", "terminus", "model", "stratum", "n_positive_bonds", "share_shared_with_training", "bond_auroc", "local_auc", "n_proteins"]


# ------------------------------------------------------------------------------------------------------------------------------- pieces of the analysis
def split_labels(lab, bp, bi, trk, key):
    """lab: int8 labels over the concatenated evaluation bonds (1 positive, 0 negative, -1 excluded); bp, bi: the protein index and the 1-based bond index of every evaluation bond; trk: the keys `protein * key + bond` of the terminal bonds (at this terminus) of the strict training rows.
    -> ({'all', 'shared', 'novel'} label arrays, {'all', 'shared', 'novel'} numbers of positive bonds). `lab` is not modified."""
    lab = np.asarray(lab)
    idx = np.flatnonzero(lab == 1)
    keys = np.asarray(bp)[idx].astype(np.int64) * key + np.asarray(bi)[idx].astype(np.int64)
    sh = np.isin(keys, np.asarray(trk, np.int64))
    out = {"all": lab, "shared": lab.copy(), "novel": lab.copy()}
    out["shared"][idx[~sh]] = -1
    out["novel"][idx[sh]] = -1
    n = {"all": int(len(idx)), "shared": int(sh.sum()), "novel": int((~sh).sum())}
    assert n["shared"] + n["novel"] == n["all"], "the shared and novel positives do not partition the positives"
    return out, n


def same_vector(a, b, tol):
    """two vectors equal within tol with equal NaN patterns (numbers or None)"""
    a = np.array([np.nan if x is None else x for x in a], np.float64)
    b = np.array([np.nan if x is None else x for x in b], np.float64)
    return len(a) == len(b) and np.array_equal(np.isnan(a), np.isnan(b)) and bool(np.all(np.abs(a[~np.isnan(a)] - b[~np.isnan(b)]) <= tol))


def summarise(pp):
    """the macro Bond-AUROC over the proteins with a defined AUROC and the pooled Local-AUC of a `metrics.per_protein` result"""
    ok = np.isfinite(pp["bond_auroc"])
    d = float(np.sum(pp["local_d"]))
    return dict(bond_auroc=float(np.mean(pp["bond_auroc"][ok])) if ok.any() else math.nan, local_auc=float(np.sum(pp["local_n"])) / d if d > 0 else math.nan, n_proteins=int(ok.sum()))


def median_rule(values, npos, min_pos=20):
    """the median of the per-study values over the studies with at least `min_pos` positive bonds in the stratum and a finite value -> (median, number included, number left out)"""
    v = np.array([np.nan if x is None else x for x in values], np.float64)
    n = np.asarray(npos)
    keep = (n >= min_pos) & np.isfinite(v)
    return (float(np.median(v[keep])) if keep.any() else math.nan), int(keep.sum()), int(len(v) - keep.sum())


def paired(a, b, npos_stratum, min_pos=20):
    """per-study differences a - b over the studies whose stratum has at least `min_pos` positives and finite values -> (median, number of studies with a positive difference, number included, number left out, the differences or NaN per study)"""
    a = np.array([np.nan if x is None else x for x in a], np.float64)
    b = np.array([np.nan if x is None else x for x in b], np.float64)
    d = a - b
    keep = (np.asarray(npos_stratum) >= min_pos) & np.isfinite(d)
    dd = np.where(keep, d, np.nan)
    return (float(np.median(d[keep])) if keep.any() else math.nan), int((d[keep] > 0).sum()), int(keep.sum()), int(len(d) - keep.sum()), dd


def jclean(o):
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, dict):
        return {k: jclean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jclean(v) for v in o]
    return o


# ------------------------------------------------------------------------------------------------------------------------------- the schema of a per-study record
# {"study", "strict_rows", "positive_bonds": [{terminus, stratum, n_positive_bonds}], "terminus_summary": [{terminus, share_shared_with_training, n_eval_proteins}],
#  "results": [{terminus, model, stratum, bond_auroc, local_auc, n_proteins}]}            (non-finite values are null)
def record_view(rec, cfg):
    """a public per-study record as the nested view that the aggregation reads: {'study', 'strict_rows', T: {'n_pos': {stratum: n}, 'share_shared', 'n_eval_proteins', 'results': {(model, stratum): {...}}}}"""
    st = cfg["strata"]
    out = dict(study=rec["study"], strict_rows=rec["strict_rows"])
    for T in "NC":
        ts = next(r for r in rec["terminus_summary"] if r["terminus"] == T)
        out[T] = dict(n_pos={s: next(r["n_positive_bonds"] for r in rec["positive_bonds"] if r["terminus"] == T and r["stratum"] == s) for s in st},
                      share_shared=ts["share_shared_with_training"], n_eval_proteins=ts["n_eval_proteins"], results={})
        for r in rec["results"]:
            if r["terminus"] == T:
                out[T]["results"][(r["model"], r["stratum"])] = dict(bond_auroc=r["bond_auroc"], local_auc=r["local_auc"], n_proteins=r["n_proteins"])
    return out


def load_records(rdir, cfg):
    per, extra = [], []
    for s in cfg["studies"]:
        with open(os.path.join(rdir, f"study_{s}.json"), encoding="utf-8") as fh:
            rec = json.load(fh)
        assert str(rec["study"]) == s, f"{s}: the record says study {rec['study']}"
        per.append((rec, record_view(rec, cfg)))
    for f in sorted(os.listdir(rdir)):
        if f.startswith("study_") and f.endswith(".json") and f[len("study_"):-len(".json")] not in set(cfg["studies"]):
            extra.append(f)
    return per, extra


# --------------------------------------------------------------------------------------------------------------------------------- summarize
def aggregate(PER, cfg):
    """the aggregation section of the analysis: PER = the nested views of the per-study records -> the summary as rows"""
    MODELS, STRATA, MIN_POS = [m["model"] for m in cfg["models"]], list(cfg["strata"]), cfg["min_positive_bonds"]
    ALL, NOVEL = STRATA[0], STRATA[2]
    share_rows, med_rows, diff_rows = [], [], []
    for T in "NC":
        share = [p[T]["share_shared"] for p in PER]
        npos = {st: [p[T]["n_pos"][st] for p in PER] for st in STRATA}
        share_rows.append(dict(terminus=T, median=float(np.median(share)), min=float(min(share)), max=float(max(share))))
        for m in MODELS:
            for st in STRATA:
                for q in cfg["metrics"]:
                    v = [p[T]["results"][(m, st)][q] for p in PER]
                    med, kept, left = median_rule(v, npos[st], MIN_POS)
                    med_rows.append(dict(terminus=T, model=m, stratum=st, metric=q, median=med, studies_included=kept, studies_left_out=left))
        for m in MODELS:
            for q in cfg["metrics"]:
                va = [p[T]["results"][(m, NOVEL)][q] for p in PER]
                vb = [p[T]["results"][(m, ALL)][q] for p in PER]
                med, npos_d, n_in, n_out, dd = paired(va, vb, npos[NOVEL], MIN_POS)
                diff_rows.append(dict(terminus=T, model_a=m, model_b=m, stratum_a=NOVEL, stratum_b=ALL, metric=q, median=med, studies_positive=npos_d, studies_included=n_in, studies_left_out=n_out))
        for q in cfg["metrics"]:
            va = [p[T]["results"][(MODELS[1], NOVEL)][q] for p in PER]
            vb = [p[T]["results"][(MODELS[0], NOVEL)][q] for p in PER]
            med, npos_d, n_in, n_out, dd = paired(va, vb, npos[NOVEL], MIN_POS)
            diff_rows.append(dict(terminus=T, model_a=MODELS[1], model_b=MODELS[0], stratum_a=NOVEL, stratum_b=NOVEL, metric=q, median=med, studies_positive=npos_d, studies_included=n_in, studies_left_out=n_out))
    return dict(label=LABEL, studies=list(cfg["studies"]), min_positive_bonds=MIN_POS, models=[dict(m) for m in cfg["models"]], strata=STRATA, guards=GUARDS,
                share_shared_with_training=share_rows, medians=med_rows, differences=diff_rows)


def f4(x):
    return "  n/a " if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def render_report(summary, records, cfg):
    """the text report from the summary (as written to JSON) and the per-study records (as read from JSON)"""
    MODELS, STRATA, MIN_POS = [m["model"] for m in cfg["models"]], list(cfg["strata"]), cfg["min_positive_bonds"]
    med = {(r["terminus"], r["model"], r["stratum"], r["metric"]): r for r in summary["medians"]}
    dif = [r for r in summary["differences"]]
    lines = [f"Overlap-stratified analysis of the development studies (post hoc, exploratory, descriptive; no gate): {len(summary['studies'])} unseen development studies, stored logits, strict training rows of each refit",
             f"guards: the strict rows equal the stored information of the refits; the per-protein values of all positives equal the records of the ensemble seed-robustness stage (1e-9); a study-terminus with fewer than {MIN_POS} positive bonds in a stratum is left out of that stratum's median"]
    for T in "NC":
        sh = next(r for r in summary["share_shared_with_training"] if r["terminus"] == T)
        lines.append(f"--- terminus {T}: the share of the observed ligand-terminal bonds that are shared with the training (a terminal bond of the same terminus of a strict training unit): median {sh['median']:.3f} (range {sh['min']:.3f} to {sh['max']:.3f})")
        lines.append("    median Bond-AUROC [studies included / left out]: " + " | ".join(
            f"{m} {st}: {f4(med[(T, m, st, 'bond_auroc')]['median'])} [{med[(T, m, st, 'bond_auroc')]['studies_included']}/{med[(T, m, st, 'bond_auroc')]['studies_left_out']}]" for m in MODELS for st in STRATA))
        lines.append("    median pooled Local-AUC:                   " + " | ".join(f"{m} {st}: {f4(med[(T, m, st, 'local_auc')]['median'])}" for m in MODELS for st in STRATA))
        for r in dif:
            if r["terminus"] != T:
                continue
            k = (f"{r['stratum_a']} - {r['stratum_b']}|{r['model_a']}|{r['metric']}" if r["model_a"] == r["model_b"] else f"{r['model_a']} - {r['model_b']}|{r['stratum_a']}|{r['metric']}")
            lines.append(f"    {k:72s}: median {f4(r['median'])}, studies with a positive difference {r['studies_positive']} of {r['studies_included']} (left out {r['studies_left_out']})")
    lines.append(f"per study (N | C): share shared with the training; Bond-AUROC {' / '.join(STRATA)} for {MODELS[0]}, then {MODELS[1]}")
    for rec in records:
        row = [f"{rec['study']}"]
        for T in "NC":
            ts = next(r for r in rec["terminus_summary"] if r["terminus"] == T)
            npos = {r["stratum"]: r["n_positive_bonds"] for r in rec["positive_bonds"] if r["terminus"] == T}
            res = {(r["model"], r["stratum"]): r for r in rec["results"] if r["terminus"] == T}
            row.append(f"{T}: share {ts['share_shared_with_training']:.2f} (pos {'/'.join(str(npos[s]) for s in STRATA)}) " + " ; ".join(f"{m} " + "/".join(f4(res[(m, s)]["bond_auroc"]).strip() for s in STRATA) for m in MODELS))
        lines.append("  " + " | ".join(row))
    return "\n".join(lines) + "\n"


def write_summary_files(summary, records, out_dir, cfg):
    """the three outputs of the aggregation (the summary JSON, the per-study table, the text report); `records` as in render_report"""
    os.makedirs(out_dir, exist_ok=True)
    summary = jclean(summary)
    with open(os.path.join(out_dir, "overlap_strata_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
        fh.write("\n")
    with open(os.path.join(out_dir, "overlap_strata_per_study.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_FIELDS)
        for rec in records:
            for T in "NC":
                ts = next(r for r in rec["terminus_summary"] if r["terminus"] == T)
                npos = {r["stratum"]: r["n_positive_bonds"] for r in rec["positive_bonds"] if r["terminus"] == T}
                for m in [x["model"] for x in cfg["models"]]:
                    for st in cfg["strata"]:
                        r = next(x for x in rec["results"] if x["terminus"] == T and x["model"] == m and x["stratum"] == st)
                        nan = r["bond_auroc"] is None
                        w.writerow([rec["study"], T, m, st, npos[st], f"{ts['share_shared_with_training']:.4f}", "" if nan else f"{r['bond_auroc']:.6f}", "" if r["local_auc"] is None else f"{r['local_auc']:.6f}", r["n_proteins"]])
    text = render_report(summary, records, cfg)
    with open(os.path.join(out_dir, "overlap_strata_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def summarize(rdir, out_dir, cfg):
    per, extra = load_records(rdir, cfg)
    assert not extra, f"the records folder holds records of studies that are not in the configuration: {extra}"
    assert len(per) == len(cfg["studies"])
    summary = aggregate([v for _, v in per], cfg)
    text = write_summary_files(summary, [r for r, _ in per], out_dir, cfg)
    print(text)
    print(f"wrote overlap_strata_summary.json, overlap_strata_per_study.csv and overlap_strata_report.txt to {out_dir}")


# ------------------------------------------------------------------------------------------------------------------------------------ study
def read_ensemble_record(path):
    """{(model, terminus): the bond-level row} of the ensemble-stage record of a study, for the models without a seed field"""
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)
    return {(r["model"], r["terminus"]): r for r in rec["bond"] if "seed" not in r}


def study_record(S, ctx, cfg, logits_dir, records_dir=None, log=print):
    t0 = time.time()
    PRIM, TOL, MIN_POS = cfg["primary_grouping"], cfg["record_tolerance"], cfg["min_positive_bonds"]
    MODELS, STRATA = [m["model"] for m in cfg["models"]], list(cfg["strata"])
    assert S in cfg["studies"], f"{S} is not one of the {len(cfg['studies'])} unseen development studies of the analysis"
    z = np.load(os.path.join(logits_dir, f"{S}.npz"), allow_pickle=False)
    assert str(z["study"]) == S, f"{S}: the stored logits are those of study {z['study']}"
    rows, sinfo = training.strict_rows(ctx, S, PRIM)
    assert len(rows) == int(z["strict_rows"]) and sinfo == dict(n_group_filtered=int(z["strict_group_filtered"]), n_dropped_overlap=int(z["strict_dropped_overlap"]), n_dropped_identical=int(z["strict_dropped_identical"])), \
        f"{S}: the strict rows differ from the stored information of the refit ({len(rows)}, {sinfo})"
    ens = {}
    if records_dir:
        ens = read_ensemble_record(os.path.join(records_dir, f"study_{S}.json"))
    out = dict(study=S, strict_rows=int(len(rows)), positive_bonds=[], terminus_summary=[], results=[])
    for T in "NC":
        es = evaluation.eval_setup(ctx, S, T)
        eprot, lab = es["eprot"], es["labs"]["primary"]
        assert np.array_equal(z[f"{T}_eval_proteins"], eprot), f"{S} {T}: the evaluation proteins of the stored logits differ from the study's"
        lg = np.asarray(z[f"{T}_ligterm_gb"], np.float64)
        x0, x1, x2 = (np.asarray(z[f"{T}_window_mlp_seed{k}"], np.float64) for k in WINDOW_SEEDS)
        sc = {MODELS[0]: lg, MODELS[1]: 0.5 * (lg + (x0 + x1 + x2) / 3.0)}
        for m, s in sc.items():
            assert len(s) == len(lab) and np.isfinite(s).all(), f"{S} {T} {m}: the scores do not cover the evaluation bonds or are not finite"
        ok, b = bond_of(ctx, rows, T)
        trk = np.unique(ctx.vp[rows][ok].astype(np.int64) * KEY + b[ok])
        labs, npos = split_labels(lab, es["bp"], es["bi"], trk, KEY)
        res = {}
        for m, s in sc.items():
            for st_i, st in enumerate(("all", "shared", "novel")):
                pp = metrics.per_protein(s, labs[st], es["ebase"], es["nb"])
                if st == "all" and ens:
                    row = ens[(m, T)]
                    for k, kr in (("bond_auroc", "bond_auroc"), ("local_n", "local_pairs_won"), ("local_d", "local_pairs"), ("npos", "n_positive_bonds")):
                        assert same_vector(pp[k].tolist(), row[kr], TOL), f"{S} {T} {m}: the per-protein {kr} of ALL positives differ from the record of the ensemble seed-robustness stage"
                res[(m, STRATA[st_i])] = summarise(pp)
        for st_i, st in enumerate(("all", "shared", "novel")):
            out["positive_bonds"].append(dict(terminus=T, stratum=STRATA[st_i], n_positive_bonds=npos[st]))
        out["terminus_summary"].append(dict(terminus=T, share_shared_with_training=npos["shared"] / npos["all"], n_eval_proteins=int(len(eprot))))
        for m in MODELS:
            for st in STRATA:
                r = res[(m, st)]
                out["results"].append(dict(terminus=T, model=m, stratum=st, bond_auroc=r["bond_auroc"], local_auc=r["local_auc"], n_proteins=r["n_proteins"]))
    log(f"study {S}: strict rows {len(rows)}; guards passed" + ("" if ens else " (no ensemble-stage records given: the all-positives guard against them was not run)") + f"; {time.time() - t0:.0f} s")
    return jclean(out)


def run_studies(a, cfg):
    ctx = load_ctx(os.path.join(a.data, "units.tsv.gz"), os.path.join(a.data, "protein_sequences.json"), os.path.join(a.data, "sequence_groups.tsv.gz"))
    os.makedirs(a.out, exist_ok=True)
    for S in a.study:
        path = os.path.join(a.out, f"study_{S}.json")
        if os.path.exists(path):
            sys.exit(f"refusing to overwrite {path}")
        rec = study_record(S, ctx, cfg, a.logits, a.records)
        with open(path + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1)
            fh.write("\n")
        os.replace(path + ".tmp", path)
        print("written", path, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("study", help="one per-study record from the stored logits")
    s1.add_argument("--data", required=True, help="datasets folder (units.tsv.gz, protein_sequences.json, sequence_groups.tsv.gz)")
    s1.add_argument("--logits", required=True, help="folder with the stored logits of the studies, <pmid>.npz")
    s1.add_argument("--study", nargs="+", required=True)
    s1.add_argument("--out", required=True)
    s1.add_argument("--records", help="folder with the records of the ensemble seed-robustness stage (the all-positives guard)")
    s1.add_argument("--config", default=DEFAULT_CONFIG)
    s2 = sub.add_parser("summarize", help="the aggregation of the per-study records")
    s2.add_argument("--records", required=True, help="folder with the per-study records study_<pmid>.json")
    s2.add_argument("--out", required=True)
    s2.add_argument("--config", default=DEFAULT_CONFIG)
    a = ap.parse_args(argv)
    with open(a.config, encoding="utf-8") as fh:
        cfg = json.load(fh)
    if a.cmd == "study":
        run_studies(a, cfg)
    else:
        summarize(a.records, a.out, cfg)

