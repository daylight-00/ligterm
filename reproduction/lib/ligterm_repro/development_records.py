"""Per-study records of the development stages and the helpers that the three record-level aggregations share (representation comparison, ensemble score scaling, ensemble seed robustness).

The helpers are the pure functions of the stage evaluator of the development protocol (the ensemble with a matched spread, the summaries of the per-protein vectors, the comparison of two vectors), copied as they were with
public names; the arithmetic is that of the stored aggregates. The per-study records are the stored outputs of the runs of these stages, in the public layout below; the aggregations read them and nothing else of the runs (the GPU fits that
produced the window-MLP and ESMC logits are never repeated).

A record is one JSON file per study (and, for the representation comparison, per part). Rows carry the model as fields (model, optional fit, optional seed), never as a composite key:
  study, seen, stage, [part], models [{model[, fit][, seed]}], variants [public variant names], units {...}, rows [{terminus, n_rows, n_pos, n_neg, n_eval_proteins, n_eval_bonds}],
  bond_proteins [{terminus, acc, group, length}], peptide_proteins {acc, group, length, n_pos},
  bond [{<model fields>, terminus, bond_auroc[], local_pairs_won[], local_pairs[], n_positive_bonds[]}]            one entry per evaluation protein of the terminus (None: not defined)
  peptide [{<model fields>, variant, peptide_auroc[], length_matched_num[], length_matched_den[], near_miss_num[], near_miss_den[]}]   one entry per evaluation protein of the peptide-level populations
  bond_summary [{<model fields>, terminus, bond_auroc, local_auc}], peptide_summary [{<model fields>, variant, peptide_auroc, length_matched_auc, near_miss_auc}],
  and the stage-specific blocks (score scaling, fits, floor, seeds, control, env).
Bond-AUROC is the protein-macro AUROC of the bond scores; Local-AUC the pooled share of local pairs won; Peptide-AUROC, Length-matched AUC and Near-miss AUC are the peptide-level measures of the peptide-ranking experiment.
"""
import csv
import json
import math

import numpy as np

BOND_KEYS = ("bond_auroc", "local_pairs_won", "local_pairs", "n_positive_bonds")
PEP_KEYS = ("peptide_auroc", "length_matched_num", "length_matched_den", "near_miss_num", "near_miss_den")
TERMINI = ("N", "C")
SEP = "|"                                        # model key inside the aggregations: 'model|fit|seed' (never written to a file)


def mkey(row):
    """the key of a model row inside the aggregations"""
    return SEP.join([row["model"], row.get("fit", ""), str(row["seed"]) if "seed" in row else ""])


def mrow(key):
    """the model fields of a key (the inverse of mkey)"""
    name, fit, seed = key.split(SEP)
    out = {"model": name}
    if fit:
        out["fit"] = fit
    if seed != "":
        out["seed"] = int(seed)
    return out


def mkey_of(name, fit=None, seed=None):
    return mkey({"model": name, **({"fit": fit} if fit else {}), **({"seed": seed} if seed is not None else {})})


def printed(key):
    """the name of a model as printed in a report"""
    r = mrow(key)
    name = r["model"] + ("-CV" if r.get("fit") == "cv" and r["model"] == "LR-Both16" else "")
    return name + (f" (seed {r['seed']})" if "seed" in r else "")


def load_record(path):
    with open(path, encoding="utf-8") as fh:
        return view(json.load(fh))


def view(o):
    """the nested view of a public record that the aggregations use: bond[model][terminus][key], peptide[model][variant][key], bond_summary[model][terminus], peptide_summary[model][variant], scale[terminus][model], fits[model][terminus], ..."""
    listed = ("models", "rows", "bond_proteins", "bond", "peptide", "bond_summary", "peptide_summary", "peptide_proteins", "score_scaling", "fits", "floor", "device", "gpu_seconds", "seeds", "control")
    r = {k: o[k] for k in o if k not in listed}
    r["models"] = [mkey(m) for m in o["models"]]
    r["rows"] = {x["terminus"]: {k: v for k, v in x.items() if k != "terminus"} for x in o["rows"]}
    r["bond_proteins"] = {x["terminus"]: {k: v for k, v in x.items() if k != "terminus"} for x in o["bond_proteins"]}
    r["peptide_proteins"] = o["peptide_proteins"]
    r["bond"], r["peptide"], r["bond_summary"], r["peptide_summary"] = {}, {}, {}, {}
    for x in o["bond"]:
        r["bond"].setdefault(mkey(x), {})[x["terminus"]] = {k: x[k] for k in BOND_KEYS}
    for x in o["peptide"]:
        r["peptide"].setdefault(mkey(x), {})[x["variant"]] = {k: x[k] for k in PEP_KEYS}
    for x in o["bond_summary"]:
        r["bond_summary"].setdefault(mkey(x), {})[x["terminus"]] = {"bond_auroc": x["bond_auroc"], "local_auc": x["local_auc"]}
    for x in o["peptide_summary"]:
        r["peptide_summary"].setdefault(mkey(x), {})[x["variant"]] = {k: x[k] for k in ("peptide_auroc", "length_matched_auc", "near_miss_auc")}
    if "score_scaling" in o:
        r["scale"] = {}
        for x in o["score_scaling"]:
            r["scale"].setdefault(x["terminus"], {})[mkey(x)] = {k: v for k, v in x.items() if k not in ("terminus", "model", "fit", "seed")}
    if "fits" in o:
        r["fits"] = {}
        for x in o["fits"]:
            r["fits"].setdefault(mkey(x), {})[x["terminus"]] = {k: v for k, v in x.items() if k not in ("terminus", "model", "fit", "seed")}
    if "floor" in o:
        r["floor"] = o["floor"] and {x["terminus"]: {k: v for k, v in x.items() if k != "terminus"} for x in o["floor"]}
    if "device" in o:
        r["device"] = {mkey(x): {k: v for k, v in x.items() if k not in ("model", "fit", "seed")} for x in o["device"]}
    if "gpu_seconds" in o:
        r["gpu_seconds"] = {mkey(x): x["seconds"] for x in o["gpu_seconds"]}
    if "seeds" in o:
        r["seeds"] = {}
        for x in o["seeds"]:
            r["seeds"].setdefault(str(x["seed_index"]), {})[x["terminus"]] = {k: v for k, v in x.items() if k not in ("seed_index", "terminus")}
    if "control" in o:
        r["control"] = o["control"] and {x["terminus"]: {k: v for k, v in x.items() if k != "terminus"} for x in o["control"]}
    return r


# ------------------------------------------------------------------------------------------------------------------ the evaluator's pure helpers
def spread_scale(ligterm_gb, x):
    """s = SD(LigTerm-GB) / SD(x) over the evaluation bonds of one terminus (float64)"""
    sd_l, sd_x = float(np.std(np.asarray(ligterm_gb, np.float64))), float(np.std(np.asarray(x, np.float64)))
    assert sd_l > 0 and sd_x > 0, "a score vector without spread cannot be matched"
    return sd_l / sd_x


def lst(a):
    return [None if (isinstance(x, float) and not np.isfinite(x)) else (int(x) if isinstance(x, (int, np.integer)) else round(float(x), 10)) for x in np.asarray(a).tolist()]


def bvec(rec, m, T, key):
    return np.array([np.nan if x is None else x for x in rec["bond"][m][T][key]], np.float64)


def pvec(rec, m, v, key):
    return np.array([np.nan if x is None else x for x in rec["peptide"][m][v][key]], np.float64)


def bond_summary(auc, e3n, e3d):
    ok = np.isfinite(auc)
    return dict(bond_auroc=float(auc[ok].mean()) if ok.any() else math.nan, local_auc=float(e3n.sum() / e3d.sum()) if e3d.sum() > 0 else math.nan)


def pep_summary(F, Mn, Md, Xn, Xd):
    okF, okM = np.isfinite(F), Md > 0
    return dict(peptide_auroc=float(F[okF].mean()) if okF.any() else math.nan, length_matched_auc=float((Mn[okM] / Md[okM]).mean()) if okM.any() else math.nan,
                near_miss_auc=float(Xn.sum() / Xd.sum()) if Xd.sum() > 0 else math.nan)


def study_stats(rec, models, variants):
    """per model: {terminus: {bond_auroc, local_auc}, variant: {peptide_auroc, length_matched_auc, near_miss_auc}} of one record"""
    out = {}
    for m in models:
        o = {}
        for T in TERMINI:
            o[T] = bond_summary(bvec(rec, m, T, "bond_auroc"), bvec(rec, m, T, "local_pairs_won"), bvec(rec, m, T, "local_pairs"))
        for v in variants:
            o[v] = pep_summary(*(pvec(rec, m, v, k) for k in PEP_KEYS))
        out[m] = o
    return out


def same_vector(a, b, tol):
    """two vectors equal within tol, NaN patterns equal (both lists of numbers or None, or arrays)"""
    a = np.array([np.nan if x is None else x for x in a], np.float64)
    b = np.array([np.nan if x is None else x for x in b], np.float64)
    return len(a) == len(b) and np.array_equal(np.isnan(a), np.isnan(b)) and bool(np.all(np.abs(a[~np.isnan(a)] - b[~np.isnan(b)]) <= tol))


def floor_diagnostic(window_mlp, reference, margin_study, margin_panel, unseen):
    """the floor of the window MLP against the reference model (LR-Both16-CV), a diagnostic: both are {study: {terminus: Bond-AUROC}}; returns the per-study differences, the medians over `unseen` and the two violation flags"""
    d = {s: {T: window_mlp[s][T] - reference[s][T] for T in TERMINI} for s in window_mlp}
    med = {T: float(np.median([d[s][T] for s in unseen])) for T in TERMINI}
    return dict(per_study=d, median_unseen=med, study_violations=sorted(s for s in d if any(d[s][T] < -margin_study for T in TERMINI)), panel_violated=bool(any(med[T] < -margin_panel for T in TERMINI)))


def med(vals):
    a = np.array(vals, np.float64)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else math.nan


def write_per_study_csv(path, rows, value_format=None):
    """the long layout of a per-study table: study, seen, model, fit, seed, terminus, variant, metric, value; rows are (study, seen, model key, terminus or variant, metric, value) in the order study, model, quantity;
    a value is written with repr unless value_format is given (the ensemble stages write six decimals, as their stored tables did)"""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["study", "seen", "model", "fit", "seed", "terminus", "variant", "metric", "value"])
        for s, seen, m, a, q, v in rows:
            f = mrow(m)
            w.writerow([s, str(bool(seen)), f["model"], f.get("fit", ""), f.get("seed", ""), a if a in TERMINI else "", a if a not in TERMINI else "", q, v if value_format is None else format(v, value_format)])


def level_fields(a):
    """a level of a quantity (terminus or variant) -> the fields of a row"""
    return {"terminus": a} if a in TERMINI else {"variant": a}
