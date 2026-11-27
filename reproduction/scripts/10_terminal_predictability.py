#!/usr/bin/env python3
"""Terminal predictability: how well simple sequence models rank the observed ligand-terminal bonds of a held-out study.

For one held-out study and each terminus (N, C): the strict training set (sequence groups touched by the study are excluded, and
so is every unit whose peptide shares a 9-mer with a peptide of the study), the eight models of the configuration, Bond-AUROC,
Bond-AUPRC, Bond recall at the top 10 % and Local-AUC with a group bootstrap, paired differences, a shuffled-label control, the
label variants of the evaluation (same scores) and the refits of the primary model only: the other eligible sequence groupings,
training without same-lab studies with a size-matched random-removal control, and a leaky control (study held out, groups not).
Nothing is tuned; every setting is read from the configuration.

usage: 10_terminal_predictability.py --data <datasets folder> --out <results folder> (--study PMID [PMID ...] | --all) [--config FILE]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from ligterm_repro import data, evaluation, metrics, models, records, training  # noqa: E402

LABEL_VARIANTS = ("other_study_bonds_as_negatives", "positives_any_study", "nested_family_positives", "opposite_terminus_bonds_excluded")


def run_study(S, cfg, ctx, keys, out_dir):
    log = records.Logger(S)
    PRIM, PM, B_ = cfg["primary_grouping"], cfg["primary_model"], cfg["bootstrap"]
    SEEN = S in cfg["seen"]
    RES, DIAG = [], []
    rows, sinfo = training.strict_rows(ctx, S, PRIM)
    training.check_disjoint(ctx, rows, S, PRIM)
    log("strict training rows:", sinfo, "->", len(rows), "units")
    STUDY = dict(strict=sinfo, strict_units=int(len(rows)))

    def rec(T, analysis, model, bt, extra=None):
        RES.append(records.cell(S, SEEN, T, analysis, model, bt, extra))

    for T in ("N", "C"):
        rng = np.random.default_rng(records.terminus_stream(cfg["seed"], S, T))
        es = evaluation.eval_setup(ctx, S, T)
        etok = data.tokens(ctx, es["bp"], es["bi"])
        epf = data.posfeat(es["bi"], ctx.L[es["bp"]])
        gid = ctx.G[PRIM][es["eprot"]]
        training.check_disjoint(ctx, rows, S, PRIM, eprot=es["eprot"], peptides=False)
        D = dict(terminus=T, eval_proteins=int(len(es["eprot"])), eval_bonds=int(es["nb"].sum()), n_pos=es["n_pos"], n_excl_other=es["n_other"], n_excl_repeat=es["n_repeat"],
                 n_nested_pos=es["n_nested_pos"], n_opposite_terminus_bonds=es["n_opposite"], n_eval_groups=int(len(np.unique(gid))), fit_info={})
        DIAG.append(D)
        P, B, y, info = training.build_train(ctx, rows, T, rng, cfg["neg_ratio"])
        info = dict(train_pos=info["n_pos"], train_neg=info["n_neg"], train_proteins=info["n_train_proteins"], train_units=int(len(rows)))
        log(T, D, info)
        # ---- the models (strict training set)
        scores, fi = models.fit_score(ctx, P, B, y, models.make_models(cfg["models"]), etok, epf)
        D["fit_info"]["primary"] = fi
        log(T, "fitted", cfg["models"])
        pps = {}
        for m in cfg["models"]:
            pps[m] = metrics.per_protein(scores[m], es["labs"]["primary"], es["ebase"], es["nb"])
            rec(T, "primary", m, metrics.boot(pps[m], gid, rng, B_), info)
        for d in cfg["paired_differences"]:                  # paired differences of the macro Bond-AUROC (same proteins, same resampled groups)
            est, lo, hi = metrics.boot_diff(pps[d["a"]], pps[d["b"]], gid, rng, B_)
            RES.append(records.paired(S, SEEN, T, "primary", d["a"], d["b"], est, lo, hi))
        # ---- shuffled-label control, label variants and one protein per group (primary model, same scores)
        mu, sd = metrics.shuffle_control(scores[PM], es["labs"]["primary"], es["ebase"], es["nb"], rng, cfg["shuffles"])
        RES.append(dict(study=S, seen=SEEN, terminus=T, analysis="shuffled_label_control", kind="model", model=PM, bond_auroc=mu, bond_auroc_sd=sd))
        for v in LABEL_VARIANTS:
            pp = metrics.per_protein(scores[PM], es["labs"][v], es["ebase"], es["nb"])
            if np.isfinite(pp["bond_auroc"]).any():
                rec(T, v, PM, metrics.boot(pp, gid, rng, B_))
        valid = np.isfinite(pps[PM]["bond_auroc"])           # per sequence group the longest protein among those with a defined primary Bond-AUROC
        order = np.lexsort((np.arange(len(es["eprot"])), -ctx.L[es["eprot"]], gid))
        order = order[valid[order]]
        keep = np.zeros(len(es["eprot"]), bool)
        keep[order[np.r_[True, gid[order][1:] != gid[order][:-1]]]] = True
        pp = metrics.per_protein(scores[PM], es["labs"]["primary"], es["ebase"], es["nb"], keep=keep)
        rec(T, "one_protein_per_group", PM, metrics.boot(pp, gid, rng, B_))
        # ---- refits of the primary model only
        pmodel = models.make_models([PM])

        def refit(analysis, rows2, gridg, extra=None):
            P2, B2_, y2, i2 = training.build_train(ctx, rows2, T, rng, cfg["neg_ratio"])
            sc, fi2 = models.fit_score(ctx, P2, B2_, y2, pmodel, etok, epf)
            D["fit_info"][analysis] = fi2
            pp2 = metrics.per_protein(sc[PM], es["labs"]["primary"], es["ebase"], es["nb"])
            e = dict(train_pos=i2["n_pos"], train_neg=i2["n_neg"], train_proteins=i2["n_train_proteins"], train_units=int(len(rows2)))
            e.update(extra or {})
            rec(T, analysis, PM, metrics.boot(pp2, ctx.G[gridg][es["eprot"]], rng, B_), e)

        for g2 in cfg["groupings"]:
            if g2 != PRIM and S in cfg["eligible"].get(g2, []):
                r2, i_g2 = training.strict_rows(ctx, S, g2)
                training.check_disjoint(ctx, r2, S, g2, eprot=es["eprot"], peptides=False)
                refit("grouping_" + g2, r2, g2, dict(dropped_overlap=i_g2["n_dropped_overlap"]))
        X, why = training.lab_exclusion(ctx, S, es["inS"], keys)
        bad = np.zeros(ctx.nv, bool)
        for s2 in X:
            bad[ctx.stud[s2]] = True
        r3 = np.setdiff1d(rows, np.flatnonzero(bad))
        training.check_disjoint(ctx, r3, S, PRIM, eprot=es["eprot"], peptides=False)
        D["lab_excluded_studies"] = {k: v for k, v in sorted(why.items())}
        D["lab_excluded_units_removed"] = int(len(rows) - len(r3))
        refit("same_lab_excluded_from_training", r3, PRIM)
        refit("size_control_random", training.size_matched_rows(rows, len(rows) - len(r3), rng), PRIM)       # the same number of training units removed, at random
        rl = np.flatnonzero(~es["inS"])                                                                       # leaky by design: the study is held out, its protein groups and peptides are not
        tp = np.unique(ctx.vp[rl])
        g_ = ctx.G[PRIM]
        leak = dict(leaky=True, train_proteins_in_study_groups=int(np.isin(g_[tp], g_[np.unique(ctx.vp[ctx.stud[S]])]).sum()),
                    train_units_sharing_9mer_with_study=int(training.peptide_overlap_mask(ctx, rl, ctx.stud[S])[0].sum()))
        D["study_held_out_control"] = leak
        refit("study_held_out_leaky_control", rl, PRIM, leak)
        log(T, "done")
    STUDY["config_lr"] = cfg["lr"]
    path = records.write_study(out_dir, S, dict(study=S, study_info=STUDY, diag=DIAG, cells=RES))
    log("written", path)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="datasets folder (units.tsv.gz, protein_sequences.json, sequence_groups.tsv.gz, studies.tsv)")
    ap.add_argument("--out", required=True, help="results folder (one study_<pmid>.json per study)")
    ap.add_argument("--config", default=os.path.join(here, "..", "configs", "terminal_predictability.json"))
    ap.add_argument("--study", nargs="+", help="PMIDs of the held-out studies")
    ap.add_argument("--all", action="store_true", help="all studies of the configuration")
    a = ap.parse_args()
    cfg = records.load_config(a.config)
    studies = cfg["studies"] if a.all else a.study
    if not studies:
        ap.error("give --study or --all")
    models.configure(cfg["lr"])
    assert cfg["kmer_overlap"] == training.KMER_OVERLAP, "configuration and library disagree on the peptide-overlap k"
    ctx, keys = records.load_study_context(a.data)
    for S in studies:
        run_study(S, cfg, ctx, keys, a.out)


if __name__ == "__main__":
    main()
