"""One held-out study of the directional decomposition and of the published-method comparison.

Both experiments use the same cohort, the same strict training set and the same primary labels as the terminal-predictability experiment, and
they share one random stream per (study, terminus) for the bootstrap. The entry points 20_directional_decomposition.py and
21_published_method_comparison.py call run_study with part="directional" or part="published"; each fits and writes only the models and rows of its
part. The stream is advanced over the evaluations of the other part (the number of sequence groups of a label set does not depend on a score), so
every interval is the one a joint run gives. For the same reason both entry points read the training windows of the released NetCleave data: the
label set netcleave_unseen_window depends on them, and the stream position of the study-exclusive rows (directional part) follows the evaluations on
that label set. The reference rows that both parts need (LR-Both4 and LR-Both8, primary labels) are written by both, with identical values.

Who writes which row:
  directional: primary rows of LR-Both4, LR-Both8, LR-Both16, LR-Out<w>, LR-In<w>; their paired differences (primary); the study-exclusive label set.
  published:   primary rows of LR-Both4, LR-Both8, Window-NB4 and the comparators (Paul-style motif, the NetCleave-style networks, NetCleave as released);
               every row and paired difference on complete_standard_4plus3_window and netcleave_unseen_window.
"""
import numpy as np

from . import comparators, data, evaluation, metrics, models, records, training

PARTS = ("directional", "published")
VW, VNC, V6 = "complete_standard_4plus3_window", "netcleave_unseen_window", "study_exclusive_positives"
REFS = ("LR-Both8", "LR-Both4", "LR-Both16", "Window-NB4")        # fitted first, in this order (the order of the joint run)


def full_order(T, widths):
    """names of all models of terminus T in the order of the joint run: the references, the side models (outward, interior) by width, the comparators."""
    names = list(REFS)
    for w in widths:
        names += [f"LR-Out{w}", f"LR-In{w}"]
    names += ["Paul-style motif"]
    if T == "C":
        names += ["NetCleave-style (released decoys)", "NetCleave-style (protocol negatives)", "NetCleave (published weights)"]
    return names


def part_names(part, widths):
    """models a part fits (and writes primary rows for)."""
    if part == "directional":
        return {"LR-Both8", "LR-Both4", "LR-Both16"} | {f"{k}{w}" for w in widths for k in ("LR-Out", "LR-In")}
    return {"LR-Both8", "LR-Both4", "Window-NB4", "Paul-style motif", "NetCleave-style (released decoys)", "NetCleave-style (protocol negatives)", "NetCleave (published weights)"}


def make_model(name, T, nc):
    if name.startswith("LR-Both"):
        return models.window_model(int(name[7:]))
    if name == "Window-NB4":
        return models.WindowNB4()
    if name.startswith("LR-Out"):
        return models.side_model(T, "outward", int(name[6:]))
    if name.startswith("LR-In"):
        return models.side_model(T, "interior", int(name[5:]))
    for m in comparators.make_models(T, nc):
        if m.name == name:
            return m
    raise ValueError(name)


def run_study(part, S, cfg, ctx, out_dir):
    assert part in PARTS
    log = records.Logger(S)
    PRIM, B_ = cfg["primary_grouping"], cfg["bootstrap"]
    widths, pub = cfg["side_widths"], cfg["published_methods"]
    SEEN, POST = S in cfg["seen"], S in cfg["post_training_date_studies"]
    mine = part_names(part, widths)
    rows, sinfo = training.strict_rows(ctx, S, PRIM)
    training.check_disjoint(ctx, rows, S, PRIM)
    log(part, "strict training rows:", sinfo, "->", len(rows), "units")
    pos_w, all_w = comparators.netcleave_training_windows(pub["netcleave"]["frozen_weights"])
    POS_ARR, ALL_ARR = np.array(sorted(pos_w), np.int64), np.array(sorted(all_w), np.int64)
    RES, DIAG, WEIGHTS = [], [], []

    def owns(analysis, name):
        if analysis == "primary":
            return name in mine
        return part == ("directional" if analysis == V6 else "published")

    for T in ("N", "C"):
        rng_tr = np.random.default_rng(records.terminus_stream(cfg["seed_training"], S, T))        # the seeding of the terminal-predictability experiment: the same training bonds
        rng_bs = np.random.default_rng(records.terminus_stream(cfg["seed_bootstrap"], S, T))
        es = evaluation.eval_setup(ctx, S, T)
        etok = data.tokens(ctx, es["bp"], es["bi"])
        epf = data.posfeat(es["bi"], ctx.L[es["bp"]])
        gid = ctx.G[PRIM][es["eprot"]]
        training.check_disjoint(ctx, rows, S, PRIM, eprot=es["eprot"], peptides=False)
        P, B, y, info = training.build_train(ctx, rows, T, rng_tr, cfg["neg_ratio"])
        # ---- label sets
        labs = dict(primary=es["labs"]["primary"])
        labs[V6], n6keep, n6drop = evaluation.study_exclusive_labels(ctx, S, T, es)
        complete = evaluation.complete_window(es["bi"], ctx.L[es["bp"]], etok)
        labs[VW] = np.where(complete, labs["primary"], -1).astype(np.int8)
        D = dict(terminus=T, eval_proteins=int(len(es["eprot"])), eval_bonds=int(es["nb"].sum()), n_pos=es["n_pos"], n_eval_groups=int(len(np.unique(gid))), fit_info={})
        if part == "directional":
            D.update(v6_pos_bonds=n6keep, v6_excluded_shared_pos_bonds=n6drop)
        else:
            D.update(vw_pos_bonds=int((labs[VW] == 1).sum()), vw_dropped_pos_bonds=int(((labs["primary"] == 1) & ~complete).sum()))
        if T == "C":
            codes = comparators.window_codes(etok)
            seen_any = np.isin(codes, ALL_ARR)
            seen_pos = np.isin(codes, POS_ARR)
            labs[VNC] = np.where(seen_any, -1, labs[VW]).astype(np.int8)
            if part == "published":
                vwp = labs[VW] == 1
                D["ncf_overlap"] = dict(pos_bonds=int(vwp.sum()), pos_window_in_released_training_set=int((vwp & seen_any).sum()), pos_window_in_released_positives=int((vwp & seen_pos).sum()),
                                        neg_bonds=int((labs[VW] == 0).sum()), neg_window_in_released_training_set=int(((labs[VW] == 0) & seen_any).sum()), vnc_pos_bonds=int((labs[VNC] == 1).sum()))
        DIAG.append(D)
        # ---- the models of this part, in the order of the joint run
        order = full_order(T, widths)
        fitted = [make_model(n, T, pub["netcleave"]) for n in order if n in mine]
        log(T, D, [m.name for m in fitted])
        scores, fi = models.fit_score(ctx, P, B, y, fitted, etok, epf)
        D["fit_info"]["all"] = fi
        ok_of = {a: metrics.valid_proteins(lab, es["ebase"], es["nb"]) for a, lab in labs.items()}

        def evaluate(analysis, names):
            """Per-protein metrics and bootstrap of the named models under one label set. A row of the other part is not computed; the stream is advanced as if it were.
            Returns {model: per-protein arrays (own rows) or True (rows of the other part with a defined Bond-AUROC)}."""
            out = {}
            for n in names:
                if n not in order:                                  # the model does not exist for this terminus
                    continue
                if owns(analysis, n):
                    pp = metrics.per_protein(scores[n], labs[analysis], es["ebase"], es["nb"], T=T)
                    if np.isfinite(pp["bond_auroc"]).any():
                        RES.append(records.cell(S, SEEN, T, analysis, n, metrics.boot(pp, gid, rng_bs, B_), post_date=POST, side_split=True))
                        out[n] = pp
                elif ok_of[analysis].any():
                    metrics.skip_boot(ok_of[analysis], gid, rng_bs, B_, side_split=True)
                    out[n] = True
            return out

        def pairs(analysis, lab_name, plist, evaluated, own):
            for d in plist:
                a, b = d["a"], d["b"]
                if a in evaluated and b in evaluated:
                    if own:
                        est, lo, hi = metrics.boot_diff(evaluated[a], evaluated[b], gid, rng_bs, B_)
                        RES.append(records.paired(S, SEEN, T, analysis, a, b, est, lo, hi, post_date=POST))
                    else:
                        metrics.skip_boot_diff(ok_of[lab_name], gid, rng_bs, B_)

        pps = evaluate("primary", order)
        pairs("primary", "primary", cfg["directional_pairs"], pps, part == "directional")           # paired differences of the macro Bond-AUROC (same proteins, same resampled groups)
        for lab_name in (VW, VNC):
            if lab_name not in labs:                                                                    # netcleave_unseen_window exists for the C terminus only
                continue
            ppl = evaluate(lab_name, pub["comparison_models"])
            pairs(lab_name, lab_name, pub["comparison_pairs"], ppl, part == "published")
        evaluate(V6, cfg["study_exclusive_models"])
        wf = cfg["weights_for"] if part == "directional" else pub["weights_for"]
        for m in fitted:
            if m.name in wf:
                w = getattr(m, "W", getattr(m, "t", None))
                if w is not None:
                    WEIGHTS.append(dict(terminus=T, model=m.name, value=np.round(np.asarray(w, np.float64), 4).tolist()))
        log(T, "done")
    path = records.write_study(out_dir, S, dict(study=S, diag=DIAG, cells=RES, weights=WEIGHTS, strict=sinfo))
    log("written", path)
