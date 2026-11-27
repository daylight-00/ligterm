#!/usr/bin/env python3
"""Fit LigTerm-GB, the boosted release model, on all verified training units, and check it against a shipped model directory.

For each terminus (N, C): the training bonds of all verified units (unique observed terminal bonds, and six times as many other bonds of the same proteins; random stream [seed, 1, terminus]),
72 features (the 32 window tokens as categorical columns, the 20 residue counts of the interior-facing and of the outward-facing 16 residues), the number of boosting iterations chosen from
(50, 100, 200, 400, 800) by the pooled out-of-fold log-loss on three group-disjoint folds of the training rows (sequence-similarity grouping u30), one fit of 800 iterations, the first n_iter
trees exported to flat arrays. The model directory is written, reloaded, and its numpy scores are checked to equal the scores of the fitted scikit-learn model exactly on 200 random proteins.

With --compare <shipped LigTerm-GB directory> every array of gb.npz is compared with the shipped one (shape, dtype, values) and the training description with the shipped metadata (except the
creation time, the machine, the library versions and the hash of the units file, whose columns differ in the public datasets). Nothing is tuned and no performance is computed; the shipped weights are
never overwritten. CPU; about 6 minutes with 8 threads (the original fit set 8 BLAS and OpenMP threads, which are part of the numerical result).

usage: 41_train_ligterm_gb.py --data <datasets folder> --out <new model folder> [--compare <models/gb>] [--ligterm-root <folder with the ligterm package>] [--config FILE]
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import scipy
import sklearn

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "lib"))
from ligterm_repro import boosted, composition as c1, data, training  # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def train_features(tok, T, w=16):
    """(X float32, indices of the categorical columns): the residue token at each of the 2w positions (categorical), then the 20 residue counts of the interior-facing and of the outward-facing w residues."""
    cols = boosted.full_cols(T, w)
    parts = [tok[:, cols].astype(np.float32), c1.counts20(tok, c1.interior_cols(T, w)).astype(np.float32), c1.counts20(tok, boosted.outward_cols(T, w)).astype(np.float32)]
    return np.concatenate(parts, axis=1), list(range(len(cols)))


def fit_gb(X, y, fid, cat, grid=boosted.GRID):
    """the number of iterations by boosted.select_iter on the folds `fid` of the training rows, then one fit of the largest grid value; the fitted model is checked against the specified settings."""
    nstar, loss = boosted.select_iter(X, y, fid, cat, grid)
    assert nstar in grid, "the chosen number of iterations is not on the grid"
    model = boosted.new_model(cat, grid[-1]).fit(X, y)
    boosted.check_model(model, len(cat), X.shape[1], grid[-1])
    return model, int(nstar), loss


def arrays_equal(a, b):
    za, zb = np.load(a), np.load(b)
    if sorted(za.files) != sorted(zb.files):
        return False, f"different arrays: {sorted(set(za.files) ^ set(zb.files))[:4]}"
    for k in za.files:
        x, y = za[k], zb[k]
        if x.shape != y.shape or x.dtype != y.dtype or not np.array_equal(x, y):
            return False, f"array {k} differs"
    return True, f"{len(za.files)} arrays equal"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="datasets folder (units.tsv.gz, protein_sequences.json, sequence_groups.tsv.gz)")
    ap.add_argument("--out", required=True, help="new model folder (refused if it exists)")
    ap.add_argument("--compare", help="shipped LigTerm-GB directory to compare with")
    ap.add_argument("--ligterm-root", default=os.path.join(here, "..", "..", "src"), help="the folder that contains the ligterm package (default: src/ of the repository)")
    ap.add_argument("--config", default=os.path.join(here, "..", "configs", "ligterm_gb.json"))
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    sys.path.insert(0, a.ligterm_root)
    from ligterm.features import window_tokens
    from ligterm.gb import GBTermModel, LEN_HI, LEN_LO, export_hgb, gb_features, predict_raw

    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists; refusing to overwrite")
    t0 = time.time()

    def log(*x):
        print(f"[{time.time() - t0:6.0f}s]", *x, flush=True)

    SEED, NEG, PRIM, W = cfg["seed"], cfg["neg_ratio"], cfg["grouping"], cfg["window"]
    units, seqs, groups = (os.path.join(a.data, f) for f in ("units.tsv.gz", "protein_sequences.json", "sequence_groups.tsv.gz"))
    c = data.load_ctx(units, seqs, groups)
    rows = np.arange(c.nv)
    log(f"all verified units: {len(rows)}")
    h = hashlib.sha256()
    for arr in (c.vp[rows], c.vs[rows], c.ve[rows]):
        h.update(np.ascontiguousarray(arr, dtype=np.int64).tobytes())
    training_sha = h.hexdigest()
    fits = {}
    for k, T in enumerate("NC"):
        rng = np.random.default_rng([SEED, 1, k])
        P, B, y, bi = training.build_train(c, rows, T, rng, NEG)
        log(T, "training set", {kk: int(v) for kk, v in bi.items()}, "rows", len(y))
        tok = data.tokens(c, P, B)
        grp = c.G[PRIM][P]
        fid = c1.fold_ids(grp)
        boosted.assert_folds(fid, grp, y)
        X, cat = train_features(tok, T, W)
        assert X.shape[1] == 72 and len(cat) == 32
        t1 = time.time()
        model, nstar, loss = fit_gb(X, y, fid, cat)
        fits[T] = dict(model=model, nstar=nstar, loss=[float(v) for v in loss], rows=int(len(y)), n_pos=int(bi["n_pos"]), n_neg=int(bi["n_neg"]), fit_seconds=round(time.time() - t1, 1))
        log(T, f"fitted: n_iter {nstar} of {boosted.GRID}, at the edge of the grid: {nstar in (boosted.GRID[0], boosted.GRID[-1])}; {fits[T]['fit_seconds']} s")
        del X, tok
    L = np.array([int(c.ve[r] - c.vs[r] + 1) for r in rows])
    cnt = np.array([int((L == ell).sum()) for ell in range(LEN_LO, LEN_HI + 1)], np.float64)
    g = np.log((cnt + 1.0) / (cnt.sum() + len(cnt)))
    ex = {T: export_hgb(fits[T]["model"], fits[T]["nstar"]) for T in "NC"}
    meta = dict(release_artefact=True, model_name="LigTerm-GB",
                recipe="LigTerm-GB: 72 features (32 residue tokens as categorical columns, 20 residue counts of the interior-facing and of the outward-facing 16 residues); the HistGradientBoosting settings of the development protocol (see hgb); the number of iterations chosen from (50, 100, 200, 400, 800) on three group-disjoint folds (sequence-similarity grouping u30), one fit of 800 iterations, the first n_iter trees kept",
                hgb=boosted.HGB, population="all", verified_units_in_database=int(c.nv), training_units=int(len(rows)), training_rows_sha256=training_sha, seeds=f"[{SEED}, 1, terminus index]",
                fits={T: {k_: v for k_, v in fits[T].items() if k_ != "model"} for T in "NC"}, units_sha256=sha(units), seqs_sha256=sha(seqs), groups_sha256=sha(groups),
                length_prior=f"log of add-one smoothed counts of verified ligand lengths {LEN_LO}..{LEN_HI} of the training units", python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                created=time.strftime("%Y-%m-%d %H:%M:%S"), schema_version=1,
                note="scores are logits that rank bonds within a protein; performance estimates of the protocol (development cohort and historical test cohort, refits on held-out or reserved studies) describe the recipe, not this all-data fit, whose measured performance is that on the external test cohort (see the model card); a measure of observed ligand-terminal propensity, not of proteolytic cleavage; the class II identity of the lung list of PMID 35051231 is unresolved (see the model card)")
    model = GBTermModel(ex["N"], ex["C"], g, LEN_LO, LEN_HI, meta)
    model.save(a.out)
    back = GBTermModel.load(a.out)
    assert all(np.array_equal(model.ex[T][k_], back.ex[T][k_]) for T in "NC" for k_ in model.ex[T]), "the model directory does not reload to the fitted arrays"
    # the numpy scores of the written model equal the scores of the fitted scikit-learn model exactly, on random real proteins
    rng1 = np.random.default_rng([SEED, 1, 99])
    cand = np.array([i for i in range(c.NP) if 50 <= int(c.L[i]) <= 3000])
    sel = np.sort(rng1.choice(cand, size=cfg["check_proteins"], replace=False))
    p_all = np.concatenate([np.full(int(c.L[i]) - 1, i, np.int64) for i in sel])
    i_all = np.concatenate([np.arange(1, int(c.L[i]), dtype=np.int64) for i in sel])
    tk = data.tokens(c, p_all, i_all)
    for T in "NC":
        Xf, _ = train_features(tk, T, W)
        ref = boosted.staged_scores(fits[T]["model"], Xf, [fits[T]["nstar"]], chunk=len(Xf))[fits[T]["nstar"]]
        mine = predict_raw(back.ex[T], gb_features(np.concatenate([window_tokens(c.seqs[c.names[i]], W) for i in sel]), T))
        assert np.array_equal(ref, mine), f"{T}: the numpy scores of the written model differ from the scikit-learn model"
    log("model directory written, reloaded and checked on", len(sel), "random proteins:", a.out, {T: model.n_trees[T] for T in "NC"})

    ok = True
    if a.compare:
        res = {"gb.npz": arrays_equal(os.path.join(a.out, "gb.npz"), os.path.join(a.compare, "gb.npz"))}
        ship = json.load(open(os.path.join(a.compare, "meta.json")))
        mine_meta = json.load(open(os.path.join(a.out, "meta.json")))
        skip = {"created", "cpu", "host", "commit", "python", "numpy", "scipy", "sklearn", "units_sha256"}
        for key in sorted((set(ship) | set(mine_meta)) - skip):
            if key == "fits":
                same = all({k_: v for k_, v in ship["fits"][T].items() if k_ != "fit_seconds"} == {k_: v for k_, v in mine_meta["fits"][T].items() if k_ != "fit_seconds"} for T in "NC")
            else:
                same = ship.get(key) == mine_meta.get(key)
            if not same:
                res[f"meta: {key}"] = (False, "differs")
        res["meta"] = res.get("meta", (not any(k.startswith("meta:") for k in res), "the training description is equal (fit seconds, creation time, machine, library versions and units-file hash not compared)"))
        for k, (good, text) in res.items():
            print(("PASS " if good else "FAIL ") + f"{k}: {text}")
            ok = ok and good
        print("EQUAL to the shipped model" if ok else "NOT EQUAL to the shipped model")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
