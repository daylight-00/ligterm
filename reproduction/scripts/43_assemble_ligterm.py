#!/usr/bin/env python3
"""Assemble the LigTerm model (family 'ens') from LigTerm-GB and the all-data window-MLP heads, and check it against a shipped model directory.

LigTerm scores a bond with (LigTerm-GB logit + mean over the seeds of the window-MLP logit) / 2 per terminus. The inputs are the LigTerm-GB model directory (models/gb)
and, for each seed and terminus, the fitted head (seed<k>_<N|C>.npz) with its info file (seed<k>_info.json) from the data package. Guards, before anything is written: the info
files describe release fits (not smoke runs); the number of epochs E* is below the cap; the numpy head reproduced the torch head within 1e-4 at fit time; the rows of the fit are
the rows of the LigTerm-GB release fit; the seeds differ. After writing, the directory is reloaded and scores random windows exactly like the in-memory model, and its LigTerm-GB
component scores exactly like the LigTerm-GB model.

With --compare <shipped LigTerm directory> the assembled model is checked against the shipped one: every array of mlp.npz and of gb/gb.npz equal (shape, dtype, values), the scores of
both on random windows equal, the metadata equal except the creation time. No label is used and no performance is computed. numpy only; nothing is fitted.

usage: 43_assemble_ligterm.py --gb <models/gb> --heads <folder with the seed files> --out <new model folder> [--seeds 0,1,2] [--compare <models/ens>] [--ligterm-root <folder with the ligterm package>]
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


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
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gb", required=True)
    ap.add_argument("--heads", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--compare")
    ap.add_argument("--ligterm-root", default=os.path.join(here, "..", "..", "src"), help="the folder that contains the ligterm package (default: src/ of the repository)")
    a = ap.parse_args()
    sys.path.insert(0, a.ligterm_root)
    from ligterm.ens import EnsTermModel
    from ligterm.gb import GBTermModel
    from ligterm.mlp import MLPHead, WEIGHT_KEYS

    seeds = [int(x) for x in a.seeds.split(",")]
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists; refusing to overwrite")
    card = json.load(open(os.path.join(a.gb, "meta.json")))
    assert card["family"] == "gb" and card["release_artefact"] is True and card["population"] == "all", "the LigTerm-GB directory is not the all-data release model"
    infos = {k: json.load(open(os.path.join(a.heads, f"seed{k}_info.json"))) for k in seeds}
    for k, i in infos.items():
        assert i["release_artefact"] is True and i["smoke_rows"] is None, f"seed {k}: not a release fit"
        assert i["training_rows_sha256"] == card["training_rows_sha256"] and i["training_units"] == card["training_units"], f"seed {k}: the rows of the fit are not the rows of the LigTerm-GB release fit"
        for T in "NC":
            t = i["termini"][T]
            assert t["at_cap"] is False, f"seed {k} {T}: E* is at the epoch cap"
            assert t["numpy_vs_torch_max_abs_logit_difference"] < 1e-4, f"seed {k} {T}: the numpy head differs from the torch head"
            assert t["n_rows"] == card["fits"][T]["rows"], f"seed {k} {T}: the number of rows differs from the LigTerm-GB release fit"
    assert len({infos[k]["termini"][T]["seed"] for k in seeds for T in "NC"}) == 2 * len(seeds), "two fitted heads share a seed"
    print("guards passed: release fits on the LigTerm-GB rows; E* below the cap; numpy head = torch head; seeds distinct")

    def load_head(k, T):
        z = np.load(os.path.join(a.heads, f"seed{k}_{T}.npz"))
        assert set(z.files) == set(WEIGHT_KEYS), f"seed {k} {T}: unexpected arrays {z.files}"
        return MLPHead({key: np.asarray(z[key]) for key in WEIGHT_KEYS})

    heads = {T: [load_head(k, T) for k in seeds] for T in "NC"}
    gb = GBTermModel.load(a.gb)
    first = infos[seeds[0]]
    meta = dict(release_artefact=True, population="all", training_units=card["training_units"], training_rows_sha256=card["training_rows_sha256"], created=time.strftime("%Y-%m-%d %H:%M:%S"),
                gb_npz_sha256=sha(os.path.join(a.gb, "gb.npz")), gb_meta_sha256=sha(os.path.join(a.gb, "meta.json")),
                mlp_recipe=first["recipe"], mlp_seed_indices=seeds, mlp_fit_seeds={T: [infos[k]["termini"][T]["seed"] for k in seeds] for T in "NC"},
                mlp_e_star={T: [infos[k]["termini"][T]["e_star"] for k in seeds] for T in "NC"},
                mlp_numpy_vs_torch_max_abs_logit_difference={T: [infos[k]["termini"][T]["numpy_vs_torch_max_abs_logit_difference"] for k in seeds] for T in "NC"}, mlp_fit_device=first["device"],
                model_name="LigTerm", schema_version=1,
                note="scores are logits that rank bonds within a protein; performance estimates of the protocol (development cohort, refits on held-out studies) describe the recipe, not this all-data fit, whose measured performance is that on the external test cohort (see the model card); a measure of observed ligand-terminal propensity, not of proteolytic cleavage")
    m = EnsTermModel(gb, heads, meta)
    m.save(a.out)
    # ---- reload and compare with the in-memory model, and the LigTerm-GB component with LigTerm-GB
    m2 = EnsTermModel.load(a.out)
    rng = np.random.default_rng([20261120, 31])
    seqs = ["".join("ACDEFGHIKLMNPQRSTVWY"[i] for i in rng.integers(0, 20, size=n)) for n in (45, 120, 400)]
    x, y = m.bond_scores_many(seqs), m2.bond_scores_many(seqs)
    assert all(np.array_equal(p[0], q[0]) and np.array_equal(p[1], q[1]) for p, q in zip(x, y)), "the reloaded model differs from the in-memory model"
    from ligterm.features import window_tokens
    from ligterm.gb import W, gb_features, predict_raw
    tok = window_tokens(seqs[2], W)
    for T in "NC":
        assert np.array_equal(np.asarray(predict_raw(gb.ex[T], gb_features(tok, T)), np.float64), np.asarray(predict_raw(m2.gb.ex[T], gb_features(tok, T)), np.float64)), f"{T}: the LigTerm-GB component of the written directory differs from LigTerm-GB"
    print("written", a.out, "| reload equal; the LigTerm-GB component equals LigTerm-GB; seeds", seeds)

    ok = True
    if a.compare:
        res = {}
        res["mlp.npz"] = arrays_equal(os.path.join(a.out, "mlp.npz"), os.path.join(a.compare, "mlp.npz"))
        res["gb/gb.npz"] = arrays_equal(os.path.join(a.out, "gb", "gb.npz"), os.path.join(a.compare, "gb", "gb.npz"))
        ship = EnsTermModel.load(a.compare)
        more = ["".join("ACDEFGHIKLMNPQRSTVWY"[i] for i in rng.integers(0, 20, size=n)) for n in (30, 77, 250, 1200)]
        s1, s2 = m2.bond_scores_many(more), ship.bond_scores_many(more)
        res["scores"] = (all(np.array_equal(p[0], q[0]) and np.array_equal(p[1], q[1]) for p, q in zip(s1, s2)), f"{sum(len(p[0]) for p in s1)} bonds of {len(more)} random proteins, both termini")
        ma, mb = dict(m2.meta), dict(ship.meta)
        ma.pop("created", None), mb.pop("created", None)
        res["meta"] = (ma == mb, "equal except the creation time" if ma == mb else f"differs in {sorted(k for k in set(ma) | set(mb) if ma.get(k) != mb.get(k))}")
        res["file hashes (information)"] = (True, "mlp.npz " + ("same bytes" if sha(os.path.join(a.out, 'mlp.npz')) == sha(os.path.join(a.compare, 'mlp.npz')) else "other bytes") + "; gb.npz " + ("same bytes" if sha(os.path.join(a.out, 'gb', 'gb.npz')) == sha(os.path.join(a.compare, 'gb', 'gb.npz')) else "other bytes"))
        for k, (good, text) in res.items():
            print(("PASS " if good else "FAIL ") + f"{k}: {text}")
            ok = ok and good
        print("EQUAL to the shipped model" if ok else "NOT EQUAL to the shipped model")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
