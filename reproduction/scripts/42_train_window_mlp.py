#!/usr/bin/env python3
"""The window MLP of LigTerm: fit the all-data heads (fit) and check fitted heads (check).

fit    For each seed and terminus, the head of ligterm_repro.window_mlp is fitted on the training bonds of the LigTerm-GB release fit (the stored rows file): fold 0 held out, E* from the
       hold-out loss, a fresh head on all rows for E* epochs. The seed of a model is a function of the seed index and the terminus only. Guards: the rows were built for this cohort file; the window tokens
       of this step equal the stored sample; every fitted head is checked against its torch twin (the numpy forward pass of ligterm.mlp must agree within 1e-4). Writes seed<k>_<N|C>.npz (the six arrays
       of the head) and seed<k>_info.json. The heads of the release were fitted on one GPU; the same fit on another device gives a different head (training is reproducible from its seed on one device and
       one software version, not across devices), so a refit is not expected to equal the shipped heads: use `check` and 43_assemble_ligterm.py for the shipped ones. --smoke ROWS fits on a random
       subset of ROWS rows and at most 2 epochs, to try the pipeline; its output is flagged and never a release artefact.
check  The stored heads are applied, in numpy (ligterm.mlp) and in torch on the CPU, to the windows of random proteins of the datasets; the two must agree within 1e-4 (the rule of the fit).
       This checks the inference of the shipped heads from their stored weights; nothing is trained.

usage: 42_train_window_mlp.py fit   --rows <release_rows.npz> --cohort <cohort.tsv.gz> --out <folder> [--seeds 0,1,2] [--device cuda|cpu] [--smoke ROWS] [--ligterm-root DIR]
       42_train_window_mlp.py check --heads <folder with seed<k>_<T>.npz> --data <datasets folder> [--seeds 0,1,2] [--proteins 300] [--ligterm-root DIR]
"""
import argparse
import gzip
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "lib"))

RECIPE = "one-hot window MLP (a per-residue projection shared over the positions, two dense layers), AdamW, 1,024 rows per step, 40 epochs at most, patience 3, E* then a fresh fit on all rows"
WEIGHT_KEYS = ("proj_w", "proj_b", "mix1_w", "mix1_b", "mix2_w", "mix2_b")


def read_cohort_sequences(path):
    """the sequences of the cohort proteins in the order of the file (columns: index, accession, length, eligible, sequence); the rows of the stored files index this order"""
    seqs = []
    with gzip.open(path, "rt", encoding="utf-8", newline="\n") as fh:
        head = fh.readline().rstrip("\n").split("\t")
        assert len(head) == 5 and head[4] == "seq", f"unexpected cohort header {head}"
        for i, ln in enumerate(fh):
            f = ln.rstrip("\n").split("\t")
            assert len(f) == 5 and int(f[0]) == i and int(f[2]) == len(f[4]), f"cohort row {i} is malformed or out of order"
            seqs.append(f[4])
    return seqs


def cmd_fit(a):
    import torch
    from ligterm.mlp import MLPHead
    from ligterm_repro import window_mlp as W

    seeds = [int(x) for x in a.seeds.split(",")]
    t0 = time.time()

    def log(*x):
        print(f"[{time.time() - t0:6.0f}s]", *x, flush=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(a.out, exist_ok=True)
    suffix = "_SMOKE" if a.smoke else ""
    seqs = read_cohort_sequences(a.cohort)
    z = np.load(a.rows)
    coh_sha = hashlib.sha256(open(a.cohort, "rb").read()).hexdigest()
    assert str(z["study"]) == "RELEASE" and str(z["cohort_sha256"]) == coh_sha, "the rows are not the release rows or were built for another cohort file"
    cx = W.Ctx(seqs, dev)
    devinfo = dict(W.device_info(dev), python=platform.python_version(), numpy=np.__version__)
    log("device:", devinfo, "| seeds", seeds, "| smoke", a.smoke or None)
    for T in "NC":                                                         # guard: the window tokens of this step equal the stored sample of training rows
        idx = z[f"{T}_tok_idx"]
        p = torch.as_tensor(z[f"{T}_P"][idx], dtype=torch.long, device=dev)
        b = torch.as_tensor(z[f"{T}_B"][idx], dtype=torch.long, device=dev)
        ii, valid = W.window_index(p, b, cx.L, cx.off)
        _, live = W.onehot_window(cx.tok, ii, valid)
        back = torch.where(live, cx.tok[ii].long(), torch.full_like(ii, W.PAD)).cpu().numpy()
        assert np.array_equal(back, z[f"{T}_tok"]), f"the window tokens of this step differ from the stored sample ({T})"
    log("guard passed: the window tokens equal the stored samples of both termini")
    for K in seeds:
        if any(os.path.exists(os.path.join(a.out, f"seed{K}_{T}{suffix}.npz")) for T in "NC"):
            log(f"seed {K}: output exists, skipped")
            continue
        info = dict(seed_index=K, smoke_rows=a.smoke or None, release_artefact=not a.smoke, recipe=RECIPE, settings=W.settings(), device=devinfo,
                    training_rows_sha256=str(z["unit_hash"]), training_units=int(z["n_units"]), termini={})
        for ti, T in enumerate("NC"):
            Pr, Br, y, fid = z[f"{T}_P"], z[f"{T}_B"], z[f"{T}_y"], z[f"{T}_fold"]
            if a.smoke:
                sel = np.sort(np.random.default_rng([20261118, 5, K, ti]).choice(len(y), size=min(a.smoke, len(y)), replace=False))
                Pr, Br, y, fid = Pr[sel], Br[sel], y[sel], fid[sel]
            rw = W.Rows(Pr, Br, y, fid, dev)
            seed = W.run_seed(910000 + K, ti)
            fit = W.fit_head(cx, rw, seed, log=log, tag=f"seed {K} {T}", e_cap=(2 if a.smoke else None))
            w = W.head_arrays(fit.head)
            ix = np.sort(np.random.default_rng([20261118, 6, K, ti]).choice(len(y), size=min(20000, len(y)), replace=False))
            p = torch.as_tensor(Pr[ix], dtype=torch.long, device=dev)
            b = torch.as_tensor(Br[ix], dtype=torch.long, device=dev)
            ii, valid = W.window_index(p, b, cx.L, cx.off)
            _, live = W.onehot_window(cx.tok, ii, valid)
            tok = torch.where(live, cx.tok[ii].long(), torch.full_like(ii, W.PAD)).cpu().numpy()
            fit.head.eval()
            with torch.no_grad():
                ref = W.rows_logits(fit.head, cx, p, b).float().cpu().numpy()
            dmax = float(np.abs(MLPHead(w).logits(tok) - ref).max())
            assert dmax < 1e-4, f"seed {K} {T}: the numpy head differs from the torch head by {dmax}"
            np.savez_compressed(os.path.join(a.out, f"seed{K}_{T}{suffix}.npz"), **w)
            fi = dict(fit.info)
            info["termini"][T] = dict(e_star=fi["e_star"], at_cap=fi["at_cap"], n_rows=fi["n_rows"], n_hold=fi["n_hold"], seconds=round(fi["seconds"], 1), seed=int(seed), numpy_vs_torch_max_abs_logit_difference=dmax,
                                      select_best_hold_loss=fi["select"]["best_hold_loss"], final_train_loss=fi["final"]["train_loss"][-1],
                                      select_trace=dict(train_loss=fi["select"]["train_loss"], hold_loss=fi["select"]["hold_loss"]))
            log(f"seed {K} {T}: E* {fi['e_star']} (cap {fi['e_cap']}, at cap {fi['at_cap']}), {fi['seconds']:.0f} s; numpy head against torch on {len(ix)} rows: {dmax:.2e}")
            del fit, rw
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        with open(os.path.join(a.out, f"seed{K}_info{suffix}.json"), "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=1, default=float)
    log("done")


def cmd_check(a):
    import torch
    from ligterm.features import window_tokens
    from ligterm.mlp import MLPHead
    from ligterm_repro import window_mlp as W

    torch.set_num_threads(2)
    seeds = [int(x) for x in a.seeds.split(",")]
    seqs = json.load(open(os.path.join(a.data, "protein_sequences.json")))
    accs = sorted(seqs)
    rng = np.random.default_rng([20261120, 42])
    pick = [accs[i] for i in sorted(rng.choice(len(accs), size=a.proteins, replace=False))]
    tok = np.concatenate([window_tokens(seqs[acc], 16) for acc in pick])
    print(f"{len(pick)} random proteins, {len(tok)} windows")
    worst, ok = 0.0, True
    for K in seeds:
        for T in "NC":
            z = np.load(os.path.join(a.heads, f"seed{K}_{T}.npz"))
            assert set(z.files) == set(WEIGHT_KEYS), f"seed {K} {T}: unexpected arrays {z.files}"
            w = {k: np.asarray(z[k]) for k in WEIGHT_KEYS}
            d = float(np.abs(MLPHead(w).logits(tok) - W.logits_from_tokens(W.head_from_arrays(w), tok)).max())
            worst = max(worst, d)
            good = d < 1e-4
            ok = ok and good
            print(("PASS " if good else "FAIL ") + f"seed {K} {T}: numpy head against torch head, largest difference {d:.2e}")
    print(f"largest difference over all heads {worst:.2e} (rule: < 1e-4)")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--rows", required=True)
    f.add_argument("--cohort", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--seeds", default="0,1,2")
    f.add_argument("--device")
    f.add_argument("--smoke", type=int, default=0)
    c = sub.add_parser("check")
    c.add_argument("--heads", required=True)
    c.add_argument("--data", required=True)
    c.add_argument("--seeds", default="0,1,2")
    c.add_argument("--proteins", type=int, default=300)
    for s in (f, c):
        s.add_argument("--ligterm-root", default=os.path.join(here, "..", "..", "src"), help="the folder that contains the ligterm package (default: src/ of the repository)")
    a = ap.parse_args()
    sys.path.insert(0, a.ligterm_root)
    {"fit": cmd_fit, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
