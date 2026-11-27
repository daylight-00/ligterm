"""The window MLP of LigTerm: the one-hot window head and its training (torch).

A bond has the window of residues i-15 .. i+16 (positions P16..P1, P1'..P16'; 32 positions). The head reads the one-hot residue letters (20 letters; a non-standard letter and a
position outside the protein are all-zero): a per-residue projection 20 -> 64 shared over the positions (GELU), the flattened 32 x 64 values with dropout 0.2, a dense layer 2048 -> 128
(GELU, dropout 0.1) and a dense layer 128 -> 1; 263,745 parameters. The model of one terminus is fitted on the training bonds of that terminus (rows with fold 0, 1, 2 from group-disjoint
folds). Selection rule: the rows of fold 0 are held out, a fresh head is trained on the other two for at most 40 epochs (AdamW, 1,024 rows per step, learning rate 1e-3 with a linear warm-up
over 100 steps, weight decay 0.01 on the weight matrices only, gradient clipping at 1, stop after 3 epochs without a new best hold-out loss), E* is the first epoch with the lowest hold-out
loss, and a fresh head is trained on all rows for E* epochs. Nothing else is selected. The numpy forward pass of a fitted head is ligterm.mlp.MLPHead.
The training is not bit-for-bit reproducible across devices (the heads of the release were fitted on a GPU); on one device and one software version it is reproducible from its seed.
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AA = "ACDEFGHIKLMNPQRSTVWY"
PAD = 20
W_SIDE = 16
NWIN = 2 * W_SIDE
D_PROJ = 64
D_MIX = 128
P_DROP_IN = 0.2
P_DROP_HID = 0.1
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
WARMUP = 100
BATCH_ROWS = 1024
E_CAP = 40
PATIENCE = 3
SEED0 = 20261118
LUT = np.full(256, PAD, np.int8)
for _i, _a in enumerate(AA):
    LUT[ord(_a)] = _i


def settings():
    """the constants of the training as one dictionary (stored in the info file of a fit)"""
    return dict(window_side=W_SIDE, d_proj=D_PROJ, d_mix=D_MIX, p_drop_in=P_DROP_IN, p_drop_hid=P_DROP_HID, lr_head=LR_HEAD, weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP, warmup=WARMUP,
                batch_rows=BATCH_ROWS, epoch_cap=E_CAP, patience=PATIENCE, seed0=SEED0)


def run_seed(study, ti):
    """the seed of one model: a function of [20261118, 0, study code, terminus (0 N, 1 C)] only; the selection run uses it and the final run seed + 7. The release heads use study = 910000 + seed index."""
    return int(np.random.default_rng([SEED0, 0, int(study), int(ti)]).integers(1, 2 ** 31 - 100))


def device_info(dev):
    if dev.type != "cuda":
        return dict(device="cpu", torch=torch.__version__)
    pr = torch.cuda.get_device_properties(dev)
    return dict(device=pr.name, capability=f"{pr.major}.{pr.minor}", memory_mib=int(pr.total_memory // (1 << 20)), torch=torch.__version__, cuda=torch.version.cuda)


def residue_arrays(seqs):
    """flat residue tokens (int8: 0..19, 20 for every other letter), the start offsets of the proteins in the flat array (length n + 1) and the lengths, in the order of `seqs`"""
    L = np.array([len(s) for s in seqs], np.int64)
    off = np.zeros(len(seqs) + 1, np.int64)
    off[1:] = np.cumsum(L)
    flat = LUT[np.frombuffer("".join(seqs).encode("ascii"), np.uint8)]
    return flat, off, L


def window_index(prot, bond, L, off):
    """(n, 32) flat residue index (clamped into the protein) and validity of the window of every row; prot, bond torch long tensors on one device, L and off torch long tensors (per protein)"""
    j = torch.arange(-(W_SIDE - 1), W_SIDE + 1, device=prot.device)
    r = (bond - 1).unsqueeze(1) + j.unsqueeze(0)
    Lp = L[prot].unsqueeze(1)
    valid = (r >= 0) & (r < Lp)
    idx = off[prot].unsqueeze(1) + torch.minimum(r.clamp(min=0), Lp - 1)
    return idx, valid


def onehot_window(tok_flat, idx, valid):
    """(n, 32, 20) float32 one-hot of the residue letters; non-standard letters and positions outside the protein are all-zero"""
    t = tok_flat[idx].long()
    live = valid & (t < PAD)
    return F.one_hot(t.clamp(max=PAD - 1), PAD).float() * live.unsqueeze(-1), live


class WindowHead(nn.Module):
    """a per-residue projection shared over the 32 positions, then a position-specific mixing of the flattened window"""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(PAD, D_PROJ)
        self.mix1 = nn.Linear(NWIN * D_PROJ, D_MIX)
        self.mix2 = nn.Linear(D_MIX, 1)
        self.drop_in = nn.Dropout(P_DROP_IN)
        self.drop_hid = nn.Dropout(P_DROP_HID)

    def forward(self, x, live):
        z = F.gelu(self.proj(x)) * live.unsqueeze(-1).to(x.dtype)
        h = self.drop_in(z.flatten(1))
        h = self.drop_hid(F.gelu(self.mix1(h)))
        return self.mix2(h).squeeze(-1)


def head_arrays(head):
    """the six arrays of a fitted head in the layout of ligterm.mlp.MLPHead"""
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in head.state_dict().items()}
    return {"proj_w": sd["proj.weight"], "proj_b": sd["proj.bias"], "mix1_w": sd["mix1.weight"], "mix1_b": sd["mix1.bias"], "mix2_w": sd["mix2.weight"], "mix2_b": sd["mix2.bias"]}


class Rows:
    """the training rows of one terminus: P (protein), B (bond), y (0/1), fold (0..2); on the device"""

    def __init__(self, P, B, y, fold, device):
        self.n = len(P)
        self.np_P, self.np_B = np.asarray(P, np.int64), np.asarray(B, np.int64)
        self.P = torch.as_tensor(self.np_P, device=device)
        self.B = torch.as_tensor(self.np_B, device=device)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=device)
        self.fold = np.asarray(fold)


class Ctx:
    """the proteins of the rows: tensors of the lengths, offsets and residue tokens on the device"""

    def __init__(self, seqs, device):
        self.flat, self.off_np, self.L_np = residue_arrays(seqs)
        self.device = device
        self.tok = torch.from_numpy(self.flat).to(device)
        self.off = torch.from_numpy(self.off_np[:-1]).to(device)
        self.L = torch.from_numpy(self.L_np).to(device)


def rows_logits(head, cx, prot, bond):
    idx, valid = window_index(prot, bond, cx.L, cx.off)
    x, live = onehot_window(cx.tok, idx, valid)
    return head(x, live)


def _loss(head, cx, rows, ids):
    """mean binary cross-entropy of the rows `ids` (numpy array)"""
    ids_t = torch.as_tensor(ids, dtype=torch.long, device=cx.device)
    return F.binary_cross_entropy_with_logits(rows_logits(head, cx, rows.P[ids_t], rows.B[ids_t]), rows.y[ids_t])


def _make_opt(head):
    groups = [dict(params=[p for p in head.parameters() if p.ndim > 1], lr=LR_HEAD, weight_decay=WEIGHT_DECAY),
              dict(params=[p for p in head.parameters() if p.ndim <= 1], lr=LR_HEAD, weight_decay=0.0)]
    for g in groups:
        g["base_lr"] = g["lr"]
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


def _set_lr(opt, step):
    f = min(1.0, (step + 1) / WARMUP)
    for g in opt.param_groups:
        g["lr"] = g["base_lr"] * f


@torch.no_grad()
def _hold_loss(head, cx, rows, hold_ids):
    head.eval()
    tot = 0.0
    for a in range(0, len(hold_ids), 8192):
        ids = hold_ids[a:a + 8192]
        tot += float(_loss(head, cx, rows, ids)) * len(ids)
    head.train()
    return tot / len(hold_ids)


class Fit:
    """one fitted head and the record of its fit (selection trace, epochs, times)"""

    def __init__(self, head, info):
        self.head, self.info = head, info


def _train_run(cx, rows, train_ids, hold_ids, e_max, seed, patience, log, tag):
    """one training run: a fresh head, up to e_max epochs over train_ids; with hold_ids the hold-out loss after every epoch and the early stop; returns (head, trace)"""
    dev = cx.device
    torch.manual_seed(seed)
    head = WindowHead().to(dev)
    opt = _make_opt(head)
    clip_params = [p for g in opt.param_groups for p in g["params"]]
    trace = dict(train_loss=[], hold_loss=[], seconds=[])
    best, best_e, step = float("inf"), 0, 0
    head.train()
    for e in range(1, e_max + 1):
        t0 = time.time()
        rng = np.random.default_rng([SEED0, seed, e])
        tot = 0.0
        perm = train_ids[rng.permutation(len(train_ids))]
        for a in range(0, len(perm), BATCH_ROWS):
            ids = perm[a:a + BATCH_ROWS]
            loss = _loss(head, cx, rows, ids)
            _set_lr(opt, step)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clip_params, GRAD_CLIP)
            opt.step()
            step += 1
            tot += float(loss.detach()) * len(ids)
        assert math.isfinite(tot), f"{tag}: the training loss is not finite in epoch {e} (a diverged run is never scored)"
        trace["train_loss"].append(tot / len(train_ids))
        if hold_ids is not None:
            hl = _hold_loss(head, cx, rows, hold_ids)
            assert math.isfinite(hl), f"{tag}: the hold-out loss is not finite in epoch {e} (a diverged run is never scored)"
            trace["hold_loss"].append(hl)
            if hl < best - 1e-12:
                best, best_e = hl, e
        trace["seconds"].append(time.time() - t0)
        if log:
            log(f"{tag} epoch {e}: train {trace['train_loss'][-1]:.5f}" + (f" hold-out {trace['hold_loss'][-1]:.5f}" if hold_ids is not None else "") + f" ({trace['seconds'][-1]:.1f}s, {step} steps)")
        if hold_ids is not None and e - best_e >= patience:
            break
    trace["best_epoch"], trace["best_hold_loss"], trace["steps"] = best_e, (best if hold_ids is not None else None), step
    if dev.type == "cuda":
        trace["peak_memory_gb"] = torch.cuda.max_memory_allocated(dev) / 1e9
    return head, trace


def fit_head(cx, rows, seed, log=None, tag="", e_cap=None, patience=PATIENCE):
    """the selection rule: fold 0 held out, a fresh head on folds 1 and 2 for at most e_cap epochs (early stop after `patience` epochs without a new best), E* = the first epoch of the lowest
    hold-out loss, then a fresh head on all rows for E* epochs"""
    e_cap = e_cap or E_CAP
    t0 = time.time()
    hold = np.flatnonzero(rows.fold == 0)
    train = np.flatnonzero(rows.fold != 0)
    assert len(hold) > 0 and len(train) > 0, "a fold is empty"
    _, tr1 = _train_run(cx, rows, train, hold, e_cap, seed, patience, log, tag + " select")
    assert tr1["best_epoch"] >= 1, f"{tag}: no epoch of the selection run has a finite hold-out loss below the initial value (E* is undefined)"
    e_star = tr1["best_epoch"]
    head2, tr2 = _train_run(cx, rows, np.arange(rows.n), None, e_star, seed + 7, patience, log, tag + " final")
    info = dict(e_star=int(e_star), e_cap=int(e_cap), at_cap=bool(tr1["best_epoch"] >= e_cap), select=tr1, final=tr2, seconds=time.time() - t0, n_rows=int(rows.n), n_hold=int(len(hold)), n_train_select=int(len(train)))
    return Fit(head2, info)


def head_from_arrays(w):
    """a torch WindowHead (eval mode) with the six arrays of a fitted head (the layout of head_arrays / ligterm.mlp.MLPHead)"""
    head = WindowHead()
    head.load_state_dict({"proj.weight": torch.from_numpy(np.asarray(w["proj_w"], np.float32)), "proj.bias": torch.from_numpy(np.asarray(w["proj_b"], np.float32)),
                          "mix1.weight": torch.from_numpy(np.asarray(w["mix1_w"], np.float32)), "mix1.bias": torch.from_numpy(np.asarray(w["mix1_b"], np.float32)),
                          "mix2.weight": torch.from_numpy(np.asarray(w["mix2_w"], np.float32)), "mix2.bias": torch.from_numpy(np.asarray(w["mix2_b"], np.float32))})
    return head.eval()


@torch.no_grad()
def logits_from_tokens(head, tok):
    """float32 logits of window token rows (n, 32; 0..19 residues, 20 = pad or non-standard) from a torch head in eval mode"""
    t = torch.as_tensor(np.asarray(tok)).long()
    live = t < PAD
    x = F.one_hot(t.clamp(max=PAD - 1), PAD).float() * live.unsqueeze(-1)
    return head(x, live).float().cpu().numpy()
