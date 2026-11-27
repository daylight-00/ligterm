"""LigTerm, the ensemble ('ens') of the scorer: per terminus, score = (LigTerm-GB logit + mean over the seeds of the window-MLP logit) / 2, a fixed scale of 1 and equal weights (nothing is estimated from the data scored).
A model FAMILY next to the linear reference model and LigTerm-GB; the command line (`cli.py`) reads it from a model directory whose meta.json says family 'ens'. Like every score of this scorer the result is a logit that ranks bonds and candidate peptides WITHIN a protein; the score of a bond depends on its 32-residue window only, never on the other proteins submitted with it
(the scale is constant, so the property of `docs/PREDICTOR_SPEC.md` section 6 holds).
Model directory: `meta.json` (family 'ens', the seeds, the training description), `gb/` (the LigTerm-GB directory: gb.npz and meta.json), `mlp.npz` (for each seed k and terminus T the six arrays of the head: s{k}_{T}_{proj_w, proj_b, mix1_w, mix1_b, mix2_w, mix2_b}).
The log length prior and the length range are those of the LigTerm-GB component. numpy only."""
import json
import os

import numpy as np

from .features import window_tokens
from .gb import GBTermModel, W, gb_features, predict_raw
from .mlp import MLPHead, WEIGHT_KEYS


class EnsTermModel:
    family = "ens"

    def __init__(self, gb, heads, meta=None):
        if not isinstance(gb, GBTermModel):
            raise ValueError("the LigTerm-GB component must be a GBTermModel")
        if set(heads) != {"N", "C"} or not heads["N"] or len(heads["N"]) != len(heads["C"]):
            raise ValueError("heads must hold the same, non-zero number of seeds for N and C")
        self.gb, self.heads, self.meta = gb, {T: list(heads[T]) for T in "NC"}, dict(meta or {})
        self.w, self.len_lo, self.len_hi, self.len_logp = W, gb.len_lo, gb.len_hi, gb.len_logp

    @property
    def n_seeds(self):
        return len(self.heads["N"])

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.gb.save(os.path.join(directory, "gb"))
        arrs = {f"s{k}_{T}_{key}": h.w[key] for T in "NC" for k, h in enumerate(self.heads[T]) for key in WEIGHT_KEYS}
        np.savez_compressed(os.path.join(directory, "mlp.npz"), **arrs)
        meta = dict(self.meta, family=self.family, window=W, mlp_seeds=self.n_seeds, scale=1.0, combination="(GB + mean of the MLP seeds) / 2 per terminus")
        with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1, sort_keys=True)

    @classmethod
    def load(cls, directory):
        with open(os.path.join(directory, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("family") != "ens":
            raise ValueError(f"{directory} is not an ensemble model directory (family {meta.get('family')!r})")
        z = np.load(os.path.join(directory, "mlp.npz"))
        K = int(meta["mlp_seeds"])
        heads = {T: [MLPHead({key: z[f"s{k}_{T}_{key}"] for key in WEIGHT_KEYS}) for k in range(K)] for T in "NC"}
        return cls(GBTermModel.load(os.path.join(directory, "gb")), heads, meta)

    def bond_scores(self, seq):
        return self.bond_scores_many([seq])[0]

    def bond_scores_many(self, seqs):
        """[(s_N, s_C), ...] over the bonds 1..L-1 of each sequence; every bond is scored from its own window, so the result does not depend on how the proteins are batched"""
        toks = [window_tokens(s, W) for s in seqs]
        sizes = [len(t) for t in toks]
        if sum(sizes) == 0:
            return [(np.empty(0, np.float64), np.empty(0, np.float64)) for _ in seqs]
        tok = np.concatenate([t for t in toks if len(t)])
        sc = {}
        for T in "NC":
            l = predict_raw(self.gb.ex[T], gb_features(tok, T))
            x = np.mean([h.logits(tok) for h in self.heads[T]], axis=0, dtype=np.float64)
            sc[T] = 0.5 * (np.asarray(l, np.float64) + x)
        out, o = [], 0
        for n in sizes:
            out.append((sc["N"][o:o + n], sc["C"][o:o + n]))
            o += n
        return out
