"""Linear reference model, bond ranks and the peptide score of the LigTerm scorer (the one-hot logistic window model is the baseline family of the scorer; the release models are in `ligterm.ens` and `ligterm.gb`).
A model directory holds `model.npz` (weights of the N and C terminal models, the log length prior) and `meta.json` (window, penalty, training-set description, library versions). Scores are logits of one-hot logistic window models; they rank bonds WITHIN a protein and are not probabilities that can be compared across proteins.
Arithmetic: the scores are summed in the dtype of the stored weights (float32 for a model fitted by the development recipe, as in the development library), so that the scorer reproduces the development scores bitwise; the length prior is float64."""
import json
import os
import numpy as np

from .features import NTOK, window_tokens, linear_score

LEN_LO, LEN_HI = 9, 30          # candidate peptide lengths (98.1% of the verified ligands)


def _weights(a):
    """the weights keep their floating dtype: a model fitted by the development recipe has float32 coefficients (the development library fits on float32 input) and its scores are then summed in float32, exactly as the development library does, so that the reproduction is bitwise; anything else becomes float64"""
    a = np.asarray(a)
    return a if a.dtype in (np.float32, np.float64) else a.astype(np.float64)


class TermModel:
    def __init__(self, w, W_N, b_N, W_C, b_C, len_logp, len_lo=LEN_LO, len_hi=LEN_HI, meta=None):
        W_N, W_C = _weights(W_N), _weights(W_C)
        if W_N.shape != (2 * w, NTOK) or W_C.shape != (2 * w, NTOK):
            raise ValueError(f"weights must have shape ({2 * w}, {NTOK}); got {W_N.shape} and {W_C.shape}")
        len_logp = np.asarray(len_logp, np.float64)
        if len(len_logp) != len_hi - len_lo + 1:
            raise ValueError("the log length prior must cover every candidate length")
        self.w, self.W = w, {"N": W_N, "C": W_C}
        self.b = {"N": float(b_N), "C": float(b_C)}
        self.len_lo, self.len_hi, self.len_logp = int(len_lo), int(len_hi), len_logp
        self.meta = dict(meta or {})

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        np.savez(os.path.join(directory, "model.npz"), w=np.array(self.w), W_N=self.W["N"], b_N=np.array(self.b["N"]), W_C=self.W["C"], b_C=np.array(self.b["C"]),
                 len_logp=self.len_logp, len_range=np.array([self.len_lo, self.len_hi]))
        with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(self.meta, fh, indent=1, sort_keys=True)

    @classmethod
    def load(cls, directory):
        z = np.load(os.path.join(directory, "model.npz"))
        meta = {}
        mp = os.path.join(directory, "meta.json")
        if os.path.exists(mp):
            with open(mp, encoding="utf-8") as fh:
                meta = json.load(fh)
        lo, hi = (int(x) for x in z["len_range"])
        return cls(int(z["w"]), z["W_N"], float(z["b_N"]), z["W_C"], float(z["b_C"]), z["len_logp"], lo, hi, meta)

    def bond_scores(self, seq):
        """(s_N, s_C): arrays over the bonds 1..L-1 of `seq` (index k = bond k+1). s_N scores bond i as the N-terminal bond of a peptide that starts at residue i+1; s_C scores it as the C-terminal bond of a peptide that ends at residue i."""
        tok = window_tokens(seq, self.w)
        return linear_score(tok, self.W["N"], self.b["N"]), linear_score(tok, self.W["C"], self.b["C"])


def competition_rank(scores):
    """rank within the protein, 1 = highest score; tied scores share the best rank (1 + the number of strictly higher scores)."""
    s = np.asarray(scores, np.float64)
    srt = np.sort(s)
    return 1 + (len(s) - np.searchsorted(srt, s, side="right"))


def peptide_candidates(sN, sC, L, lo, hi, g, top=None):
    """candidate peptides (start, end) with both terminal bonds inside the protein (start >= 2, end <= L-1) and length lo..hi; score = s_N(start-1) + s_C(end) + g(length), g = the log length prior.
    sN and sC are indexed by bond-1 (bond k at index k-1). Returns arrays (start, end, length, score) of the `top` best (all if top is None), best first; ties keep the order of increasing length, then start."""
    starts, ends, lens, scs = [], [], [], []
    for ell in range(lo, hi + 1):
        s = np.arange(2, L - ell + 1)                 # end = s + ell - 1 <= L - 1
        if len(s) == 0:
            continue
        e = s + ell - 1
        sc = sN[s - 2] + sC[e - 1] + g[ell - lo]
        starts.append(s), ends.append(e), lens.append(np.full(len(s), ell)), scs.append(sc)
    if not starts:
        z = np.empty(0, np.int64)
        return z, z, z, np.empty(0, np.float64)
    st, en, ln, sc = (np.concatenate(x) for x in (starts, ends, lens, scs))
    order = np.argsort(-sc, kind="stable")
    if top is not None:
        order = order[:top]
    return st[order], en[order], ln[order], sc[order]
