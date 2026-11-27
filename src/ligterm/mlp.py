"""The window MLP of the LigTerm ensemble: a numpy-only forward pass of the one-hot window MLP (a per-residue projection shared over the positions, two dense layers).
Architecture (eval mode, dropout off): the 32 window tokens P16..P1, P1'..P16' (0..19 residue, 20 = pad or a non-standard letter) -> a per-residue projection 20 -> 64 shared over the positions and a GELU, zeroed at positions that are not live (pad, non-standard letter or outside the protein) ->
the flattened 32 x 64 values -> Linear 2048 -> 128, GELU -> Linear 128 -> 1: the logit of a bond for one terminus.
Exactness: the input is one-hot, so the projection is a table of 20 vectors: T[a] = GELU(W_proj[:, a] + b_proj), and the first mixing layer is a sum over positions of table entries V[p, a] = T[a] @ W_mix1[:, p, :]^T; the pre-activation of a bond is b_mix1 + sum_p V[p, token_p] (a dead position adds nothing). This is the same arithmetic as the torch head up to float32 summation order (tested against the torch module to 1e-5 in the logit).
GELU is the exact (erf) form of torch's default, evaluated by a dense table of the normal CDF (built with math.erf) and linear interpolation (error below 2e-9). numpy only."""
import math

import numpy as np

W_SIDE = 16
NWIN = 2 * W_SIDE
PAD = 20
D_PROJ = 64
D_MIX = 128
_GRID = np.linspace(-12.0, 12.0, 120001)
_PHI = np.array([0.5 * (1.0 + math.erf(x / math.sqrt(2.0))) for x in _GRID], np.float64)
WEIGHT_KEYS = ("proj_w", "proj_b", "mix1_w", "mix1_b", "mix2_w", "mix2_b")


def gelu(x):
    """exact GELU, x * Phi(x), by interpolation of a dense table of Phi (float64 inside, the dtype of x out)"""
    x64 = np.asarray(x, np.float64)
    return (x64 * np.interp(x64, _GRID, _PHI)).astype(np.asarray(x).dtype, copy=False)


class MLPHead:
    """one fitted head (one terminus, one seed): the six arrays of the one-hot window MLP (proj_w (64, 20), proj_b (64,), mix1_w (128, 2048), mix1_b (128,), mix2_w (1, 128), mix2_b (1,))"""

    def __init__(self, w):
        for k, shp in (("proj_w", (D_PROJ, PAD)), ("proj_b", (D_PROJ,)), ("mix1_w", (D_MIX, NWIN * D_PROJ)), ("mix1_b", (D_MIX,)), ("mix2_w", (1, D_MIX)), ("mix2_b", (1,))):
            if k not in w or tuple(np.shape(w[k])) != shp:
                raise ValueError(f"head array {k} missing or not of shape {shp}")
        self.w = {k: np.asarray(w[k], np.float32) for k in WEIGHT_KEYS}
        tab = gelu(self.w["proj_w"].T + self.w["proj_b"][None, :])                              # (20, 64): the projection of each one-hot letter
        m1 = self.w["mix1_w"].reshape(D_MIX, NWIN, D_PROJ)                                       # (128, 32, 64), flatten order position-major
        V = np.zeros((NWIN, PAD + 1, D_MIX), np.float32)                                         # V[p, a] = T[a] @ m1[:, p, :]^T; the row a = PAD is zero (a dead position)
        for p in range(NWIN):
            V[p, :PAD] = tab @ m1[:, p, :].T
        self.V = V
        self.b1 = self.w["mix1_b"]
        self.w2 = self.w["mix2_w"][0]
        self.b2 = float(self.w["mix2_b"][0])
        self._pos = np.arange(NWIN)[None, :]

    def logits(self, tok, block=4000):
        """float32 logits of n bonds from their window tokens tok (n, 32) with letters 0..19 and PAD = 20 (or any value above 19 for a dead position)"""
        tok = np.asarray(tok)
        if tok.ndim != 2 or tok.shape[1] != NWIN:
            raise ValueError(f"tokens must have shape (n, {NWIN}); got {tok.shape}")
        t = np.minimum(tok.astype(np.int64), PAD)
        out = np.empty(len(t), np.float32)
        for a in range(0, len(t), block):
            pre = self.b1[None, :] + self.V[self._pos, t[a:a + block]].sum(axis=1, dtype=np.float32)       # a sequential sum over the 32 positions: the order does not depend on the block
            out[a:a + block] = (gelu(pre) * self.w2[None, :]).sum(axis=1, dtype=np.float32) + np.float32(self.b2)   # an elementwise product and a row-wise sum, NOT a BLAS product: the result of a row does not depend on which other rows are in the call
        return out
