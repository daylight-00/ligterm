"""The boosted local-window model (LigTerm-GB, family 'gb') of the LigTerm scorer: a numpy-only evaluator of the trees of a fitted scikit-learn HistGradientBoostingClassifier.
`cli.py` is the command line that reads any model family from a model directory.

Model directory: `gb.npz` (for each terminus T in N, C: the flat node arrays of the first n_iter trees, the bitsets of the categorical splits and of the known categories, the baseline and the root of every tree; the log length prior)
and `meta.json` (family 'gb', window 16, the training description, library versions). Scores are logits (log-odds) of the boosted model; like the logits of the linear model they rank bonds WITHIN a protein and are not probabilities that can be compared across proteins.
Features of one bond for terminus T (the design of the development library at w = 16): the 32 residue tokens P16..P1, P1'..P16' as categorical columns (20 residues + the pad token 20), then the 20 residue counts of the INTERIOR-facing 16 residues and the 20 counts of the OUTWARD-facing 16 residues
(N bond: interior = P1'..P16', outward = P16..P1; C bond: interior = P16..P1, outward = P1'..P16'; pad positions and non-standard letters count for no residue): 72 float32 columns.
Arithmetic: raw score = baseline + the leaf values summed tree by tree in float64, in the order of scikit-learn's own prediction loop, so that the score equals `staged_decision_function` after n_iter iterations EXACTLY (tested bitwise, with unseen and negative categories and NaN).
Only numpy is needed to score; scikit-learn is needed only by `export_hgb` (fitting time), whose private attributes are those of the pinned version recorded in the meta file."""
import json
import os
import numpy as np

from .features import window_tokens, NTOK
from .model import LEN_LO, LEN_HI

W = 16                      # window per side: 32 token columns
NCAT = 2 * W                # categorical columns = the token columns
NCNT = 20                   # residue types counted
NFEAT = NCAT + 2 * NCNT     # 72
NODE_KEYS = ("value", "feature_idx", "num_threshold", "missing_go_to_left", "left", "right", "is_leaf", "is_categorical", "bitset_idx")


def gb_features(tok, T):
    """tok (n, 32) int tokens of the bonds (P16..P1 in columns 0..15, P1'..P16' in 16..31; 20 = pad) -> (n, 72) float32 features of terminus T ('N' or 'C')."""
    tok = np.asarray(tok)
    if tok.ndim != 2 or tok.shape[1] != 2 * W:
        raise ValueError(f"tokens must have shape (n, {2 * W}); got {tok.shape}")
    if T not in ("N", "C"):
        raise ValueError(f"terminus must be 'N' or 'C', got {T!r}")
    interior = tok[:, W:] if T == "N" else tok[:, :W]
    outward = tok[:, :W] if T == "N" else tok[:, W:]
    X = np.empty((len(tok), NFEAT), np.float32)
    X[:, :NCAT] = tok
    for a in range(NCNT):
        X[:, NCAT + a] = (interior == a).sum(1)
        X[:, NCAT + NCNT + a] = (outward == a).sum(1)
    return X


def export_hgb(model, nstar):
    """flat arrays of the first nstar trees of a fitted binary HistGradientBoostingClassifier (needs scikit-learn; uses its private predictor attributes: the version is recorded in the meta file and the export is checked bitwise against the model by the tests and by the agreement script)"""
    if not 1 <= nstar <= len(model._predictors):
        raise ValueError(f"nstar {nstar} is outside 1..{len(model._predictors)}")
    base = float(np.asarray(model._baseline_prediction).ravel()[0])
    known, fmap = model._bin_mapper.make_known_categories_bitsets()
    cols = {k: [] for k in NODE_KEYS}
    raw_rows, roots, off, boff = [], [], 0, 0
    for it in range(nstar):
        tr = model._predictors[it][0]
        nd = tr.nodes
        roots.append(off)
        for k in ("value", "feature_idx", "num_threshold", "missing_go_to_left", "is_leaf", "is_categorical"):
            cols[k].append(np.asarray(nd[k]))
        cols["left"].append(np.asarray(nd["left"]).astype(np.int64) + off)
        cols["right"].append(np.asarray(nd["right"]).astype(np.int64) + off)
        cols["bitset_idx"].append(np.asarray(nd["bitset_idx"]).astype(np.int64) + boff)
        raw_rows.append(np.asarray(tr.raw_left_cat_bitsets))
        off += len(nd)
        boff += len(tr.raw_left_cat_bitsets)
    out = {k: np.concatenate(v) for k, v in cols.items()}
    n_nodes = len(out["value"])
    if n_nodes >= 2 ** 31 or boff >= 2 ** 31:
        raise ValueError("too many nodes for int32 indices")
    for k in ("left", "right", "bitset_idx"):
        out[k] = out[k].astype(np.int32)
    out["value"] = out["value"].astype(np.float64)
    out["num_threshold"] = out["num_threshold"].astype(np.float64)
    out["feature_idx"] = out["feature_idx"].astype(np.int32)
    out["missing_go_to_left"] = out["missing_go_to_left"].astype(np.uint8)
    out["is_leaf"] = out["is_leaf"].astype(np.uint8)
    out["is_categorical"] = out["is_categorical"].astype(np.uint8)
    out["baseline"] = np.array(base, np.float64)
    out["roots"] = np.array(roots, np.int32)
    out["raw_left"] = np.concatenate(raw_rows, 0).astype(np.uint32) if boff else np.zeros((0, 8), np.uint32)
    out["known"] = np.asarray(known).astype(np.uint32)
    out["fmap"] = np.asarray(fmap).astype(np.int32)
    return out


def _in_bitset(bits, val, row):
    return ((bits[row, val >> 5] >> (val & 31).astype(np.uint32)) & np.uint32(1)).astype(bool)


def predict_raw(ex, X, block=50_000):
    """raw scores (log-odds) of the rows of X: baseline + the leaf values of the trees, summed tree by tree in float64 (the order of scikit-learn's prediction loop)"""
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != NFEAT:
        raise ValueError(f"features must have shape (n, {NFEAT}); got {X.shape}")
    X = X.astype(np.float64, copy=False)                                   # scikit-learn reads the raw data as float64
    out = np.empty(len(X), np.float64)
    nt = len(ex["roots"])
    val, fidx, thr = ex["value"], ex["feature_idx"].astype(np.int64), ex["num_threshold"]
    mgl, left, right = ex["missing_go_to_left"].astype(bool), ex["left"].astype(np.int64), ex["right"].astype(np.int64)
    leafm, catm, bidx = ex["is_leaf"].astype(bool), ex["is_categorical"].astype(bool), ex["bitset_idx"].astype(np.int64)
    rawl, known, fmap = ex["raw_left"], ex["known"], ex["fmap"].astype(np.int64)
    base = float(ex["baseline"])
    for b0 in range(0, len(X), block):
        Xb = X[b0:b0 + block]
        n = len(Xb)
        acc = np.full(n, base, np.float64)
        for t in range(nt):
            idx = np.full(n, int(ex["roots"][t]), np.int64)
            act = np.arange(n)
            while act.size:
                nid = idx[act]
                leaf = leafm[nid]
                if leaf.any():
                    acc[act[leaf]] += val[nid[leaf]]
                act, nid = act[~leaf], nid[~leaf]
                if not act.size:
                    break
                f = fidx[nid]
                v = Xb[act, f]
                go_left = np.empty(act.size, bool)
                miss = np.isnan(v)
                cat = catm[nid] & ~miss
                num = ~miss & ~cat
                go_left[miss] = mgl[nid[miss]]
                go_left[num] = v[num] <= thr[nid[num]]
                if cat.any():
                    vi = v[cat]
                    neg = vi < 0
                    vint = np.where(neg, 0, vi).astype(np.int64)
                    inleft = _in_bitset(rawl, vint, bidx[nid[cat]])
                    kn = _in_bitset(known, vint, fmap[f[cat]])
                    mg = mgl[nid[cat]]
                    go_left[cat] = np.where(neg, mg, np.where(inleft, True, np.where(kn, False, mg)))
                idx[act] = np.where(go_left, left[nid], right[nid])
        out[b0:b0 + block] = acc
    return out


class GBTermModel:
    """LigTerm-GB: one boosted model per terminus (N and C) on the 72 features, plus the log length prior of the peptide score (as in the linear model)."""
    family = "gb"

    def __init__(self, ex_N, ex_C, len_logp, len_lo=LEN_LO, len_hi=LEN_HI, meta=None):
        len_logp = np.asarray(len_logp, np.float64)
        if len(len_logp) != len_hi - len_lo + 1:
            raise ValueError("the log length prior must cover every candidate length")
        for ex in (ex_N, ex_C):
            missing = [k for k in NODE_KEYS + ("baseline", "roots", "raw_left", "known", "fmap") if k not in ex]
            if missing:
                raise ValueError(f"tree arrays missing: {missing}")
        self.w, self.ex = W, {"N": ex_N, "C": ex_C}
        self.len_lo, self.len_hi, self.len_logp = int(len_lo), int(len_hi), len_logp
        self.meta = dict(meta or {})

    @property
    def n_trees(self):
        return {T: int(len(self.ex[T]["roots"])) for T in "NC"}

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        arrs = {"len_logp": self.len_logp, "len_range": np.array([self.len_lo, self.len_hi])}
        for T in "NC":
            for k, v in self.ex[T].items():
                arrs[f"{T}_{k}"] = np.asarray(v)
        np.savez_compressed(os.path.join(directory, "gb.npz"), **arrs)
        meta = dict(self.meta, family=self.family, window=W, n_trees=self.n_trees)
        with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1, sort_keys=True)

    @classmethod
    def load(cls, directory):
        z = np.load(os.path.join(directory, "gb.npz"))
        with open(os.path.join(directory, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("family") != "gb":
            raise ValueError(f"{directory} is not a LigTerm-GB model directory (family {meta.get('family')!r})")
        ex = {T: {k[2:]: z[k] for k in z.files if k.startswith(f"{T}_")} for T in "NC"}
        lo, hi = (int(x) for x in z["len_range"])
        return cls(ex["N"], ex["C"], z["len_logp"], lo, hi, meta)

    def bond_scores(self, seq):
        """(s_N, s_C): arrays over the bonds 1..L-1 of `seq` (index k = bond k+1), exactly as TermModel.bond_scores of the linear model."""
        return self.bond_scores_many([seq])[0]

    def bond_scores_many(self, seqs):
        """[(s_N, s_C), ...] for several proteins in ONE evaluation of the trees (the cost of the tree loop is per call, not per row, so the command line batches proteins). Every row is scored independently and its leaf values are summed in the same order, so the
        result does not depend on how the proteins are batched (tested bitwise)."""
        toks = [window_tokens(s, W) for s in seqs]
        sizes = [len(t) for t in toks]
        if sum(sizes) == 0:
            return [(np.empty(0, np.float64), np.empty(0, np.float64)) for _ in seqs]
        tok = np.concatenate([t for t in toks if len(t)])
        sN = predict_raw(self.ex["N"], gb_features(tok, "N"))
        sC = predict_raw(self.ex["C"], gb_features(tok, "C"))
        out, o = [], 0
        for n in sizes:
            out.append((sN[o:o + n], sC[o:o + n]))
            o += n
        return out


def load_model(directory):
    """the model of a directory, by the family in meta.json: 'gb' -> GBTermModel; 'ens' -> EnsTermModel; no family key (a linear reference model) -> the linear TermModel."""
    from .model import TermModel
    mp = os.path.join(directory, "meta.json")
    fam = "linear"
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as fh:
            fam = json.load(fh).get("family", "linear")
    if fam == "gb":
        return GBTermModel.load(directory)
    if fam == "ens":                                                          # the LigTerm ensemble (ens.py)
        from .ens import EnsTermModel
        return EnsTermModel.load(directory)
    if fam == "linear":
        return TermModel.load(directory)
    raise ValueError(f"unknown model family {fam!r} in {mp}")
