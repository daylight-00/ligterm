"""Training sets and the leave-one-study-out selection of training units.

For a held-out study S the training units are the verified units in sequence groups that contain no unit of S and
whose peptide shares no 9-mer with any peptide of S. The training bonds are the unique T-bonds of those units
(positives) and 6 times as many bonds drawn uniformly from the other bonds of the proteins that carry a training unit.
"""
import numpy as np

from .data import KEY, bond_of

KMER_OVERLAP = 9


def build_train(c, train_rows, T, rng, neg_ratio=6):
    """Rows of verified units to training bonds. Positives: the unique T-bonds of the units. Negatives: `neg_ratio` times |positives|
    bonds drawn uniformly from all other bonds of the proteins that carry at least one training unit."""
    p = c.vp[train_rows]
    ok, b = bond_of(c, train_rows, T)
    keys = np.unique(p[ok].astype(np.int64) * KEY + b[ok])
    tprot = np.unique(p)
    nb = c.L[tprot] - 1
    cum = np.cumsum(nb)
    n_neg = int(neg_ratio * len(keys))
    draw = rng.integers(0, cum[-1], size=int(n_neg * 1.25) + 1000)          # oversample, then keep distinct non-positive bonds (sampling without replacement)
    k = np.searchsorted(cum, draw, side="right")
    bi = draw - (cum[k] - nb[k]) + 1
    npro = tprot[k]
    nkey = npro.astype(np.int64) * KEY + bi
    ukey = np.unique(nkey[~np.isin(nkey, keys)])
    assert len(ukey) >= n_neg, "oversampling produced too few distinct negative bonds"
    ukey = ukey[rng.permutation(len(ukey))[:n_neg]]
    npro, bi = ukey // KEY, ukey % KEY
    P = np.concatenate([keys // KEY, npro])
    B = np.concatenate([keys % KEY, bi])
    y = np.concatenate([np.ones(len(keys), np.int8), np.zeros(len(npro), np.int8)])
    return P, B, y, dict(n_pos=int(len(keys)), n_neg=int(len(npro)), n_train_proteins=int(len(tprot)))


def peptide_overlap_mask(c, cand_rows, S_rows, k=KMER_OVERLAP):
    """For every candidate row: (shares a k-mer with some peptide of the held-out study, has a peptide identical to one of them).
    A peptide shorter than k counts as overlapping if it lies inside a peptide of the study."""
    speps = {c.vseq[r] for r in S_rows}
    skm = set()
    for p in speps:
        for i in range(len(p) - k + 1):
            skm.add(p[i:i + k])
    shortS = list(speps)
    ov = np.zeros(len(cand_rows), bool)
    ident = np.zeros(len(cand_rows), bool)
    for j, r in enumerate(cand_rows):
        p = c.vseq[r]
        ident[j] = p in speps
        if len(p) >= k:
            ov[j] = any(p[i:i + k] in skm for i in range(len(p) - k + 1))
        else:
            ov[j] = any(p in sp for sp in shortS)
    return ov, ident


def strict_rows(c, S, grouping, drop_overlap=True):
    """Training rows for the held-out study S: verified units in groups that contain no unit of S and, in addition, whose peptide
    shares no 9-mer with any peptide of S. Returns (rows, info)."""
    g = c.G[grouping]
    S_rows = c.stud[S]
    eg = np.unique(g[c.vp[S_rows]])
    base = np.flatnonzero(~np.isin(g[c.vp], eg))
    info = dict(n_group_filtered=int(len(base)), n_dropped_overlap=0, n_dropped_identical=0)
    if not drop_overlap:
        return base, info
    ov, ident = peptide_overlap_mask(c, base, S_rows)
    info.update(n_dropped_overlap=int(ov.sum()), n_dropped_identical=int(ident.sum()))
    return base[~ov], info


def check_disjoint(c, rows, S, grouping, eprot=None, peptides=True):
    """Raises AssertionError if any path of information from the held-out study S into training exists at the protein, group, unit or 9-mer level."""
    S_rows = c.stud[S]
    tp = np.unique(c.vp[rows])
    sp_all = np.unique(c.vp[S_rows])
    g = c.G[grouping]
    assert not np.isin(tp, sp_all).any(), "a training protein is a protein of the held-out study"
    if eprot is not None:
        assert not np.isin(tp, eprot).any(), "a training protein is an evaluation protein"
    inS = np.zeros(c.nv, bool)
    inS[S_rows] = True
    assert not inS[rows].any(), "a training unit belongs to the held-out study"
    assert not np.isin(g[tp], g[sp_all]).any(), "a training protein shares a group with a protein of the held-out study"
    if peptides:
        assert not peptide_overlap_mask(c, rows, S_rows)[0].any(), "a training peptide shares a 9-mer with a held-out peptide"


def lab_key(authors):
    """(surname, set of given-name initials) of the last listed author of an IEDB author string ('A B; C D; ...')."""
    parts = [p.strip() for p in authors.replace("\n", " ").split(";") if p.strip()]
    if not parts:
        return None
    toks = parts[-1].replace(".", " ").replace(",", " ").split()
    if not toks:
        return None
    return toks[-1].lower(), frozenset(t[0].lower() for t in toks[:-1]) or frozenset(["?"])


def same_lab(k1, k2):
    return k1 is not None and k2 is not None and k1[0] == k2[0] and bool(k1[1] & k2[1])


def lab_exclusion(c, S, inS, keys, share_thr=0.05):
    """Studies that share the last-listed author with S or share at least `share_thr` of S's units: (set of PMIDs, reasons)."""
    X, why = set(), {}
    nS = max(int(inS.sum()), 1)
    for s2, rows in c.stud.items():
        if s2 == S:
            continue
        same = same_lab(keys.get(S), keys.get(s2))
        share = float(inS[rows].sum()) / nS
        if same or share >= share_thr:
            X.add(s2)
            why[s2] = dict(same_last_author=bool(same), share_of_study_units=round(share, 4))
    return X, why


def size_matched_rows(rows, n_remove, rng):
    """Random removal of n_remove training units (the size control of the same-lab-excluded variant)."""
    return rows[rng.permutation(len(rows))[:len(rows) - n_remove]]
