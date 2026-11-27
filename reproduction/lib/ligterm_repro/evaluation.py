"""Evaluation targets of a held-out study and the label variants of the experiments.

For study S and terminus T the evaluation proteins are the proteins that carry an observed T-bond of S; every bond of such a protein
is an evaluation bond. Labels: 1 observed terminal bond of S, 0 other bond, -1 excluded.
"""
import collections

import numpy as np

from .data import KEY, WMAX, bond_of


def eval_setup(c, S, T):
    """Targets of study S for terminus T: the evaluation proteins, the concatenated bond table and the labels of every variant
    (primary and the variants other_study_bonds_as_negatives, positives_any_study, nested_family_positives, opposite_terminus_bonds_excluded)."""
    rows = c.stud[S]
    ok, b = bond_of(c, rows, T)
    p = c.vp[rows]
    posk = np.unique(p[ok].astype(np.int64) * KEY + b[ok])
    eprot = np.unique(posk // KEY)
    nb = c.L[eprot] - 1
    ebase = np.concatenate([[0], np.cumsum(nb)[:-1]])
    tot = int(nb.sum())
    bp = np.repeat(eprot, nb)
    bi = np.arange(tot) - np.repeat(ebase, nb) + 1

    def gpos(keys):
        k = np.searchsorted(eprot, keys // KEY)
        assert np.all(eprot[k] == keys // KEY)
        return ebase[k] + keys % KEY - 1

    inS = np.zeros(c.nv, bool)
    inS[rows] = True
    onp = np.isin(c.vp, eprot) & ~inS
    ro = np.flatnonzero(onp)
    ok2, b2 = bond_of(c, ro, T)
    otherk = np.setdiff1d(np.unique(c.vp[ro][ok2].astype(np.int64) * KEY + b2[ok2]), posk)
    # repeats: other occurrences of a peptide of S inside the same protein
    repk = []
    ineval = np.zeros(c.NP, bool)
    ineval[eprot] = True
    for r in rows.tolist():
        pn = c.vp[r]
        if not ineval[pn]:
            continue
        sq = c.seqs[c.names[pn]]
        pep = c.vseq[r]
        k = sq.find(pep)
        while k >= 0:
            if k + 1 != c.vs[r]:
                bd = (k + 1 - 1) if T == "N" else (k + len(pep))
                if 1 <= bd <= c.L[pn] - 1:
                    repk.append(pn * KEY + bd)
            k = sq.find(pep, k + 1)
    repk = np.setdiff1d(np.unique(np.array(repk, np.int64)), posk) if repk else np.empty(0, np.int64)
    # nested family: a unit of S that overlaps another unit of S (different span) of the same protein by at least 9 residues
    byp = collections.defaultdict(list)
    for r in rows.tolist():
        byp[c.vp[r]].append(r)
    nested = []
    for pn, rs in byp.items():
        if len(rs) < 2:
            continue
        s_, e_ = c.vs[rs], c.ve[rs]
        ov = np.minimum(e_[:, None], e_[None, :]) - np.maximum(s_[:, None], s_[None, :]) + 1
        differ = (s_[:, None] != s_[None, :]) | (e_[:, None] != e_[None, :])
        nested += [rs[j] for j in np.flatnonzero(((ov >= 9) & differ).any(1))]
    nested = np.array(sorted(set(nested)), np.int64)
    okn, bn = bond_of(c, nested, T) if len(nested) else (np.zeros(0, bool), np.zeros(0, np.int64))
    v3k = np.unique(c.vp[nested][okn].astype(np.int64) * KEY + bn[okn]) if len(nested) else np.empty(0, np.int64)
    singk = np.setdiff1d(posk, v3k)

    def lab(pos, excl_sets):
        y = np.zeros(tot, np.int8)
        for ks in excl_sets:
            if len(ks):
                y[gpos(ks)] = -1
        if len(pos):
            y[gpos(pos)] = 1
        return y
    # bonds that are the opposite terminus of any verified unit on the evaluation proteins: other bonds in the primary labels, excluded in the last variant
    To = "C" if T == "N" else "N"
    alr = np.flatnonzero(np.isin(c.vp, eprot))
    ok3, b3 = bond_of(c, alr, To)
    oppk = np.setdiff1d(np.unique(c.vp[alr][ok3].astype(np.int64) * KEY + b3[ok3]), posk)
    labs = {
        "primary": lab(posk, [otherk, repk]),
        "other_study_bonds_as_negatives": lab(posk, [repk]),
        "positives_any_study": lab(np.union1d(posk, otherk), [repk]),
        "nested_family_positives": lab(v3k, [singk, otherk, repk]),
        "opposite_terminus_bonds_excluded": lab(posk, [otherk, repk, oppk]),
    }
    return dict(eprot=eprot, nb=nb, ebase=ebase, bp=bp, bi=bi, labs=labs, n_pos=len(posk), n_other=len(otherk), n_repeat=len(repk), n_nested_pos=len(v3k),
                n_opposite=len(np.setdiff1d(oppk, np.union1d(otherk, repk))), inS=inS)


def study_exclusive_labels(c, S, T, es):
    """Primary labels, but positives only for bonds that are the T-bond of at least one study-exclusive unit (listed under no other PMID);
    the T-bonds of units that are all shared with other studies are excluded. The evaluated ligands are then exclusive to the study."""
    rows = c.stud[S]
    excl_rows = np.array([r for r in rows.tolist() if len(c.vpm[r]) == 1], np.int64)
    ok, b = bond_of(c, excl_rows, T)
    keys_excl = np.unique(c.vp[excl_rows][ok].astype(np.int64) * KEY + b[ok]) if len(excl_rows) else np.empty(0, np.int64)
    lab = es["labs"]["primary"].copy()
    pos = np.flatnonzero(lab == 1)
    pk = es["bp"][pos].astype(np.int64) * KEY + es["bi"][pos]                 # key of every positive bond
    keep = np.isin(pk, keys_excl)
    lab[pos[~keep]] = -1
    return lab, int(keep.sum()), int((~keep).sum())


def complete_window(bi, L, etok):
    """Evaluation bonds whose NetCleave window exists: the bond is >= 4 (P4 lies inside the protein), <= L-3 (P3' lies inside the protein)
    and all seven window tokens are standard residues (a non-standard residue, a pad or a missing residue makes the window incomplete)."""
    return (bi >= 4) & (bi <= L - 3) & (etok[:, np.arange(WMAX - 4, WMAX + 3)] < 20).all(1)
