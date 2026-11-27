"""Labels of the evaluation on the external cohort. numpy only.

The construction is that of the primary labels of the development bond-level evaluation (evaluation.eval_setup) applied to a stratum of the units of ONE study in ONE protein, for ONE terminus:
  positive (1)  = the terminal bond of a unit of the study that is in the evaluated stratum (N: bond start - 1, valid for start >= 2; C: bond end, valid for end <= L - 1; bond i lies between residues i and i + 1);
  excluded (-1) = (a) the bond of a unit of the same study in another stratum, (b) a bond of the same terminus of a unit of another study (the training unit table or another candidate study) on the protein,
                  (c) the bond of another occurrence of a peptide of the study inside the protein; a bond that is also positive stays positive (the positives are set last);
  negative (0)  = every other bond of the protein.
"""
import numpy as np

# Strata of the candidate units (column `stratum` of datasets/external/candidate_units.tsv.gz), a partition in this order: a unit of a study outside the training units is
#   training_unit               the same unit (protein, start, end) is a training unit
#   training_peptide            not that, but the peptide sequence is a training peptide
#   training_terminal_bond      neither, but at least one of its terminal bonds is a terminal bond of a training unit
#   new_peptide_known_protein   none of these, on a protein that has training units
#   unseen_protein_no_shared_9mer  a protein without training units that shares no 9-mer with the training peptides
#   unseen_protein_shared_9mer     a protein without training units that shares a 9-mer with a training peptide
NEW_PEPTIDE_KNOWN_PROTEIN = "new_peptide_known_protein"
UNSEEN_PROTEIN = "unseen_protein_no_shared_9mer"
UNSEEN_PROTEIN_SHARED_9MER = "unseen_protein_shared_9mer"


def bond_of(start, end, L, T):
    """the terminal bond of the unit (start, end) in a protein of L residues at terminus T, or None"""
    if T == "N":
        b, ok = start - 1, start >= 2
    else:
        b, ok = end, end <= L - 1
    return int(b) if ok and 1 <= b <= L - 1 else None


def repeat_bonds(seq, pep, start, T):
    """the bonds of the OTHER occurrences of the peptide inside the protein (the occurrence at `start` is the unit itself); only valid bonds"""
    L = len(seq)
    out = set()
    k = seq.find(pep)
    while k >= 0:
        if k + 1 != start:
            bd = k if T == "N" else k + len(pep)
            if 1 <= bd <= L - 1:
                out.add(int(bd))
        k = seq.find(pep, k + 1)
    return out


def build_labels(seq, units, in_stratum, other_bonds, T):
    """units: list of (start, end, peptide, stratum) of the study in this protein; in_stratum(stratum) -> bool; other_bonds: set of bonds of the same terminus of the units of other studies on the protein;
    -> int8 array of length L - 1 (index i = bond i + 1)"""
    L = len(seq)
    pos, own_other, rep = set(), set(), set()
    for (s, e, pep, st) in units:
        b = bond_of(s, e, L, T)
        if b is not None:
            (pos if in_stratum(st) else own_other).add(b)
        rep |= repeat_bonds(seq, pep, s, T)
    excl = (own_other | rep | set(other_bonds)) - pos
    lab = np.zeros(L - 1, np.int8)
    for b in excl:
        lab[b - 1] = -1
    for b in pos:
        lab[b - 1] = 1
    return lab


def auroc_bruteforce(scores, lab):
    """the AUROC of the positives (1) against the negatives (0) with ties 0.5, excluded (-1) ignored; NaN if either class is empty (the definition of metrics.auroc written naively)"""
    s = np.asarray(scores, np.float64)
    p, n = s[lab == 1], s[lab == 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    tot = 0.0
    for x in p:
        tot += float((n < x).sum()) + 0.5 * float((n == x).sum())
    return tot / (len(p) * len(n))
