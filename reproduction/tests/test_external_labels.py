"""Planted-oracle tests of ligterm_repro.external: the label construction on proteins whose labels are known by construction, and the brute-force AUROC.
usage: python3 -B test_external_labels.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from ligterm_repro.external import auroc_bruteforce, bond_of, build_labels, repeat_bonds     # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"
rng = np.random.default_rng(20261120)

# 1. bond_of: the convention of the project (bond i lies between residues i and i + 1)
assert bond_of(10, 22, 60, "N") == 9 and bond_of(10, 22, 60, "C") == 22
assert bond_of(1, 22, 60, "N") is None and bond_of(10, 60, 60, "C") is None and bond_of(2, 22, 60, "N") == 1 and bond_of(10, 59, 60, "C") == 59
print("1. bond_of: N = start - 1 (start >= 2), C = end (end <= L - 1)")

# 2. a planted protein: L = 80; unit A (the stratum) 10-22; unit B (another stratum of the same study) 30-40; an other-study unit 50-60 and one that shares the N bond of A (start 10, end 70); a repeat of A's peptide
seq = list("".join(rng.choice(list(AA), 80)))
pepA = "".join(rng.choice(list(AA), 13))
seq[9:22] = list(pepA)                                                      # A at 10-22
seq[64:77] = list(pepA)                                                     # a second occurrence at 65-77 (start 65: N bond 64, C bond 77)
seq = "".join(seq)
assert seq[9:22] == pepA and seq[64:77] == pepA and seq.find(pepA) == 9
units = [(10, 22, pepA, "unseen_protein_no_shared_9mer"), (30, 40, seq[29:40], "training_peptide")]
other = {49, 60, 9}                                                         # other-study N bonds: 49 (start 50), 9 (start 10 = the same bond as A's), C-like numbers are checked below
labN = build_labels(seq, units, lambda st: st == "unseen_protein_no_shared_9mer", other, "N")
assert labN.shape == (79,)
assert labN[9 - 1] == 1, "A's N bond must be positive even though another study has it too (the positives are set last)"
assert labN[29 - 1] == -1, "B's N bond (another stratum of the same study) must be excluded"
assert labN[49 - 1] == -1 and labN[60 - 1] == -1, "the other-study bonds must be excluded"
assert labN[64 - 1] == -1, "the N bond of the other occurrence of A's peptide must be excluded"
assert (labN == 1).sum() == 1 and (labN == -1).sum() == 4 and (labN == 0).sum() == 79 - 5
labC = build_labels(seq, units, lambda st: st == "unseen_protein_no_shared_9mer", {40, 22}, "C")
assert labC[22 - 1] == 1 and labC[40 - 1] == -1 and labC[77 - 1] == -1 and (labC == 1).sum() == 1 and (labC == -1).sum() == 2
print("2. planted protein: positive bond kept although another study has it, other strata, other studies and repeats excluded, the rest negative")

# 3. a unit whose bond is invalid (start = 1) gives no positive at N but its C bond is positive at C; a protein end
seq2 = "".join(rng.choice(list(AA), 40))
u = [(1, 15, seq2[0:15], "unseen_protein_no_shared_9mer")]
assert (build_labels(seq2, u, lambda st: True, set(), "N") == 1).sum() == 0
assert build_labels(seq2, u, lambda st: True, set(), "C")[15 - 1] == 1
print("3. a unit at the start of the protein has no N bond; its C bond is positive")

# 4. repeat_bonds on a peptide that occurs twice, and a peptide that occurs once
assert repeat_bonds(seq, pepA, 10, "N") == {64} and repeat_bonds(seq, pepA, 10, "C") == {77} and repeat_bonds(seq, pepA, 65, "N") == {9}
assert repeat_bonds("ACDEFGHIK", "CDE", 2, "N") == set()
print("4. repeat_bonds finds the other occurrences only")

# 5. the brute-force AUROC: a perfect ranking, a reversed one, ties, excluded ignored
lab = np.array([1, 0, 0, -1, 0], np.int8)
assert auroc_bruteforce([9, 1, 2, 100, 3], lab) == 1.0 and auroc_bruteforce([0, 1, 2, -100, 3], lab) == 0.0
assert auroc_bruteforce([1, 1, 1, 7, 1], lab) == 0.5
assert np.isnan(auroc_bruteforce([1, 2, 3], np.array([0, 0, 0], np.int8)))
print("5. brute-force AUROC: perfect 1, reversed 0, all tied 0.5, excluded bonds ignored, undefined without both classes")
print("EXTERNAL LABEL TESTS: passed")
