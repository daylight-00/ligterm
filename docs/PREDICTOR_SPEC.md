# Specification

This document describes the package `ligterm` and the two models [`models/gb`](MODEL_CARD_gb.md) (LigTerm-GB) and [`models/ens`](MODEL_CARD_ens.md) (LigTerm) as implemented. Performance: [Evaluation](EVALUATION.md).

## 1. Score

A score is a logit that ranks the bonds, and the candidate peptides, within one protein by the observed ligand-terminal propensity of the training data (human MHC class II ligands of the IEDB). It is not a measure of cleavage, not a binding, presentation or immunogenicity prediction, not a probability, and not comparable across proteins or models. The score of a protein depends only on its own sequence and the model.

## 2. Encoding

The 20 standard residues are tokens 0 to 19. Every other character is the pad token 20, one token per character, as is every position outside the protein; a non-ASCII character first becomes `?`, so the length never changes. Lower case is read as upper case. FASTA parsing rules are in [Usage](USAGE.md).

## 3. Coordinates and window

Residues are 1-based. Bond i lies between residue i and residue i+1 (P1 = residue i, P1' = residue i+1), for i = 1 to L-1. The N-terminal bond of a peptide (start, end) is bond start-1 (start >= 2); its C-terminal bond is bond end (end <= L-1). `N_score` of bond i scores it as the N-terminal bond of a peptide starting at residue i+1; `C_score` as the C-terminal bond of a peptide ending at residue i. The window of a bond is P16..P1, P1'..P16': 32 tokens.

## 4. Models

One interface; the family is read from `meta.json`. The package also reads a linear reference family, which only its tests use; none is shipped.

**LigTerm-GB** (family `gb`). One boosted model per terminus on 72 features per bond: the 32 window tokens as categorical columns (20 residues and the pad token), the 20 residue counts of the interior-facing 16 residues and the 20 counts of the outward-facing 16 residues. For an N bond the interior is P1'..P16' and the outward side P16..P1; for a C bond the reverse. Pad positions and non-standard letters count for no residue. The release keeps 800 trees per terminus. Scoring needs only NumPy: the leaf values are summed tree by tree in float64 in scikit-learn's own order, equal to scikit-learn's staged score after the kept iterations (tested bitwise).

**LigTerm** (family `ens`). Per terminus, score = (LigTerm-GB logit + mean over three seeds of the window-MLP logit) / 2: scale 1, equal weights, nothing estimated from the data scored.

**Window MLP head.** A projection of the one-hot letter (20 to 64 dimensions, GELU) shared over the 32 window positions, where a pad or non-standard position contributes the zero vector; the flattened 32 x 64 window; a 2,048 to 128 layer (GELU); a 128 to 1 layer. Scoring is NumPy only: the projection of a one-hot letter is a table, the first mixing layer a sum of per-position table entries, GELU the exact erf form by a dense table. The result of a row does not depend on which other rows are in the call.

## 5. Candidate peptides

A candidate (start, end) has length 9 to 30 and both terminal bonds inside the protein. Its score is

`s_N(start - 1) + s_C(end) + g(length)`

where g is the log of the add-one-smoothed distribution of the verified ligand lengths 9 to 30 of the training units (98.1 % of the verified ligands lie in 9 to 30). Candidates are written best first; ties keep the order of increasing length, then start. Bond ranks are competition ranks (1 = highest score; tied scores share the best rank).

## 6. Batching

Every row is scored independently and its leaf values are summed in the same order, so the scores do not depend on how proteins are batched or on the other proteins of a request. The consistency check scores a toy FASTA with batches of 100,000 and of 7 bonds and gets byte-identical outputs.
