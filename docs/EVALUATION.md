# Evaluation

## Cohorts

The checkpoints ([`models/gb`](MODEL_CARD_gb.md), [`models/ens`](MODEL_CARD_ens.md)) are fitted on all 417,415 verified training units. The development and historical studies are among them, so those cohorts evaluate refits of the recipe, not the checkpoints.

| Cohort | Evaluates |
|---|---|
| External test cohort (4 studies, 152 items) | The checkpoints |
| Development cohort (19 studies) | Refits of the recipe |
| Historical test cohort (2 reserved studies) | Refits of the recipe; a checkpoint must not be scored on these studies |

## Training data

Human MHC class II ligands of the IEDB: cellular MHC, mass spectrometry, positive, with a UniProt parent protein and consistent coordinates.

- 417,415 verified units from 125 studies and 16,873 proteins (the unit table lists 131 PMIDs, 6 without a verified unit).
- 1,516,627 N rows (216,661 positive bonds) and 1,551,697 C rows (221,671 positive bonds), 6 sampled negatives per positive; both models use the same rows.
- 22,774 units come only from the lung class II list of PMID 35051231, whose class II identity is unresolved. The curated alternative without them is a sensitivity analysis, not the default.
- The IEDB export is pinned only by its sha256; its release number was not recorded.
- The audit of one study showed that the class label of a mass-spectrometry record coincides with the elution antibody of its evidence code and is not a peptide-level assignment; no study other than PMID 35051231 was reconciled with its paper.
- Input is local sequence only: no structure, allele, tissue, assay or expression.

## External evaluation

152 items (study-protein pairs) from four studies whose source proteins are absent from the training data under the authors' own criteria: no identical 9-mer with a training protein, and no hit of identity >= 0.40 over >= 0.50 of the query. Bond-AUROC, macro over proteins, 95 % bootstrap intervals over protein clusters:

| Model | N | C |
|---|---|---|
| LigTerm | 0.703 [0.661, 0.740] | 0.710 [0.672, 0.749] |
| LigTerm-GB | 0.697 [0.658, 0.737] | 0.704 [0.666, 0.741] |

- Small, with strongly differing studies; below the development medians.
- Not homology-free: 68 of the 152 items have a significant homology hit below the criterion.
- Not blind: its labels were profiled before the run.
- Local-window audit (positive and sampled negative windows): median identity 0.375 to the training windows, the same as for shuffled windows; 13 of 325 positive windows (4.0 %) lie above the 99th percentile of the shuffled windows, one at 0.72.
- The gain of LigTerm over LigTerm-GB was not resolved on this cohort.

## Development evaluation

Each number is from a refit of the recipe on the strict training rows of one study, leaving that study and its relatives out (sequence groups u30, the 9-mer rule, 6 negatives per positive bond, three group-disjoint folds for the number of iterations), scored on that study. These are exposed development studies, not blind validation.

Metrics: Bond-AUROC (protein-macro, ties 0.5); Local-AUC (share of pairs in which an observed terminal bond outscores a non-observed bond 1 to 3 positions away, pooled; chance 0.5); Peptide-AUROC, Length-matched AUC (negatives of the positive's own length) and Near-miss AUC (start and end each shifted by at most 3), for the sum of the two bond scores (NC) and with the length prior (NCL), clean negatives.

| Development cohort, median Bond-AUROC | N | C |
|---|---|---|
| LigTerm-GB | 0.776 | 0.761 |
| LigTerm | 0.7955 | 0.7752 |

LigTerm minus LigTerm-GB (medians): Bond-AUROC +0.0139 (N) and +0.0143 (C); Length-matched AUC (NC) +0.0145; Near-miss AUC (NC) +0.0158; each positive in 19 of 19 studies. LigTerm adds the window-MLP logit with the fixed scale 1; matching the scale over the evaluation bonds of a study instead changes the Bond-AUROC of a study by at most 0.0006. One seed of the head instead of three gives +0.0108 to +0.0132 in Bond-AUROC, positive in 19 of 19 studies at both termini.

Historical test cohort (the class II identity of most units of one study is unresolved): LigTerm-GB Bond-AUROC N / C is 0.771 / 0.742 in one study and 0.545 / 0.538 in the other. The development medians are not the expected performance on a new study.

## Limitations

Human class II ligand data only; peptides at a protein end and peptides shorter than 9 or longer than 30 residues are not scored. The window-MLP heads are bitwise reproducible from their seed on the same device and software, not tested across devices. Further limits of use (calibration, very long proteins): [Usage](USAGE.md).

Software checks: [Verification](VERIFICATION.md). Records behind these numbers: [Reproducing](../REPRODUCING.md).
