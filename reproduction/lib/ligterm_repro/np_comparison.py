"""Study-level values of the comparison with the published tool (MHCII-NP) from per-protein values.

Per protein and model the table holds: Peptide-AUROC (all candidate peptides of the protein; the share of pairs won), the numerator and denominator of the length-matched pairs (each observed peptide
against the candidates of its own length) and of the near-miss pairs (negatives whose start and end are each shifted by at most 3 residues; and of the same length), and the length-free variants
(the scores of the models are compared with the score of the tool divided by the length factor). A study value is the macro mean over proteins for Peptide-AUROC (and its length-free variant),
the macro mean of the per-protein share of length-matched pairs won, and the pooled ratio of the pairs for the near-miss variants. numpy only.
"""
import numpy as np

# metric -> (kind, per-protein column(s))
METRIC_SOURCE = {
    "peptide_auroc": ("mean", "peptide_auroc"),
    "peptide_auroc_length_free": ("mean", "peptide_auroc_length_free"),
    "length_matched_auc": ("share", "length_matched"),
    "near_miss_auc": ("pooled", "near_miss"),
    "near_miss_auc_same_length": ("pooled", "near_miss_same_length"),
    "near_miss_auc_length_free": ("pooled", "near_miss_length_free"),
    "recall_at_10": ("mean", "recall_at_10"),                      # retrieval measures of the supplement: the share of the observed peptides of a protein among its top 10 / 50 candidates,
    "recall_at_50": ("mean", "recall_at_50"),                      # and whether one of the top candidates lies within three residues of an observed peptide at both ends (macro means)
    "tolerance_hit_at_10": ("mean", "tolerance_hit_at_10"),
    "tolerance_hit_at_50": ("mean", "tolerance_hit_at_50"),
}


def study_value(a, idx, metric):
    """a: dict of per-protein arrays; idx: protein indices (1-D) or a (B, n) matrix of resampled indices -> the study-level value (an array over the first axis of a matrix idx)"""
    kind, col = METRIC_SOURCE[metric]
    if kind == "mean":
        return np.nanmean(a[col][idx], axis=-1)
    if kind == "share":
        num, den = a[f"{col}_num"], a[f"{col}_den"]
        share = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        return np.nanmean(share[idx], axis=-1)
    return a[f"{col}_num"][idx].sum(axis=-1) / a[f"{col}_den"][idx].sum(axis=-1)
