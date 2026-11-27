"""ligterm: scorer of observed MHC class II ligand-terminal propensity. LigTerm (the ensemble, `ligterm.ens`) and LigTerm-GB (the boosted model, `ligterm.gb`) rank, within a protein, the bonds that are the N- or C-terminal bond of an observed ligand, and the candidate peptides between them; the scores do not measure proteolytic cleavage."""
from .features import AA, PAD, PADW, NTOK, encode, window_tokens, linear_score, read_fasta
from .model import TermModel, competition_rank, peptide_candidates, LEN_LO, LEN_HI

__all__ = ["AA", "PAD", "PADW", "NTOK", "encode", "window_tokens", "linear_score", "read_fasta", "TermModel", "competition_rank", "peptide_candidates", "LEN_LO", "LEN_HI"]
