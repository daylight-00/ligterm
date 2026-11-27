# Usage

Run every command from the repository root.

## Requirements

Python 3.10 or newer and NumPy; CPU only. Scoring imports only the standard library and NumPy (scikit-learn, scipy and torch were used to fit the models and are not needed). Only Python 3.12 has been run. Install the package from the repository root with `python3 -m pip install .`; it installs `ligterm` and NumPy, not the models, which stay in `models/`. Without installing, `PYTHONPATH=src python3 -m ligterm ...` runs the same code.

## Command

```bash
ligterm score --fasta FILE --model DIR --out DIR [--top-peptides K] [--batch-bonds N]
```

| Option | Meaning |
|---|---|
| `--fasta` | Input proteins |
| `--model` | `models/ens` ([LigTerm](MODEL_CARD_ens.md)) or `models/gb` ([LigTerm-GB](MODEL_CARD_gb.md)); required |
| `--out` | Output folder |
| `--top-peptides K` | Candidate peptides written per protein (default 20; 0 writes none) |
| `--batch-bonds N` | Memory and speed only (default 100,000); the scores do not depend on it |

`python3 -m ligterm score ...` is the same command (add `-B` to keep Python from writing `__pycache__` into the folder).

## Input

- The identifier is the first whitespace-delimited token of the header. A duplicate identifier, or a sequence before the first header, is an error.
- Whitespace at the ends of a line, spaces inside a line and `*` are dropped. Empty sequences are skipped.
- A protein of fewer than 2 residues is skipped and listed in `run.json`.
- The 20 standard residues (upper or lower case) are read as such. Every other character is one pad token, as is every position outside the protein.
- The code has no length limit; behaviour was checked up to 3,000 residues.

## Output

| File | Content |
|---|---|
| `bonds.tsv` | One row per bond 1 to L-1: `protein`, `bond`, `P1`, `P1prime`, `N_score`, `N_rank`, `C_score`, `C_rank` |
| `peptides.tsv` | The K best candidate peptides per protein, best first: `protein`, `start`, `end`, `length`, `score`, `rank` |
| `run.json` | Counts, window, model family and folder, numpy version, skipped proteins, wall and CPU seconds, tree counts of the boosted model, `mlp_seeds` (LigTerm), a note on how to read the scores |

## Reading a score

- Residues are numbered from 1. Bond i lies between residue i and residue i+1 (`P1` is residue i, `P1prime` residue i+1).
- `N_score` of bond i scores it as the N-terminal bond of a peptide that starts at residue i+1. `C_score` scores it as the C-terminal bond of a peptide that ends at residue i.
- `N_rank` and `C_rank` are competition ranks within the protein: 1 is the highest score, tied scores share the best rank.
- A candidate peptide (start, end) has 9 to 30 residues and both terminal bonds inside the protein (start >= 2, end <= L-1). Its score is defined in the [specification](PREDICTOR_SPEC.md); its `rank` is its position in the written order.
- A score is a logit of observed ligand-terminal propensity. It ranks bonds and peptides within one protein, carries no calibration, and does not depend on the other proteins of the file. It is not comparable across models.

## Cost

CPU only. LigTerm-GB takes about 0.19 s per 1,000 bonds (300 proteins of 50 to 3,000 residues, 181,542 bonds: 34.4 s wall, 33.6 s CPU). LigTerm takes about 1.5 times as long: 86.8 s CPU (0.478 s per 1,000 bonds) against 58.1 s (0.320 s) for LigTerm-GB in the same run (one thread, a loaded node, so only the ratio is meaningful).
