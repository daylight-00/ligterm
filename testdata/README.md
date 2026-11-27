# Test data

Four proteins and the expected outputs of the scorer, used by the [regression test](../docs/VERIFICATION.md).

## Contents

- `proteins.fasta`: four UniProt sequences, accession first in the header, 60 residues per line.
- `expected/ens/` and `expected/gb/`: for each model the `bonds.tsv` and `peptides.tsv` that `ligterm score --fasta testdata/proteins.fasta --model models/<model> --out <dir>` writes (1,304 bonds; the 20 best peptides of each protein, 80 rows), and `run_stable.json`, the `run.json` of that run without `model_dir`, `numpy`, `wall_seconds` and `cpu_seconds`.

| Accession | Residues | Verified training units of the protein (all units) | Non-standard letters |
|---|---|---|---|
| P07203 | 203 | 68 of 74 | U at residue 49 |
| P04233 | 296 | 852 of 871 | none |
| P60709 | 375 | 997 of 1,168 | none |
| P06733 | 434 | 905 of 1,121 | none |

## What the expected files are

Regression expectations: the scorer's own output when they were written (2026-11-21; Python 3.12.3, numpy 2.4.1, Linux x86-64). They are not an independent truth, and not a measure of performance.

The four proteins are part of the training population: each is the parent of many verified ligand units (third column; the rule that counts them, the peptide equals the UniProt sequence at the stated positions, reproduces the 417,415 verified units and 16,873 proteins of the [Evaluation](../docs/EVALUATION.md)). Their scores are in-sample.

P07203 contains a non-standard letter on purpose: the scorer reads every character other than the 20 standard residues as the pad token ([Specification](../docs/PREDICTOR_SPEC.md)), so the expectations cover that path.

## Source

UniProt sequences (UniProt Consortium, CC BY 4.0), fetched from the UniProt REST service on 2026-11-13; the UniProt release was not recorded. They come from the table of the parent proteins of the training data (`expanded_uniprot_seqs.json` of the reproduction record, sha256 `23a07da98eeba33ba9a57c413085f21c03a80c4fd758f60ffcf3868ac09449a1`). The sequences are unchanged; the header lines were written for this file and the lines are wrapped to 60 residues. Attribution and licence: [NOTICE.md](../NOTICE.md).
