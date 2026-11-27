# LigTerm

Sequence-based ranking of observed MHC class II ligand termini.

LigTerm gives every peptide bond of a protein an N-terminal and a C-terminal score and ranks candidate peptides of 9 to 30 residues. It runs on CPU with NumPy, without protein structures or language models.

## Quick start

Python 3.10+ (only 3.12 has been run). From the repository root:

```bash
python3 -m pip install .
ligterm score --fasta testdata/proteins.fasta --model models/ens --out out/ens
```

The models are read from `models/` and are not installed; `python3 -m ligterm` is the same command. Use your own FASTA instead of `testdata/proteins.fasta`.

## Models

- LigTerm ([`models/ens`](models/ens)): LigTerm-GB and window MLP ensemble
- LigTerm-GB ([`models/gb`](models/gb)): gradient boosting

## Outputs

- `bonds.tsv`: N and C scores and ranks of every bond
- `peptides.tsv`: ranked candidate peptides
- `run.json`: run metadata

Scores rank bonds and peptides within one protein. They are not cleavage, binding or presentation probabilities, and they are not comparable across proteins or models.

## Documentation

[Usage](docs/USAGE.md) · [Specification](docs/PREDICTOR_SPEC.md) · Model cards: [LigTerm](docs/MODEL_CARD_ens.md), [LigTerm-GB](docs/MODEL_CARD_gb.md) · [Evaluation](docs/EVALUATION.md) · [Verification](docs/VERIFICATION.md) · [Reproducing the paper](REPRODUCING.md) (research data on Zenodo: in preparation)

## License

MIT ([LICENSE](LICENSE)). Data attribution: [NOTICE.md](NOTICE.md).
