# Verification

## Regression test

Run from the repository root with the package installed ([Usage](USAGE.md)). The first two lines check that the model files equal their hash lists, the next two score the four proteins of [`testdata/`](../testdata/README.md) with both models, and the last four compare the outputs with the expected files in `testdata/expected/`.

```clean-room
(cd models/ens && sha256sum -c sha256.txt)
(cd models/gb && sha256sum -c sha256.txt)
ligterm score --fasta testdata/proteins.fasta --model models/ens --out out/ens
ligterm score --fasta testdata/proteins.fasta --model models/gb --out out/gb
cmp out/ens/bonds.tsv testdata/expected/ens/bonds.tsv
cmp out/ens/peptides.tsv testdata/expected/ens/peptides.tsv
cmp out/gb/bonds.tsv testdata/expected/gb/bonds.tsv
cmp out/gb/peptides.tsv testdata/expected/gb/peptides.tsv
```

A line that prints nothing and returns 0 (`cmp`) or `OK` (`sha256sum`) passes.

## What the expected files are

Regression expectations: the scorer's own output on the four test proteins, which belong to the training population. They show that the package and the models reproduce that output; they are not a test of performance ([Evaluation](EVALUATION.md)). Scores are printed with six decimals. The files are byte-identical to the output on the machine where they were written.

If `cmp` reports a difference, check the model files first (`sha256sum -c`), then compare the numbers: a difference of the order of the last printed digit (1e-6) points to floating-point arithmetic, a larger one to a different package or model file.

## Model files

Each model folder carries `sha256.txt`. The weights of `models/ens/gb/` equal those of `models/gb` byte for byte.

## Beyond the regression test

The experiments that evaluate the models, and the scripts that refit LigTerm-GB and re-assemble LigTerm from stored components, are in [`reproduction/`](../reproduction/README.md); [Reproducing](../REPRODUCING.md) says what each level of reproduction covers.
