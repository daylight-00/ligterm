# LigTerm-GB (`models/gb`)

Two boosted models, one per terminal bond, on 72 features of the 16 residues on each side of a bond. It is the reference implementation and the boosted component of [LigTerm](MODEL_CARD_ens.md). Features and scoring: [specification](PREDICTOR_SPEC.md). Training population, performance and limits: [Evaluation](EVALUATION.md); only the external test cohort can evaluate this checkpoint.

## Fit

All verified units of the database, the same population as LigTerm. Settings of the development protocol (`hgb` in `meta.json`): scikit-learn `HistGradientBoostingClassifier`, learning rate 0.1, at most 31 leaves, at least 200 samples per leaf, L2 1.0, 255 bins, no early stopping, `random_state` 0.

The number of iterations was chosen from (50, 100, 200, 400, 800) on three group-disjoint folds (sequence-similarity grouping u30). Both termini select 800, the upper end of the grid. The pooled out-of-fold log-loss was still falling at 800 (N 0.3674, 0.3624, 0.3598, 0.3587, 0.3567; C 0.3703, 0.3652, 0.3627, 0.3612, 0.3597 at 50, 100, 200, 400, 800), so the grid, not the data, limits the number of trees; it was fixed beforehand and not extended. The fit took 6 min 18 s on an Intel Xeon Gold 6526Y.

## Files

| File | Content |
|---|---|
| `gb.npz` | The model, 1,341,453 bytes |
| `meta.json` | Name, schema version, family `gb`, window 16, 800 trees per terminus, training description, software versions, hashes of the units, sequences and groups files |
| `sha256.txt` | Hashes of the files above |

## Numerical verification

- The written model was reloaded; its NumPy scores equal the training-reference path (the development library's tokens, features and staged scores) bitwise on 200 random real proteins at write time.
- Command line on 300 random real proteins (50 to 3,000 residues; 181,542 bonds): every protein scored, every score finite. Cost: [Usage](USAGE.md).

Training records: [Reproducing](../REPRODUCING.md).
