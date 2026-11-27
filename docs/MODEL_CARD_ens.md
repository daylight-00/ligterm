# LigTerm (`models/ens`)

Ensemble of [LigTerm-GB](MODEL_CARD_gb.md) and three seeds of a window MLP, fitted on all verified training units. Architecture and score: [specification](PREDICTOR_SPEC.md). Training population, performance and limits: [Evaluation](EVALUATION.md); only the external test cohort can evaluate this checkpoint.

## Files

`models/ens/` is 7.2 MB; hashes are in `sha256.txt`.

- `gb/`: the LigTerm-GB model, byte-identical in its weights to `models/gb`.
- `mlp.npz`: per seed k = 0, 1, 2 and terminus T the arrays `s{k}_{T}_{proj_w, proj_b, mix1_w, mix1_b, mix2_w, mix2_b}`.
- `meta.json`: name, schema version, family `ens`, seeds, E*, fit device, hashes of the boosted files and of the training rows, status and note.

## Training

Rows: those of LigTerm-GB (`training_rows_sha256` `caa5606e...`, asserted at build time). Head: AdamW, learning rate 1e-3, weight decay 0.01, 1,024 rows per step, at most 40 epochs, patience 3. E* is the first epoch of the lowest hold-out loss on one fold; the head is then refitted on all rows for E* epochs. The recipe is the development protocol's, not re-optimised.

One H100 PCIe GPU (torch 2.15.0+cu132, CUDA 13.2, numpy 2.5.4, Python 3.12.3), about 100 s per head. No E* is at the cap. A head was reproduced bit for bit from its seed on the same device and software only.

| Seed index | N: seed, E* | C: seed, E* | NumPy vs torch head, 20,000 rows (max abs logit difference, N / C) |
|---|---|---|---|
| 0 | 578757013, 6 | 2070612744, 6 | 4.8e-06 / 5.1e-06 |
| 1 | 118025102, 6 | 393222615, 6 | 5.2e-06 / 1.0e-05 |
| 2 | 711601904, 5 | 434791113, 6 | 6.2e-06 / 4.6e-06 |

## Numerical verification

- NumPy head against torch head: 1.05e-5 at most in the logit, every head. Build guards before writing: release fits, E* below the cap, distinct seeds, head agreement < 1e-4.
- Reloaded after writing: random windows score bitwise equal to the in-memory model; the boosted component equals the boosted release model bitwise.
- Command line on 300 real proteins (181,542 bonds): every score finite; the mean head logit equals the torch heads to 2.8e-06 (N) and 3.0e-06 (C).
- Code tests: GELU table against torch 1.2e-6; head against the torch module on 5 x 3,000 windows with pads 5.7e-6; window tokens against the training code; bitwise batching invariance and save/reload.

Training records: [Reproducing](../REPRODUCING.md).
