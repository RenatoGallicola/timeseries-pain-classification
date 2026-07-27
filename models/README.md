# Model weights

The trained checkpoints are **not committed** — about 212 MB across ~85 files,
most of them per-fold intermediates that nothing downstream reads.

They are not needed to reproduce the submitted predictions: the per-subject
logits every model produced are committed under `artifacts/`, and
`python scripts/rebuild_submission.py` replays the final ensemble and gate from
them, verifying the result against `submissions/submission_final.csv`.

## Weights the pipeline actually loads

| File | Produced by | Purpose |
|---|---|---|
| `MODEL_A_FULL_FULL.pt` | `notebooks/01_full_pipeline.ipynb`, *A and B training on full training set* | CNN-GRU, window 96 / stride 32, refit on 100% of the training data |
| `MODEL_B_FULL_FULL.pt` | same section | CNN-GRU, window 200 / stride 50, refit on 100% |
| `MODEL_C_FULL.pt` | *C training on full training set* | InceptionTime on the 92-feature representation |
| `binary_classifier_model.pt` | *Binary classifier — model training* | low/high specialist feeding the gate |

The `KFOLD_F{1..5}_{A,B}.pt` checkpoints are the per-fold models used only to
generate the out-of-fold logits in `artifacts/oof_logits/`. Once those logits
exist, the checkpoints are disposable.

## Regenerating them

Open `notebooks/01_full_pipeline.ipynb` on a GPU runtime, add the dataset (see
`data/README.md`) and run top to bottom. On a Colab T4 the full pipeline —
5-fold cross-validation for A and B, the Optuna study for C, the full refits, and
the binary model — takes several hours.

Expect the numbers to differ slightly from `docs/RESULTS.md`. The two notebooks
in this repository are themselves two independent runs of the same design and
their fold-level scores differ by up to a point; cuDNN kernel selection and
`WeightedRandomSampler` draws are not fully deterministic even with the seed
fixed in `src/config.py:set_seed`.
