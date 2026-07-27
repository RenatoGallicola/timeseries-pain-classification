# Artifacts

Per-subject logits produced by the submitted run. These are what make the final
stage of the pipeline reproducible without the dataset, a GPU, or the trained
weights — `scripts/rebuild_submission.py` and `scripts/ablation.py` read only
this directory.

## Out-of-fold logits — `oof_logits/`

Predictions for the 661 training subjects, each made by a model that never saw
that subject during training (5-fold, grouped by `sample_index`). These are the
logits the CMA-ES ensemble weights are fitted on.

| File | Shape | Contents |
|---|---|---|
| `oof_logits_A.npy` | (661, 3) | CNN-GRU, window 96 / stride 32 |
| `oof_logits_B.npy` | (661, 3) | CNN-GRU, window 200 / stride 50 |
| `oof_logits_C.npy` | (661, 3) | InceptionTime, 92 features, aligned to the A/B subject order |
| `oof_labels.npy` | (661,) | Ground truth, 0 = `no_pain`, 1 = `low_pain`, 2 = `high_pain` |
| `oof_user_ids.npy` | (661,) | Subject ids — the reference order for A, B and C |

## Test logits — `test_logits/`

Predictions for the 1324 test subjects, from the models refit on 100% of the
training data, after multi-stride test-time augmentation. Rows are already
aligned to the order of `data/sample_submission.csv`.

| File | Shape |
|---|---|
| `test_logits_A.npy` | (1324, 3) |
| `test_logits_B.npy` | (1324, 3) |
| `test_logits_C.npy` | (1324, 3) |

## Binary head — `binary_logits/`

Logits of the `low_pain` vs `high_pain` specialist. Column 0 is `low_pain`,
column 1 is `high_pain`.

| File | Shape | Contents |
|---|---|---|
| `train_logits_bin_user.npy` | (150, 2) | The 150 training subjects that are actually `low_pain` or `high_pain` |
| `train_users_bin_user.npy` | (150,) | Their subject ids |
| `test_logits_bin_user.npy` | (1324, 2) | All test subjects, aligned to `sample_submission.csv` |

## Alignment

Models A, B and C window the signal differently and therefore emit their subjects
in different orders. Every file here has already been passed through
`src/data.py:align_to`, so the *n*-th row refers to the same subject across all
of them. If you regenerate any of these, align before combining.

Misalignment raises no exception — the shapes still match — so it is worth
knowing what it costs. Measured on these out-of-fold logits:

| Scenario | weighted-F1 |
|---|---:|
| Correctly aligned | 0.9526 |
| Model C alone misaligned | 0.9473 |
| All three models misaligned | 0.6794 |
| Final predictions written in the wrong subject order | 0.6242 |

The first failure mode is the dangerous one: a single misaligned model still
scores 0.947, which looks entirely plausible and would have been shipped without
anyone noticing.
