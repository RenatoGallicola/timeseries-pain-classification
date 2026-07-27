# Data

The Pirate Pain dataset belongs to the AN2DL course and is **not redistributed
here**. The raw CSVs are also too large for a git repository (`pirate_pain_test.csv`
alone is 125 MB, above GitHub's 100 MB per-file limit).

## What is committed

| File | Description |
|---|---|
| `sample_submission.csv` | Submission template — 1324 rows. Its row order is authoritative: all logits are aligned to it before the final `argmax`. |
| `feature_list_model_C.txt` | The 92 engineered feature names consumed by model C. |
| `outliers_model_C.txt` | The 19 subject ids excluded from model C's training set. |

## What you need to add

Place the four competition files here (or point `PIRATE_DATA_DIR` at them):

```
data/
  pirate_pain_train.csv          61 MB   105,760 rows
  pirate_pain_train_labels.csv           661 rows
  pirate_pain_test.csv          125 MB   211,840 rows
  sample_submission.csv                  1,324 rows   (already committed)
```

They come from the AN2DL 2025/26 Challenge 1 Kaggle competition, which is
restricted to enrolled students.

## Schema

One row per (subject, time step). 40 columns:

| Column(s) | Meaning |
|---|---|
| `sample_index` | Subject id — the grouping key for every train/validation split |
| `time` | Time step, 0–159 |
| `pain_survey_1` … `pain_survey_4` | Rule-based sensor aggregations estimating perceived pain |
| `n_legs`, `n_hands`, `n_eyes` | Static subject attributes; fed to the binary model through a small MLP |
| `joint_00` … `joint_30` | Body-joint angle trajectories — the bulk of the signal |

Labels live in a separate file: `sample_index,label` with `label` in
`{no_pain, low_pain, high_pain}`.

### Facts worth knowing before you model

* **Every subject has exactly 160 time steps**, train and test alike. No padding
  or truncation is needed for windows shorter than 160; model B's window of 200
  does get zero-padded, deliberately, so that each subject yields exactly one
  full-context window.
* **661 training subjects, 1324 test subjects** — the test set is twice the size
  of the training set, which is the main reason the solution leans so heavily on
  variance reduction (windowing, K-fold, TTA, ensembling).
* **`joint_30` is constant** and carries no information; `src/data.py:joint_columns`
  drops zero-variance joint columns automatically.
* **Classes are imbalanced 511 / 94 / 56.** A model that always predicts
  `no_pain` scores 0.77 accuracy and 0.29 macro-F1 — worth keeping in mind when
  reading any headline accuracy figure.
* **No validation split is provided.** Build your own, and group it by
  `sample_index`: an ungrouped random split puts windows of the same recording on
  both sides of the boundary and inflates validation scores by several points.
