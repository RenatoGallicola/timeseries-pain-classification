# Experiments

The eight notebooks in `notebooks/experiments/` are the branches that shaped the
final design — including the ones that failed. They are kept because the reason
something did *not* work is usually the more transferable half of the result.

All of them ran on Google Colab GPUs and keep their original outputs. Scores are
weighted F1 on a subject-grouped validation split unless noted.

---

## Sequence encoders

### `01_cnn_gru_baseline.ipynb` — 0.9301 · **kept, became models A and B**

`Conv1d(37→64, k=5) → BatchNorm → ReLU → Conv1d(64→128, k=3) → BatchNorm → ReLU
→ Dropout → GRU → pooling → MLP`, on windows of 200 steps with stride 50.

Set the bar every later model was measured against. Two findings from this
notebook shaped everything downstream: subject-grouped splits are mandatory (an
ungrouped split inflated validation F1 by several points, since windows of the
same recording landed on both sides), and mean pooling over the GRU states
matched attention pooling while being cheaper — so the shipped models A and B run
with `use_attention=False`.

### `02_cnn_gru_optuna.ipynb` — 0.9262 · discarded

Optuna over hidden size, layers, dropout, learning rate, weight decay and label
smoothing. Did not beat the hand-tuned baseline. The conclusion was that the
architecture was already at the capacity the dataset supports, so the search
budget moved to the *windowing* parameters instead — which is where models A
(96/32) and B (200/50) come from.

### `03_cnn_gru_multibranch_se.ipynb` — 0.9413 · discarded

A deliberately heavier encoder: `MultiBranchConv1D` with parallel kernel widths,
`SqueezeExcite1D` channel attention, and `GaussianNoise` input augmentation.

The best single run beat the baseline, but the Optuna trials plateaued near 0.86
and the spread across trials was much wider than for the plain CNN-GRU — the
classic signature of a model with more capacity than the data supports. None of
these blocks reached the final pipeline. The multi-scale idea did survive, but in
the cleaner form of InceptionTime (model C).

### `04_tcn_hybrid.ipynb` — 0.9272 · discarded

Dilated causal TCN: `Chomp1d`, `TemporalBlock`, `TemporalAttentionPooling`,
`TCNHybrid`, with an Optuna study.

Competitive but never better than CNN-GRU, and considerably slower to train. The
temporal dependencies in this dataset appear to be local and short-range, so the
large receptive field a TCN buys through dilation is not worth its cost here.

### LSTM / BiLSTM / CNN+LSTM · discarded, not kept as notebooks

Tried early and abandoned: LSTMs overfit within a few epochs on 661 subjects, and
the bidirectional variants doubled the parameter count for no gain. The
TensorBoard traces of these runs survive in the original archive
(`lstm_bidirectional`, `lstm_monodirectional_2hiddenlayers`, `rnn`) but the
notebooks were superseded before the final submission. The one surviving
LSTM-based notebook is `07_binary_lstm_focal_loss.ipynb` below.

---

## The low/high subproblem

The three-class ensemble separates `no_pain` almost perfectly and makes nearly
all its remaining errors between `low_pain` (94 subjects) and `high_pain` (56).
Three notebooks attack that subproblem directly. Their scores are on the
two-class problem and are **not** comparable with the three-class figures.

### `05_binary_cnn_gru_optuna.ipynb` · **kept, feeds the gate**

CNN-GRU trunk — deliberately the same shape as the main classifier, so the two
sets of logits live on a comparable scale and fuse cleanly — plus a small MLP
over the static subject attributes (`n_legs`, `n_hands`, `n_eyes`) concatenated
before the classifier head. Tuned with Optuna over StratifiedGroupKFold splits.
This is the model whose logits sit in `artifacts/binary_logits/`.

### `06_binary_cnn1d_kfold.ipynb` — 0.9586 · not shipped

A purely convolutional binary classifier with a full Optuna + StratifiedGroupKFold
protocol. It scored slightly *higher* than the CNN-GRU variant in isolation, but
was not shipped: its logits are less well calibrated against the main ensemble,
and the gate fuses probabilities rather than picking a winner, so agreement of
scale matters more than a fraction of a point of standalone F1.

### `07_binary_lstm_focal_loss.ipynb` · discarded

`LSTMAttentionBinary` trained with a custom `FocalLoss`. The hypothesis was that
down-weighting easy examples would help on the badly under-represented
`high_pain` class.

It did the opposite. On a dataset this small, focal loss amplifies label noise:
by construction it concentrates gradient on the hardest examples, which here are
disproportionately outliers rather than informative minority-class cases.
Training became visibly less stable. The final models use class-weighted
cross-entropy with label smoothing (0.031) instead — a much gentler correction
for the same imbalance. This is the experiment described in section 4 of the
report.

### `08_binary_feature_analysis.ipynb` — 0.9594 · **fed model C's feature set**

Feature-level study: per-channel distributions, correlations between joint groups
and the `pain_survey_*` variables, and two slimmer model variants
(`CNN_GRU_Slim`, `CNN_GRU_Binary_Optim`) trained on reduced feature sets.

The transferable result is the feature engineering, not the models: first and
second derivatives (velocity, acceleration) and rolling mean/std over a 3-step
window carry signal the convolutions were not extracting on their own from a
160-step sequence. That representation — 24 informative joint channels expanded
to 92 features — is exactly what model C consumes in the final pipeline. This
notebook also produced the per-subject outlier criterion in
`src/data.py:detect_outlier_users` (19 subjects with near-flat joint traces,
listed in `data/outliers_model_C.txt`).

---

## Ensembling strategies

Tried, in order of increasing sophistication:

| Strategy | Outcome |
|---|---|
| Simple logit averaging (0.5 / 0.5) | 0.9478 out-of-fold — already better than either model alone |
| Manually tuned `(alpha, 1-alpha)` | Marginal gain, tedious to extend past two models |
| **CMA-ES on out-of-fold logits** | **Shipped.** Scales to three models, no manual sweep |
| 5-fold CV models + three no-split models | Discarded (below) |

**The CV + no-split combination failed.** The idea was that cross-validated models
would capture different decision boundaries while models trained on 100% of the
data would reduce variance, and that mixing them would counter both overfitting
and underfitting. In practice the models were not different *enough* — similar
architectures on similar data produce highly correlated errors — and some folds
contributed unstable logits, which on 661 subjects is enough to drag the average
down.

One caveat on CMA-ES that only became visible when the artifacts were re-examined:
F1 is a step function of the ensemble weights, so the objective landscape is a
set of wide plateaus. Two searches in the same notebook returned clearly
different weight vectors with *identical* F1 to 16 digits. CMA-ES is finding a
plateau, not a unique optimum, and the specific weights it reports carry less
meaning than their score suggests. See `docs/RESULTS.md`.

---

## What we would do differently

* **Fix the metric up front.** Weighted F1, macro F1 and micro F1 answer different
  questions on a 511/94/56 split, and switching between them late in a challenge
  is how inconsistent numbers get reported. Macro-F1 (0.8957) is arguably the
  honest headline here: it refuses to let 511 easy `no_pain` subjects hide the
  0.8148 on `high_pain`.
* **Tune the gate on out-of-fold logits, and check it fires.** The shipped
  threshold leaves the gate dormant — it changes 1 prediction in 1324. A
  mechanism that never activates is indistinguishable from one that does not
  exist, and we did not measure that until after the fact. `src/gate.py` now
  ships a `tune_gate()` grid search and returns intervention statistics, so this
  is visible by construction.
* **Attack `high_pain` recall directly.** 7 of 56 `high_pain` subjects are
  predicted `no_pain` — the single largest block of costly errors. A specialist
  for that boundary would have been a better use of time than a third generalist
  model.
* **Record the provenance of every submitted artifact.** The final weights were
  chosen interactively and pasted into the inference cells; reconstructing which
  run produced them afterwards was harder than it should have been.
