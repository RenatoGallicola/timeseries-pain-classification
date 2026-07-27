"""The confidence-based gate that fuses the binary head into the main ensemble.

Motivation: the three-class ensemble separates `no_pain` from the rest almost
perfectly, and makes most of its remaining errors deciding between `low_pain`
and `high_pain` — the two rarest classes. So instead of letting the binary
specialist touch every prediction, it is consulted only where the ensemble is
demonstrably unsure:

    ensemble says no_pain                     -> keep it, gate never fires
    ensemble says low/high, confident         -> keep it
    ensemble says low/high, not confident     -> fuse with the binary head

Confidence is measured *after* renormalising over {low, high} only, so it asks
"given that this subject is in pain, how sure are we which kind" rather than
"how sure are we overall".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ensemble import softmax_np


@dataclass
class GateStats:
    """Bookkeeping for one gate pass. `fixed`/`worsened` need true labels."""
    n_total: int = 0
    n_interventions: int = 0
    fixed: int = 0
    worsened: int = 0
    both_correct: int = 0
    both_wrong: int = 0

    @property
    def net_gain(self) -> int:
        return self.fixed - self.worsened

    def __str__(self) -> str:
        return (f"gate fired on {self.n_interventions}/{self.n_total} subjects | "
                f"fixed {self.fixed}, worsened {self.worsened} "
                f"(net {self.net_gain:+d})")


def apply_gate(logits_ens: np.ndarray, logits_bin: np.ndarray,
               bin_mask: np.ndarray | None = None,
               y_true: np.ndarray | None = None,
               alpha: float = 0.524, t_conf: float = 0.59):
    """Refine ensemble predictions with the binary low/high head.

    Args:
        logits_ens: ``(N, 3)`` logits of the A+B+C ensemble.
        logits_bin: ``(N, 2)`` logits of the binary head, aligned subject-wise.
        bin_mask:   ``(N,)`` True where a binary prediction exists. Defaults to all True.
        y_true:     ``(N,)`` optional ground truth; only used to fill in `GateStats`.
        alpha:      weight of the binary head in the fusion, in ``[0, 1]``.
        t_conf:     low/high confidence at or above which the gate stays out.

    Returns:
        ``(predictions, stats)`` — predictions are 0/1/2.
    """
    logits_ens = np.asarray(logits_ens, dtype=np.float64)
    logits_bin = np.asarray(logits_bin, dtype=np.float64)
    n = len(logits_ens)

    if logits_bin.shape[0] != n:
        raise ValueError(
            f"binary logits ({logits_bin.shape[0]}) are not aligned with the "
            f"ensemble logits ({n}); align them by subject id first")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    if bin_mask is None:
        bin_mask = np.ones(n, dtype=bool)

    probs_ens = softmax_np(logits_ens)
    probs_bin = softmax_np(logits_bin)
    ens_preds = probs_ens.argmax(axis=1)

    preds = ens_preds.copy()
    stats = GateStats(n_total=n)

    # Renormalise over {low, high} for the subjects the ensemble put in pain.
    in_pain = ens_preds != 0
    lh = probs_ens[:, 1:]
    lh = lh / np.maximum(lh.sum(axis=1, keepdims=True), 1e-12)
    uncertain = in_pain & bin_mask & (lh.max(axis=1) < t_conf)

    if uncertain.any():
        fused = alpha * probs_bin[uncertain] + (1.0 - alpha) * lh[uncertain]
        preds[uncertain] = 1 + (fused[:, 1] > fused[:, 0]).astype(int)
        stats.n_interventions = int(uncertain.sum())

    if y_true is not None:
        y_true = np.asarray(y_true)
        idx = np.flatnonzero(uncertain)
        ens_ok = ens_preds[idx] == y_true[idx]
        gate_ok = preds[idx] == y_true[idx]
        stats.fixed = int((~ens_ok & gate_ok).sum())
        stats.worsened = int((ens_ok & ~gate_ok).sum())
        stats.both_correct = int((ens_ok & gate_ok).sum())
        stats.both_wrong = int((~ens_ok & ~gate_ok).sum())

    return preds, stats


def tune_gate(logits_ens: np.ndarray, logits_bin: np.ndarray, y_true: np.ndarray,
              bin_mask: np.ndarray | None = None,
              alphas=np.arange(0.0, 1.01, 0.02),
              confidences=np.arange(0.50, 0.96, 0.01)):
    """Grid-search ``(alpha, t_conf)`` on out-of-fold logits.

    Returns ``(best_alpha, best_t_conf, best_f1)``. Run this on OOF logits only —
    tuning the gate on the same predictions you report is how a gate that does
    nothing ends up looking like it helps.
    """
    from sklearn.metrics import f1_score

    best = (0.5, 0.5, -1.0)
    for alpha in alphas:
        for t_conf in confidences:
            preds, _ = apply_gate(logits_ens, logits_bin, bin_mask, None,
                                  float(alpha), float(t_conf))
            f1 = f1_score(y_true, preds, average="weighted")
            if f1 > best[2]:
                best = (float(alpha), float(t_conf), f1)
    return best
