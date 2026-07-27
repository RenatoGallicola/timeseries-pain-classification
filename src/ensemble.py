"""Ensembling: CMA-ES weight search, temperature scaling, test-time augmentation.

Models A, B and C make different mistakes — A and B differ in how much temporal
context each window carries, C sees an explicitly enriched feature set — so a
weighted average of their logits beats any of them alone. The weights are fitted
on out-of-fold logits, never on the fold a model was trained on.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def softmax_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis (works for 1-D and 2-D)."""
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def weighted_logits(logits: list[np.ndarray], weights) -> np.ndarray:
    """Normalise weights to sum to 1 and combine the logit matrices."""
    w = np.asarray(weights, dtype=np.float64)
    if np.any(w < 0):
        raise ValueError("ensemble weights must be non-negative")
    total = w.sum()
    if total <= 0:
        raise ValueError("ensemble weights must not sum to zero")
    w = w / total
    return sum(wi * li for wi, li in zip(w, logits))


def score_weights(logits: list[np.ndarray], weights, y_true: np.ndarray,
                  average: str = "weighted") -> float:
    """F1 of the ensemble defined by `weights`."""
    preds = weighted_logits(logits, weights).argmax(axis=1)
    return f1_score(y_true, preds, average=average)


def optimise_weights_cmaes(logits: list[np.ndarray], y_true: np.ndarray, *,
                           sigma: float = 0.15, popsize: int = 30,
                           maxiter: int = 200, seed: int = 42,
                           verbose: bool = True):
    """Search the ensemble weights with CMA-ES, maximising out-of-fold weighted F1.

    Returns ``(weights, best_f1)`` with the weights normalised to sum to 1.

    A caveat worth knowing before you trust the result: F1 is a step function of
    the weights, so the objective has wide flat regions. Different runs can land
    on visibly different weight vectors that score *identically* — see
    docs/RESULTS.md. Treat the weights as one point on a plateau, not as a
    uniquely optimal solution.
    """
    import cma  # optional dependency, only needed to re-fit the weights

    n = len(logits)
    x0 = [1.0 / n] * n

    def objective(w):
        if np.any(np.asarray(w) < 0) or np.sum(w) <= 0:
            return 10.0
        return -score_weights(logits, w, y_true)

    es = cma.CMAEvolutionStrategy(
        x0, sigma,
        {"bounds": [[0.0] * n, [1.0] * n], "popsize": popsize,
         "maxiter": maxiter, "seed": seed, "verbose": -9 if not verbose else 1},
    )
    es.optimize(objective)

    best = np.asarray(es.result.xbest, dtype=np.float64)
    best = best / best.sum()
    return best, score_weights(logits, best, y_true)


def tune_temperature(logits: np.ndarray, y_true: np.ndarray,
                     grid=np.arange(0.5, 3.01, 0.05)):
    """Pick the temperature that maximises weighted F1 on the given logits.

    Scaling logits by 1/T cannot change an argmax on its own, so this only moves
    the score when the scaled logits are subsequently *mixed* with another head
    (as happens in the gate). Returns ``(best_T, best_f1)``.
    """
    best_t, best_f1 = 1.0, -1.0
    for t in grid:
        f1 = f1_score(y_true, (logits / t).argmax(axis=1), average="weighted")
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


def tta_logits(predict_fn, strides, *, aggregate=np.mean) -> np.ndarray:
    """Average subject-level logits obtained with several window strides.

    `predict_fn(stride)` must return a ``(n_users, n_classes)`` array with the
    subjects in a consistent order. Different strides cut the signal at different
    offsets, so this is a cheap variance reduction at inference time.
    """
    stacked = np.stack([predict_fn(s) for s in strides], axis=0)
    return aggregate(stacked, axis=0)
