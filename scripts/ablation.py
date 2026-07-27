#!/usr/bin/env python3
"""Recompute the ablation table in docs/RESULTS.md from the committed OOF logits.

Every number printed here comes from `artifacts/oof_logits/`, so the table can be
checked without the dataset or a GPU:

    python scripts/ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ENSEMBLE, LABELS   # noqa: E402
from src.gate import apply_gate           # noqa: E402


def main() -> int:
    oof = ROOT / "artifacts" / "oof_logits"
    a = np.load(oof / "oof_logits_A.npy")
    b = np.load(oof / "oof_logits_B.npy")
    c = np.load(oof / "oof_logits_C.npy")
    y = np.load(oof / "oof_labels.npy")
    users = np.load(oof / "oof_user_ids.npy")

    binary = ROOT / "artifacts" / "binary_logits"
    bin_logits = np.load(binary / "train_logits_bin_user.npy")
    bin_users = np.load(binary / "train_users_bin_user.npy")

    w = ENSEMBLE
    ensemble = w.w_a * a + w.w_b * b + w.w_c * c

    # Scatter the binary logits onto the full subject list.
    mask = np.isin(users, bin_users)
    index = {uid: i for i, uid in enumerate(bin_users)}
    bin_full = np.zeros((len(users), 2))
    for i, uid in enumerate(users):
        if mask[i]:
            bin_full[i] = bin_logits[index[uid]]

    gated, stats = apply_gate(ensemble, bin_full, mask, y,
                              w.gate_alpha, w.gate_confidence)

    rows = [
        ("Model A alone (window 96 / stride 32)", a.argmax(1)),
        ("Model B alone (window 200 / stride 50)", b.argmax(1)),
        ("Model C alone (InceptionTime)", c.argmax(1)),
        ("A + B, equal weights", (0.5 * a + 0.5 * b).argmax(1)),
        ("A + B + C, submission weights", ensemble.argmax(1)),
        ("A + B + C + binary gate", gated),
    ]

    print(f"{len(y)} subjects, out-of-fold predictions\n")
    print(f"{'':42s} {'weighted-F1':>12s} {'macro-F1':>10s}")
    for name, preds in rows:
        print(f"{name:42s} {f1_score(y, preds, average='weighted'):12.4f} "
              f"{f1_score(y, preds, average='macro'):10.4f}")

    print(f"\nGate: {stats}")
    print(f"Subjects with a binary prediction available: {int(mask.sum())} of {len(users)}")

    print("\nFinal ensemble, per class:")
    print(classification_report(y, gated, target_names=list(LABELS), digits=4,
                                zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y, gated))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
