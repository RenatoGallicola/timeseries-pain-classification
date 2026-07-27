#!/usr/bin/env python3
"""Rebuild the exact submitted predictions from the committed logits.

This does not require the dataset, a GPU, or the trained weights: it replays the
last stage of the pipeline (ensemble + gate) from the per-subject logits stored
in `artifacts/`, and verifies the result byte-for-byte against the submitted
`submissions/submission_final.csv`.

    python scripts/rebuild_submission.py            # verify only
    python scripts/rebuild_submission.py -o out.csv # verify and write

Only numpy, pandas and scikit-learn are needed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ENSEMBLE                       # noqa: E402
from src.gate import apply_gate                       # noqa: E402
from src.submission import build_submission, check_submission  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", help="also write the rebuilt submission here")
    args = ap.parse_args()

    test = ROOT / "artifacts" / "test_logits"
    logits_a = np.load(test / "test_logits_A.npy")
    logits_b = np.load(test / "test_logits_B.npy")
    logits_c = np.load(test / "test_logits_C.npy")
    logits_bin = np.load(ROOT / "artifacts" / "binary_logits" / "test_logits_bin_user.npy")

    print(f"Loaded logits: A{logits_a.shape} B{logits_b.shape} "
          f"C{logits_c.shape} binary{logits_bin.shape}")

    ensemble = (ENSEMBLE.w_a * logits_a
                + ENSEMBLE.w_b * logits_b
                + ENSEMBLE.w_c * logits_c)
    print(f"Ensemble weights: wA={ENSEMBLE.w_a} wB={ENSEMBLE.w_b} wC={ENSEMBLE.w_c}")

    preds, stats = apply_gate(ensemble, logits_bin,
                              alpha=ENSEMBLE.gate_alpha,
                              t_conf=ENSEMBLE.gate_confidence)
    print(f"Gate (alpha={ENSEMBLE.gate_alpha}, t_conf={ENSEMBLE.gate_confidence}): {stats}")

    sample = pd.read_csv(ROOT / "data" / "sample_submission.csv")
    rebuilt = build_submission(sample, preds, out_path=args.output)
    check_submission(rebuilt, sample)

    reference = pd.read_csv(ROOT / "submissions" / "submission_final.csv")
    if rebuilt["label"].equals(reference["label"]):
        print("\nOK — rebuilt predictions match submissions/submission_final.csv exactly.")
        return 0

    n_diff = int((rebuilt["label"].values != reference["label"].values).sum())
    print(f"\nMISMATCH — {n_diff} of {len(reference)} predictions differ.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
