"""Build and sanity-check the competition submission file."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ID_TO_LABEL


def build_submission(sample_submission: pd.DataFrame, predictions: np.ndarray,
                     out_path: str | None = None) -> pd.DataFrame:
    """Write ``sample_index,label`` in the order given by `sample_submission`.

    The row order of `sample_submission` is authoritative: all logits must have
    been aligned to it before this point, otherwise predictions land on the
    wrong subjects and the score silently collapses.
    """
    if len(predictions) != len(sample_submission):
        raise ValueError(
            f"{len(predictions)} predictions for "
            f"{len(sample_submission)} rows in sample_submission")

    submission = sample_submission.copy()
    submission["label"] = [ID_TO_LABEL[int(p)] for p in predictions]

    if out_path:
        submission.to_csv(out_path, index=False)
        print(f"Submission written to {out_path}")
        print(submission["label"].value_counts().to_string())

    return submission


def check_submission(submission: pd.DataFrame,
                     sample_submission: pd.DataFrame) -> None:
    """Fail loudly on the mistakes that cost a leaderboard slot."""
    problems = []

    if list(submission.columns) != ["sample_index", "label"]:
        problems.append(f"unexpected columns: {list(submission.columns)}")
    if len(submission) != len(sample_submission):
        problems.append(f"{len(submission)} rows, expected {len(sample_submission)}")
    if submission["sample_index"].duplicated().any():
        problems.append("duplicated sample_index values")
    if not submission["sample_index"].equals(sample_submission["sample_index"]):
        problems.append("sample_index order differs from sample_submission")

    unknown = set(submission["label"]) - set(ID_TO_LABEL.values())
    if unknown:
        problems.append(f"unknown labels: {sorted(unknown)}")
    if submission["label"].isna().any():
        problems.append("missing labels")

    if problems:
        raise ValueError("Invalid submission:\n  - " + "\n  - ".join(problems))

    print(f"Submission OK: {len(submission)} rows.")
    print(submission["label"].value_counts().to_string())
