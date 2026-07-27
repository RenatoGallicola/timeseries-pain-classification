"""Loading, preprocessing and windowing of the Pirate Pain dataset.

The raw data is one row per (subject, time step): 661 training subjects and
1324 test subjects, 160 time steps each. Models consume fixed-length windows,
so every subject yields several training examples and predictions are pooled
back to subject level at inference time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LABEL_TO_ID, NUM_CLASSES

ID_COLS = ["sample_index", "time"]
STATIC_COLS = ["n_legs", "n_hands", "n_eyes"]
SURVEY_COLS = [f"pain_survey_{i}" for i in range(1, 5)]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw(data_dir: str = "data"):
    """Read the four competition CSVs. See data/README.md on how to obtain them."""
    x_train = pd.read_csv(f"{data_dir}/pirate_pain_train.csv")
    y_train = pd.read_csv(f"{data_dir}/pirate_pain_train_labels.csv")
    x_test = pd.read_csv(f"{data_dir}/pirate_pain_test.csv")
    sample_sub = pd.read_csv(f"{data_dir}/sample_submission.csv")
    return x_train, y_train, x_test, sample_sub


def encode_labels(y_df: pd.DataFrame) -> pd.DataFrame:
    """Map the textual labels to 0/1/2 in place-safe fashion."""
    y_df = y_df.copy()
    if y_df["label"].dtype == object:
        y_df["label"] = y_df["label"].map(LABEL_TO_ID)
    return y_df


def joint_columns(df: pd.DataFrame) -> list[str]:
    """Joint-angle columns, dropping constant ones (e.g. joint_30 in this dataset)."""
    cols = [c for c in df.columns if c.startswith("joint_")]
    return [c for c in cols if df[c].nunique() > 1]


# --------------------------------------------------------------------------- #
# Preprocessing for model C
# --------------------------------------------------------------------------- #
def detect_outlier_users(df: pd.DataFrame, joint_cols: list[str],
                         threshold: float = 3.0) -> list[int]:
    """Subjects whose mean per-joint standard deviation is `threshold` sigmas
    away from the population mean — typically flat or saturated recordings.

    These subjects are excluded from model C's training set (but still predicted
    on at test time).
    """
    user_stats = df.groupby("sample_index")[joint_cols].std().mean(axis=1)
    mean, std = user_stats.mean(), user_stats.std()
    mask = (user_stats < mean - threshold * std) | (user_stats > mean + threshold * std)
    return user_stats[mask].index.tolist()


def add_temporal_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Append first/second derivative and rolling mean/std (window 3) per subject.

    This is what takes model C's input from ~24 joint channels to 92 features:
    the convolutions get explicit velocity and acceleration instead of having to
    learn them from a 160-step signal on 661 subjects.
    """
    df = df.copy().sort_values(["sample_index", "time"])
    grouped = df.groupby("sample_index")

    for col in feature_cols:
        df[f"{col}_d1"] = grouped[col].diff().fillna(0)
        df[f"{col}_d2"] = df.groupby("sample_index")[f"{col}_d1"].diff().fillna(0)
        roll = df.groupby("sample_index")[col].rolling(window=3, min_periods=1)
        df[f"{col}_roll3"] = roll.mean().reset_index(0, drop=True)
        df[f"{col}_roll3_std"] = roll.std().reset_index(0, drop=True).fillna(0)

    return df


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def build_sequences_with_ids(df: pd.DataFrame, y_df: pd.DataFrame | None,
                             window: int, stride: int, is_test: bool = False):
    """Cut every subject's signal into overlapping windows.

    Returns ``(sequences, labels, user_ids)`` for training data and
    ``(sequences, user_ids)`` for test data. `user_ids` is what allows the
    window-level predictions to be pooled back per subject and, during
    cross-validation, keeps all windows of a subject inside the same fold.
    """
    all_seqs, all_labels, all_users = [], [], []
    rel_cols = [c for c in df.columns if c not in ID_COLS]

    for uid in df["sample_index"].unique():
        temp = df[df["sample_index"] == uid][rel_cols].values
        label = None
        if not is_test:
            label = y_df[y_df["sample_index"] == uid]["label"].values[0]

        # zero-pad so the last window is complete (relevant when window > 160)
        pad = window - len(temp) % window
        if pad < window:
            temp = np.concatenate(
                [temp, np.zeros((pad, temp.shape[1]), dtype=np.float32)], axis=0)

        idx = 0
        while idx + window <= len(temp):
            all_seqs.append(temp[idx:idx + window])
            all_users.append(uid)
            if not is_test:
                all_labels.append(label)
            idx += stride

    sequences = np.asarray(all_seqs, dtype=np.float32)
    users = np.asarray(all_users)
    if is_test:
        return sequences, users
    return sequences, np.asarray(all_labels, dtype=np.int64), users


def aggregate_user_logits(seq_logits: np.ndarray, seq_user_ids: np.ndarray):
    """Average window-level logits into one prediction per subject.

    Returns ``(user_ids_sorted, user_logits)``.
    """
    cols = [f"logit_{i}" for i in range(NUM_CLASSES)]
    df = pd.DataFrame(seq_logits, columns=cols)
    df["user_id"] = seq_user_ids
    user_df = df.groupby("user_id")[cols].mean().reset_index().sort_values("user_id")
    return user_df["user_id"].values, user_df[cols].values


def align_to(logits: np.ndarray, user_ids: np.ndarray,
             target_user_ids: np.ndarray) -> np.ndarray:
    """Reorder per-subject logits to match `target_user_ids`.

    Models A, B and C window the data differently and therefore emit their
    subjects in different orders; every ensemble step goes through here first.
    """
    index = {uid: i for i, uid in enumerate(user_ids)}
    return np.vstack([logits[index[uid]] for uid in target_user_ids])


def class_weights(labels: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights, used in the weighted cross-entropy."""
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    weights = len(labels) / (NUM_CLASSES * np.maximum(counts, 1.0))
    return weights
