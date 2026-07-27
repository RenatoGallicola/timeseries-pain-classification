"""Hyperparameters of the submitted solution.

Every value here is the one actually used for the final run; they are collected
in one place so the notebooks and the modules cannot drift apart.
"""
from dataclasses import dataclass, field

SEED = 42
NUM_CLASSES = 3
SEQ_LEN = 160  # time steps per subject, fixed by the dataset

LABELS = ("no_pain", "low_pain", "high_pain")
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
ID_TO_LABEL = {i: name for i, name in enumerate(LABELS)}


@dataclass(frozen=True)
class WindowConfig:
    """Sliding-window parameters. Windows longer than SEQ_LEN are zero-padded."""
    window: int
    stride: int


# Models A and B share an architecture and differ only in how the signal is cut.
WINDOW_A = WindowConfig(window=96, stride=32)
WINDOW_B = WindowConfig(window=200, stride=50)
WINDOW_C = WindowConfig(window=128, stride=16)


@dataclass(frozen=True)
class CnnGruConfig:
    """CNN-GRU classifier (models A and B).

    Note that `use_attention` is False in the submitted configuration: the
    attention head is implemented and available, but mean pooling over the GRU
    states scored better on the out-of-fold folds, so the final models use it.
    """
    num_layers: int = 1
    hidden_size: int = 64
    bidirectional: bool = False
    use_attention: bool = False
    dropout_rate: float = 0.2902960296649729
    learning_rate: float = 0.004160390932108078
    l2_lambda: float = 1.6641256227825607e-05
    label_smoothing: float = 0.030744716461245148
    batch_size: int = 128
    max_epochs_kfold: int = 200
    patience_kfold: int = 30
    max_epochs_full: int = 200


@dataclass(frozen=True)
class InceptionConfig:
    """InceptionTime classifier (model C), trained on the enriched feature set."""
    channels: int = 32
    n_blocks: int = 3
    batch_size: int = 64
    outlier_threshold: float = 3.0  # std-devs for per-subject outlier detection


@dataclass(frozen=True)
class EnsembleConfig:
    """Weights and gate parameters used to build `submissions/submission_final.csv`.

    These values were selected on out-of-fold logits during the Colab session and
    then hard-coded for the final inference pass; `notebooks/01_full_pipeline.ipynb`
    re-runs the CMA-ES search but the submission uses the constants below.
    See docs/RESULTS.md.
    """
    w_a: float = 0.4412
    w_b: float = 0.3246
    w_c: float = 0.2342
    gate_alpha: float = 0.524      # weight of the binary head in the fusion
    gate_confidence: float = 0.59  # above this low/high confidence the gate stays out
    tta_strides_ab: tuple = (11, 53, 50)
    tta_strides_c: tuple = (24, 32)


CNN_GRU = CnnGruConfig()
INCEPTION = InceptionConfig()
ENSEMBLE = EnsembleConfig()


def set_seed(seed: int = SEED) -> None:
    """Seed python, numpy and torch. Called at the top of every notebook."""
    import os
    import random

    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
