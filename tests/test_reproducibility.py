"""End-to-end check that the committed artifacts still produce the submitted file.

This is the test that gives the repository its main claim. If `src/config.py`,
`src/gate.py` or any file under `artifacts/` changes in a way that moves a single
prediction, this fails.
"""
import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.config import ENSEMBLE
from src.gate import apply_gate
from src.submission import build_submission, check_submission

N_TEST_SUBJECTS = 1324
N_TRAIN_SUBJECTS = 661


@pytest.fixture(scope="module")
def test_logits(repo_root):
    d = repo_root / "artifacts" / "test_logits"
    return {k: np.load(d / f"test_logits_{k}.npy") for k in "ABC"}


@pytest.fixture(scope="module")
def oof(repo_root):
    d = repo_root / "artifacts" / "oof_logits"
    return {
        "A": np.load(d / "oof_logits_A.npy"),
        "B": np.load(d / "oof_logits_B.npy"),
        "C": np.load(d / "oof_logits_C.npy"),
        "y": np.load(d / "oof_labels.npy"),
        "users": np.load(d / "oof_user_ids.npy"),
    }


class TestArtifactIntegrity:
    def test_test_logits_have_the_expected_shape(self, test_logits):
        for name, logits in test_logits.items():
            assert logits.shape == (N_TEST_SUBJECTS, 3), f"model {name}"
            assert np.isfinite(logits).all(), f"model {name} contains NaN or inf"

    def test_oof_logits_have_the_expected_shape(self, oof):
        for name in "ABC":
            assert oof[name].shape == (N_TRAIN_SUBJECTS, 3), f"model {name}"
        assert oof["y"].shape == (N_TRAIN_SUBJECTS,)

    def test_oof_user_ids_are_unique(self, oof):
        assert len(np.unique(oof["users"])) == N_TRAIN_SUBJECTS

    def test_label_distribution_matches_the_dataset(self, oof):
        counts = np.bincount(oof["y"], minlength=3)
        assert list(counts) == [511, 94, 56]

    def test_binary_logits(self, repo_root):
        d = repo_root / "artifacts" / "binary_logits"
        train = np.load(d / "train_logits_bin_user.npy")
        users = np.load(d / "train_users_bin_user.npy")
        test = np.load(d / "test_logits_bin_user.npy")
        assert train.shape == (150, 2)
        assert users.shape == (150,)
        assert test.shape == (N_TEST_SUBJECTS, 2)


class TestSubmissionIsReproducible:
    def test_rebuilds_the_submitted_predictions_exactly(self, repo_root, test_logits):
        ensemble = (ENSEMBLE.w_a * test_logits["A"]
                    + ENSEMBLE.w_b * test_logits["B"]
                    + ENSEMBLE.w_c * test_logits["C"])
        binary = np.load(repo_root / "artifacts" / "binary_logits"
                         / "test_logits_bin_user.npy")

        preds, _ = apply_gate(ensemble, binary, alpha=ENSEMBLE.gate_alpha,
                              t_conf=ENSEMBLE.gate_confidence)

        sample = pd.read_csv(repo_root / "data" / "sample_submission.csv")
        rebuilt = build_submission(sample, preds)
        submitted = pd.read_csv(repo_root / "submissions" / "submission_final.csv")

        n_diff = int((rebuilt["label"].values != submitted["label"].values).sum())
        assert n_diff == 0, f"{n_diff} predictions differ from the submitted file"

    def test_submitted_file_passes_its_own_validation(self, repo_root):
        sample = pd.read_csv(repo_root / "data" / "sample_submission.csv")
        submitted = pd.read_csv(repo_root / "submissions" / "submission_final.csv")
        check_submission(submitted, sample)

    def test_submitted_class_distribution(self, repo_root):
        submitted = pd.read_csv(repo_root / "submissions" / "submission_final.csv")
        counts = submitted["label"].value_counts().to_dict()
        assert counts == {"no_pain": 1028, "low_pain": 176, "high_pain": 120}


class TestDocumentedResults:
    """Pin the numbers quoted in docs/RESULTS.md."""

    @pytest.mark.parametrize("model,expected", [("A", 0.9454), ("B", 0.9286),
                                                ("C", 0.8803)])
    def test_single_model_scores(self, oof, model, expected):
        f1 = f1_score(oof["y"], oof[model].argmax(1), average="weighted")
        assert f1 == pytest.approx(expected, abs=5e-5)

    def test_equal_weight_ab_ensemble(self, oof):
        f1 = f1_score(oof["y"], (0.5 * oof["A"] + 0.5 * oof["B"]).argmax(1),
                      average="weighted")
        assert f1 == pytest.approx(0.9478, abs=5e-5)

    def test_full_ensemble(self, oof):
        ensemble = (ENSEMBLE.w_a * oof["A"] + ENSEMBLE.w_b * oof["B"]
                    + ENSEMBLE.w_c * oof["C"])
        assert f1_score(oof["y"], ensemble.argmax(1),
                        average="weighted") == pytest.approx(0.9526, abs=5e-5)

    def test_adding_c_helps_despite_its_weak_standalone_score(self, oof):
        """The central claim of the ablation in docs/RESULTS.md."""
        ab = f1_score(oof["y"], (0.5 * oof["A"] + 0.5 * oof["B"]).argmax(1),
                      average="weighted")
        abc = f1_score(oof["y"], (ENSEMBLE.w_a * oof["A"] + ENSEMBLE.w_b * oof["B"]
                                  + ENSEMBLE.w_c * oof["C"]).argmax(1),
                       average="weighted")
        c_alone = f1_score(oof["y"], oof["C"].argmax(1), average="weighted")
        assert c_alone < ab, "C should be the weakest model on its own"
        assert abc > ab, "yet adding it should improve the ensemble"

    def test_gate_is_dormant_at_the_shipped_operating_point(self, oof, repo_root):
        """documented in docs/RESULTS.md: it fires on exactly one subject."""
        d = repo_root / "artifacts" / "binary_logits"
        bin_logits = np.load(d / "train_logits_bin_user.npy")
        bin_users = np.load(d / "train_users_bin_user.npy")

        mask = np.isin(oof["users"], bin_users)
        index = {uid: i for i, uid in enumerate(bin_users)}
        full = np.zeros((len(oof["users"]), 2))
        for i, uid in enumerate(oof["users"]):
            if mask[i]:
                full[i] = bin_logits[index[uid]]

        ensemble = (ENSEMBLE.w_a * oof["A"] + ENSEMBLE.w_b * oof["B"]
                    + ENSEMBLE.w_c * oof["C"])
        _, stats = apply_gate(ensemble, full, mask, oof["y"],
                              ENSEMBLE.gate_alpha, ENSEMBLE.gate_confidence)
        assert stats.n_interventions == 1


def load_script(repo_root, name):
    """Import a file from `scripts/` as a module.

    The scripts are imported and called in-process rather than spawned as
    subprocesses: a child process racing the parent over the same working
    directory made these two tests intermittently fail on Windows, with no
    usable diagnostic. CI still invokes both scripts as real commands, so the
    command-line path stays covered.
    """
    path = repo_root / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestScripts:
    def test_rebuild_submission_reports_a_match(self, repo_root, capsys,
                                                monkeypatch):
        monkeypatch.chdir(repo_root)
        monkeypatch.setattr(sys, "argv", ["rebuild_submission.py"])
        exit_code = load_script(repo_root, "rebuild_submission.py").main()
        out = capsys.readouterr().out
        assert exit_code == 0, out
        assert "match" in out, out

    def test_ablation_reports_the_documented_score(self, repo_root, capsys,
                                                  monkeypatch):
        monkeypatch.chdir(repo_root)
        exit_code = load_script(repo_root, "ablation.py").main()
        out = capsys.readouterr().out
        assert exit_code == 0, out
        assert "0.9526" in out, out
