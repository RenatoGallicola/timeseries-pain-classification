"""Submission writing and validation.

`check_submission` exists because every failure it catches is silent at write
time and only shows up as a collapsed leaderboard score.
"""
import numpy as np
import pandas as pd
import pytest

from src.submission import build_submission, check_submission


@pytest.fixture
def sample():
    return pd.DataFrame({"sample_index": [0, 1, 2, 3], "label": ["no_pain"] * 4})


class TestBuildSubmission:
    def test_maps_ids_to_label_names(self, sample):
        out = build_submission(sample, np.array([0, 1, 2, 0]))
        assert list(out["label"]) == ["no_pain", "low_pain", "high_pain", "no_pain"]

    def test_preserves_sample_index_order(self, sample):
        out = build_submission(sample, np.array([0, 0, 0, 0]))
        assert list(out["sample_index"]) == [0, 1, 2, 3]

    def test_does_not_mutate_the_template(self, sample):
        before = sample.copy()
        build_submission(sample, np.array([2, 2, 2, 2]))
        pd.testing.assert_frame_equal(sample, before)

    def test_length_mismatch_raises(self, sample):
        with pytest.raises(ValueError, match="predictions"):
            build_submission(sample, np.array([0, 1]))

    def test_writes_the_file(self, sample, tmp_path):
        path = tmp_path / "sub.csv"
        build_submission(sample, np.array([0, 1, 2, 0]), out_path=str(path))
        reread = pd.read_csv(path)
        assert list(reread.columns) == ["sample_index", "label"]
        assert len(reread) == 4


class TestCheckSubmission:
    def test_accepts_a_valid_submission(self, sample):
        check_submission(build_submission(sample, np.array([0, 1, 2, 0])), sample)

    def test_rejects_wrong_row_count(self, sample):
        bad = build_submission(sample, np.array([0, 1, 2, 0])).iloc[:2]
        with pytest.raises(ValueError, match="rows"):
            check_submission(bad, sample)

    def test_rejects_duplicated_ids(self, sample):
        bad = build_submission(sample, np.array([0, 1, 2, 0]))
        bad.loc[3, "sample_index"] = 0
        with pytest.raises(ValueError, match="duplicated"):
            check_submission(bad, sample)

    def test_rejects_reordered_ids(self, sample):
        bad = build_submission(sample, np.array([0, 1, 2, 0])).iloc[::-1]
        with pytest.raises(ValueError, match="order"):
            check_submission(bad, sample)

    def test_rejects_unknown_labels(self, sample):
        bad = build_submission(sample, np.array([0, 1, 2, 0]))
        bad.loc[0, "label"] = "some_pain"
        with pytest.raises(ValueError, match="unknown labels"):
            check_submission(bad, sample)

    def test_rejects_missing_labels(self, sample):
        bad = build_submission(sample, np.array([0, 1, 2, 0]))
        bad.loc[0, "label"] = None
        with pytest.raises(ValueError):
            check_submission(bad, sample)
