"""Windowing, aggregation and alignment.

These are the functions where a silent bug is most expensive: a wrong window
boundary or a lost subject id does not raise, it just quietly degrades the score.
"""
import numpy as np
import pandas as pd
import pytest

from src.data import (add_temporal_features, aggregate_user_logits, align_to,
                      build_sequences_with_ids, class_weights,
                      detect_outlier_users, joint_columns)


@pytest.fixture
def toy_frame():
    """Three subjects, 20 time steps each, two joint channels."""
    rows = []
    for uid in (7, 3, 11):
        for t in range(20):
            rows.append({"sample_index": uid, "time": t,
                         "joint_00": float(uid + t), "joint_01": float(t)})
    return pd.DataFrame(rows)


@pytest.fixture
def toy_labels():
    return pd.DataFrame({"sample_index": [7, 3, 11], "label": [0, 1, 2]})


class TestBuildSequences:
    def test_shapes_and_labels(self, toy_frame, toy_labels):
        seqs, labels, users = build_sequences_with_ids(
            toy_frame, toy_labels, window=10, stride=10)
        assert seqs.shape == (6, 10, 2)          # 3 subjects x 2 windows
        assert labels.shape == users.shape == (6,)
        assert seqs.dtype == np.float32

    def test_label_follows_its_subject(self, toy_frame, toy_labels):
        _, labels, users = build_sequences_with_ids(
            toy_frame, toy_labels, window=10, stride=10)
        mapping = dict(zip(toy_labels["sample_index"], toy_labels["label"]))
        for user, label in zip(users, labels):
            assert label == mapping[user], "a window carries another subject's label"

    def test_overlapping_stride_produces_more_windows(self, toy_frame, toy_labels):
        few, _, _ = build_sequences_with_ids(toy_frame, toy_labels, 10, 10)
        many, _, _ = build_sequences_with_ids(toy_frame, toy_labels, 10, 5)
        assert len(many) > len(few)

    def test_window_content_is_contiguous(self, toy_frame, toy_labels):
        seqs, _, users = build_sequences_with_ids(toy_frame, toy_labels, 10, 10)
        first = seqs[list(users).index(7)]
        # joint_01 is just the time step, so the window must be 0..9
        assert np.allclose(first[:, 1], np.arange(10))

    def test_window_longer_than_signal_is_zero_padded(self, toy_frame, toy_labels):
        """Model B uses a 200-step window on a 160-step signal; this is that case."""
        seqs, _, users = build_sequences_with_ids(toy_frame, toy_labels, 30, 30)
        assert len(seqs) == 3, "each subject should yield exactly one padded window"
        assert np.all(seqs[:, 20:, :] == 0), "the tail should be zero padding"

    def test_test_mode_returns_no_labels(self, toy_frame):
        out = build_sequences_with_ids(toy_frame, None, 10, 10, is_test=True)
        assert len(out) == 2
        assert out[0].shape == (6, 10, 2)

    def test_id_columns_are_not_features(self, toy_frame, toy_labels):
        seqs, _, _ = build_sequences_with_ids(toy_frame, toy_labels, 10, 10)
        assert seqs.shape[2] == 2, "sample_index/time leaked into the features"


class TestAggregation:
    def test_averages_windows_per_subject(self):
        logits = np.array([[1.0, 0, 0], [3.0, 0, 0], [0, 2.0, 0]])
        users = np.array([5, 5, 9])
        ids, pooled = aggregate_user_logits(logits, users)
        assert list(ids) == [5, 9]
        assert np.allclose(pooled[0], [2.0, 0, 0])   # mean of 1 and 3

    def test_output_is_sorted_by_user_id(self):
        logits = np.zeros((3, 3))
        ids, _ = aggregate_user_logits(logits, np.array([9, 2, 5]))
        assert list(ids) == [2, 5, 9]


class TestAlignment:
    def test_reorders_to_target(self):
        logits = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        aligned = align_to(logits, np.array([10, 20, 30]), np.array([30, 10, 20]))
        assert np.allclose(aligned, [[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]])

    def test_identity_when_orders_match(self):
        logits = np.random.default_rng(0).normal(size=(4, 3))
        users = np.array([1, 2, 3, 4])
        assert np.allclose(align_to(logits, users, users), logits)

    def test_missing_subject_raises(self):
        with pytest.raises(KeyError):
            align_to(np.zeros((2, 3)), np.array([1, 2]), np.array([1, 99]))


class TestPreprocessing:
    def test_detects_the_flat_subject(self):
        rng = np.random.default_rng(0)
        rows = []
        for uid in range(30):
            # subject 0 is flat; the rest vary normally
            values = np.zeros(20) if uid == 0 else rng.normal(0, 1 + uid * 0.05, 20)
            for t, v in enumerate(values):
                rows.append({"sample_index": uid, "time": t,
                             "joint_00": v, "joint_01": v * 0.5})
        df = pd.DataFrame(rows)
        assert 0 in detect_outlier_users(df, ["joint_00", "joint_01"], threshold=2.0)

    def test_temporal_features_expand_the_frame(self, toy_frame):
        out = add_temporal_features(toy_frame, ["joint_00"])
        for suffix in ("_d1", "_d2", "_roll3", "_roll3_std"):
            assert f"joint_00{suffix}" in out.columns
        assert len(out) == len(toy_frame)

    def test_derivative_is_correct_and_does_not_cross_subjects(self, toy_frame):
        out = add_temporal_features(toy_frame, ["joint_00"]).sort_values(
            ["sample_index", "time"])
        for _, group in out.groupby("sample_index"):
            d1 = group["joint_00_d1"].values
            assert d1[0] == 0, "first step of a subject must not use the previous one"
            assert np.allclose(d1[1:], 1.0)     # joint_00 increases by 1 per step

    def test_constant_joint_columns_are_dropped(self):
        df = pd.DataFrame({"sample_index": [1, 1], "time": [0, 1],
                           "joint_00": [1.0, 2.0], "joint_30": [5.0, 5.0]})
        assert joint_columns(df) == ["joint_00"]


class TestClassWeights:
    def test_rare_classes_weigh_more(self):
        weights = class_weights(np.array([0] * 511 + [1] * 94 + [2] * 56))
        assert weights[2] > weights[1] > weights[0]

    def test_balanced_data_gives_equal_weights(self):
        weights = class_weights(np.array([0, 1, 2] * 10))
        assert np.allclose(weights, 1.0)
