"""The ensemble weighting and the confidence gate.

The gate has three branches and they are easy to get subtly wrong, so each is
pinned down with a hand-built case rather than a fixture.
"""
import numpy as np
import pytest

from src.ensemble import (optimise_weights_cmaes, score_weights, softmax_np,
                          tta_logits, tune_temperature, weighted_logits)
from src.gate import GateStats, apply_gate, tune_gate


class TestSoftmax:
    def test_rows_sum_to_one(self):
        probs = softmax_np(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_stable_on_large_values(self):
        probs = softmax_np(np.array([[1000.0, 1001.0, 999.0]]))
        assert np.isfinite(probs).all()
        assert np.allclose(probs.sum(), 1.0)

    def test_handles_one_dimensional_input(self):
        assert np.allclose(softmax_np(np.array([1.0, 1.0])), [0.5, 0.5])


class TestWeightedLogits:
    def test_weights_are_normalised(self):
        a, b = np.ones((2, 3)), np.zeros((2, 3))
        # 2 and 2 must behave like 0.5 and 0.5
        assert np.allclose(weighted_logits([a, b], [2.0, 2.0]), 0.5)

    def test_negative_weights_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            weighted_logits([np.zeros((2, 3))] * 2, [-1.0, 2.0])

    def test_zero_sum_rejected(self):
        with pytest.raises(ValueError, match="sum to zero"):
            weighted_logits([np.zeros((2, 3))] * 2, [0.0, 0.0])

    def test_score_matches_manual_f1(self):
        logits = np.array([[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]])
        y = np.array([0, 1, 2])
        assert score_weights([logits], [1.0], y) == pytest.approx(1.0)


class TestTemperature:
    def test_returns_a_temperature_from_the_grid(self):
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(50, 3))
        y = logits.argmax(axis=1)
        t, f1 = tune_temperature(logits, y)
        assert t > 0 and f1 == pytest.approx(1.0)


class TestTTA:
    def test_averages_over_strides(self):
        calls = []

        def predict(stride):
            calls.append(stride)
            return np.full((4, 3), float(stride))

        out = tta_logits(predict, [10, 20, 30])
        assert calls == [10, 20, 30]
        assert np.allclose(out, 20.0)


class TestGate:
    """0 = no_pain, 1 = low_pain, 2 = high_pain."""

    def test_no_pain_prediction_is_never_gated(self):
        ensemble = np.array([[10.0, 0.0, 0.0]])
        # a binary head screaming "high_pain" must not override a no_pain call
        binary = np.array([[-10.0, 10.0]])
        preds, stats = apply_gate(ensemble, binary, alpha=1.0, t_conf=0.99)
        assert preds[0] == 0
        assert stats.n_interventions == 0

    def test_confident_low_high_prediction_is_kept(self):
        # renormalised over {low, high} this is ~0.999 confident on low
        ensemble = np.array([[0.0, 10.0, 3.0]])
        binary = np.array([[-10.0, 10.0]])
        preds, stats = apply_gate(ensemble, binary, alpha=1.0, t_conf=0.59)
        assert preds[0] == 1
        assert stats.n_interventions == 0

    def test_uncertain_prediction_is_fused_with_the_binary_head(self):
        # low and high are nearly tied for the ensemble -> the gate fires
        ensemble = np.array([[0.0, 5.0, 5.0]])
        binary = np.array([[-5.0, 5.0]])          # binary says high
        preds, stats = apply_gate(ensemble, binary, alpha=1.0, t_conf=0.9)
        assert stats.n_interventions == 1
        assert preds[0] == 2, "the binary head should have decided this one"

    def test_alpha_zero_ignores_the_binary_head(self):
        ensemble = np.array([[0.0, 5.01, 5.0]])
        binary = np.array([[-5.0, 5.0]])
        preds, _ = apply_gate(ensemble, binary, alpha=0.0, t_conf=0.9)
        assert preds[0] == 1, "with alpha=0 the ensemble must win"

    def test_missing_binary_prediction_falls_back_to_the_ensemble(self):
        ensemble = np.array([[0.0, 5.0, 5.0]])
        binary = np.array([[-5.0, 5.0]])
        preds, stats = apply_gate(ensemble, binary, bin_mask=np.array([False]),
                                  alpha=1.0, t_conf=0.9)
        assert stats.n_interventions == 0
        assert preds[0] in (1, 2)

    def test_misaligned_binary_logits_raise(self):
        with pytest.raises(ValueError, match="not aligned"):
            apply_gate(np.zeros((5, 3)), np.zeros((4, 2)))

    def test_alpha_outside_unit_interval_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            apply_gate(np.zeros((2, 3)), np.zeros((2, 2)), alpha=1.5)

    def test_stats_count_fixes_and_regressions(self):
        # subject 0: ensemble wrong (low), binary right (high) -> a fix
        ensemble = np.array([[0.0, 5.0, 5.0]])
        binary = np.array([[-5.0, 5.0]])
        _, stats = apply_gate(ensemble, binary, None, np.array([2]),
                              alpha=1.0, t_conf=0.9)
        assert stats.fixed == 1 and stats.worsened == 0
        assert stats.net_gain == 1

    def test_gate_is_a_no_op_when_threshold_is_zero(self):
        rng = np.random.default_rng(0)
        ensemble = rng.normal(size=(50, 3))
        binary = rng.normal(size=(50, 2))
        preds, stats = apply_gate(ensemble, binary, alpha=1.0, t_conf=0.0)
        assert stats.n_interventions == 0
        assert np.array_equal(preds, ensemble.argmax(axis=1))

    def test_tune_gate_finds_parameters_that_do_not_hurt(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 3, size=60)
        ensemble = np.eye(3)[y] * 2 + rng.normal(0, 0.8, size=(60, 3))
        binary = np.eye(2)[np.clip(y - 1, 0, 1)] * 2
        alpha, t_conf, f1 = tune_gate(ensemble, binary, y,
                                      alphas=np.arange(0, 1.01, 0.25),
                                      confidences=np.arange(0.5, 0.96, 0.1))
        baseline = score_weights([ensemble], [1.0], y)
        assert 0.0 <= alpha <= 1.0 and 0.5 <= t_conf <= 1.0
        assert f1 >= baseline, "tuning should never select a harmful configuration"


class TestGateStats:
    def test_net_gain_and_repr(self):
        stats = GateStats(n_total=10, n_interventions=4, fixed=3, worsened=1)
        assert stats.net_gain == 2
        assert "4/10" in str(stats)


class TestCmaesOptional:
    def test_import_error_is_explicit_when_cma_is_absent(self):
        """`cma` is only needed to refit the weights, so it is imported lazily."""
        pytest.importorskip("cma")
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, size=40)
        good = np.eye(3)[y] * 5.0
        noise = rng.normal(size=(40, 3))
        weights, f1 = optimise_weights_cmaes([good, noise], y, maxiter=15,
                                             verbose=False)
        assert weights[0] > weights[1], "the informative model should win"
        assert f1 == pytest.approx(1.0)
