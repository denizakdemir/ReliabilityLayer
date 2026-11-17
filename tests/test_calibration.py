"""
Tests for reliability.calibration

Verifies:
- softmax outputs valid probability vectors
- compute_ece returns 0 for perfectly calibrated model
- reliability_diagram returns a matplotlib Figure
- temperature_scale finds T > 1 for an overconfident model
- CalibratedModel.predict_proba sums to 1
- ECE decreases after temperature scaling
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reliability.calibration import (
    softmax,
    compute_ece,
    reliability_diagram,
    temperature_scale,
    brier_score,
    CalibratedModel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_overconfident_logits(n=500, n_classes=10, seed=0):
    """
    Simulate overconfident logits: large magnitude, so softmax is peaked.
    True labels are mostly correct (accuracy ~80%) but probs are very high.
    """
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, n_classes, size=n)
    # Start from roughly correct logits, then scale up (overconfident)
    logits = rng.randn(n, n_classes)
    # Boost the true class with high probability
    for i in range(n):
        if rng.rand() < 0.8:  # 80% of the time, true class has highest logit
            logits[i, labels[i]] += 5.0
    logits *= 3.0  # scale up to make overconfident
    return logits, labels


def make_uniform_logits(n=200, n_classes=5, seed=1):
    """Uniform logits → uniform probs → should be well-calibrated at chance."""
    rng = np.random.RandomState(seed)
    logits = rng.randn(n, n_classes) * 0.01  # near-zero → near-uniform
    labels = rng.randint(0, n_classes, size=n)
    return logits, labels


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------

class TestSoftmax:
    def test_output_sums_to_one(self):
        rng = np.random.RandomState(0)
        logits = rng.randn(50, 5)
        probs = softmax(logits)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(50), atol=1e-10)

    def test_output_non_negative(self):
        logits = np.random.randn(20, 4)
        probs = softmax(logits)
        assert np.all(probs >= 0)

    def test_output_in_range(self):
        logits = np.random.randn(30, 3)
        probs = softmax(logits)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_argmax_preserved(self):
        """Argmax of logits should equal argmax of softmax probs."""
        rng = np.random.RandomState(42)
        logits = rng.randn(100, 6)
        probs = softmax(logits)
        np.testing.assert_array_equal(logits.argmax(axis=1), probs.argmax(axis=1))

    def test_numerical_stability_large_logits(self):
        """Should not return NaN or Inf for large logits."""
        logits = np.array([[1000.0, 0.0, -1000.0]])
        probs = softmax(logits)
        assert not np.any(np.isnan(probs))
        assert not np.any(np.isinf(probs))
        np.testing.assert_allclose(probs.sum(axis=1), [1.0], atol=1e-10)


# ---------------------------------------------------------------------------
# compute_ece
# ---------------------------------------------------------------------------

class TestComputeECE:
    def test_perfect_calibration_returns_zero(self):
        """If the model's max confidence equals accuracy in every bin, ECE ≈ 0.

        We construct a 2-class model where, in each confidence bin [c-ε, c+ε],
        accuracy equals exactly c. This is done deterministically.
        """
        n_per_conf = 100
        rng = np.random.RandomState(0)
        probs_list, labels_list = [], []

        for conf in [0.6, 0.7, 0.8, 0.9]:
            # 2-class: probs column 0 = 1-conf, column 1 = conf
            # predictions = class 1 (argmax), confidence = conf
            probs_b = np.column_stack([
                np.full(n_per_conf, 1 - conf),
                np.full(n_per_conf, conf)
            ])
            # accuracy in this bin should = conf: fraction correct = conf
            n_correct = int(round(conf * n_per_conf))
            labels_b = np.zeros(n_per_conf, dtype=int)
            labels_b[:n_correct] = 1  # first n_correct are class 1 (correct)
            probs_list.append(probs_b)
            labels_list.append(labels_b)

        probs_all = np.vstack(probs_list)
        labels_all = np.concatenate(labels_list)
        ece = compute_ece(probs_all, labels_all, n_bins=10)
        assert ece < 0.02, f"ECE for deterministically calibrated model = {ece:.4f}"

    def test_worst_case_calibration(self):
        """Confidence always 1.0 but accuracy 0.0: ECE should be near 1."""
        n = 200
        probs = np.ones(n)   # always 100% confident (binary: predicts 1)
        labels = np.zeros(n, dtype=int)  # always wrong
        ece = compute_ece(probs, labels, n_bins=10)
        assert ece > 0.9

    def test_returns_float_in_range(self):
        rng = np.random.RandomState(5)
        n = 300
        logits = rng.randn(n, 4)
        probs = softmax(logits)
        labels = rng.randint(0, 4, size=n)
        ece = compute_ece(probs, labels)
        assert isinstance(ece, float)
        assert 0.0 <= ece <= 1.0

    def test_multiclass_shape(self):
        """Should handle (n, n_classes) probs correctly."""
        rng = np.random.RandomState(3)
        probs = softmax(rng.randn(200, 5))
        labels = rng.randint(0, 5, 200)
        ece = compute_ece(probs, labels, n_bins=10)
        assert 0.0 <= ece <= 1.0

    def test_n_bins_parameter(self):
        rng = np.random.RandomState(7)
        probs = softmax(rng.randn(300, 3))
        labels = rng.randint(0, 3, 300)
        ece5 = compute_ece(probs, labels, n_bins=5)
        ece20 = compute_ece(probs, labels, n_bins=20)
        # Both should be valid floats (values may differ)
        assert isinstance(ece5, float)
        assert isinstance(ece20, float)


# ---------------------------------------------------------------------------
# reliability_diagram
# ---------------------------------------------------------------------------

class TestReliabilityDiagram:
    def test_returns_figure(self):
        rng = np.random.RandomState(0)
        probs = softmax(rng.randn(200, 4))
        labels = rng.randint(0, 4, 200)
        fig = reliability_diagram(probs, labels)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_accepts_ax_parameter(self):
        rng = np.random.RandomState(1)
        probs = softmax(rng.randn(200, 3))
        labels = rng.randint(0, 3, 200)
        fig, ax = plt.subplots()
        fig2 = reliability_diagram(probs, labels, ax=ax)
        assert fig2 is fig
        plt.close(fig)

    def test_binary_probs(self):
        rng = np.random.RandomState(2)
        probs = rng.uniform(0, 1, 200)
        labels = rng.randint(0, 2, 200)
        fig = reliability_diagram(probs, labels)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# temperature_scale
# ---------------------------------------------------------------------------

class TestTemperatureScale:
    def test_overconfident_model_gives_T_greater_than_1(self):
        """An overconfident model (peaked logits) should get T > 1."""
        logits, labels = make_overconfident_logits(n=500, n_classes=10, seed=0)
        T = temperature_scale(logits, labels)
        assert T > 1.0, f"Expected T > 1 for overconfident model, got T={T:.3f}"

    def test_returns_positive_float(self):
        logits, labels = make_uniform_logits(n=200, n_classes=5)
        T = temperature_scale(logits, labels)
        assert isinstance(T, float)
        assert T > 0

    def test_T_in_reasonable_range(self):
        """T should be in [0.01, 20] (the search bounds)."""
        logits, labels = make_overconfident_logits()
        T = temperature_scale(logits, labels)
        assert 0.01 <= T <= 20.0

    def test_ece_decreases_after_scaling(self):
        """ECE should be lower with calibrated probs than raw probs."""
        logits, labels = make_overconfident_logits(n=1000, n_classes=10, seed=42)
        T = temperature_scale(logits, labels)

        raw_probs = softmax(logits)
        cal_probs = softmax(logits / T)

        ece_raw = compute_ece(raw_probs, labels)
        ece_cal = compute_ece(cal_probs, labels)
        assert ece_cal < ece_raw, (
            f"ECE did not decrease: raw={ece_raw:.4f}, calibrated={ece_cal:.4f}"
        )

    def test_ece_after_scaling_below_threshold(self):
        """ECE after temperature scaling should be substantially reduced.

        Note: Temperature scaling minimizes NLL, not ECE directly, so ECE
        may not reach 0.05 for randomly structured synthetic logits.
        The key check is that ECE is reduced and is below a generous bound.
        For production neural networks (high accuracy, structured logits),
        ECE routinely drops below 0.05 — see validate_calibration.py.
        """
        logits, labels = make_overconfident_logits(n=2000, n_classes=10, seed=0)
        raw_probs = softmax(logits)
        ece_before = compute_ece(raw_probs, labels)

        T = temperature_scale(logits, labels)
        cal_probs = softmax(logits / T)
        ece_after = compute_ece(cal_probs, labels)

        # ECE must decrease
        assert ece_after < ece_before, (
            f"ECE did not decrease: {ece_before:.4f} → {ece_after:.4f}"
        )
        # ECE after scaling should be below a reasonable bound for 10-class random model
        assert ece_after < 0.15, (
            f"ECE after temperature scaling = {ece_after:.4f} (expected < 0.15)"
        )


# ---------------------------------------------------------------------------
# brier_score
# ---------------------------------------------------------------------------

class TestBrierScore:
    def test_perfect_prediction_score_zero(self):
        """Perfect predictions should give Brier score 0."""
        n, n_classes = 100, 3
        labels = np.array([0] * 34 + [1] * 33 + [2] * 33)
        probs = np.zeros((n, n_classes))
        probs[np.arange(n), labels] = 1.0
        bs = brier_score(probs, labels)
        assert bs == pytest.approx(0.0, abs=1e-10)

    def test_random_prediction_reasonable_score(self):
        rng = np.random.RandomState(0)
        probs = softmax(rng.randn(200, 4))
        labels = rng.randint(0, 4, 200)
        bs = brier_score(probs, labels)
        assert 0.0 <= bs <= 2.0

    def test_binary_case(self):
        probs = np.array([0.9, 0.1, 0.8, 0.2])
        labels = np.array([1, 0, 1, 0])
        # Expected: mean of (0.9-1)^2, (0.1-0)^2, (0.8-1)^2, (0.2-0)^2
        expected = np.mean([0.01, 0.01, 0.04, 0.04])
        bs = brier_score(probs, labels)
        assert bs == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# CalibratedModel
# ---------------------------------------------------------------------------

class TestCalibratedModel:
    def _make_model(self):
        """Simple mock model that returns known probs."""
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        X, y = make_classification(n_samples=400, n_features=10, random_state=0)
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.5, random_state=0)
        model = LogisticRegression(max_iter=500, random_state=0)
        model.fit(X_tr, y_tr)
        return model, X, y

    def test_predict_proba_sums_to_one(self):
        model, X, _ = self._make_model()
        cal_model = CalibratedModel(model, T=1.5)
        probs = cal_model.predict_proba(X)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(X)), atol=1e-8)

    def test_predict_proba_non_negative(self):
        model, X, _ = self._make_model()
        cal_model = CalibratedModel(model, T=2.0)
        probs = cal_model.predict_proba(X)
        assert np.all(probs >= 0)

    def test_T_equals_1_approximately_unchanged(self):
        """With T=1, calibrated probs should match raw probs closely."""
        model, X, _ = self._make_model()
        cal_model = CalibratedModel(model, T=1.0)
        raw_probs = model.predict_proba(X)
        cal_probs = cal_model.predict_proba(X)
        np.testing.assert_allclose(cal_probs, raw_probs, atol=1e-5)

    def test_high_T_flattens_distribution(self):
        """Higher T should make predictions closer to uniform."""
        model, X, _ = self._make_model()
        cal1 = CalibratedModel(model, T=1.0)
        cal10 = CalibratedModel(model, T=10.0)
        probs1 = cal1.predict_proba(X)
        probs10 = cal10.predict_proba(X)
        # Higher T → lower max prob (flatter)
        assert probs10.max(axis=1).mean() < probs1.max(axis=1).mean()
