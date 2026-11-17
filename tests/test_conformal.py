"""
Tests for reliability.conformal

Verifies:
- conformal_quantile computes the correct order statistic
- compute_nonconformity_scores returns correct values for all methods
- ConformalClassifier achieves >= 1-alpha coverage across multiple seeds
- ConformalRegressor achieves >= 1-alpha coverage
- Edge cases: empty sets, single-class prediction, large alpha
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reliability.conformal import (
    conformal_quantile,
    compute_nonconformity_scores,
    ConformalClassifier,
    ConformalRegressor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_classifier():
    """Fitted LogisticRegression on a 2-class problem."""
    X, y = make_classification(n_samples=500, n_features=10, random_state=0)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.4, random_state=0)
    model = LogisticRegression(max_iter=500, random_state=0)
    model.fit(X_train, y_train)
    return model, X, y


@pytest.fixture
def multiclass_classifier():
    """Fitted LogisticRegression on a 3-class problem."""
    X, y = make_classification(
        n_samples=800, n_features=10, n_classes=3, n_informative=5, random_state=1
    )
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=1)
    model = LogisticRegression(max_iter=1000, random_state=1)
    model.fit(X_train, y_train)
    return model, X, y


@pytest.fixture
def simple_regressor():
    """Fitted LinearRegression."""
    X, y = make_regression(n_samples=500, n_features=5, noise=10.0, random_state=0)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.4, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X, y


# ---------------------------------------------------------------------------
# conformal_quantile
# ---------------------------------------------------------------------------

class TestConformalQuantile:
    def test_basic_formula(self):
        """Threshold should be the ceil((1-alpha)(n+1))-th order statistic."""
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])  # n=5
        alpha = 0.1
        # k = ceil(0.9 * 6) = ceil(5.4) = 6, capped at 5 -> scores[4] = 0.9
        q = conformal_quantile(scores, alpha)
        assert q == pytest.approx(0.9)

    def test_exact_coverage_property(self):
        """At alpha=0, threshold should be the maximum score."""
        scores = np.arange(1.0, 11.0)  # [1, 2, ..., 10]
        q = conformal_quantile(scores, alpha=0.0)
        assert q == pytest.approx(10.0)

    def test_returns_float(self):
        scores = np.random.RandomState(0).uniform(0, 1, 100)
        q = conformal_quantile(scores, 0.1)
        assert isinstance(q, float)

    def test_monotone_in_alpha(self):
        """Higher alpha → lower (more permissive) threshold."""
        scores = np.random.RandomState(0).uniform(0, 1, 200)
        q1 = conformal_quantile(scores, 0.05)
        q2 = conformal_quantile(scores, 0.20)
        assert q1 >= q2

    def test_large_calibration_set(self):
        """With n=1000 and alpha=0.1, threshold should be near 90th percentile."""
        rng = np.random.RandomState(42)
        scores = rng.uniform(0, 1, 1000)
        q = conformal_quantile(scores, 0.1)
        # Should be close to empirical 90th percentile
        assert abs(q - np.percentile(scores, 90)) < 0.05


# ---------------------------------------------------------------------------
# compute_nonconformity_scores
# ---------------------------------------------------------------------------

class TestComputeNonconformityScores:
    def test_softmax_range(self, simple_classifier):
        model, X, y = simple_classifier
        _, X_cal, _, y_cal = train_test_split(X, y, test_size=0.3, random_state=0)
        scores = compute_nonconformity_scores(model, X_cal, y_cal, method='softmax')
        assert scores.shape == (len(y_cal),)
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)

    def test_softmax_true_class_high_confidence(self, simple_classifier):
        """For a very confident correct prediction, softmax score should be near 0."""
        model, X, y = simple_classifier
        # Get a sample where model is very confident
        proba = model.predict_proba(X)
        confident_mask = proba.max(axis=1) > 0.95
        if confident_mask.sum() > 0:
            X_conf = X[confident_mask]
            y_conf = y[confident_mask]
            # Make y_conf be the predicted class (so score ≈ 0)
            y_pred = proba[confident_mask].argmax(axis=1)
            scores = compute_nonconformity_scores(model, X_conf, y_pred, 'softmax')
            assert np.all(scores < 0.1)

    def test_margin_non_negative(self, simple_classifier):
        model, X, y = simple_classifier
        _, X_cal, _, y_cal = train_test_split(X, y, test_size=0.3, random_state=0)
        scores = compute_nonconformity_scores(model, X_cal, y_cal, method='margin')
        assert scores.shape == (len(y_cal),)
        assert np.all(scores >= 0)

    def test_residual_regression(self, simple_regressor):
        model, X, y = simple_regressor
        _, X_cal, _, y_cal = train_test_split(X, y, test_size=0.3, random_state=0)
        scores = compute_nonconformity_scores(model, X_cal, y_cal, method='residual')
        assert scores.shape == (len(y_cal),)
        assert np.all(scores >= 0)

    def test_raps_range(self, simple_classifier):
        model, X, y = simple_classifier
        _, X_cal, _, y_cal = train_test_split(X, y, test_size=0.3, random_state=0)
        scores = compute_nonconformity_scores(model, X_cal, y_cal, method='raps')
        assert scores.shape == (len(y_cal),)
        assert np.all(scores >= 0)
        assert np.all(scores <= 1 + 1e-9)  # cumsum of probs ≤ 1

    def test_invalid_method_raises(self, simple_classifier):
        model, X, y = simple_classifier
        with pytest.raises(ValueError, match="Unknown nonconformity method"):
            compute_nonconformity_scores(model, X[:10], y[:10], method='invalid')


# ---------------------------------------------------------------------------
# ConformalClassifier
# ---------------------------------------------------------------------------

class TestConformalClassifier:
    def test_calibrate_sets_threshold(self, simple_classifier):
        model, X, y = simple_classifier
        X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.6, random_state=0)
        X_cal, X_test, y_cal, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)
        clf = ConformalClassifier(model, alpha=0.10)
        clf.calibrate(X_cal, y_cal)
        assert clf.threshold_ is not None
        assert isinstance(clf.threshold_, float)
        assert clf.n_cal_ == len(y_cal)

    def test_predict_returns_list_of_arrays(self, simple_classifier):
        model, X, y = simple_classifier
        _, X_rest, _, y_rest = train_test_split(X, y, test_size=0.6, random_state=0)
        X_cal, X_test, y_cal, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)
        clf = ConformalClassifier(model, alpha=0.10)
        clf.calibrate(X_cal, y_cal)
        sets = clf.predict(X_test)
        assert len(sets) == len(X_test)
        for s in sets:
            assert isinstance(s, np.ndarray)

    def test_predict_before_calibrate_raises(self, simple_classifier):
        model, X, y = simple_classifier
        clf = ConformalClassifier(model, alpha=0.10)
        with pytest.raises(RuntimeError):
            clf.predict(X[:10])

    def test_coverage_binary_softmax(self):
        """ConformalClassifier should achieve >= 1-alpha coverage across seeds.

        Each seed uses a freshly trained model on its own train split to
        avoid data contamination between train and cal/test sets.
        """
        alpha = 0.10
        coverages = []
        for seed in range(5):
            X, y = make_classification(
                n_samples=1000, n_features=10, n_informative=5, random_state=seed
            )
            X_train, X_rest, y_train, y_rest = train_test_split(
                X, y, test_size=0.6, random_state=seed
            )
            X_cal, X_test, y_cal, y_test = train_test_split(
                X_rest, y_rest, test_size=0.5, random_state=seed
            )
            model = LogisticRegression(max_iter=500, random_state=seed)
            model.fit(X_train, y_train)
            clf = ConformalClassifier(model, alpha=alpha, score='softmax')
            clf.calibrate(X_cal, y_cal)
            cov = clf.empirical_coverage(X_test, y_test)
            coverages.append(cov)
        # Average coverage should be >= 1-alpha; individual seeds allow finite-sample slack
        assert np.mean(coverages) >= 1 - alpha - 0.02, (
            f"Mean coverage {np.mean(coverages):.3f} < {1-alpha - 0.02:.3f}"
        )
        assert min(coverages) >= 1 - alpha - 0.05, (
            f"Min coverage {min(coverages):.3f} < {1-alpha - 0.05:.3f}"
        )

    def test_coverage_multiclass_softmax(self):
        """Multiclass coverage should be >= 1-alpha (fresh model per seed)."""
        alpha = 0.10
        coverages = []
        for seed in range(5):
            X, y = make_classification(
                n_samples=1200, n_features=15, n_classes=3,
                n_informative=8, random_state=seed
            )
            X_train, X_rest, y_train, y_rest = train_test_split(
                X, y, test_size=0.6, random_state=seed
            )
            X_cal, X_test, y_cal, y_test = train_test_split(
                X_rest, y_rest, test_size=0.5, random_state=seed
            )
            model = LogisticRegression(max_iter=1000, random_state=seed)
            model.fit(X_train, y_train)
            clf = ConformalClassifier(model, alpha=alpha)
            clf.calibrate(X_cal, y_cal)
            cov = clf.empirical_coverage(X_test, y_test)
            coverages.append(cov)
        assert np.mean(coverages) >= 1 - alpha - 0.01
        assert min(coverages) >= 1 - alpha - 0.05

    def test_coverage_alpha_05(self):
        """At alpha=0.5 (50% coverage), empirical coverage should be >= 0.48."""
        alpha = 0.5
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
        X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.6, random_state=42)
        X_cal, X_test, y_cal, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)
        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        clf = ConformalClassifier(model, alpha=alpha)
        clf.calibrate(X_cal, y_cal)
        cov = clf.empirical_coverage(X_test, y_test)
        assert cov >= 1 - alpha - 0.05, f"Coverage {cov:.3f} < {1-alpha - 0.05:.3f}"

    def test_average_set_size(self, simple_classifier):
        """Set size should be between 1 and n_classes."""
        model, X, y = simple_classifier
        _, X_rest, _, y_rest = train_test_split(X, y, test_size=0.6, random_state=0)
        X_cal, X_test, y_cal, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)
        clf = ConformalClassifier(model, alpha=0.10)
        clf.calibrate(X_cal, y_cal)
        avg_size = clf.average_set_size(X_test)
        n_classes = len(np.unique(y))
        assert 1.0 <= avg_size <= n_classes

    def test_coverage_raps(self):
        """RAPS score should also achieve >= 1-alpha coverage (fresh model)."""
        alpha = 0.10
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=7)
        X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.6, random_state=7)
        X_cal, X_test, y_cal, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=7)
        model = LogisticRegression(max_iter=500, random_state=7)
        model.fit(X_train, y_train)
        clf = ConformalClassifier(model, alpha=alpha, score='raps')
        clf.calibrate(X_cal, y_cal)
        cov = clf.empirical_coverage(X_test, y_test)
        assert cov >= 1 - alpha - 0.05

    def test_small_cal_warning(self, simple_classifier):
        """Should warn when calibration set < 100 samples."""
        model, X, y = simple_classifier
        clf = ConformalClassifier(model, alpha=0.10)
        with pytest.warns(UserWarning, match="Small calibration set"):
            clf.calibrate(X[:50], y[:50])


# ---------------------------------------------------------------------------
# ConformalRegressor
# ---------------------------------------------------------------------------

class TestConformalRegressor:
    def test_calibrate_sets_threshold(self, simple_regressor):
        model, X, y = simple_regressor
        _, X_rest, _, y_rest = train_test_split(X, y, test_size=0.6, random_state=0)
        X_cal, X_test, y_cal, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)
        reg = ConformalRegressor(model, alpha=0.10)
        reg.calibrate(X_cal, y_cal)
        assert reg.threshold_ is not None
        assert reg.threshold_ >= 0

    def test_predict_returns_intervals(self, simple_regressor):
        model, X, y = simple_regressor
        _, X_rest, _, y_rest = train_test_split(X, y, test_size=0.6, random_state=0)
        X_cal, X_test, y_cal, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)
        reg = ConformalRegressor(model, alpha=0.10)
        reg.calibrate(X_cal, y_cal)
        lower, upper = reg.predict(X_test)
        assert lower.shape == (len(X_test),)
        assert upper.shape == (len(X_test),)
        assert np.all(lower <= upper)

    def test_coverage_standard(self, simple_regressor):
        """Standard regressor should achieve >= 1-alpha coverage across seeds."""
        model, X, y = simple_regressor
        alpha = 0.10
        coverages = []
        for seed in range(5):
            _, X_rest, _, y_rest = train_test_split(X, y, test_size=0.6, random_state=seed)
            X_cal, X_test, y_cal, y_test = train_test_split(
                X_rest, y_rest, test_size=0.5, random_state=seed
            )
            reg = ConformalRegressor(model, alpha=alpha)
            reg.calibrate(X_cal, y_cal)
            cov = reg.empirical_coverage(X_test, y_test)
            coverages.append(cov)
        for cov in coverages:
            assert cov >= 1 - alpha - 0.02

    def test_predict_before_calibrate_raises(self, simple_regressor):
        model, X, y = simple_regressor
        reg = ConformalRegressor(model, alpha=0.10)
        with pytest.raises(RuntimeError):
            reg.predict(X[:10])
