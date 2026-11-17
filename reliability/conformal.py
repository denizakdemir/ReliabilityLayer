"""
reliability.conformal
=====================

Split conformal prediction for classification and regression.

Theory
------
Split conformal prediction (Papadopoulos et al. 2002; Vovk et al. 2005):

Given a calibration set {(X_1, Y_1), ..., (X_n, Y_n)} and a nonconformity
score function s(x, y), the calibration threshold is:

    q̂ = Quantile_{(1-α)(1 + 1/n)}(s(X_1, Y_1), ..., s(X_n, Y_n))

The prediction set for a new point x is:

    C_α(x) = {y : s(x, y) ≤ q̂}

Guarantee: If (X_1, Y_1), ..., (X_{n+1}, Y_{n+1}) are exchangeable,
then P(Y_{n+1} ∈ C_α(X_{n+1})) ≥ 1 − α.

This guarantee is model-agnostic and distribution-free.

Connection to Le Cam theory (Paper 5)
--------------------------------------
The guarantee holds exactly when P_cal ≈ P_test (exchangeability).
When P_cal ≠ P_test, the coverage loss is at most TV(P_cal, P_test), the joint
total variation (coefficient one, unimprovable). The deficiency estimate in
drift.py is a monitor of shift severity, not an upper bound on that TV.

References
----------
- Angelopoulos & Bates (2021) "A Gentle Introduction to Conformal Prediction"
- Romano, Patterson, Candès (2019) "Conformalized Quantile Regression"
"""

import warnings
import numpy as np
from typing import List, Tuple


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Compute the conformal calibration threshold.

    The threshold is the ⌈(1-alpha)(n+1)⌉-th order statistic of the
    calibration nonconformity scores. This is the standard split-conformal
    correction that ensures ≥ 1-alpha coverage.

    Parameters
    ----------
    scores : np.ndarray, shape (n,)
        Nonconformity scores on the calibration set.
    alpha : float in (0, 1)
        Target miscoverage rate (e.g., 0.10 for 90% coverage).

    Returns
    -------
    float : The calibration threshold q̂.
    """
    n = len(scores)
    k = int(np.ceil((1 - alpha) * (n + 1)))
    k = min(k, n)  # cap: if k > n, threshold = +inf; we cap at max score
    sorted_scores = np.sort(scores)
    return float(sorted_scores[k - 1])


def compute_nonconformity_scores(
    model,
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'softmax'
) -> np.ndarray:
    """
    Compute nonconformity scores for a calibration set.

    Parameters
    ----------
    model : object
        For classification: must have predict_proba(X) method.
        For regression: must have predict(X) method.
    X : np.ndarray, shape (n, d)
        Feature matrix.
    y : np.ndarray, shape (n,)
        True labels (classification) or values (regression).
    method : str
        Nonconformity score method:
        - 'softmax'   : 1 - p(y_true | x)  [classification]
        - 'margin'    : p(y_1 | x) - p(y_true | x)  [classification]
        - 'residual'  : |y_true - y_pred|  [regression]
        - 'raps'      : Cumulative softmax score up to true class rank

    Returns
    -------
    np.ndarray, shape (n,) : Nonconformity scores. Higher = less conforming.
    """
    y = np.asarray(y)
    n = len(y)

    if method == 'softmax':
        proba = model.predict_proba(X)
        return 1.0 - proba[np.arange(n), y]

    elif method == 'margin':
        proba = model.predict_proba(X)
        true_proba = proba[np.arange(n), y]
        top_proba = np.sort(proba, axis=1)[:, -1]  # max probability
        # If true class is the top class, margin = 0; else = top - true
        return np.maximum(top_proba - true_proba, 0.0)

    elif method == 'residual':
        preds = model.predict(X)
        return np.abs(y.astype(float) - preds.astype(float))

    elif method == 'raps':
        # Cumulative sum score: sum probabilities of classes ranked above
        # (and including) the true class (sorted by descending probability).
        proba = model.predict_proba(X)
        scores = np.zeros(n)
        for i in range(n):
            p = proba[i]
            sorted_idx = np.argsort(p)[::-1]  # descending rank order
            rank = int(np.where(sorted_idx == y[i])[0][0])
            scores[i] = p[sorted_idx[:rank + 1]].sum()
        return scores

    else:
        raise ValueError(
            f"Unknown nonconformity method '{method}'. "
            "Choose from: 'softmax', 'margin', 'residual', 'raps'."
        )


class ConformalClassifier:
    """
    Split conformal prediction sets for classification.

    Wraps any classifier with a predict_proba method and adds a calibration
    step that produces coverage-guaranteed prediction sets.

    Parameters
    ----------
    model : object
        A fitted classifier with predict_proba(X) → (n, n_classes) method.
    alpha : float in (0, 1)
        Target miscoverage rate. Default 0.10 gives 90% coverage.
    score : str
        Nonconformity score: 'softmax' (default), 'margin', 'raps'.

    Usage
    -----
    clf = ConformalClassifier(model, alpha=0.10)
    clf.calibrate(X_cal, y_cal)
    sets = clf.predict(X_test)           # list of sets (one per sample)
    cov  = clf.empirical_coverage(X_test, y_test)  # check coverage

    Attributes (after calibrate)
    ----------------------------
    threshold_ : float
        The calibration quantile q̂.
    calibration_scores_ : np.ndarray
        Nonconformity scores on the calibration set.
    n_cal_ : int
        Number of calibration samples.
    """

    def __init__(self, model, alpha: float = 0.10, score: str = 'softmax'):
        self.model = model
        self.alpha = alpha
        self.score = score
        self.threshold_ = None
        self.calibration_scores_ = None
        self.n_cal_ = None

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> 'ConformalClassifier':
        """
        Calibrate the conformal predictor on a held-out calibration set.

        This must be called before predict(). The calibration set must be
        independent of the training set (split conformal requirement).

        Parameters
        ----------
        X_cal : np.ndarray, shape (n_cal, d)
        y_cal : np.ndarray, shape (n_cal,)

        Returns
        -------
        self
        """
        if len(y_cal) < 100:
            warnings.warn(
                f"Small calibration set ({len(y_cal)} samples). "
                "Coverage guarantees may be loose. Recommend ≥ 100 samples.",
                UserWarning
            )
        scores = compute_nonconformity_scores(self.model, X_cal, y_cal, self.score)
        self.calibration_scores_ = scores
        self.n_cal_ = len(y_cal)
        self.threshold_ = conformal_quantile(scores, self.alpha)
        return self

    def predict(self, X_test: np.ndarray) -> List[np.ndarray]:
        """
        Produce conformal prediction sets for new data.

        Parameters
        ----------
        X_test : np.ndarray, shape (n_test, d)

        Returns
        -------
        list of np.ndarray : One array per sample, containing predicted class labels.
                             e.g., [array([0, 1]), array([2]), array([0, 1, 2])]

        Notes
        -----
        A sample's prediction set contains all classes y such that
        s(x, y) ≤ threshold_, where s is the nonconformity score.
        """
        if self.threshold_ is None:
            raise RuntimeError("Must call calibrate() before predict().")

        proba = self.model.predict_proba(X_test)
        n_test, n_classes = proba.shape
        prediction_sets = []

        for i in range(n_test):
            p = proba[i]

            if self.score == 'softmax':
                # Include j if 1 - p_j <= threshold_ ↔ p_j >= 1 - threshold_
                in_set = np.where(1.0 - p <= self.threshold_)[0]

            elif self.score == 'margin':
                # Calibration score for (x, y): max_k(p_k) - p_y.
                # Include j if max_k(p_k) - p_j <= threshold_
                top = p.max()
                in_set = np.where(top - p <= self.threshold_)[0]

            elif self.score == 'raps':
                # Include classes greedily from top probability until
                # cumulative sum of included classes exceeds threshold_
                sorted_idx = np.argsort(p)[::-1]
                cumsum = np.cumsum(p[sorted_idx])
                n_included = int(np.sum(cumsum <= self.threshold_))
                # Always include at least the top class
                n_included = max(n_included, 1)
                in_set = np.sort(sorted_idx[:n_included])

            else:
                # Fallback: softmax-style
                scores_i = 1.0 - p
                in_set = np.where(scores_i <= self.threshold_)[0]

            prediction_sets.append(in_set)

        return prediction_sets

    def empirical_coverage(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Compute empirical marginal coverage on a labeled test set.

        Parameters
        ----------
        X_test : np.ndarray, shape (n_test, d)
        y_test : np.ndarray, shape (n_test,)

        Returns
        -------
        float : Fraction of test samples where true label is in prediction set.
                Should be ≥ 1 - alpha when calibration and test are exchangeable.
        """
        sets = self.predict(X_test)
        covered = [int(y_test[i] in sets[i]) for i in range(len(y_test))]
        return float(np.mean(covered))

    def average_set_size(self, X_test: np.ndarray) -> float:
        """
        Compute average prediction set size (efficiency metric).

        Smaller is better (more informative). A set size of n_classes means
        the predictor is completely uninformative for that sample.
        """
        sets = self.predict(X_test)
        return float(np.mean([len(s) for s in sets]))


class ConformalRegressor:
    """
    Split conformal prediction intervals for regression.

    Parameters
    ----------
    model : object
        A fitted regressor with predict(X) method.
        For CQR variant: must also have predict_quantile(X, q) method,
        or pass lower_model and upper_model separately.
    alpha : float in (0, 1)
        Target miscoverage rate.
    variant : str
        'standard' : Uses |y - ŷ| as nonconformity score (homoscedastic).
        'cqr'      : Conformalized Quantile Regression (Romano et al. 2019).
                     Better for heteroscedastic data. Requires model to have
                     predict_quantile(X, q) for q = alpha/2 and 1 - alpha/2.

    Usage
    -----
    reg = ConformalRegressor(model, alpha=0.10)
    reg.calibrate(X_cal, y_cal)
    lower, upper = reg.predict(X_test)  # coverage-guaranteed intervals
    """

    def __init__(self, model, alpha: float = 0.10, variant: str = 'standard'):
        self.model = model
        self.alpha = alpha
        self.variant = variant
        self.threshold_ = None

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> 'ConformalRegressor':
        """
        Calibrate on held-out data.

        For 'standard': scores = |y_cal - model.predict(X_cal)|
        For 'cqr': scores = max(q_lo - y, y - q_hi) where q_lo, q_hi are
                   predicted quantiles at alpha/2 and 1-alpha/2.
        """
        if len(y_cal) < 100:
            warnings.warn(
                f"Small calibration set ({len(y_cal)} samples). "
                "Coverage guarantees may be loose.",
                UserWarning
            )

        if self.variant == 'standard':
            preds = self.model.predict(X_cal)
            scores = np.abs(y_cal.astype(float) - preds.astype(float))

        elif self.variant == 'cqr':
            q_lo = self.model.predict_quantile(X_cal, self.alpha / 2)
            q_hi = self.model.predict_quantile(X_cal, 1 - self.alpha / 2)
            scores = np.maximum(q_lo - y_cal, y_cal - q_hi)

        else:
            raise ValueError(f"Unknown variant '{self.variant}'. Choose 'standard' or 'cqr'.")

        self.threshold_ = conformal_quantile(scores, self.alpha)
        self._calibration_scores = scores
        return self

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Produce conformal prediction intervals.

        Returns
        -------
        lower : np.ndarray, shape (n_test,)
        upper : np.ndarray, shape (n_test,)

        Guarantee: P(lower_i ≤ Y_i ≤ upper_i) ≥ 1 - alpha under exchangeability.
        """
        if self.threshold_ is None:
            raise RuntimeError("Must call calibrate() before predict().")

        if self.variant == 'standard':
            preds = self.model.predict(X_test).astype(float)
            lower = preds - self.threshold_
            upper = preds + self.threshold_

        elif self.variant == 'cqr':
            q_lo = self.model.predict_quantile(X_test, self.alpha / 2).astype(float)
            q_hi = self.model.predict_quantile(X_test, 1 - self.alpha / 2).astype(float)
            lower = q_lo - self.threshold_
            upper = q_hi + self.threshold_

        else:
            raise ValueError(f"Unknown variant '{self.variant}'.")

        return lower, upper

    def empirical_coverage(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Fraction of test samples where true value falls inside the interval."""
        lower, upper = self.predict(X_test)
        covered = (y_test >= lower) & (y_test <= upper)
        return float(np.mean(covered))
