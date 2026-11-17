"""
reliability.calibration
=======================

Probability calibration diagnostics and correction.

Why calibration matters
-----------------------
A classifier is well-calibrated if P(Y=1 | p̂ = p) ≈ p for all p.
Conformal prediction sets are valid regardless of calibration — a poorly
calibrated model still produces valid sets. But calibrated probabilities
are essential for *risk-based decision making*: "act if P(event) > 0.7".

The standard diagnostic is the Expected Calibration Error (ECE):
    ECE = Σ_b (n_b / n) |acc_b - conf_b|
where bins b partition [0,1] by predicted confidence.

The standard correction is temperature scaling (Guo et al. 2017):
    calibrated_prob = softmax(logits / T)
where T is chosen to minimize NLL on a calibration set.

References
----------
- Guo, Pleiss, Sun, Weinberger (2017) "On Calibration of Modern Neural Networks"
- Niculescu-Mizil & Caruana (2005) "Predicting Good Probabilities with SVMs"
"""

import numpy as np
from typing import Optional, Tuple
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')  # non-interactive backend; change to 'TkAgg' if needed
import matplotlib.pyplot as plt


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.

    Parameters
    ----------
    logits : np.ndarray, shape (n, n_classes)

    Returns
    -------
    np.ndarray, shape (n, n_classes) : probability vectors
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / exp_shifted.sum(axis=1, keepdims=True)


def _extract_confidences(probs: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract (confidences, predictions) from probs array.

    For 2D probs: confidence = max prob, prediction = argmax.
    For 1D probs (binary): confidence = probs, prediction = probs > 0.5.
    """
    if probs.ndim == 2:
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
    else:
        # Binary case: probs is P(Y=1). Confidence = P(predicted class).
        predictions = (probs >= 0.5).astype(int)
        confidences = np.where(probs >= 0.5, probs, 1.0 - probs)
    return confidences, predictions


def _calibration_bins(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-bin accuracy, confidence, and counts.

    Returns
    -------
    bin_accs : np.ndarray, shape (n_bins,) — mean accuracy per bin (nan if empty)
    bin_confs : np.ndarray, shape (n_bins,) — mean confidence per bin (nan if empty)
    bin_counts : np.ndarray, shape (n_bins,) — number of samples per bin
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = np.full(n_bins, np.nan)
    bin_confs = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        # Use (lo, hi] for all bins except the first (use [lo, hi])
        if b == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences > lo) & (confidences <= hi)
        count = mask.sum()
        bin_counts[b] = count
        if count > 0:
            bin_accs[b] = (predictions[mask] == labels[mask]).mean()
            bin_confs[b] = confidences[mask].mean()

    return bin_accs, bin_confs, bin_counts


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Expected Calibration Error (ECE).

    Parameters
    ----------
    probs : np.ndarray, shape (n,) or (n, n_classes)
        For binary: (n,) of positive class probabilities.
        For multiclass: (n, n_classes); uses max probability.
    labels : np.ndarray, shape (n,) of int
        True class labels.
    n_bins : int
        Number of equal-width confidence bins (default 15).

    Returns
    -------
    float : ECE ∈ [0, 1]. 0 = perfectly calibrated.

    Notes
    -----
    ECE = Σ_b (n_b / n) |accuracy_b - confidence_b|
    where:
      - bins b are equal-width partitions of [0, 1]
      - accuracy_b = fraction of correct predictions in bin b
      - confidence_b = mean predicted confidence in bin b
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels)
    n = len(labels)

    confidences, predictions = _extract_confidences(probs, labels)
    bin_accs, bin_confs, bin_counts = _calibration_bins(
        confidences, predictions, labels, n_bins
    )

    # ECE = weighted average of |acc - conf| over non-empty bins
    valid = bin_counts > 0
    ece = np.sum(
        bin_counts[valid] / n * np.abs(bin_accs[valid] - bin_confs[valid])
    )
    return float(ece)


def reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    ax: Optional[plt.Axes] = None,
    title: str = 'Reliability Diagram'
) -> plt.Figure:
    """
    Plot a reliability diagram (calibration curve).

    A perfectly calibrated model produces a line along y=x.
    Points above y=x: underconfident. Points below: overconfident.

    Parameters
    ----------
    probs : np.ndarray, shape (n,) or (n, n_classes)
    labels : np.ndarray, shape (n,)
    n_bins : int
    ax : optional matplotlib Axes. If None, a new figure is created.
    title : str

    Returns
    -------
    plt.Figure
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels)

    confidences, predictions = _extract_confidences(probs, labels)
    bin_accs, bin_confs, bin_counts = _calibration_bins(
        confidences, predictions, labels, n_bins
    )
    ece = compute_ece(probs, labels, n_bins=n_bins)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    valid = bin_counts > 0
    ax.bar(
        bin_centers[valid],
        bin_accs[valid],
        width=bin_width * 0.9,
        align='center',
        edgecolor='black',
        color='steelblue',
        alpha=0.7,
        label='Empirical accuracy'
    )
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Perfect calibration')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{title}\nECE = {ece:.4f}')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left')
    fig.tight_layout()

    return fig


def temperature_scale(
    logits: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Find optimal temperature T by minimizing NLL on calibration data.

    Temperature scaling is the simplest post-hoc calibration method.
    It learns a single scalar T such that softmax(logits / T) is well-calibrated.

    Parameters
    ----------
    logits : np.ndarray, shape (n, n_classes)
        Raw logits (pre-softmax) from the model on the calibration set.
    labels : np.ndarray, shape (n,) of int
        True class labels on the calibration set.

    Returns
    -------
    float : Optimal temperature T > 0.
            T > 1: model was overconfident (predictions too sharp).
            T < 1: model was underconfident.
            T = 1: no change needed.

    Usage
    -----
    T = temperature_scale(logits_cal, y_cal)
    calibrated_probs = softmax(logits_test / T)
    """
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels)
    n = len(labels)

    def nll(T: float) -> float:
        scaled = logits / T
        # log-sum-exp for numerical stability
        log_probs = scaled - logsumexp(scaled, axis=1, keepdims=True)
        return -float(log_probs[np.arange(n), labels].mean())

    result = minimize_scalar(nll, bounds=(0.01, 20.0), method='bounded')
    return float(result.x)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Brier score: mean squared error between predicted probs and one-hot labels.

    Parameters
    ----------
    probs : np.ndarray, shape (n, n_classes) or (n,) for binary
    labels : np.ndarray, shape (n,) of int

    Returns
    -------
    float : Brier score ∈ [0, 2] (multiclass) or [0, 1] (binary).
            Lower is better. 0 = perfect.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels)

    if probs.ndim == 1:
        # Binary case
        y_onehot = labels.astype(float)
        return float(np.mean((probs - y_onehot) ** 2))
    else:
        n, n_classes = probs.shape
        y_onehot = np.zeros((n, n_classes))
        y_onehot[np.arange(n), labels] = 1.0
        return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


class CalibratedModel:
    """
    Wrapper that applies temperature scaling to any model.

    Parameters
    ----------
    model : object with predict_proba(X) method
    T : float, temperature parameter (from temperature_scale())

    Usage
    -----
    T = temperature_scale(logits_cal, y_cal)
    cal_model = CalibratedModel(model, T)
    cal_probs = cal_model.predict_proba(X_test)  # calibrated probabilities
    """

    def __init__(self, model, T: float):
        self.model = model
        self.T = T

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return temperature-scaled probabilities.

        Uses log(probs) as surrogate logits, then rescales by T.
        This is exact when the model exposes true logits via
        model.predict_logits(X); otherwise approximate.
        """
        raw_probs = self.model.predict_proba(X)
        raw_probs = np.clip(raw_probs, 1e-10, 1.0)  # avoid log(0)
        # Convert probs back to logits using log (approximate when not true logits)
        log_probs = np.log(raw_probs)
        return softmax(log_probs / self.T)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions (argmax of calibrated probabilities)."""
        return self.predict_proba(X).argmax(axis=1)
