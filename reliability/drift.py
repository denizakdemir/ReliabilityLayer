"""
reliability.drift
=================

Distribution shift detection and Le Cam deficiency certificates.

Architecture: two-tier
----------------------
Tier 1 — Fast tests (< 1 second):
    mmd_test()  : Maximum Mean Discrepancy two-sample test
    ks_test()   : Per-feature Kolmogorov-Smirnov test

Tier 2 — Deficiency certificate (seconds to minutes):
    DeficiencyMonitor : Estimates δ(E_cal, E_test) using the built-in
                        deficiency solver; provides coverage validity warning.

The two tiers are complementary. Use Tier 1 for real-time monitoring (flag
when drift is detected), and Tier 2 for auditing (certify how much coverage
has degraded).

Theory
------
Paper 5 Theorem 1:

    Let C_α be a conformal predictor calibrated on distribution P_cal.
    If δ(P_cal, P_test) ≤ ε, then:
        P_test(Y ∈ C_α(X)) ≥ (1 − α) − 2ε

The DeficiencyMonitor.estimate() method computes δ̂ and the resulting
coverage bound (1 − α) − 2δ̂.
"""

import numpy as np
from scipy.stats import ks_2samp
from typing import Optional
import warnings

from ._deficiency import compute_deficiency as _compute_deficiency

_DEFICIENCY_AVAILABLE = True


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> np.ndarray:
    """Compute RBF kernel matrix K[i,j] = exp(-||X[i]-Y[j]||^2 / (2*bandwidth))."""
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x.y
    XX = np.sum(X ** 2, axis=1, keepdims=True)  # (n, 1)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)  # (m, 1)
    sq_dists = XX + YY.T - 2.0 * X @ Y.T       # (n, m)
    return np.exp(-sq_dists / (2.0 * bandwidth))


def _mmd_squared(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> float:
    """
    Biased MMD² estimator.
    MMD²(P,Q) = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    """
    Kxx = _rbf_kernel(X, X, bandwidth)
    Kyy = _rbf_kernel(Y, Y, bandwidth)
    Kxy = _rbf_kernel(X, Y, bandwidth)
    return Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()


def mmd_test(
    X_ref: np.ndarray,
    X_new: np.ndarray,
    kernel: str = 'rbf',
    n_permutations: int = 500,
    random_state: Optional[int] = None
) -> dict:
    """
    Maximum Mean Discrepancy (MMD) two-sample test.

    Tests H0: X_ref and X_new are from the same distribution.
    Uses a permutation test to compute the p-value.

    Parameters
    ----------
    X_ref : np.ndarray, shape (n_ref, d) — reference (calibration) data
    X_new : np.ndarray, shape (n_new, d) — new (test/deployment) data
    kernel : str — 'rbf' (Gaussian kernel, bandwidth = median heuristic)
    n_permutations : int — permutations for p-value estimation
    random_state : optional int seed

    Returns
    -------
    dict with keys:
        'statistic'   : float, MMD² value
        'p_value'     : float, p-value under H0 (< 0.05 suggests drift)
        'significant' : bool, True if p_value < 0.05
        'bandwidth'   : float, RBF kernel bandwidth used

    Notes
    -----
    MMD²(P, Q) = E_{x,x'~P}[k(x,x')] + E_{y,y'~Q}[k(y,y')] - 2·E_{x~P,y~Q}[k(x,y)]
    where k is the kernel function. For H0: P=Q, MMD² ≈ 0.

    RBF kernel: k(x,y) = exp(-||x-y||² / (2σ²))
    Bandwidth σ² = median(pairwise sq. distances in combined sample) / 2
    """
    rng = np.random.default_rng(random_state)

    X_ref = np.atleast_2d(X_ref)
    X_new = np.atleast_2d(X_new)
    if X_ref.ndim == 1:
        X_ref = X_ref.reshape(-1, 1)
    if X_new.ndim == 1:
        X_new = X_new.reshape(-1, 1)

    n_ref, n_new = len(X_ref), len(X_new)

    # Sub-sample for large datasets to keep permutation test tractable
    max_n = 2000
    if n_ref + n_new > max_n:
        idx_ref = rng.choice(n_ref, size=min(n_ref, max_n // 2), replace=False)
        idx_new = rng.choice(n_new, size=min(n_new, max_n // 2), replace=False)
        X_ref_sub = X_ref[idx_ref]
        X_new_sub = X_new[idx_new]
    else:
        X_ref_sub, X_new_sub = X_ref, X_new

    n_ref_s, n_new_s = len(X_ref_sub), len(X_new_sub)

    # Bandwidth: median heuristic on combined sample
    combined = np.vstack([X_ref_sub, X_new_sub])
    # Compute pairwise squared distances via broadcasting (efficient for moderate n)
    diff = combined[:, np.newaxis, :] - combined[np.newaxis, :, :]   # (n, n, d)
    sq_dists = (diff ** 2).sum(axis=-1)  # (n, n)
    # Take upper triangle to avoid redundancy
    upper_idx = np.triu_indices(len(combined), k=1)
    median_sq = np.median(sq_dists[upper_idx])
    bandwidth = max(median_sq / 2.0, 1e-8)

    # Observed MMD²
    observed_stat = _mmd_squared(X_ref_sub, X_new_sub, bandwidth)

    # Permutation test
    combined_data = np.vstack([X_ref_sub, X_new_sub])
    n_total = n_ref_s + n_new_s
    permuted_stats = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(n_total)
        X_perm_ref = combined_data[perm[:n_ref_s]]
        X_perm_new = combined_data[perm[n_ref_s:]]
        permuted_stats[i] = _mmd_squared(X_perm_ref, X_perm_new, bandwidth)

    p_value = float(np.mean(permuted_stats >= observed_stat))

    return {
        'statistic': float(observed_stat),
        'p_value': p_value,
        'significant': p_value < 0.05,
        'bandwidth': float(bandwidth),
    }


def ks_test(
    X_ref: np.ndarray,
    X_new: np.ndarray
) -> dict:
    """
    Per-feature Kolmogorov-Smirnov two-sample test.

    Tests each feature independently. Useful for identifying *which* features
    have drifted, not just whether drift has occurred.

    Parameters
    ----------
    X_ref : np.ndarray, shape (n_ref, d)
    X_new : np.ndarray, shape (n_new, d)

    Returns
    -------
    dict with keys:
        'statistics'      : np.ndarray, shape (d,), KS statistic per feature
        'p_values'        : np.ndarray, shape (d,), p-value per feature
        'drifted_features': list of int, feature indices with p_value < 0.05
        'max_statistic'   : float, max KS statistic across features
        'n_drifted'       : int, count of drifted features

    Notes
    -----
    Uses scipy.stats.ks_2samp for each feature.
    The multiple-testing issue (d tests) is acknowledged but not corrected
    here — users should apply Bonferroni or BH correction if needed.
    """
    X_ref = np.asarray(X_ref)
    X_new = np.asarray(X_new)
    if X_ref.ndim == 1:
        X_ref = X_ref.reshape(-1, 1)
    if X_new.ndim == 1:
        X_new = X_new.reshape(-1, 1)

    d = X_ref.shape[1]
    statistics = np.empty(d)
    p_values = np.empty(d)

    for j in range(d):
        stat, pval = ks_2samp(X_ref[:, j], X_new[:, j])
        statistics[j] = stat
        p_values[j] = pval

    drifted_features = [j for j in range(d) if p_values[j] < 0.05]

    return {
        'statistics': statistics,
        'p_values': p_values,
        'drifted_features': drifted_features,
        'max_statistic': float(statistics.max()),
        'n_drifted': len(drifted_features),
    }


def covariate_shift_detector(
    X_ref: np.ndarray,
    X_new: np.ndarray
) -> dict:
    """
    Comprehensive covariate shift detection combining MMD and KS tests.

    Parameters
    ----------
    X_ref : np.ndarray, shape (n_ref, d)
    X_new : np.ndarray, shape (n_new, d)

    Returns
    -------
    dict with keys:
        'mmd'            : result of mmd_test()
        'ks'             : result of ks_test()
        'drift_detected' : bool, True if either test is significant
        'severity'       : str, 'none' | 'mild' | 'moderate' | 'severe'
                           based on MMD statistic magnitude
    """
    mmd_result = mmd_test(X_ref, X_new)
    ks_result = ks_test(X_ref, X_new)

    drift_detected = mmd_result['significant'] or ks_result['n_drifted'] > 0

    # Severity based on MMD statistic quartiles (heuristic thresholds)
    stat = mmd_result['statistic']
    if not drift_detected:
        severity = 'none'
    elif stat < 0.01:
        severity = 'mild'
    elif stat < 0.05:
        severity = 'moderate'
    else:
        severity = 'severe'

    return {
        'mmd': mmd_result,
        'ks': ks_result,
        'drift_detected': drift_detected,
        'severity': severity,
    }


class DeficiencyMonitor:
    """
    Estimates Le Cam deficiency δ(E_cal, E_test) from data.

    This is the "exact tier" drift detector. It estimates δ(E_cal, E_test)
    using the built-in Le Cam deficiency solver and certifies the bound:

        P_test(Y ∈ C_α(X)) ≥ (1 − α) − δ(P_cal, P_test)

    at the theorem level. Operationally, the SDK applies a fixed safety
    multiplier to the empirical estimator (default: c=2).

    Parameters
    ----------
    n_bins : int
        Number of histogram bins for feature discretization (default 20).
        Higher = more precise but slower and requires more data.
        Rule of thumb: n_bins ≤ sqrt(min(n_cal, n_test)).
    solver : str
        Backend for deficiency computation: 'auto' (default), 'sinkhorn',
        'cvxpy', 'scipy'. 'auto' uses Sinkhorn (fast). 'scipy' uses exact LP.
    n_bootstrap : int
        Bootstrap samples for confidence interval (default 200).
    method : str
        Histogram construction method:
            - 'class_conditional' (default): columns correspond to labels/classes.
            - 'marginal': legacy behavior, columns correspond to feature marginals.
    n_label_buckets : int
        Number of quantile buckets used when labels are continuous/regression-like
        under method='class_conditional' (default 4).
    hard_reject_threshold : float
        Operational threshold above which the raw deficiency estimate is treated
        as severe enough to require halt or human review (default 0.30).
    bound_multiplier : float
        Fixed operational multiplier applied to the empirical deficiency
        estimate (default 2.0).

    Usage
    -----
    monitor = DeficiencyMonitor(n_bins=20, method='class_conditional')
    monitor.fit(X_cal, y_cal)                      # store calibration distribution
    result = monitor.estimate(X_new, y_new=y_test) # estimate δ

    # result dict:
    # {
    #   'delta':                  float,   # deficiency estimate δ̂
    #   'ci_lower':               float,   # 95% bootstrap CI lower bound
    #   'ci_upper':               float,   # 95% bootstrap CI upper bound
    #   'coverage_bound':         float,   # (1-alpha) - c_multiplier*delta
    #   'coverage_bound_clipped': float,   # max(0, coverage_bound)
    #   'conservative_bound_note': str,    # note on the bound formula
    #   'warning':                bool,    # True if delta > warning threshold
    #   'hard_reject':            bool,    # severe-shift operational halt flag
    #   'c_multiplier':           float,   # multiplier applied to delta
    #   'operational_action':     str,     # monitor / recalibrate / reject_or_human_review
    #   'message':                str,     # human-readable explanation
    #   'method_used':            str,     # 'class_conditional' or 'marginal'
    #   'fallback_reason':        str|None # reason if fallback happened
    # }

    Implementation notes
    --------------------
    The deficiency solver takes two matrices:
        P : (n_rows, n_thetas)
        Q : (n_rows, n_thetas)
    where n_thetas is the number of parameter values θ.

    method='class_conditional' (default):
        θ is label/class.
        Column k is the normalized histogram of X restricted to {i: y_i = k}.
        For continuous targets, y is bucketed into quantile bins.
        This aligns with Theorem 1's experiment interpretation.

    method='marginal' (legacy):
        θ is feature index.
        Column j is the feature-j marginal histogram.
        This is retained for backward compatibility.

    Raises
    ------
    RuntimeError : If fit() has not been called before estimate().
    """

    def __init__(
        self,
        n_bins: int = 20,
        solver: str = "auto",
        n_bootstrap: int = 200,
        method: str = "class_conditional",
        n_label_buckets: int = 4,
        hard_reject_threshold: float = 0.30,
        bound_multiplier: float = 2.0,
    ):
        self.n_bins = n_bins
        self.solver = solver
        self.n_bootstrap = n_bootstrap
        self.method = method
        self.n_label_buckets = n_label_buckets
        self.hard_reject_threshold = float(hard_reject_threshold)
        self.bound_multiplier = float(bound_multiplier)
        self._ref_data = None
        self._ref_labels = None
        self._bin_edges = None
        self._label_mode = None
        self._label_bucket_edges = None

    @staticmethod
    def _as_2d_array(X: np.ndarray) -> np.ndarray:
        """Ensure feature matrix shape (n, d)."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    @staticmethod
    def _as_1d_labels(y: np.ndarray, n_samples: int, name: str) -> np.ndarray:
        """Validate and flatten labels to shape (n,)."""
        y = np.asarray(y)
        if y.ndim > 1:
            if y.shape[1] == 1:
                y = y.ravel()
            else:
                raise ValueError(f"{name} must be 1D. Got shape {y.shape}.")
        if len(y) != n_samples:
            raise ValueError(
                f"{name} length mismatch: expected {n_samples}, got {len(y)}."
            )
        return y

    def _should_bucketize_labels(self, y: np.ndarray) -> bool:
        """
        Heuristic: treat labels as continuous/regression-like when they have
        many unique numeric values.
        """
        unique_count = np.unique(y).size
        kind = y.dtype.kind

        if kind in ('f', 'c') and unique_count > self.n_label_buckets:
            return True
        if kind in ('i', 'u') and unique_count > max(20, 2 * self.n_label_buckets):
            return True
        return False

    def _fit_label_transform(self, y_ref: np.ndarray) -> np.ndarray:
        """Fit label transform on calibration labels and return transformed labels."""
        if self._should_bucketize_labels(y_ref):
            self._label_mode = 'quantile'
            y_float = np.asarray(y_ref, dtype=float)
            quantiles = np.linspace(0.0, 1.0, self.n_label_buckets + 1)
            edges = np.quantile(y_float, quantiles)
            edges = np.unique(edges)
            if edges.size < 2:
                center = float(np.mean(y_float))
                edges = np.array([center - 0.5, center + 0.5], dtype=float)
            self._label_bucket_edges = edges
            return np.digitize(y_float, edges[1:-1], right=True)

        self._label_mode = 'categorical'
        self._label_bucket_edges = None
        return np.asarray(y_ref)

    def _transform_labels(self, y: np.ndarray, n_samples: int, name: str) -> np.ndarray:
        """Transform labels with fit-time transform rules."""
        y = self._as_1d_labels(y, n_samples=n_samples, name=name)
        if self._label_mode == 'quantile':
            if self._label_bucket_edges is None:
                raise RuntimeError("Label bucket edges are not initialized. Call fit() first.")
            y_float = np.asarray(y, dtype=float)
            return np.digitize(y_float, self._label_bucket_edges[1:-1], right=True)
        return np.asarray(y)

    def fit(
        self,
        X_ref: np.ndarray,
        y_ref: Optional[np.ndarray] = None
    ) -> 'DeficiencyMonitor':
        """
        Store reference (calibration) distribution and compute bin edges.

        Parameters
        ----------
        X_ref : np.ndarray, shape (n_ref, d)
        y_ref : optional np.ndarray, shape (n_ref,)
            Labels for class-conditional histogram estimation. Required when
            method='class_conditional'. If omitted, fit falls back to method='marginal'
            with a warning.
        """
        if self.method not in ('class_conditional', 'marginal'):
            raise ValueError(
                f"Unknown method '{self.method}'. Use 'class_conditional' or 'marginal'."
            )

        X_ref = self._as_2d_array(X_ref)

        self._ref_data = X_ref.copy()
        self._ref_labels = None
        self._label_mode = None
        self._label_bucket_edges = None
        d = X_ref.shape[1]

        # Compute bin edges from reference data for each feature
        # Add small padding to include edge points
        self._bin_edges = []
        for j in range(d):
            col = X_ref[:, j]
            lo, hi = col.min(), col.max()
            # Expand range by 1% to include boundary points
            margin = max((hi - lo) * 0.01, 1e-8)
            edges = np.linspace(lo - margin, hi + margin, self.n_bins + 1)
            self._bin_edges.append(edges)

        if self.method == 'class_conditional':
            if y_ref is None:
                warnings.warn(
                    "DeficiencyMonitor(method='class_conditional') requires y_ref in fit(). "
                    "Falling back to marginal histograms.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                y_ref = self._as_1d_labels(y_ref, n_samples=len(X_ref), name='y_ref')
                self._ref_labels = self._fit_label_transform(y_ref)

        return self

    def _build_marginal_histogram(self, X: np.ndarray) -> np.ndarray:
        """Legacy histogram: each column is a per-feature marginal distribution."""
        return self._build_marginal_histogram_with_edges(X, bin_edges=self._bin_edges)

    def _make_bin_edges(self, n_bins: int) -> list:
        """Construct evenly spaced bin edges over the fit-time feature ranges."""
        if self._bin_edges is None:
            raise RuntimeError("DeficiencyMonitor must be fitted before bin-edge construction.")

        edges_list = []
        for edges in self._bin_edges:
            lo = float(edges[0])
            hi = float(edges[-1])
            edges_list.append(np.linspace(lo, hi, int(n_bins) + 1))
        return edges_list

    @staticmethod
    def _support_fraction(P: np.ndarray, Q: np.ndarray) -> float:
        """Fraction of histogram cells occupied by either calibration or test mass."""
        occupied = (P > 0.0) | (Q > 0.0)
        return float(np.mean(occupied))

    def _coverage_multiplier(self) -> float:
        """
        Operational inflation applied to the deficiency estimate.

        The sharp transfer constant for TV is 1. This multiplier is not part of
        the theorem; it is an engineering hedge applied to the empirical
        estimator.
        """
        return float(self.bound_multiplier)

    def _build_marginal_histogram_with_edges(
        self,
        X: np.ndarray,
        bin_edges: list,
    ) -> np.ndarray:
        """Legacy histogram with explicit bin edges."""
        d = X.shape[1]
        n_bins = len(bin_edges[0]) - 1
        hist_matrix = np.zeros((n_bins, d))

        for j in range(d):
            edges = bin_edges[j]
            col = np.clip(X[:, j], edges[0], edges[-1])
            hist, _ = np.histogram(col, bins=edges)
            total = hist.sum()
            if total > 0:
                hist_matrix[:, j] = hist / total
            else:
                hist_matrix[:, j] = 1.0 / n_bins

        return hist_matrix

    def _build_class_conditional_histogram(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        theta_values: np.ndarray
    ) -> np.ndarray:
        return self._build_class_conditional_histogram_with_edges(
            X,
            labels,
            theta_values,
            bin_edges=self._bin_edges,
        )

    def _build_class_conditional_histogram_with_edges(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        theta_values: np.ndarray,
        bin_edges: list,
    ) -> np.ndarray:
        """
        Class-conditional histogram: each column k is the histogram of X
        restricted to the subset {i : label_i = theta_values[k]}.

        We discretize each feature separately and stack feature-bin blocks into
        one long vector per class, giving shape (n_bins * d, K).
        """
        d = X.shape[1]
        n_bins = len(bin_edges[0]) - 1
        n_rows = n_bins * d
        K = len(theta_values)

        if K == 0:
            return np.full((n_rows, 1), 1.0 / n_rows)

        hist_matrix = np.zeros((n_rows, K))
        uniform_col = np.full(n_rows, 1.0 / n_rows)

        for k, theta in enumerate(theta_values):
            mask = labels == theta
            if not np.any(mask):
                hist_matrix[:, k] = uniform_col
                continue

            X_theta = X[mask]
            col_hist = np.zeros(n_rows)

            for j in range(d):
                edges = bin_edges[j]
                values = np.clip(X_theta[:, j], edges[0], edges[-1])
                hist, _ = np.histogram(values, bins=edges)
                start = j * n_bins
                col_hist[start:start + n_bins] = hist.astype(float)

            total = col_hist.sum()
            if total > 0:
                hist_matrix[:, k] = col_hist / total
            else:
                hist_matrix[:, k] = uniform_col

        return hist_matrix

    def _to_histogram(
        self,
        X: np.ndarray,
        labels: Optional[np.ndarray] = None,
        method: Optional[str] = None,
        theta_values: Optional[np.ndarray] = None,
        bin_edges: Optional[list] = None,
    ) -> np.ndarray:
        """
        Convert data to a probability matrix compatible with compute_deficiency.

        method='marginal':
            shape (n_bins, d), one column per feature.
        method='class_conditional':
            shape (n_bins * d, K), one column per label/class bucket.
        """
        if self._bin_edges is None and bin_edges is None:
            raise RuntimeError("DeficiencyMonitor must be fitted before histogram conversion.")

        X = self._as_2d_array(X)
        active_method = self.method if method is None else method
        active_edges = self._bin_edges if bin_edges is None else bin_edges

        if active_method == 'marginal' or labels is None:
            return self._build_marginal_histogram_with_edges(X, bin_edges=active_edges)

        if theta_values is None:
            theta_values = np.unique(labels)
        return self._build_class_conditional_histogram_with_edges(
            X,
            labels,
            np.asarray(theta_values),
            bin_edges=active_edges,
        )

    def estimate(
        self,
        X_new: np.ndarray,
        y_new: Optional[np.ndarray] = None,
        alpha: float = 0.10,
        warning_threshold: float = 0.05
    ) -> dict:
        """
        Estimate deficiency of X_new relative to reference calibration data.

        Parameters
        ----------
        X_new : np.ndarray, shape (n_new, d)
        y_new : optional np.ndarray, shape (n_new,)
            Labels for class-conditional estimation. Required when
            method='class_conditional'. If omitted, falls back to marginal
            estimation with a warning.
        alpha : float, conformal miscoverage rate (for coverage_bound)
        warning_threshold : float, issue warning if δ̂ > this value

        Returns
        -------
        dict with keys: delta, ci_lower, ci_upper, coverage_bound,
            coverage_bound_clipped, conservative_bound_note, warning, hard_reject,
            c_multiplier, operational_action,
            message, method_used, fallback_reason
        """
        if self._ref_data is None or self._bin_edges is None:
            raise RuntimeError("DeficiencyMonitor must be fitted before calling estimate(). "
                               "Call monitor.fit(X_cal) first.")

        X_new = self._as_2d_array(X_new)
        if X_new.shape[1] != self._ref_data.shape[1]:
            raise ValueError(
                f"Feature mismatch: X_new has {X_new.shape[1]} features, "
                f"expected {self._ref_data.shape[1]} from fit()."
            )

        rng = np.random.default_rng(42)
        used_method = self.method
        fallback_reason = None
        y_new_transformed = None
        theta_values = None

        if self.method == 'class_conditional':
            if self._ref_labels is None:
                used_method = 'marginal'
                fallback_reason = "missing y_ref in fit()"
                warnings.warn(
                    "estimate() falling back to marginal histograms because fit() "
                    "did not receive y_ref under method='class_conditional'.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            elif y_new is None:
                used_method = 'marginal'
                fallback_reason = "missing y_new in estimate()"
                warnings.warn(
                    "estimate() falling back to marginal histograms because y_new "
                    "was not provided under method='class_conditional'.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                y_new_transformed = self._transform_labels(
                    y_new,
                    n_samples=len(X_new),
                    name='y_new'
                )
                theta_values = np.unique(
                    np.concatenate([np.asarray(self._ref_labels), np.asarray(y_new_transformed)])
                )

        if used_method == 'class_conditional':
            P = self._to_histogram(
                self._ref_data,
                labels=np.asarray(self._ref_labels),
                method='class_conditional',
                theta_values=theta_values,
            )
            Q = self._to_histogram(
                X_new,
                labels=np.asarray(y_new_transformed),
                method='class_conditional',
                theta_values=theta_values,
            )
        else:
            P = self._to_histogram(self._ref_data, method='marginal')
            Q = self._to_histogram(X_new, method='marginal')

        delta = _compute_deficiency(P, Q, solver=self.solver)
        if delta is None:
            delta = float('nan')

        # Bootstrap confidence interval
        n_ref = len(self._ref_data)
        n_new = len(X_new)
        deltas_boot = []
        for _ in range(self.n_bootstrap):
            idx_ref = rng.integers(0, n_ref, size=n_ref)
            idx_new = rng.integers(0, n_new, size=n_new)
            X_ref_b = self._ref_data[idx_ref]
            X_new_b = X_new[idx_new]

            if used_method == 'class_conditional':
                y_ref_b = np.asarray(self._ref_labels)[idx_ref]
                y_new_b = np.asarray(y_new_transformed)[idx_new]
                theta_values_b = np.unique(np.concatenate([y_ref_b, y_new_b]))
                P_b = self._to_histogram(
                    X_ref_b,
                    labels=y_ref_b,
                    method='class_conditional',
                    theta_values=theta_values_b,
                )
                Q_b = self._to_histogram(
                    X_new_b,
                    labels=y_new_b,
                    method='class_conditional',
                    theta_values=theta_values_b,
                )
            else:
                P_b = self._to_histogram(X_ref_b, method='marginal')
                Q_b = self._to_histogram(X_new_b, method='marginal')

            d_b = _compute_deficiency(P_b, Q_b, solver=self.solver)
            if d_b is not None:
                deltas_boot.append(d_b)

        if deltas_boot:
            ci_lower = float(np.percentile(deltas_boot, 2.5))
            ci_upper = float(np.percentile(deltas_boot, 97.5))
        else:
            ci_lower = ci_upper = float('nan')

        c_multiplier = self._coverage_multiplier()
        coverage_bound = (1.0 - alpha) - c_multiplier * delta
        coverage_bound_clipped = max(0.0, float(coverage_bound))

        warning = bool(delta > warning_threshold)
        hard_reject = bool(delta > self.hard_reject_threshold or coverage_bound < 0.0)

        if hard_reject:
            operational_action = "reject_or_human_review"
        elif warning:
            operational_action = "recalibrate_or_investigate"
        else:
            operational_action = "monitor"

        multiplier_desc = f"fixed c={c_multiplier:.4f}"

        if hard_reject:
            message = (
                f"SEVERE SHIFT: Estimated deficiency δ̂ = {delta:.4f}. "
                f"Operational certificate uses {multiplier_desc}, giving bound "
                f"{coverage_bound:.4f} (clipped display: {coverage_bound_clipped:.4f}; "
                f"target: {1 - alpha:.4f}). This exceeds the hard-reject policy "
                f"threshold {self.hard_reject_threshold:.4f}; halt or route to human review. "
                f"Method used: {used_method}."
            )
        elif warning:
            message = (
                f"WARNING: Estimated deficiency δ̂ = {delta:.4f} exceeds threshold "
                f"{warning_threshold:.4f}. Operational certificate uses {multiplier_desc}, "
                f"giving bound {coverage_bound:.4f} (target: {1 - alpha:.4f}). "
                f"Recalibrate or investigate distribution shift. Method used: {used_method}."
            )
        else:
            message = (
                f"OK: Estimated deficiency δ̂ = {delta:.4f} is below threshold "
                f"{warning_threshold:.4f}. Operational certificate uses {multiplier_desc}, "
                f"giving bound {coverage_bound:.4f} (target: {1 - alpha:.4f}). "
                f"Method used: {used_method}."
            )

        if fallback_reason is not None:
            message += f" Fallback reason: {fallback_reason}."

        return {
            'delta': float(delta),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'coverage_bound': float(coverage_bound),
            'coverage_bound_clipped': coverage_bound_clipped,
            'conservative_bound_note': (
                "The theorem-level TV transfer constant is 1. "
                "coverage_bound applies a fixed operational safety multiplier to the "
                "empirical deficiency estimate: "
                "coverage_bound = (1-alpha) - c_multiplier*delta."
            ),
            'warning': warning,
            'hard_reject': hard_reject,
            'c_multiplier': float(c_multiplier),
            'operational_action': operational_action,
            'message': message,
            'method_used': used_method,
            'fallback_reason': fallback_reason,
        }

    def coverage_validity_warning(
        self,
        X_new: np.ndarray,
        y_new: Optional[np.ndarray] = None,
        alpha: float = 0.10,
        delta_threshold: float = 0.05
    ) -> dict:
        """
        High-level: is the conformal coverage still valid?

        Returns a simplified dict suitable for a monitoring dashboard.
        """
        result = self.estimate(
            X_new,
            y_new=y_new,
            alpha=alpha,
            warning_threshold=delta_threshold
        )
        return {
            'valid': not (result.get('warning', True) or result.get('hard_reject', False)),
            'delta': result.get('delta'),
            'coverage_bound': result.get('coverage_bound'),
            'coverage_bound_clipped': result.get('coverage_bound_clipped'),
            'operational_action': result.get('operational_action'),
            'message': result.get('message', ''),
        }
