"""
Tests for reliability.drift

Tests for mmd_test, ks_test, covariate_shift_detector, and DeficiencyMonitor.
The deficiency solver is self-contained within the package; all tests run
unconditionally.
"""

import numpy as np
import pytest
from reliability.drift import (
    mmd_test,
    ks_test,
    covariate_shift_detector,
    DeficiencyMonitor,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def same_dist(rng):
    """Two samples from the same N(0,1) distribution."""
    X_ref = rng.standard_normal((200, 3))
    X_new = rng.standard_normal((200, 3))
    return X_ref, X_new


@pytest.fixture
def shifted_dist(rng):
    """Two samples with a clear mean shift (μ=2)."""
    X_ref = rng.standard_normal((200, 3))
    X_new = rng.standard_normal((200, 3)) + 2.0
    return X_ref, X_new


# ============================================================
# mmd_test
# ============================================================

class TestMMDTest:
    def test_returns_expected_keys(self, same_dist):
        X_ref, X_new = same_dist
        result = mmd_test(X_ref, X_new, n_permutations=50, random_state=0)
        assert set(result.keys()) >= {'statistic', 'p_value', 'significant', 'bandwidth'}

    def test_statistic_nonnegative(self, same_dist):
        X_ref, X_new = same_dist
        result = mmd_test(X_ref, X_new, n_permutations=50, random_state=0)
        assert result['statistic'] >= 0.0

    def test_bandwidth_positive(self, same_dist):
        X_ref, X_new = same_dist
        result = mmd_test(X_ref, X_new, n_permutations=50, random_state=0)
        assert result['bandwidth'] > 0.0

    def test_p_value_in_01(self, same_dist):
        X_ref, X_new = same_dist
        result = mmd_test(X_ref, X_new, n_permutations=50, random_state=0)
        assert 0.0 <= result['p_value'] <= 1.0

    def test_significant_matches_p_value(self, same_dist):
        X_ref, X_new = same_dist
        result = mmd_test(X_ref, X_new, n_permutations=50, random_state=0)
        assert result['significant'] == (result['p_value'] < 0.05)

    def test_same_dist_low_statistic(self, rng):
        """Same distribution → small MMD statistic."""
        X = rng.standard_normal((500, 2))
        Y = rng.standard_normal((500, 2))
        result = mmd_test(X, Y, n_permutations=50, random_state=0)
        assert result['statistic'] < 0.05

    def test_large_shift_detected(self, rng):
        """Large distribution shift → significant test."""
        X_ref = rng.standard_normal((300, 2))
        X_new = rng.standard_normal((300, 2)) + 3.0
        result = mmd_test(X_ref, X_new, n_permutations=200, random_state=0)
        assert result['significant'], f"Expected drift to be detected, got p={result['p_value']}"

    def test_1d_input(self, rng):
        """1D array input is handled (reshaped internally)."""
        X_ref = rng.standard_normal(100)
        X_new = rng.standard_normal(100) + 1.0
        result = mmd_test(X_ref, X_new, n_permutations=50, random_state=0)
        assert 'statistic' in result

    def test_reproducible_with_seed(self, same_dist):
        X_ref, X_new = same_dist
        r1 = mmd_test(X_ref, X_new, n_permutations=50, random_state=7)
        r2 = mmd_test(X_ref, X_new, n_permutations=50, random_state=7)
        assert r1['statistic'] == r2['statistic']
        assert r1['p_value'] == r2['p_value']


# ============================================================
# ks_test
# ============================================================

class TestKSTest:
    def test_returns_expected_keys(self, same_dist):
        X_ref, X_new = same_dist
        result = ks_test(X_ref, X_new)
        assert set(result.keys()) >= {
            'statistics', 'p_values', 'drifted_features', 'max_statistic', 'n_drifted'
        }

    def test_statistics_shape(self, same_dist):
        X_ref, X_new = same_dist
        d = X_ref.shape[1]
        result = ks_test(X_ref, X_new)
        assert result['statistics'].shape == (d,)
        assert result['p_values'].shape == (d,)

    def test_statistics_in_01(self, same_dist):
        X_ref, X_new = same_dist
        result = ks_test(X_ref, X_new)
        assert np.all(result['statistics'] >= 0.0)
        assert np.all(result['statistics'] <= 1.0)

    def test_p_values_in_01(self, same_dist):
        X_ref, X_new = same_dist
        result = ks_test(X_ref, X_new)
        assert np.all(result['p_values'] >= 0.0)
        assert np.all(result['p_values'] <= 1.0)

    def test_max_statistic(self, same_dist):
        X_ref, X_new = same_dist
        result = ks_test(X_ref, X_new)
        assert result['max_statistic'] == pytest.approx(result['statistics'].max())

    def test_n_drifted_matches_list(self, same_dist):
        X_ref, X_new = same_dist
        result = ks_test(X_ref, X_new)
        assert result['n_drifted'] == len(result['drifted_features'])

    def test_shifted_dist_detects_drift(self, shifted_dist):
        X_ref, X_new = shifted_dist
        result = ks_test(X_ref, X_new)
        assert result['n_drifted'] >= 1, "Expected at least one drifted feature"
        assert result['max_statistic'] > 0.1

    def test_1d_input(self, rng):
        X_ref = rng.standard_normal(200)
        X_new = rng.standard_normal(200) + 1.5
        result = ks_test(X_ref, X_new)
        assert result['statistics'].shape == (1,)
        assert result['n_drifted'] >= 0

    def test_drifted_features_indices_valid(self, shifted_dist):
        X_ref, X_new = shifted_dist
        d = X_ref.shape[1]
        result = ks_test(X_ref, X_new)
        for idx in result['drifted_features']:
            assert 0 <= idx < d


# ============================================================
# covariate_shift_detector
# ============================================================

class TestCovariateShiftDetector:
    def test_returns_expected_keys(self, same_dist):
        X_ref, X_new = same_dist
        result = covariate_shift_detector(X_ref, X_new)
        assert set(result.keys()) >= {'mmd', 'ks', 'drift_detected', 'severity'}

    def test_severity_values(self, same_dist):
        X_ref, X_new = same_dist
        result = covariate_shift_detector(X_ref, X_new)
        assert result['severity'] in ('none', 'mild', 'moderate', 'severe')

    def test_no_drift_when_same(self, rng):
        """Identical data → no drift detected."""
        X = rng.standard_normal((300, 2))
        # Use same array as ref and new — MMD should be exactly 0
        result = covariate_shift_detector(X, X.copy())
        # MMD should be ~0, ks should pass
        assert result['mmd']['statistic'] < 0.01

    def test_large_shift_detected(self, shifted_dist):
        X_ref, X_new = shifted_dist
        result = covariate_shift_detector(X_ref, X_new)
        assert result['drift_detected'] is True


# ============================================================
# DeficiencyMonitor — fit and histogram (no robustea needed)
# ============================================================

class TestDeficiencyMonitorFit:
    def test_fit_stores_ref_data(self, rng):
        X_ref = rng.standard_normal((100, 2))
        mon = DeficiencyMonitor(n_bins=10, method='marginal')
        mon.fit(X_ref)
        assert mon._ref_data is not None
        assert mon._bin_edges is not None
        assert len(mon._bin_edges) == 2

    def test_fit_returns_self(self, rng):
        X_ref = rng.standard_normal((100, 2))
        mon = DeficiencyMonitor(n_bins=10, method='marginal')
        result = mon.fit(X_ref)
        assert result is mon

    def test_bin_edges_shape(self, rng):
        X_ref = rng.standard_normal((100, 3))
        mon = DeficiencyMonitor(n_bins=15, method='marginal')
        mon.fit(X_ref)
        assert len(mon._bin_edges) == 3
        for edges in mon._bin_edges:
            assert len(edges) == 16  # n_bins + 1

    def test_to_histogram_shape(self, rng):
        X_ref = rng.standard_normal((200, 4))
        mon = DeficiencyMonitor(n_bins=10, method='marginal')
        mon.fit(X_ref)
        H = mon._to_histogram(X_ref)
        assert H.shape == (10, 4)

    def test_to_histogram_sums_to_one(self, rng):
        X_ref = rng.standard_normal((200, 3))
        mon = DeficiencyMonitor(n_bins=10, method='marginal')
        mon.fit(X_ref)
        H = mon._to_histogram(X_ref)
        col_sums = H.sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-10)

    def test_to_histogram_nonnegative(self, rng):
        X_ref = rng.standard_normal((200, 3))
        mon = DeficiencyMonitor(n_bins=10, method='marginal')
        mon.fit(X_ref)
        H = mon._to_histogram(X_ref)
        assert np.all(H >= 0.0)

    def test_histogram_out_of_range_clipped(self, rng):
        """Out-of-range test data is clipped to boundary bins, not ignored."""
        X_ref = rng.standard_normal((200, 2))
        mon = DeficiencyMonitor(n_bins=10, method='marginal')
        mon.fit(X_ref)
        X_extreme = np.full((50, 2), 100.0)  # way outside calibration range
        H = mon._to_histogram(X_extreme)
        col_sums = H.sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-10)

    def test_estimate_raises_before_fit(self, rng):
        X_new = rng.standard_normal((50, 2))
        mon = DeficiencyMonitor(n_bins=10, method='marginal')
        with pytest.raises(RuntimeError, match="fitted"):
            mon.estimate(X_new)

    def test_1d_fit(self, rng):
        """1D input (n_samples,) is handled."""
        X_ref = rng.standard_normal(100)
        mon = DeficiencyMonitor(n_bins=10, method='marginal')
        mon.fit(X_ref)
        assert mon._ref_data.shape == (100, 1)

    def test_class_conditional_fit_requires_labels(self, rng):
        X_ref = rng.standard_normal((120, 3))
        mon = DeficiencyMonitor(n_bins=8, method='class_conditional')
        with pytest.warns(RuntimeWarning, match="requires y_ref"):
            mon.fit(X_ref)
        assert mon._ref_labels is None

    def test_class_conditional_histogram_shape(self, rng):
        X_ref = rng.standard_normal((200, 3))
        y_ref = (X_ref[:, 0] > 0.0).astype(int)
        mon = DeficiencyMonitor(n_bins=8, method='class_conditional')
        mon.fit(X_ref, y_ref=y_ref)
        theta = np.unique(mon._ref_labels)
        H = mon._to_histogram(
            X_ref,
            labels=mon._ref_labels,
            method='class_conditional',
            theta_values=theta,
        )
        assert H.shape == (8 * 3, len(theta))
        np.testing.assert_allclose(H.sum(axis=0), 1.0, atol=1e-10)


# ============================================================
# DeficiencyMonitor — full computation
# ============================================================

class TestDeficiencyMonitorFull:
    def test_estimate_returns_expected_keys(self, rng):
        X_ref = rng.standard_normal((200, 2))
        X_new = rng.standard_normal((100, 2))
        mon = DeficiencyMonitor(n_bins=10, n_bootstrap=10, method='marginal')
        mon.fit(X_ref)
        result = mon.estimate(X_new)
        required_keys = {
            'delta', 'ci_lower', 'ci_upper', 'coverage_bound',
            'coverage_bound_clipped', 'conservative_bound_note', 'warning',
            'hard_reject', 'c_multiplier', 'operational_action', 'message',
            'method_used', 'fallback_reason'
        }
        assert required_keys <= set(result.keys())

    def test_delta_nonnegative(self, rng):
        X_ref = rng.standard_normal((200, 2))
        X_new = rng.standard_normal((100, 2))
        mon = DeficiencyMonitor(n_bins=10, n_bootstrap=5, method='marginal')
        mon.fit(X_ref)
        result = mon.estimate(X_new)
        assert result['delta'] >= 0.0

    def test_ci_ordering(self, rng):
        X_ref = rng.standard_normal((200, 2))
        X_new = rng.standard_normal((100, 2))
        mon = DeficiencyMonitor(n_bins=10, n_bootstrap=10, method='marginal')
        mon.fit(X_ref)
        result = mon.estimate(X_new)
        assert result['ci_lower'] <= result['delta'] or result['ci_lower'] <= result['ci_upper']
        assert result['ci_lower'] <= result['ci_upper']

    def test_same_dist_low_delta(self, rng):
        """Same distribution → near-zero deficiency."""
        X_ref = rng.standard_normal((300, 2))
        X_new = rng.standard_normal((300, 2))
        mon = DeficiencyMonitor(n_bins=10, n_bootstrap=5, method='marginal')
        mon.fit(X_ref)
        result = mon.estimate(X_new)
        assert result['delta'] < 0.3, f"Expected small delta for same dist, got {result['delta']}"

    def test_warning_triggered_for_large_shift(self, rng):
        """Warning fires when threshold is set below the observed delta."""
        X_ref = rng.standard_normal((200, 2))
        X_new = rng.standard_normal((200, 2)) + 3.0
        mon = DeficiencyMonitor(n_bins=10, n_bootstrap=5, method='marginal')
        mon.fit(X_ref)
        # First get the delta, then verify the warning fires below it
        result_no_warn = mon.estimate(X_new, warning_threshold=1.0)
        delta = result_no_warn['delta']
        # Warning should fire if threshold is set below delta
        result = mon.estimate(X_new, warning_threshold=delta - 1e-6)
        assert result['warning'] is True

    def test_warning_not_triggered_for_same_dist(self, rng):
        """No shift → no warning."""
        X_ref = rng.standard_normal((300, 2))
        X_new = rng.standard_normal((300, 2))
        mon = DeficiencyMonitor(n_bins=10, n_bootstrap=5, method='marginal')
        mon.fit(X_ref)
        result = mon.estimate(X_new, warning_threshold=0.3)
        assert result['warning'] is False

    def test_coverage_bound_formula(self, rng):
        """Fixed-mode coverage_bound = (1 - alpha) - c * delta."""
        X_ref = rng.standard_normal((200, 2))
        X_new = rng.standard_normal((100, 2)) + 1.0
        mon = DeficiencyMonitor(
            n_bins=10,
            n_bootstrap=5,
            method='marginal',
            bound_multiplier=2.0,
        )
        mon.fit(X_ref)
        alpha = 0.1
        result = mon.estimate(X_new, alpha=alpha)
        expected = (1 - alpha) - 2.0 * result['delta']
        assert result['coverage_bound'] == pytest.approx(expected, abs=1e-9)

    def test_conservative_bound_note_present(self, rng):
        X_ref = rng.standard_normal((100, 2))
        X_new = rng.standard_normal((50, 2))
        mon = DeficiencyMonitor(n_bins=10, n_bootstrap=5, method='marginal')
        mon.fit(X_ref)
        result = mon.estimate(X_new)
        note = result['conservative_bound_note']
        assert 'transfer constant is 1' in note

    def test_coverage_validity_warning_valid(self, rng):
        """When delta is small, coverage_validity_warning returns valid=True."""
        X_ref = rng.standard_normal((300, 2))
        X_new = rng.standard_normal((300, 2))
        mon = DeficiencyMonitor(n_bins=10, n_bootstrap=5, method='marginal')
        mon.fit(X_ref)
        result = mon.coverage_validity_warning(X_new, alpha=0.1, delta_threshold=0.5)
        assert result['valid'] is True
        assert 'delta' in result
        assert 'coverage_bound' in result
        assert 'message' in result

    def test_coverage_validity_warning_invalid(self, rng):
        """coverage_validity_warning returns valid=False when threshold is below delta."""
        X_ref = rng.standard_normal((200, 2))
        X_new = rng.standard_normal((200, 2)) + 3.0
        mon = DeficiencyMonitor(n_bins=10, n_bootstrap=5, method='marginal')
        mon.fit(X_ref)
        # Get delta first, set threshold below it
        result_ref = mon.estimate(X_new, warning_threshold=1.0)
        delta = result_ref['delta']
        threshold_below_delta = max(delta - 1e-6, 0.0)
        result = mon.coverage_validity_warning(X_new, alpha=0.1, delta_threshold=threshold_below_delta)
        assert result['valid'] is False

    def test_result_keys_present(self, rng):
        X_ref = rng.standard_normal((100, 2))
        X_new = rng.standard_normal((50, 2))
        mon = DeficiencyMonitor(n_bins=10, n_bootstrap=5, method='marginal')
        mon.fit(X_ref)
        result = mon.estimate(X_new)
        assert 'delta' in result
        assert 'coverage_bound' in result
        assert result['delta'] is not None

    def test_same_distribution_low_delta_both_methods(self, rng):
        X_ref = rng.standard_normal((300, 4))
        X_new = rng.standard_normal((300, 4))
        y_ref = (X_ref[:, 0] > 0.0).astype(int)
        y_new = (X_new[:, 0] > 0.0).astype(int)

        mon_marg = DeficiencyMonitor(n_bins=10, n_bootstrap=0, method='marginal')
        mon_marg.fit(X_ref)
        d_marg = mon_marg.estimate(X_new)['delta']

        mon_cc = DeficiencyMonitor(n_bins=10, n_bootstrap=0, method='class_conditional')
        mon_cc.fit(X_ref, y_ref=y_ref)
        d_cc = mon_cc.estimate(X_new, y_new=y_new)['delta']

        assert d_marg < 0.3
        assert d_cc < 0.3

    def test_homogeneous_shift_positive_delta_class_conditional(self, rng):
        X_ref = rng.standard_normal((400, 4))
        y_ref = (X_ref[:, 0] > 0.0).astype(int)

        X_new = rng.standard_normal((400, 4)) + 1.25
        y_new = (X_new[:, 0] > 0.0).astype(int)

        mon = DeficiencyMonitor(n_bins=12, n_bootstrap=0, method='class_conditional')
        mon.fit(X_ref, y_ref=y_ref)
        result = mon.estimate(X_new, y_new=y_new)

        assert result['method_used'] == 'class_conditional'
        assert result['delta'] > 0.01

    def test_class_conditional_missing_y_new_falls_back(self, rng):
        X_ref = rng.standard_normal((200, 3))
        y_ref = (X_ref[:, 0] > 0.0).astype(int)
        X_new = rng.standard_normal((200, 3)) + 1.0

        mon = DeficiencyMonitor(n_bins=10, n_bootstrap=0, method='class_conditional')
        mon.fit(X_ref, y_ref=y_ref)

        with pytest.warns(RuntimeWarning, match="falling back to marginal"):
            result = mon.estimate(X_new)

        assert result['method_used'] == 'marginal'
        assert result['fallback_reason'] == "missing y_new in estimate()"
