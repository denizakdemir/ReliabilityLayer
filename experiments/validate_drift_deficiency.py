"""
validate_drift_deficiency.py
============================

Numerical validation: does DeficiencyMonitor.estimate() track empirical
coverage loss under distribution shift?

Theory under test (Paper 5 Theorem 1):
    Let C_α be calibrated on distribution P_cal. If δ(P_cal, P_test) ≤ ε,
    then P_test(Y ∈ C_α(X)) ≥ (1 - α) - c*ε  (c=2 used here).

Experiment design
-----------------
We vary a shift parameter μ ∈ {0, 0.25, 0.5, 0.75, 1.0} and measure:
  - Empirical coverage loss: max(0, (1-α) - empirical_coverage)
  - Deficiency estimate δ̂: from DeficiencyMonitor(method='class_conditional',
    solver='exact')

Label rule: Y = quantile-bucket(X[:,0]) ∈ {0,...,7} using 8 equal-probability
bins fitted on a large reference sample. This gives K=8 columns in the
experiment matrix, providing sufficient column diversity for the LP to
produce a non-degenerate deficiency estimate under Gaussian covariate shift.
Coverage is measured on the binary event {Y_binary ∈ C_α(X)} where
Y_binary = 1[X[:,0] > 0], keeping the conformal task unchanged.

Success criterion: Pearson r > 0.9 between coverage loss and δ̂ across μ levels.

OUTPUT:
    - Console: table of μ, empirical coverage, coverage loss, δ̂, c=2 bound
    - Figure: saved to experiments/figures/validate_drift_deficiency.png
    - Extra stress check: μ=2.0 deficiency magnitude (δ̂ > 0.05 target)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reliability.conformal import ConformalClassifier
from reliability.drift import DeficiencyMonitor


# ====================================================================
# Label rule: 8 quantile buckets of X[:,0]
# ====================================================================

_LABEL_EDGES: np.ndarray = np.quantile(
    np.random.default_rng(0).standard_normal(200_000),
    np.linspace(0.0, 1.0, 9),
)


def _label_from_x0(x0: np.ndarray) -> np.ndarray:
    """Map X[:,0] to one of 8 quantile-based label classes {0,...,7}."""
    return np.digitize(x0, _LABEL_EDGES[1:-1]).astype(int)


# ====================================================================
# Model
# ====================================================================

class SigmoidClassifier:
    """
    Binary classifier: P(Y=1|X) = sigmoid(-X[:,0]).

    Used for conformal coverage measurement only (binary label).
    The deficiency estimator uses the 8-class label rule.
    """
    def predict_proba(self, X):
        z = -X[:, 0]
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p, p])


# ====================================================================
# Data generation
# ====================================================================

def generate_calibration_data(n, d, rng):
    """
    Calibration: X ~ N(0, I_d).
    Returns X and 8-class label y = quantile-bucket(X[:,0]).
    """
    X = rng.standard_normal((n, d))
    y = _label_from_x0(X[:, 0])
    return X, y


def generate_test_data(n, d, mu, rng):
    """
    Test: X ~ N(mu, I_d), same 8-class label rule.
    """
    X = rng.standard_normal((n, d)) + mu
    y = _label_from_x0(X[:, 0])
    return X, y


# ====================================================================
# Main experiment
# ====================================================================

def run_experiment(
    alpha=0.10,
    n_cal=500,
    n_test=500,
    n_trials=50,
    d=4,
    n_bins=20,
    n_bootstrap=0,
    mu_grid=None,
    verbose=True,
):
    """
    For each μ in mu_grid: measure empirical coverage loss and δ̂.

    Returns
    -------
    dict with lists: mu, coverage, coverage_loss, delta, bound_c2, r_pearson
    """
    if mu_grid is None:
        mu_grid = [0.0, 0.25, 0.5, 0.75, 1.0]

    model = SigmoidClassifier()
    results = {
        'mu': [],
        'coverage': [],
        'coverage_loss': [],
        'delta': [],
        'bound_c2': [],
    }

    if verbose:
        print(f"{'μ':>5}  {'coverage':>10}  {'loss':>8}  {'δ̂':>10}  {'bound(c=2)':>12}  {'status':>8}")
        print("-" * 65)

    for mu in mu_grid:
        coverages = []
        deltas = []

        for seed in range(n_trials):
            rng = np.random.default_rng(seed)
            X_cal, y_cal = generate_calibration_data(n_cal, d, rng)
            X_test, y_test = generate_test_data(n_test, d, mu, rng)

            # Coverage on binary label (the conformal task)
            y_cal_binary = (X_cal[:, 0] > 0).astype(int)
            y_test_binary = (X_test[:, 0] > 0).astype(int)
            clf = ConformalClassifier(model, alpha=alpha)
            clf.calibrate(X_cal, y_cal_binary)
            cov = clf.empirical_coverage(X_test, y_test_binary)
            coverages.append(cov)

            # Deficiency on 8-class label
            mon = DeficiencyMonitor(
                n_bins=n_bins,
                solver='exact',
                n_bootstrap=n_bootstrap,
                method='class_conditional'
            )
            mon.fit(X_cal, y_ref=y_cal)
            est = mon.estimate(X_test, y_new=y_test, alpha=alpha)
            delta = est.get('delta')
            if delta is not None and not np.isnan(delta):
                deltas.append(delta)

        mean_cov = float(np.mean(coverages))
        mean_loss = float(max(0.0, (1.0 - alpha) - mean_cov))
        mean_delta = float(np.mean(deltas)) if deltas else float('nan')
        bound_c2 = float((1.0 - alpha) - 2.0 * mean_delta) if not np.isnan(mean_delta) else float('nan')

        results['mu'].append(mu)
        results['coverage'].append(mean_cov)
        results['coverage_loss'].append(mean_loss)
        results['delta'].append(mean_delta)
        results['bound_c2'].append(bound_c2)

        if verbose:
            status = ""
            if not np.isnan(bound_c2):
                status = "OK" if mean_cov >= bound_c2 else "VIOLATED"
            delta_str = f"{mean_delta:.5f}" if not np.isnan(mean_delta) else "  N/A  "
            bound_str = f"{bound_c2:.5f}" if not np.isnan(bound_c2) else "  N/A  "
            print(f"{mu:>5.2f}  {mean_cov:>10.4f}  {mean_loss:>8.4f}  "
                  f"{delta_str:>10}  {bound_str:>12}  {status:>8}")

    from scipy.stats import pearsonr
    losses = np.array(results['coverage_loss'])
    deltas_arr = np.array(results['delta'])
    valid = ~np.isnan(deltas_arr)
    if valid.sum() >= 3:
        r, p = pearsonr(losses[valid], deltas_arr[valid])
        results['r_pearson'] = float(r)
        results['p_pearson'] = float(p)
    else:
        r, p = float('nan'), float('nan')
        results['r_pearson'] = r
        results['p_pearson'] = p

    bound_valid = all(
        results['coverage'][i] >= results['bound_c2'][i]
        for i in range(len(mu_grid))
        if not np.isnan(results['bound_c2'][i])
    )

    if verbose:
        print()
        print(f"Pearson r(coverage_loss, δ̂) = {r:.4f}  (p={p:.4f})")
        criterion = "PASS (r > 0.9)" if r > 0.9 else "FAIL (r <= 0.9)"
        print(f"Success criterion r > 0.9: {criterion}")
        print()
        if bound_valid:
            print("Theorem 1 bound (c=2): holds for all μ values.")
        else:
            print("WARNING: Theorem 1 bound VIOLATED for some μ values.")

    return results


def stress_check_mu2(
    n_trials=30,
    n_cal=500,
    n_test=500,
    d=4,
    n_bins=20
):
    """Verify δ̂ > 0.05 for the larger homogeneous shift μ=2.0."""
    deltas = []
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        X_cal, y_cal = generate_calibration_data(n_cal, d, rng)
        X_test, y_test = generate_test_data(n_test, d, mu=2.0, rng=rng)
        mon = DeficiencyMonitor(n_bins=n_bins, solver='exact',
                                n_bootstrap=0, method='class_conditional')
        mon.fit(X_cal, y_ref=y_cal)
        est = mon.estimate(X_test, y_new=y_test, alpha=0.10)
        if est.get('delta') is not None and not np.isnan(est['delta']):
            deltas.append(est['delta'])
    return float(np.mean(deltas)) if deltas else float('nan')


# ====================================================================
# Plotting
# ====================================================================

def plot_results(results, save_path=None, show=False):
    """
    Three-panel figure:
    1. Coverage and Theorem 1 bound vs μ
    2. Deficiency δ̂ vs μ
    3. Coverage loss vs δ̂ (scatter with Pearson r annotation)
    """
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    C_COV  = '#1f77b4'
    C_DEF  = '#ff7f0e'
    C_C2   = '#d62728'
    C_REF  = '#888888'

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    mu_arr    = np.array(results['mu'])
    cov_arr   = np.array(results['coverage'])
    loss_arr  = np.array(results['coverage_loss'])
    delta_arr = np.array(results['delta'])
    bound_arr = np.array(results['bound_c2'])
    valid = ~np.isnan(bound_arr)

    ax = axes[0]
    ax.plot(mu_arr, cov_arr, color=C_COV, marker='o', linewidth=2, label='Empirical coverage')
    if valid.any():
        ax.plot(mu_arr[valid], bound_arr[valid], color=C_C2, marker='s',
                linestyle='--', linewidth=2, label='Theorem 1 bound (c=2)')
    ax.axhline(0.9, color=C_REF, linestyle=':', linewidth=1.2, label='Target 1\u2013\u03b1=0.9')
    ax.set_xlabel('Shift level \u03bc', fontsize=12)
    ax.set_ylabel('Coverage', fontsize=12)
    ax.set_title('Coverage vs Distribution Shift', fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(mu_arr[valid], delta_arr[valid], color=C_DEF, marker='^', linewidth=2)
    ax.set_xlabel('Shift level \u03bc', fontsize=12)
    ax.set_ylabel('Deficiency estimate \u03b4\u0302', fontsize=12)
    ax.set_title('Deficiency Estimate vs Shift\n(class-conditional, exact LP)', fontsize=13)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.scatter(delta_arr[valid], loss_arr[valid], color=C_COV, s=100, zorder=5)
    for i, mu in enumerate(mu_arr[valid]):
        ax.annotate(f'\u03bc={mu:.2f}', (delta_arr[valid][i], loss_arr[valid][i]),
                    textcoords="offset points", xytext=(8, 4), fontsize=8)
    r = results.get('r_pearson', float('nan'))
    ax.set_xlabel('Deficiency estimate \u03b4\u0302', fontsize=12)
    ax.set_ylabel('Coverage loss', fontsize=12)
    ax.set_title(f'Coverage Loss vs \u03b4\u0302\nPearson r = {r:.3f}', fontsize=13)
    ax.grid(True, alpha=0.3)
    criterion_str = "PASS" if r > 0.9 else "FAIL"
    criterion_color = '#2ca02c' if r > 0.9 else '#d62728'
    ax.text(0.05, 0.95, f'r > 0.9: {criterion_str}',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top', color=criterion_color,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    if show:
        plt.show()

    plt.close()
    return fig


# ====================================================================
# Entry point
# ====================================================================

if __name__ == '__main__':
    print("=" * 65)
    print("Validating: Coverage loss vs Le Cam Deficiency estimate")
    print("=" * 65)
    print()
    print("Experiment setup:")
    print("  Calibration: X ~ N(0, I_4), Y = quantile-bucket(X[:,0]) in {0,...,7}")
    print("  Test:        X ~ N(mu, I_4), same label rule")
    print("  Coverage:    measured on binary event Y_binary = 1[X[:,0] > 0]")
    print("  Solver:      exact LP")
    print()

    results = run_experiment(
        alpha=0.10,
        n_cal=500,
        n_test=500,
        n_trials=50,
        d=4,
        n_bins=20,
        n_bootstrap=0,
        mu_grid=[0.0, 0.25, 0.5, 0.75, 1.0],
        verbose=True,
    )

    if results is not None:
        # optional first argument: output PNG path (reproduce.sh uses this)
        save_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
            os.path.dirname(__file__), 'figures', 'validate_drift_deficiency.png'
        )
        plot_results(results, save_path=save_path)
        r = results.get('r_pearson', float('nan'))
        print()
        if r > 0.9:
            print(f"Success criterion MET: Pearson r = {r:.4f} > 0.9")
        else:
            print(f"Success criterion NOT MET: Pearson r = {r:.4f} <= 0.9")

        delta_mu2 = stress_check_mu2(n_trials=30, n_cal=500, n_test=500, d=4, n_bins=20)
        print(f"Stress check (mu=2.0, class-conditional): mean delta = {delta_mu2:.4f}")
        if delta_mu2 > 0.05:
            print("Stress target MET: delta > 0.05 for N(0,1) -> N(2,1).")
        else:
            print("Stress target NOT MET: delta <= 0.05 for N(0,1) -> N(2,1).")
