"""
Numerical validation of Theorem 1 (Paper 5).

This script compares empirical conformal coverage against candidate lower bounds
under distribution shift using class-conditional deficiency estimation:

  bound(c=1):      (1 - alpha) - delta_hat
  bound(c=2):      (1 - alpha) - 2 * delta_hat
  bound(c=2,cert): (1 - alpha) - 2 * delta_cert
  bound(op):       (1 - alpha) - c_op * delta_hat
  bound(op,cert):  (1 - alpha) - c_op * delta_cert

where delta_cert is either:
  - delta_hat (certificate='mean'), or
  - ci_upper  (certificate='ci_upper') when bootstrap CIs are available.
Here c_op is a fixed operational multiplier supplied to DeficiencyMonitor.
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reliability.conformal import ConformalClassifier
from reliability.drift import DeficiencyMonitor


# Quantile edges for 8-class label rule, fitted once on a large reference sample.
# Using a fixed seed ensures reproducibility across all experiment calls.
_LABEL_EDGES: np.ndarray = np.quantile(
    np.random.default_rng(0).standard_normal(200_000),
    np.linspace(0.0, 1.0, 9),
)
_N_LABEL_CLASSES: int = 8


def _label_from_x0(x0: np.ndarray) -> np.ndarray:
    """Map X[:,0] to one of 8 quantile-based label classes."""
    return np.digitize(x0, _LABEL_EDGES[1:-1]).astype(int)


def generate_calibration_data(n: int, d: int, rng: np.random.Generator):
    """Calibration split: X ~ N(0, I), y = quantile-bucket(X0) in {0,...,7}."""
    X = rng.standard_normal((n, d))
    y = _label_from_x0(X[:, 0])
    return X, y


def generate_test_data(n: int, d: int, mu: float, rng: np.random.Generator):
    """Test split: homogeneous covariate shift X ~ N(mu, I), same label rule."""
    X = rng.standard_normal((n, d)) + mu
    y = _label_from_x0(X[:, 0])
    return X, y


class SigmoidClassifier:
    """Intentional misspecification: P(Y=1|X) = sigmoid(-X0).

    The conformal predictor wraps this binary classifier; coverage is measured
    on the binary event {Y_binary in C_alpha(X)} where Y_binary = 1[X0 > 0].
    The deficiency estimator uses the 8-class label to build a richer
    experiment matrix with sufficient column diversity for the LP.
    """

    def predict_proba(self, X):
        z = -X[:, 0]
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p, p])


def _finite_mean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float('nan')
    return float(np.mean(arr))


def _pearson_safe(x: np.ndarray, y: np.ndarray):
    from scipy.stats import pearsonr

    if x.size < 3 or y.size < 3:
        return float('nan'), float('nan')
    if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return float('nan'), float('nan')
    r, p = pearsonr(x, y)
    return float(r), float(p)


def _certificate_delta(delta: float, ci_upper: Optional[float], certificate: str) -> float:
    if certificate == 'ci_upper' and ci_upper is not None and np.isfinite(ci_upper):
        return float(ci_upper)
    return float(delta)


def run_experiment(
    alpha: float,
    n_cal: int,
    n_test: int,
    n_trials: int,
    d: int,
    n_bins: int,
    n_bootstrap: int,
    mu_grid: List[float],
    certificate: str,
    bound_multiplier: float,
    strict_class_conditional: bool = True,
    verbose: bool = True,
) -> Optional[Dict[str, object]]:
    """Run theorem-validation experiment across a shift grid."""
    if certificate not in ('mean', 'ci_upper'):
        raise ValueError("certificate must be 'mean' or 'ci_upper'.")

    model = SigmoidClassifier()
    target = 1.0 - alpha

    results = {
        'mu': [],
        'coverage': [],
        'coverage_loss': [],
        'delta': [],
        'delta_cert': [],
        'bound_c1': [],
        'bound_c2': [],
        'bound_c2_cert': [],
        'bound_op': [],
        'bound_op_cert': [],
        'c_multiplier': [],
        'gap_c1': [],
        'gap_c2': [],
        'gap_c2_cert': [],
        'gap_op': [],
        'gap_op_cert': [],
        'violations_c1': [],
        'violations_c2': [],
        'violations_c2_cert': [],
        'violations_op': [],
        'violations_op_cert': [],
        'valid_trials': [],
        'total_trials': int(len(mu_grid) * n_trials),
        'total_valid_trials': 0,
        'total_violations_c1': 0,
        'total_violations_c2': 0,
        'total_violations_c2_cert': 0,
        'total_violations_op': 0,
        'total_violations_op_cert': 0,
        'alpha': float(alpha),
        'n_cal': int(n_cal),
        'n_test': int(n_test),
        'n_trials': int(n_trials),
        'd': int(d),
        'n_bins': int(n_bins),
        'n_bootstrap': int(n_bootstrap),
        'certificate': certificate,
        'bound_multiplier': float(bound_multiplier),
    }

    if verbose:
        print(
            f"{'mu':>5}  {'coverage':>10}  {'loss':>8}  {'delta':>9}  {'d_cert':>9}  "
            f"{'b(c=1)':>9}  {'b(c=2)':>9}  {'b(op)':>9}  {'b(op,cert)':>11}  "
            f"{'v1':>4}  {'v2':>4}  {'vo':>4}  {'voc':>4}  {'valid':>6}"
        )
        print("-" * 148)

    for mu in mu_grid:
        coverages = []
        deltas = []
        deltas_cert = []
        violations_c1 = 0
        violations_c2 = 0
        violations_c2_cert = 0
        violations_op = 0
        violations_op_cert = 0
        valid_trials = 0
        c_multipliers = []

        for seed in range(n_trials):
            rng = np.random.default_rng(seed)
            X_cal, y_cal = generate_calibration_data(n_cal, d, rng)
            X_test, y_test = generate_test_data(n_test, d, mu, rng)

            # Coverage is measured on the binary label (the conformal task).
            y_cal_binary = (X_cal[:, 0] > 0.0).astype(int)
            y_test_binary = (X_test[:, 0] > 0.0).astype(int)
            clf = ConformalClassifier(model, alpha=alpha)
            clf.calibrate(X_cal, y_cal_binary)
            coverage = float(clf.empirical_coverage(X_test, y_test_binary))

            monitor = DeficiencyMonitor(
                n_bins=n_bins,
                solver='exact',
                n_bootstrap=n_bootstrap,
                method='class_conditional',
                bound_multiplier=bound_multiplier,
            )
            monitor.fit(X_cal, y_ref=y_cal)
            estimate = monitor.estimate(X_test, y_new=y_test, alpha=alpha)

            method_used = estimate.get('method_used')
            fallback_reason = estimate.get('fallback_reason')
            delta = estimate.get('delta')
            ci_upper = estimate.get('ci_upper')
            c_multiplier = estimate.get('c_multiplier')

            if strict_class_conditional and (method_used != 'class_conditional' or fallback_reason is not None):
                raise RuntimeError(
                    "Expected class_conditional deficiency estimation, "
                    f"got method_used={method_used}, fallback_reason={fallback_reason}."
                )

            if delta is None or not np.isfinite(delta):
                continue

            delta = float(delta)
            delta_cert = _certificate_delta(delta, ci_upper, certificate=certificate)
            c_multiplier = float(c_multiplier) if c_multiplier is not None else float('nan')

            coverages.append(coverage)
            deltas.append(delta)
            deltas_cert.append(delta_cert)
            c_multipliers.append(c_multiplier)
            valid_trials += 1

            bound_c1_trial = target - delta
            bound_c2_trial = target - 2.0 * delta
            bound_c2_cert_trial = target - 2.0 * delta_cert
            bound_op_trial = target - c_multiplier * delta
            bound_op_cert_trial = target - c_multiplier * delta_cert

            if coverage < bound_c1_trial:
                violations_c1 += 1
            if coverage < bound_c2_trial:
                violations_c2 += 1
            if coverage < bound_c2_cert_trial:
                violations_c2_cert += 1
            if coverage < bound_op_trial:
                violations_op += 1
            if coverage < bound_op_cert_trial:
                violations_op_cert += 1

        mean_cov = _finite_mean(coverages)
        mean_delta = _finite_mean(deltas)
        mean_delta_cert = _finite_mean(deltas_cert)
        mean_c_multiplier = _finite_mean(c_multipliers)

        if np.isfinite(mean_cov):
            mean_loss = float(max(0.0, target - mean_cov))
        else:
            mean_loss = float('nan')

        if np.isfinite(mean_delta):
            bound_c1 = float(target - mean_delta)
            bound_c2 = float(target - 2.0 * mean_delta)
        else:
            bound_c1 = float('nan')
            bound_c2 = float('nan')

        if np.isfinite(mean_delta_cert):
            bound_c2_cert = float(target - 2.0 * mean_delta_cert)
        else:
            bound_c2_cert = float('nan')

        if np.isfinite(mean_delta) and np.isfinite(mean_c_multiplier):
            bound_op = float(target - mean_c_multiplier * mean_delta)
        else:
            bound_op = float('nan')

        if np.isfinite(mean_delta_cert) and np.isfinite(mean_c_multiplier):
            bound_op_cert = float(target - mean_c_multiplier * mean_delta_cert)
        else:
            bound_op_cert = float('nan')

        gap_c1 = float(mean_cov - bound_c1) if np.isfinite(mean_cov) and np.isfinite(bound_c1) else float('nan')
        gap_c2 = float(mean_cov - bound_c2) if np.isfinite(mean_cov) and np.isfinite(bound_c2) else float('nan')
        gap_c2_cert = float(mean_cov - bound_c2_cert) if np.isfinite(mean_cov) and np.isfinite(bound_c2_cert) else float('nan')
        gap_op = float(mean_cov - bound_op) if np.isfinite(mean_cov) and np.isfinite(bound_op) else float('nan')
        gap_op_cert = float(mean_cov - bound_op_cert) if np.isfinite(mean_cov) and np.isfinite(bound_op_cert) else float('nan')

        results['mu'].append(float(mu))
        results['coverage'].append(mean_cov)
        results['coverage_loss'].append(mean_loss)
        results['delta'].append(mean_delta)
        results['delta_cert'].append(mean_delta_cert)
        results['c_multiplier'].append(mean_c_multiplier)
        results['bound_c1'].append(bound_c1)
        results['bound_c2'].append(bound_c2)
        results['bound_c2_cert'].append(bound_c2_cert)
        results['bound_op'].append(bound_op)
        results['bound_op_cert'].append(bound_op_cert)
        results['gap_c1'].append(gap_c1)
        results['gap_c2'].append(gap_c2)
        results['gap_c2_cert'].append(gap_c2_cert)
        results['gap_op'].append(gap_op)
        results['gap_op_cert'].append(gap_op_cert)
        results['violations_c1'].append(int(violations_c1))
        results['violations_c2'].append(int(violations_c2))
        results['violations_c2_cert'].append(int(violations_c2_cert))
        results['violations_op'].append(int(violations_op))
        results['violations_op_cert'].append(int(violations_op_cert))
        results['valid_trials'].append(int(valid_trials))

        results['total_valid_trials'] += int(valid_trials)
        results['total_violations_c1'] += int(violations_c1)
        results['total_violations_c2'] += int(violations_c2)
        results['total_violations_c2_cert'] += int(violations_c2_cert)
        results['total_violations_op'] += int(violations_op)
        results['total_violations_op_cert'] += int(violations_op_cert)

        if verbose:
            print(
                f"{mu:>5.2f}  {mean_cov:>10.4f}  {mean_loss:>8.4f}  {mean_delta:>9.4f}  {mean_delta_cert:>9.4f}  "
                f"{bound_c1:>9.4f}  {bound_c2:>9.4f}  {bound_op:>9.4f}  {bound_op_cert:>11.4f}  "
                f"{violations_c1:>4d}  {violations_c2:>4d}  {violations_op:>4d}  {violations_op_cert:>4d}  {valid_trials:>6d}"
            )

    loss_arr = np.asarray(results['coverage_loss'], dtype=float)
    delta_arr = np.asarray(results['delta'], dtype=float)
    valid_mu = np.isfinite(loss_arr) & np.isfinite(delta_arr)

    r, p = _pearson_safe(loss_arr[valid_mu], delta_arr[valid_mu])
    results['r_pearson'] = r
    results['p_pearson'] = p

    total_valid = int(results['total_valid_trials'])
    if total_valid > 0:
        results['violation_rate_c1'] = float(results['total_violations_c1']) / float(total_valid)
        results['violation_rate_c2'] = float(results['total_violations_c2']) / float(total_valid)
        results['violation_rate_c2_cert'] = float(results['total_violations_c2_cert']) / float(total_valid)
        results['violation_rate_op'] = float(results['total_violations_op']) / float(total_valid)
        results['violation_rate_op_cert'] = float(results['total_violations_op_cert']) / float(total_valid)
    else:
        results['violation_rate_c1'] = float('nan')
        results['violation_rate_c2'] = float('nan')
        results['violation_rate_c2_cert'] = float('nan')
        results['violation_rate_op'] = float('nan')
        results['violation_rate_op_cert'] = float('nan')

    if verbose:
        print()
        print("Summary:")
        print(f"  Pearson r(coverage_loss, delta) = {r:.4f} (p={p:.4f})")
        print(
            "  c=1 violations: "
            f"{results['total_violations_c1']}/{total_valid} "
            f"({results['violation_rate_c1']:.2%} of valid trials)"
        )
        print(
            "  c=2 violations: "
            f"{results['total_violations_c2']}/{total_valid} "
            f"({results['violation_rate_c2']:.2%} of valid trials)"
        )
        label = "c=2(cert=ci_upper)" if certificate == 'ci_upper' else "c=2(cert=mean)"
        print(
            f"  {label} violations: "
            f"{results['total_violations_c2_cert']}/{total_valid} "
            f"({results['violation_rate_c2_cert']:.2%} of valid trials)"
        )
        print(
            "  operational violations: "
            f"{results['total_violations_op']}/{total_valid} "
            f"({results['violation_rate_op']:.2%} of valid trials)"
        )
        op_label = "operational(cert=ci_upper)" if certificate == 'ci_upper' else "operational(cert=mean)"
        print(
            f"  {op_label} violations: "
            f"{results['total_violations_op_cert']}/{total_valid} "
            f"({results['violation_rate_op_cert']:.2%} of valid trials)"
        )

    return results


def plot_results(results: Dict[str, object], save_path: Optional[str] = None, show: bool = False):
    """Create a compact manuscript-oriented figure."""
    import matplotlib

    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Unified color palette (consistent across all paper figures)
    C_COV   = '#1f77b4'  # blue   — empirical coverage
    C_C1    = '#d62728'  # red    — bound c=1 (tighter, more violations)
    C_C2    = '#ff7f0e'  # orange — bound c=2
    C_CERT  = '#8B1A1A'  # dark red — certified bound (CI upper)
    C_REF   = '#888888'  # gray   — target reference line

    mu = np.asarray(results['mu'], dtype=float)
    coverage = np.asarray(results['coverage'], dtype=float)
    loss = np.asarray(results['coverage_loss'], dtype=float)
    delta = np.asarray(results['delta'], dtype=float)
    bound_c1 = np.asarray(results['bound_c1'], dtype=float)
    bound_c2 = np.asarray(results['bound_c2'], dtype=float)
    bound_c2_cert = np.asarray(results['bound_c2_cert'], dtype=float)
    bound_op = np.asarray(results['bound_op'], dtype=float)
    bound_op_cert = np.asarray(results['bound_op_cert'], dtype=float)

    valid_bounds = np.isfinite(bound_c1) & np.isfinite(bound_c2)
    valid_cert = np.isfinite(bound_c2_cert)
    valid_scatter = np.isfinite(loss) & np.isfinite(delta)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # Panel 1: Coverage and candidate bounds
    ax = axes[0]
    ax.plot(mu, coverage, color=C_COV, marker='o', linewidth=2, label='Empirical coverage')
    if valid_bounds.any():
        ax.plot(mu[valid_bounds], bound_c1[valid_bounds], color=C_C1, marker='^',
                linestyle='--', linewidth=1.7, label='Bound c=1')
        ax.plot(mu[valid_bounds], bound_c2[valid_bounds], color=C_C2, marker='s',
                linestyle='--', linewidth=1.7, label='Bound c=2')
    valid_op = np.isfinite(bound_op)
    if valid_op.any():
        ax.plot(mu[valid_op], bound_op[valid_op], color='#2ca02c', marker='P',
                linestyle='-.', linewidth=1.7, label='Operational bound')
    # Only plot cert line when it differs from c=2 bound (i.e., CI-upper mode)
    cert_differs = (
        valid_cert.any()
        and valid_bounds.any()
        and not np.allclose(
            bound_c2_cert[valid_cert & valid_bounds],
            bound_c2[valid_cert & valid_bounds],
            atol=1e-8,
        )
    )
    if cert_differs:
        cert_label = 'Bound c=2 (CI cert)'
        ax.plot(mu[valid_cert], bound_c2_cert[valid_cert], color=C_CERT, marker='d',
                linestyle='--', linewidth=1.6, label=cert_label)
    valid_op_cert = np.isfinite(bound_op_cert)
    op_cert_differs = (
        valid_op_cert.any()
        and valid_op.any()
        and not np.allclose(
            bound_op_cert[valid_op_cert & valid_op],
            bound_op[valid_op_cert & valid_op],
            atol=1e-8,
        )
    )
    if op_cert_differs:
        ax.plot(mu[valid_op_cert], bound_op_cert[valid_op_cert], color='#006d2c', marker='X',
                linestyle=':', linewidth=1.5, label='Operational (CI cert)')
    ax.axhline(1.0 - float(results['alpha']), color=C_REF, linestyle=':', linewidth=1.2, label='Target 1\u2013\u03b1')
    ax.set_xlabel('Shift level \u03bc', fontsize=12)
    ax.set_ylabel('Coverage', fontsize=12)
    ax.set_title('Coverage vs Candidate Lower Bounds', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 2: Coverage loss vs delta with slope references
    ax = axes[1]
    if valid_scatter.any():
        ax.scatter(delta[valid_scatter], loss[valid_scatter], color=C_COV, s=90, zorder=5)
        for i, mv in enumerate(mu[valid_scatter]):
            ax.annotate(f'\u03bc={mv:.2f}', (delta[valid_scatter][i], loss[valid_scatter][i]),
                        textcoords='offset points', xytext=(6, 3), fontsize=8)
        max_delta = float(np.max(delta[valid_scatter]))
        grid = np.linspace(0.0, max(1e-8, max_delta * 1.05), 80)
        ax.plot(grid, grid, color=C_C1, linestyle='--', linewidth=1.5, label='loss = \u03b4 (c=1)')
        ax.plot(grid, 2.0 * grid, color=C_C2, linestyle='--', linewidth=1.5, label='loss = 2\u03b4 (c=2)')
    r = float(results.get('r_pearson', float('nan')))
    ax.set_xlabel('Deficiency estimate \u03b4\u0302', fontsize=12)
    ax.set_ylabel('Coverage loss', fontsize=12)
    ax.set_title(f'Coverage Loss vs Deficiency Estimate\nPearson r = {r:.3f}', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    if show:
        plt.show()

    plt.close(fig)
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description='Validate Theorem 1 bounds under synthetic shift.')
    parser.add_argument('--mode', choices=['fast', 'final', 'cert'], default='fast')
    parser.add_argument('--alpha', type=float, default=0.10)
    parser.add_argument('--n-cal', type=int, default=None)
    parser.add_argument('--n-test', type=int, default=None)
    parser.add_argument('--n-trials', type=int, default=None)
    parser.add_argument('--d', type=int, default=None)
    parser.add_argument('--n-bins', type=int, default=None)
    parser.add_argument('--n-bootstrap', type=int, default=None)
    parser.add_argument('--mu-grid', type=float, nargs='*', default=None)
    parser.add_argument('--certificate', choices=['mean', 'ci_upper'], default=None)
    parser.add_argument('--bound-multiplier', type=float, default=2.0)
    parser.add_argument('--save-path', type=str, default=None)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    return parser.parse_args()


def _mode_defaults(mode: str) -> Dict[str, object]:
    if mode == 'final':
        return {
            'n_cal': 700,
            'n_test': 700,
            'n_trials': 80,
            'd': 4,
            'n_bins': 20,
            'n_bootstrap': 0,
            'mu_grid': [0.0, 0.25, 0.5, 0.75, 1.0],
            'certificate': 'mean',
        }

    if mode == 'cert':
        return {
            'n_cal': 500,
            'n_test': 500,
            'n_trials': 30,
            'd': 4,
            'n_bins': 20,
            'n_bootstrap': 30,
            'mu_grid': [0.0, 0.25, 0.5, 0.75, 1.0],
            'certificate': 'ci_upper',
        }

    # fast mode (default)
    return {
        'n_cal': 500,
        'n_test': 500,
        'n_trials': 30,
        'd': 4,
        'n_bins': 20,
        'n_bootstrap': 0,
        'mu_grid': [0.0, 0.25, 0.5, 0.75, 1.0],
        'certificate': 'mean',
    }


def main():
    args = parse_args()
    defaults = _mode_defaults(args.mode)

    n_cal = int(args.n_cal) if args.n_cal is not None else int(defaults['n_cal'])
    n_test = int(args.n_test) if args.n_test is not None else int(defaults['n_test'])
    n_trials = int(args.n_trials) if args.n_trials is not None else int(defaults['n_trials'])
    d = int(args.d) if args.d is not None else int(defaults['d'])
    n_bins = int(args.n_bins) if args.n_bins is not None else int(defaults['n_bins'])
    n_bootstrap = int(args.n_bootstrap) if args.n_bootstrap is not None else int(defaults['n_bootstrap'])
    mu_grid = args.mu_grid if args.mu_grid is not None and len(args.mu_grid) > 0 else list(defaults['mu_grid'])
    certificate = args.certificate if args.certificate is not None else str(defaults['certificate'])

    save_path = args.save_path
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'figures', 'validate_theorem1.png')

    if not args.quiet:
        print("=" * 84)
        print("Validating Theorem 1 Candidate Bounds (class-conditional deficiency estimator)")
        print("=" * 84)
        print(f"mode={args.mode} | alpha={args.alpha} | n_cal={n_cal} | n_test={n_test} | n_trials={n_trials}")
        print(f"d={d} | n_bins={n_bins} | n_bootstrap={n_bootstrap} | mu_grid={mu_grid}")
        print(f"certificate={certificate} | strict_class_conditional=True")
        print(f"bound_multiplier={args.bound_multiplier}")
        print()

    results = run_experiment(
        alpha=args.alpha,
        n_cal=n_cal,
        n_test=n_test,
        n_trials=n_trials,
        d=d,
        n_bins=n_bins,
        n_bootstrap=n_bootstrap,
        mu_grid=mu_grid,
        certificate=certificate,
        bound_multiplier=args.bound_multiplier,
        strict_class_conditional=True,
        verbose=not args.quiet,
    )

    if results is None:
        return

    plot_results(results, save_path=save_path, show=args.show)

    print()
    print("MANUSCRIPT_SUMMARY")
    print(
        " ".join(
            [
                f"mode={args.mode}",
                f"certificate={certificate}",
                f"alpha={results['alpha']:.3f}",
                f"total_valid_trials={results['total_valid_trials']}",
                f"c1_violations={results['total_violations_c1']}",
                f"c2_violations={results['total_violations_c2']}",
                f"c2_cert_violations={results['total_violations_c2_cert']}",
                f"op_violations={results['total_violations_op']}",
                f"op_cert_violations={results['total_violations_op_cert']}",
                f"c1_violation_rate={results['violation_rate_c1']:.6f}",
                f"c2_violation_rate={results['violation_rate_c2']:.6f}",
                f"c2_cert_violation_rate={results['violation_rate_c2_cert']:.6f}",
                f"op_violation_rate={results['violation_rate_op']:.6f}",
                f"op_cert_violation_rate={results['violation_rate_op_cert']:.6f}",
                f"bound_multiplier={results['bound_multiplier']:.6f}",
                f"pearson_r={results['r_pearson']:.6f}",
                f"pearson_p={results['p_pearson']:.6f}",
            ]
        )
    )


if __name__ == '__main__':
    main()
