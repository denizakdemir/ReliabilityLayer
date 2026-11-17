"""
Diagnose whether the binary label degeneracy is the root cause.

With K=2 classes and a deterministic label rule, the class-conditional
experiment matrix has only 2 columns. The LP can find a near-perfect
Markov kernel trivially because it only needs to match 2 distributions.

Test: vary K (number of classes) and see if deficiency/TV ratio improves.
Also test: what happens with a probabilistic (non-deterministic) label rule.
"""
import numpy as np
from scipy.stats import norm
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from reliability.drift import DeficiencyMonitor
from reliability.conformal import ConformalClassifier

MU_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
N_CAL = N_TEST = 700
D = 4
N_TRIALS = 30
N_BINS = 20


def true_joint_tv(mu):
    return 2.0 * norm.cdf(abs(mu) / 2.0) - 1.0


def run(label_mode, n_classes=2):
    """
    label_mode:
      'binary_det'   : Y = 1[X0 > 0]  (original, K=2, deterministic)
      'multiclass_det': Y = digitize(X0, K quantiles)  (K classes, deterministic)
      'binary_prob'  : Y ~ Bernoulli(sigmoid(X0))  (K=2, probabilistic)
    """
    print(f"\nlabel_mode={label_mode}  n_classes={n_classes}  n_bins={N_BINS}")
    print(f"{'mu':>5}  {'true_TV':>9}  {'hist_def':>10}  {'def/TV':>8}  {'c2_viol%':>9}")
    print("-" * 50)

    # Fit quantile edges for multiclass on a large sample
    if label_mode == 'multiclass_det':
        rng0 = np.random.default_rng(999)
        X0_ref = rng0.standard_normal(50_000)
        edges = np.quantile(X0_ref, np.linspace(0, 1, n_classes + 1))
        edges = np.unique(edges)

    for mu in MU_GRID:
        true_tv = true_joint_tv(mu)
        defs, viols, valid = [], 0, 0

        for seed in range(N_TRIALS):
            rng = np.random.default_rng(seed)
            X_cal = rng.standard_normal((N_CAL, D))
            X_test = rng.standard_normal((N_TEST, D)) + mu

            if label_mode == 'binary_det':
                y_cal = (X_cal[:, 0] > 0).astype(int)
                y_test = (X_test[:, 0] > 0).astype(int)
            elif label_mode == 'multiclass_det':
                y_cal = np.digitize(X_cal[:, 0], edges[1:-1])
                y_test = np.digitize(X_test[:, 0], edges[1:-1])
            elif label_mode == 'binary_prob':
                p_cal = 1.0 / (1.0 + np.exp(-X_cal[:, 0]))
                y_cal = rng.binomial(1, p_cal).astype(int)
                p_test = 1.0 / (1.0 + np.exp(-X_test[:, 0]))
                y_test = rng.binomial(1, p_test).astype(int)

            mon = DeficiencyMonitor(n_bins=N_BINS, solver='exact',
                                    n_bootstrap=0, method='class_conditional')
            mon.fit(X_cal, y_ref=y_cal)
            est = mon.estimate(X_test, y_new=y_test, alpha=0.10)
            d_hat = est.get('delta')
            if d_hat is None or not np.isfinite(d_hat):
                continue

            # coverage with sigmoid classifier
            class _M:
                def predict_proba(self, X):
                    z = -X[:, 0]; p = 1.0 / (1.0 + np.exp(-z))
                    return np.column_stack([1.0 - p, p])

            clf = ConformalClassifier(_M(), alpha=0.10)
            clf.calibrate(X_cal, (X_cal[:, 0] > 0).astype(int))
            cov = float(clf.empirical_coverage(X_test, (X_test[:, 0] > 0).astype(int)))

            defs.append(d_hat)
            if cov < 0.9 - 2.0 * d_hat:
                viols += 1
            valid += 1

        md = float(np.mean(defs)) if defs else float('nan')
        ratio = md / true_tv if true_tv > 1e-9 else float('nan')
        vrate = viols / valid if valid else float('nan')
        print(f"{mu:>5.2f}  {true_tv:>9.4f}  {md:>10.4f}  {ratio:>8.4f}  {vrate:>9.1%}")


if __name__ == '__main__':
    print("=" * 60)
    print("Root cause: binary label degeneracy vs multiclass")
    print("=" * 60)
    run('binary_det', n_classes=2)
    run('multiclass_det', n_classes=4)
    run('multiclass_det', n_classes=8)
    run('multiclass_det', n_classes=16)
    run('binary_prob', n_classes=2)
