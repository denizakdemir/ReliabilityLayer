"""
Diagnose the estimator gap:
- true joint TV(P_cal, P_test)
- histogram TV (sum of column-wise TV between P and Q matrices)
- histogram deficiency (exact LP)
at each shift level, to understand what the class-conditional estimator
is actually measuring vs what the theorem needs.
"""
import numpy as np
from scipy.stats import norm
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from reliability.drift import DeficiencyMonitor

MU_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
N_CAL = N_TEST = 700
D = 4
N_TRIALS = 30
N_BINS_LIST = [12, 30, 50, 100]


def true_joint_tv(mu):
    # Joint TV = TV(N(0,I_d), N(mu*e1, I_d)) because label is deterministic given X.
    # For shift only along e1: TV = 2*Phi(|mu|/2) - 1
    return 2.0 * norm.cdf(abs(mu) / 2.0) - 1.0


def run_sweep(n_bins):
    print(f"\nn_bins={n_bins}")
    print(f"{'mu':>5}  {'true_TV':>9}  {'hist_def_exact':>15}  {'def/TV':>8}  {'c2_viol%':>9}")
    print("-" * 55)
    for mu in MU_GRID:
        true_tv = true_joint_tv(mu)
        defs, viols, valid = [], 0, 0
        for seed in range(N_TRIALS):
            rng = np.random.default_rng(seed)
            X_cal = rng.standard_normal((N_CAL, D))
            y_cal = (X_cal[:, 0] > 0).astype(int)
            X_test = rng.standard_normal((N_TEST, D)) + mu
            y_test = (X_test[:, 0] > 0).astype(int)

            # coverage
            from reliability.conformal import ConformalClassifier
            class _M:
                def predict_proba(self, X):
                    z = -X[:, 0]; p = 1.0 / (1.0 + np.exp(-z))
                    return np.column_stack([1.0 - p, p])
            clf = ConformalClassifier(_M(), alpha=0.10)
            clf.calibrate(X_cal, y_cal)
            cov = float(clf.empirical_coverage(X_test, y_test))

            mon = DeficiencyMonitor(n_bins=n_bins, solver='exact',
                                    n_bootstrap=0, method='class_conditional')
            mon.fit(X_cal, y_ref=y_cal)
            est = mon.estimate(X_test, y_new=y_test, alpha=0.10)
            d_hat = est.get('delta')
            if d_hat is None or not np.isfinite(d_hat):
                continue
            defs.append(d_hat)
            if cov < 0.9 - 2.0 * d_hat:
                viols += 1
            valid += 1

        md = float(np.mean(defs)) if defs else float('nan')
        ratio = md / true_tv if true_tv > 1e-9 else float('nan')
        vrate = viols / valid if valid else float('nan')
        print(f"{mu:>5.2f}  {true_tv:>9.4f}  {md:>15.4f}  {ratio:>8.4f}  {vrate:>9.1%}")


if __name__ == '__main__':
    print("Diagnosing estimator gap: histogram deficiency vs true joint TV")
    print("=" * 60)
    for nb in N_BINS_LIST:
        run_sweep(nb)
