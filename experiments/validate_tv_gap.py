"""
Ground-truth check of the deficiency estimate against the true joint TV.

Setting: X_cal ~ N(0, I_d), X_test ~ N(mu * e_1, I_d), Y = 1[X_1 > 0].
Because Y is a function of X, the joint TV equals the TV of the covariates:
    TV(P_cal, P_test) = 2 * Phi(|mu| / 2) - 1   (shift along one axis).

For each (n_bins, mu) this script reports the empirical coverage loss of a
split conformal classifier, the exact-LP class-conditional deficiency
estimate, the true TV, and how often the claimed conservative bound
coverage >= (1 - alpha) - 2 * delta_hat is violated.

It exists because the paper's theorem needs TV(P_cal, P_test) <= 2 * delta
and that hypothesis is checkable here. Results go to
experiments/results/tv_gap.csv.

Usage:
    python experiments/validate_tv_gap.py [--n-trials 30] [--out PATH]
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reliability.conformal import ConformalClassifier  # noqa: E402
from reliability.drift import DeficiencyMonitor  # noqa: E402

MU_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
N_CAL = N_TEST = 700
D = 4
ALPHA = 0.10


class _Model:
    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(X[:, 0]))
        return np.column_stack([1.0 - p, p])


def true_joint_tv(mu):
    return 2.0 * norm.cdf(abs(mu) / 2.0) - 1.0


def run(n_bins, n_trials):
    rows = []
    for mu in MU_GRID:
        tv = true_joint_tv(mu)
        loss, delta, hyp_ok, bound_viol = [], [], 0, 0
        for seed in range(n_trials):
            rng = np.random.default_rng(seed)
            X_cal = rng.standard_normal((N_CAL, D))
            y_cal = (X_cal[:, 0] > 0).astype(int)
            X_test = rng.standard_normal((N_TEST, D))
            X_test[:, 0] += mu
            y_test = (X_test[:, 0] > 0).astype(int)

            clf = ConformalClassifier(_Model(), alpha=ALPHA)
            clf.calibrate(X_cal, y_cal)
            cov = float(clf.empirical_coverage(X_test, y_test))

            mon = DeficiencyMonitor(
                n_bins=n_bins, solver="exact", n_bootstrap=0,
                method="class_conditional",
            )
            mon.fit(X_cal, y_ref=y_cal)
            d_hat = mon.estimate(X_test, y_new=y_test, alpha=ALPHA).get("delta")
            if d_hat is None or not np.isfinite(d_hat):
                continue
            loss.append((1.0 - ALPHA) - cov)
            delta.append(d_hat)
            hyp_ok += int(tv <= 2.0 * d_hat)
            bound_viol += int(cov < (1.0 - ALPHA) - 2.0 * d_hat)
        n = len(delta)
        rows.append({
            "n_bins": n_bins, "mu": mu, "n_valid": n,
            "true_tv": tv,
            "mean_coverage_loss": float(np.mean(loss)),
            "mean_delta_hat": float(np.mean(delta)),
            "delta_over_tv": float(np.mean(delta) / tv) if tv > 1e-9 else float("nan"),
            "hypothesis_tv_le_2delta_rate": hyp_ok / n,
            "c2_bound_violation_rate": bound_viol / n,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--n-bins", type=int, nargs="+", default=[12, 50])
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "results", "tv_gap.csv"),
    )
    args = ap.parse_args()

    rows = []
    for nb in args.n_bins:
        print(f"n_bins={nb}", flush=True)
        part = run(nb, args.n_trials)
        for r in part:
            print("  mu={mu:4.2f} TV={true_tv:.4f} delta_hat={mean_delta_hat:.4f} "
                  "hyp_holds={hypothesis_tv_le_2delta_rate:.0%} "
                  "c2_viol={c2_bound_violation_rate:.0%}".format(**r), flush=True)
        rows.extend(part)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
