"""Deficiency is dominated by the column-wise TV, so it cannot certify TV.

Facts used by the paper's reframing (2026-10-28):

1. delta(P, Q) <= max_theta TV(P_theta, Q_theta)    (take K = identity)
2. If the columns of P have disjoint support, delta(P, Q) = 0 for every Q,
   even when the joint total variation is large.

(2) is the deterministic-label case: the class-conditional experiment of X
given Y is exactly equivalent before and after a covariate shift, while
coverage can drop by an amount up to the joint TV. A deficiency estimate
therefore carries no coverage guarantee on its own.
"""
import numpy as np
import pytest

defkit = pytest.importorskip("defkit")
from defkit import deficiency_exact, tv_discrete  # noqa: E402


def _random_experiment(rng, m, t):
    A = rng.random((m, t)) + 1e-3
    return A / A.sum(axis=0, keepdims=True)


@pytest.mark.parametrize("seed", range(5))
def test_deficiency_never_exceeds_identity_kernel_tv(seed):
    rng = np.random.default_rng(seed)
    P = _random_experiment(rng, 6, 3)
    Q = _random_experiment(rng, 6, 3)
    max_tv = max(tv_discrete(P[:, k], Q[:, k]) for k in range(P.shape[1]))
    assert deficiency_exact(P, Q) <= max_tv + 1e-7


def test_disjoint_support_columns_have_zero_deficiency_despite_large_tv():
    # Two classes with disjoint supports in cal; test shifts mass within each.
    P = np.array([[0.5, 0.0], [0.5, 0.0], [0.0, 0.5], [0.0, 0.5]])
    Q = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    max_tv = max(tv_discrete(P[:, k], Q[:, k]) for k in range(2))
    assert max_tv == pytest.approx(0.5)
    assert deficiency_exact(P, Q) == pytest.approx(0.0, abs=1e-7)
