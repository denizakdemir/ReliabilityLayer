"""
reliability._deficiency
=======================

Deficiency computation for :class:`reliability.drift.DeficiencyMonitor`.

This is an internal module — import via reliability.drift.DeficiencyMonitor,
not directly.

This module used to carry its own copy of the LP and projected-gradient
solvers. That copy was one of four in the corpus: byte-identical in its LP to
the copy in ``robustea``, and mathematically identical to the transposed one
in ``transfert``. They now all delegate to ``defkit``, so there is a single
implementation to test and a single place a fix has to land.

The public signature here is unchanged, including the ``None`` return on a
failed solve, so ``drift.py`` needs no adjustment.

Theory
------
δ(P, Q) = min_{K column-stochastic} max_θ TV(KP_θ, Q_θ)

where P, Q are experiment matrices (dim × n_thetas) whose columns are
probability distributions over their respective sample spaces.

A note on the default solver
----------------------------
``solver="auto"`` resolves to :func:`defkit.deficiency_upper_bound`, which is
what this package has always computed — a fast upper bound, not the exact
deficiency. The value can sit materially above the true δ (up to ~0.1 absolute
on dense problems).

That is the safe direction *here specifically*: ``DeficiencyMonitor`` reports
``coverage >= (1 - alpha) - c * delta``, so an over-estimated δ lowers the
certified coverage and makes the validity warning fire sooner rather than
later. Keeping it as the default also keeps existing figures reproducible.
Pass ``solver="exact"`` for the true deficiency when the looseness matters
more than the speed.

Reference: Le Cam (1986), "Asymptotic Methods in Statistical Decision Theory".
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from defkit.solvers import (
    deficiency_exact,
    deficiency_upper_bound,
    evaluate_kernel as _eval_deficiency,  # noqa: F401  (re-exported for experiments/)
    project_column_stochastic as _project_stochastic,  # noqa: F401
)

__all__ = ["compute_deficiency"]

_UPPER_BOUND_ALIASES = ("auto", "sinkhorn", "upper_bound", "gradient")
_EXACT_ALIASES = ("exact", "lp", "scipy", "cvxpy")


def compute_deficiency(
    P: np.ndarray,
    Q: np.ndarray,
    solver: str = "auto",
    sinkhorn_reg: float = 0.1,
    sinkhorn_max_iter: int = 100,
    sinkhorn_tol: float = 1e-6,
) -> Optional[float]:
    """
    Compute a deficiency reading for experiments ``P`` and ``Q``.

    Parameters
    ----------
    P : np.ndarray, shape (dim_X, n_thetas)
        Source experiment matrix; each column is a probability distribution P_θ.
    Q : np.ndarray, shape (dim_Y, n_thetas)
        Target experiment matrix; each column is a probability distribution Q_θ.
    solver : str
        ``'auto'`` / ``'sinkhorn'`` return a fast upper bound on δ (this
        package's historical behaviour — see the module docstring for why that
        is the conservative choice for coverage certification).
        ``'exact'`` / ``'lp'`` / ``'scipy'`` / ``'cvxpy'`` solve the linear
        program and return δ itself.
    sinkhorn_reg : float
        Gradient step scale for the upper-bound path. Despite the name no
        entropic regularisation is involved; the name is kept so existing call
        sites keep working.
    sinkhorn_max_iter : int
        Maximum gradient iterations.
    sinkhorn_tol : float
        Convergence tolerance (max element-wise change in K).

    Returns
    -------
    float or None
        The deficiency reading, or ``None`` if the solve failed.
    """
    if solver in _UPPER_BOUND_ALIASES:
        chosen = lambda: deficiency_upper_bound(  # noqa: E731
            P, Q,
            step_scale=sinkhorn_reg,
            max_iter=sinkhorn_max_iter,
            tol=sinkhorn_tol,
        )
    elif solver in _EXACT_ALIASES:
        chosen = lambda: deficiency_exact(P, Q)  # noqa: E731
    else:
        raise ValueError(
            f"unknown solver {solver!r}; expected one of "
            f"{sorted(_UPPER_BOUND_ALIASES + _EXACT_ALIASES)}."
        )

    try:
        return chosen()
    except (RuntimeError, ValueError):
        # Historical contract: a failed solve is reported as None, and the
        # caller in drift.py substitutes NaN.
        return None
