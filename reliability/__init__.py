"""
ReliabilityLayer
================

A Python SDK for statistically grounded AI/LLM reliability.

Modules
-------
conformal    : Coverage-guaranteed prediction sets and intervals (split conformal, CQR)
calibration  : Probability calibration (ECE, temperature scaling, reliability diagrams)
drift        : Distribution shift detection (MMD, KS) and Le Cam deficiency certificates
report       : End-to-end reliability report generator (HTML output)

Quick start
-----------
from reliability import ReliabilityReport

report = ReliabilityReport(model, X_cal, y_cal, X_test, y_test, alpha=0.10)
report.compute()
report.to_html('reliability_report.html')

Theory
------
This SDK connects two complementary frameworks:

1. Conformal prediction (Vovk 2005; Angelopoulos & Bates 2021):
   Provides finite-sample coverage guarantees P(Y ∈ C_α(X)) ≥ 1−α
   under exchangeability of calibration and test data.

2. Le Cam deficiency (Le Cam 1986): δ(P_cal, P_test) measures how well the
   calibration experiment can simulate the test experiment. The accompanying
   paper proves that coverage loss is bounded by the joint total variation
   TV(P_cal, P_test) with coefficient one, and shows that the class-conditional
   deficiency estimate does NOT bound that TV. The estimate is therefore an
   uncertified shift-severity monitor, not a coverage certificate.

The deficiency solver lives in the shared `defkit` package; reliability._deficiency
is a thin adapter over it. Note that DeficiencyMonitor's default solver returns an
upper bound on delta rather than delta itself, which is the conservative direction
for the coverage certificate above (see reliability._deficiency for the detail).
"""

from .conformal import ConformalClassifier, ConformalRegressor
from .calibration import compute_ece, temperature_scale, reliability_diagram
from .drift import mmd_test, ks_test, DeficiencyMonitor
from .report import ReliabilityReport

__all__ = [
    'ConformalClassifier',
    'ConformalRegressor',
    'compute_ece',
    'temperature_scale',
    'reliability_diagram',
    'mmd_test',
    'ks_test',
    'DeficiencyMonitor',
    'ReliabilityReport',
]

__version__ = '0.1.0'
