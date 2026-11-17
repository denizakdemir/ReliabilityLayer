# ReliabilityLayer

**A Python SDK for statistically grounded ML reliability: coverage-guaranteed prediction sets, calibration diagnostics, drift detection, and deficiency-based coverage certificates.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-109%20passing-brightgreen.svg)]()

---

## What this project is

ReliabilityLayer is the fifth project in the CorpusLOKAM Le Cam series (companion solver: [defkit](https://github.com/denizakdemir/defkit)), translating statistical decision theory into a working Python library.

**One-sentence pitch:** Wrap any predictive model and get back statistically grounded trust signals — coverage-guaranteed prediction sets, calibrated probabilities, drift alarms, and a deficiency certificate that tells you when the other guarantees are breaking down.

**Theory in one sentence:** Conformal prediction gives finite-sample coverage guarantees when exchangeability holds; Le Cam deficiency measures exactly how much exchangeability has broken — so the two are complementary, not competing.

The guarantee (Paper 5): conformal coverage loss is at most the joint total variation between calibration and test distributions (coefficient 1, unimprovable). The deficiency estimate is an **uncertified monitor**: it is not an upper bound on TV (it is dominated by the class-conditional TV and is zero for classes with disjoint support), it needs test labels, and it can fail to track coverage loss (see `experiments/validate_tv_gap.py` and the 20 Newsgroups benchmark). An earlier "coverage degrades by at most 2ε" claim was withdrawn on 2026-10-28.

---

## Installation

**Not yet on PyPI.** The package depends on `defkit`, which is also unpublished, so install both from source. Python 3.11 is the tested version (numpy 2.4, scipy 1.16, scikit-learn 1.8, pandas 2.3, matplotlib 3.10; see `repro_out/logs/versions.txt`).

```bash
pip install git+https://github.com/denizakdemir/defkit.git
pip install git+https://github.com/denizakdemir/ReliabilityLayer.git
```

Or, for development, clone both and `pip install -e` each.

Then check the install with `pytest tests/ -q` (109 tests).

Optional exact LP solver backend (slower but provably optimal):

```bash
pip install "reliability-layer[exact]"
```

**Dependencies:** numpy, scipy, scikit-learn, matplotlib, jinja2. No external corpus dependencies.

---

## Quick start

```python
from reliability import ReliabilityReport

report = ReliabilityReport(
    model=model,
    X_cal=X_cal, y_cal=y_cal,
    X_test=X_test, y_test=y_test,
    alpha=0.10,
    task='classification'
)
report.compute()
report.to_html('reliability_report.html')
metrics = report.to_dict()
```

See [examples/tabular_classification.ipynb](examples/tabular_classification.ipynb) for a runnable end-to-end walkthrough.

---

## SDK modules

```
reliability/
    conformal.py     — split conformal classification/regression, CQR, RAPS
    calibration.py   — ECE, temperature scaling, reliability diagrams
    drift.py         — MMD, KS tests + DeficiencyMonitor (Le Cam deficiency certificates)
    report.py        — one-page HTML reliability report generator
    _deficiency.py   — thin adapter over the shared `defkit` deficiency solvers
```

### Conformal prediction

```python
from reliability import ConformalClassifier

clf = ConformalClassifier(model, alpha=0.10)
clf.calibrate(X_cal, y_cal)
prediction_sets = clf.predict(X_test)   # list of sets
coverage = clf.empirical_coverage(X_test, y_test)
```

### Calibration

```python
from reliability import compute_ece, temperature_scale

ece = compute_ece(probs, labels)
tau = temperature_scale(logits, labels)   # optimal temperature
```

### Drift and deficiency

```python
from reliability import DeficiencyMonitor
from reliability.drift import mmd_test

# Fast tier: MMD two-sample test
result = mmd_test(X_cal, X_test)
print(result['significant'])   # True if drift detected

# Deficiency certificate.
# solver='exact' uses the sparse LP (HiGHS interior-point); recommended for
# all reported results. The default 'auto' returns a fast upper bound which
# is conservative for the coverage certificate but masks small drift amounts.
monitor = DeficiencyMonitor(n_bins=20, method='class_conditional', solver='exact')
monitor.fit(X_cal, y_ref=y_cal)
estimate = monitor.estimate(X_test, y_new=y_test, alpha=0.10)

# estimate['delta']          — deficiency estimate δ̂
# estimate['coverage_bound'] — certified lower bound on conformal coverage
# estimate['warning']        — True if δ̂ exceeds threshold
```

---

## Paper

**"Coverage Validity Under Distribution Shift: Deficiency as a Shift-Severity Monitor for Conformal Predictors"**

Deniz Akdemir. arXiv: _to be filled in after posting_ (v1 package prepared 2026-11-16; see `arxiv_submission/ARXIV_VERSION_NOTES.txt`).

The manuscript source is `arxiv_submission/main.tex` (build with `latexmk -pdf` in that directory). The independent proof check is `proofs/THEORY_REVIEW_2026-11-16.md`.

**Guarantee (Theorem 1, Corollary 2):** coverage loss ≤ joint TV(P_cal, P_test) (coefficient 1, unimprovable; Proposition 3).

**Deficiency monitor (uncertified).** In a Gaussian-shift study with known true TV, the exact-LP deficiency estimate is 4–8% of the true TV, and the hypothesis TV ≤ 2δ̂ fails in every trial for μ ≥ 0.25 (`experiments/validate_tv_gap.py`). On 20 Newsgroups the only under-covering domain has the smallest δ̂ (7 domains; coverage loss is clipped at 0 by the benchmark script, so the quoted correlation r = −0.90 is driven by that one domain).

---

## Repository structure

```
ReliabilityLayer/
├── reliability/                 — Python SDK (pip-installable)
│   ├── conformal.py
│   ├── calibration.py
│   ├── drift.py
│   ├── report.py
│   └── _deficiency.py           — adapter over shared `defkit` solvers
├── arxiv_submission/            — manuscript source of truth + arXiv tarball
├── proofs/                      — theory review and number audit script
├── experiments/                 — validation scripts
│   ├── validate_theorem1.py     — Theorem 1 numerical validation
│   ├── validate_drift_deficiency.py
│   └── wilds_benchmark.py
├── examples/
│   ├── tabular_classification.ipynb
│   └── rag_pipeline.ipynb
├── tests/                       — 109 tests, all passing
├── pyproject.toml
└── requirements.txt
```

---

## Reproducing the paper

```bash
# every number and figure in arxiv_submission/main.tex, written to repro_out/ (about 25 min; exact LP + bootstrap)
PYTHON=python bash reproduce.sh
QUICK=1 bash reproduce.sh            # smoke run
python proofs/check_manuscript_numbers.py repro_out/results repro_out/logs   # prints every quoted value
```

## Running the experiments individually

```bash
# Theorem 1 numerical validation (final mode: 400 trials, ~5 min)
python experiments/validate_theorem1.py --mode final

# Certificate mode with bootstrap CIs (cert mode: 150 trials)
python experiments/validate_theorem1.py --mode cert

# Coverage loss vs deficiency tracking
python experiments/validate_drift_deficiency.py
```

Figures are written to `experiments/figures/`.

---

## Tests

```bash
pip install pytest
pytest tests/ -q
# 109 passed
```

---

## Position in the CorpusLOKAM series

```
Paper 1: LeCamDistortion/           — Le Cam deficiency for representation equivalence
Paper 2: SufficientLearning/        — Likelihood-preserving embeddings
Paper 3: Optimization_DecisionTheory/ — Optimization as a decision problem
Paper 4: CausalTheory/              — Causal inference as Le Cam experiment comparison
Paper 5: ReliabilityLayer/ (this)   — SDK + conformal-deficiency bridge
```

---

## License

MIT
