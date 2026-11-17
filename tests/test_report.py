"""
Tests for reliability.report

Verifies:
- compute() runs end-to-end on a classification workflow
- to_dict() returns expected structure and serializable values
- to_html() writes an HTML file with required sections
- to_dict()/to_html() fail before compute()
"""

import json
import os
import sys

import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reliability.report import ReliabilityReport


@pytest.fixture
def fitted_classification_case():
    """Train a small deterministic classifier and return cal/test splits."""
    X, y = make_classification(
        n_samples=420,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        random_state=7,
    )
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.5, random_state=7
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=7
    )
    model = LogisticRegression(max_iter=1000, random_state=7)
    model.fit(X_train, y_train)
    return model, X_cal, y_cal, X_test, y_test


class TestReliabilityReport:
    def test_compute_success_path_classification(self, fitted_classification_case):
        model, X_cal, y_cal, X_test, y_test = fitted_classification_case
        report = ReliabilityReport(
            model=model,
            X_cal=X_cal,
            y_cal=y_cal,
            X_test=X_test,
            y_test=y_test,
            alpha=0.10,
            task='classification',
            deficiency_n_bins=10,
            deficiency_n_bootstrap=5,
        )

        out = report.compute()
        metrics = report.to_dict()

        assert out is report
        assert metrics['metadata']['task'] == 'classification'
        assert metrics['conformal']['avg_set_size'] >= 1.0
        assert metrics['summary']['status'] in ('valid', 'warning', 'invalid')
        assert metrics['summary']['empirical_coverage'] is not None

        assert metrics['deficiency']['method_used'] == 'class_conditional'
        assert metrics['deficiency']['fallback_reason'] is None
        assert 'c_multiplier' in metrics['deficiency']

    def test_to_dict_structure_and_serializable(self, fitted_classification_case):
        model, X_cal, y_cal, X_test, y_test = fitted_classification_case
        report = ReliabilityReport(
            model=model,
            X_cal=X_cal,
            y_cal=y_cal,
            X_test=X_test,
            y_test=y_test,
            deficiency_n_bins=8,
            deficiency_n_bootstrap=3,
        )
        report.compute()

        metrics = report.to_dict()
        required_top_keys = {
            'summary',
            'conformal',
            'calibration',
            'drift',
            'deficiency',
            'metadata',
        }
        assert required_top_keys <= set(metrics.keys())
        assert 'reliability_figure' not in metrics.get('calibration', {})
        json.dumps(metrics)
        assert metrics['metadata']['deficiency_bound_multiplier'] == pytest.approx(2.0)

    def test_to_html_writes_file_with_expected_sections(self, fitted_classification_case, tmp_path):
        model, X_cal, y_cal, X_test, y_test = fitted_classification_case
        report = ReliabilityReport(
            model=model,
            X_cal=X_cal,
            y_cal=y_cal,
            X_test=X_test,
            y_test=y_test,
            deficiency_n_bins=8,
            deficiency_n_bootstrap=3,
        )
        report.compute()

        out_path = tmp_path / 'reliability_report.html'
        report.to_html(str(out_path))

        assert out_path.exists()
        html_text = out_path.read_text(encoding='utf-8')
        assert 'Reliability Report' in html_text
        assert 'Coverage' in html_text
        assert 'Calibration' in html_text
        assert 'Drift' in html_text
        assert 'Deficiency Certificate' in html_text
        assert 'Applied multiplier' in html_text
        assert 'Assumptions' in html_text
        assert 'data:image/png;base64' in html_text

    def test_to_dict_and_to_html_fail_before_compute(self, fitted_classification_case, tmp_path):
        model, X_cal, y_cal, X_test, y_test = fitted_classification_case
        report = ReliabilityReport(
            model=model,
            X_cal=X_cal,
            y_cal=y_cal,
            X_test=X_test,
            y_test=y_test,
        )

        with pytest.raises(RuntimeError, match="Call compute"):
            report.to_dict()
        with pytest.raises(RuntimeError, match="Call compute"):
            report.to_html(str(tmp_path / 'precompute.html'))
