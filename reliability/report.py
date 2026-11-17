"""
reliability.report
==================

End-to-end reliability report generator.

Combines conformal prediction, calibration, and drift/deficiency diagnostics
into a single HTML report. Designed for maximum simplicity: 10 lines of code
from model to report.
"""

import base64
import copy
import io
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .calibration import compute_ece, reliability_diagram
from .conformal import ConformalClassifier, ConformalRegressor
from .drift import DeficiencyMonitor, covariate_shift_detector


def _fmt(value: Any, precision: int = 4) -> str:
    """Format scalar values for HTML output."""
    if value is None:
        return "N/A"
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "N/A"
        return f"{float(value):.{precision}f}"
    return str(value)


class ReliabilityReport:
    """
    End-to-end reliability assessment for a deployed model.

    Parameters
    ----------
    model : sklearn-compatible estimator
    X_cal, y_cal : calibration data
    X_test : test/deployment covariates
    y_test : optional labels for test/deployment data
    alpha : conformal miscoverage target
    task : 'classification' or 'regression'
    deficiency_n_bins : histogram bins used by DeficiencyMonitor
    deficiency_n_bootstrap : bootstrap rounds for deficiency CI
    deficiency_solver : backend solver for deficiency estimation ('auto', 'sinkhorn', 'scipy')
    deficiency_bound_multiplier : fixed operational multiplier for the deficiency certificate
    deficiency_hard_reject_threshold : hard-reject threshold on deficiency
    """

    _STATUS_COLORS = {
        'valid': '#2e7d32',
        'warning': '#f9a825',
        'invalid': '#c62828',
    }

    def __init__(
        self,
        model,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        X_test: np.ndarray,
        y_test: Optional[np.ndarray] = None,
        alpha: float = 0.10,
        task: str = 'classification',
        deficiency_n_bins: int = 20,
        deficiency_n_bootstrap: int = 30,
        deficiency_solver: str = 'auto',
        deficiency_bound_multiplier: float = 2.0,
        deficiency_hard_reject_threshold: float = 0.30,
    ):
        self.model = model
        self.X_cal = np.asarray(X_cal)
        self.y_cal = np.asarray(y_cal)
        self.X_test = np.asarray(X_test)
        self.y_test = None if y_test is None else np.asarray(y_test)
        self.alpha = float(alpha)
        self.task = task
        self.deficiency_n_bins = int(deficiency_n_bins)
        self.deficiency_n_bootstrap = int(deficiency_n_bootstrap)
        self.deficiency_solver = deficiency_solver
        self.deficiency_bound_multiplier = float(deficiency_bound_multiplier)
        self.deficiency_hard_reject_threshold = float(deficiency_hard_reject_threshold)
        self._results: Optional[Dict[str, Any]] = None
        self._artifacts: Dict[str, Any] = {}

    @staticmethod
    def _to_serializable(obj: Any) -> Any:
        """Recursively convert numpy objects to plain Python types."""
        if isinstance(obj, dict):
            return {k: ReliabilityReport._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [ReliabilityReport._to_serializable(v) for v in obj]
        if isinstance(obj, tuple):
            return [ReliabilityReport._to_serializable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    def _compute_summary(self, conformal: Dict[str, Any], drift: Dict[str, Any], deficiency: Dict[str, Any]) -> Dict[str, Any]:
        """Compute overall status and summary metrics."""
        target = 1.0 - self.alpha
        empirical_coverage = conformal.get('empirical_coverage')
        drift_detected = bool(drift.get('drift_detected', False))
        deficiency_warning = bool(deficiency.get('warning', False))
        hard_reject = bool(deficiency.get('hard_reject', False))

        if hard_reject or deficiency_warning or (
            empirical_coverage is not None and empirical_coverage < target - 0.05
        ):
            status = 'invalid'
            reason = "coverage is materially below target, deficiency warning is active, or hard-reject policy fired"
        elif drift_detected or (
            empirical_coverage is not None and empirical_coverage < target
        ):
            status = 'warning'
            reason = "drift detected or empirical coverage is slightly below target"
        else:
            status = 'valid'
            reason = "coverage is on target and no strong deficiency warning triggered"

        if empirical_coverage is None:
            coverage_gap = None
        else:
            coverage_gap = float(target - empirical_coverage)

        return {
            'status': status,
            'status_reason': reason,
            'status_color': self._STATUS_COLORS[status],
            'target_coverage': float(target),
            'empirical_coverage': empirical_coverage,
            'coverage_gap': coverage_gap,
            'drift_detected': drift_detected,
            'deficiency_warning': deficiency_warning,
            'hard_reject': hard_reject,
        }

    def compute(self) -> 'ReliabilityReport':
        """
        Run conformal, calibration, drift, and deficiency diagnostics.

        Returns
        -------
        self
        """
        if self.task not in ('classification', 'regression'):
            raise ValueError(
                f"Unknown task '{self.task}'. Choose 'classification' or 'regression'."
            )

        if self.task == 'classification':
            conformal = ConformalClassifier(self.model, alpha=self.alpha, score='softmax')
            conformal.calibrate(self.X_cal, self.y_cal)
            prediction_sets = conformal.predict(self.X_test)
            set_sizes = np.array([len(s) for s in prediction_sets], dtype=float)

            conformal_results = {
                'threshold': float(conformal.threshold_),
                'n_cal': int(conformal.n_cal_),
                'avg_set_size': float(set_sizes.mean()),
                'median_set_size': float(np.median(set_sizes)),
                'p90_set_size': float(np.percentile(set_sizes, 90)),
                'empirical_coverage': (
                    float(conformal.empirical_coverage(self.X_test, self.y_test))
                    if self.y_test is not None else None
                ),
            }

            probs_cal = self.model.predict_proba(self.X_cal)
            calibration_results = {
                'ece': float(compute_ece(probs_cal, self.y_cal)),
                'ece_test': None,
                'n_bins': 15,
            }
            self._artifacts['reliability_figure'] = reliability_diagram(
                probs_cal, self.y_cal, n_bins=15, title='Reliability Diagram (Calibration)'
            )

            if self.y_test is not None:
                probs_test = self.model.predict_proba(self.X_test)
                calibration_results['ece_test'] = float(compute_ece(probs_test, self.y_test))

        else:
            conformal = ConformalRegressor(self.model, alpha=self.alpha, variant='standard')
            conformal.calibrate(self.X_cal, self.y_cal)
            lower, upper = conformal.predict(self.X_test)
            widths = np.asarray(upper) - np.asarray(lower)

            conformal_results = {
                'threshold': float(conformal.threshold_),
                'n_cal': int(len(self.X_cal)),
                'avg_interval_width': float(np.mean(widths)),
                'median_interval_width': float(np.median(widths)),
                'empirical_coverage': (
                    float(conformal.empirical_coverage(self.X_test, self.y_test))
                    if self.y_test is not None else None
                ),
            }
            calibration_results = {
                'ece': None,
                'ece_test': None,
                'n_bins': None,
                'note': 'ECE/reliability diagram are only reported for classification tasks.',
            }
            self._artifacts['reliability_figure'] = None

        drift_results = covariate_shift_detector(self.X_cal, self.X_test)

        monitor = DeficiencyMonitor(
            n_bins=self.deficiency_n_bins,
            solver=self.deficiency_solver,
            n_bootstrap=self.deficiency_n_bootstrap,
            method='class_conditional',
            hard_reject_threshold=self.deficiency_hard_reject_threshold,
            bound_multiplier=self.deficiency_bound_multiplier,
        )
        monitor.fit(self.X_cal, y_ref=self.y_cal)
        deficiency_results = monitor.estimate(
            self.X_test,
            y_new=self.y_test,
            alpha=self.alpha,
        )

        summary = self._compute_summary(
            conformal=conformal_results,
            drift=drift_results,
            deficiency=deficiency_results,
        )

        self._results = {
            'summary': summary,
            'conformal': conformal_results,
            'calibration': calibration_results,
            'drift': drift_results,
            'deficiency': deficiency_results,
            'metadata': {
                'task': self.task,
                'alpha': float(self.alpha),
                'n_cal': int(len(self.X_cal)),
                'n_test': int(len(self.X_test)),
                'y_test_available': bool(self.y_test is not None),
                'deficiency_method_requested': 'class_conditional',
                'deficiency_n_bins': int(self.deficiency_n_bins),
                'deficiency_n_bootstrap': int(self.deficiency_n_bootstrap),
                'deficiency_solver': self.deficiency_solver,
                'deficiency_bound_multiplier': float(self.deficiency_bound_multiplier),
                'deficiency_hard_reject_threshold': float(self.deficiency_hard_reject_threshold),
            },
        }
        return self

    def to_dict(self) -> dict:
        """Return report metrics as a plain-serializable dictionary."""
        if self._results is None:
            raise RuntimeError("Call compute() before to_dict().")
        return self._to_serializable(copy.deepcopy(self._results))

    @staticmethod
    def _html_fallback(context: Dict[str, Any]) -> str:
        """Deterministic non-Jinja2 HTML fallback."""
        top_features = context['top_drifted_features']
        top_features_txt = ", ".join(str(x) for x in top_features) if top_features else "None"
        img_block = ""
        if context['reliability_plot_b64']:
            img_block = (
                "<h2>Calibration</h2>"
                "<img alt='Reliability Diagram' style='max-width: 560px; width: 100%;'"
                f" src='data:image/png;base64,{context['reliability_plot_b64']}' />"
            )

        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Reliability Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #1f2937; }}
    .status {{ color: white; padding: 0.8rem 1rem; border-radius: 6px; margin-bottom: 1rem; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 1rem; margin: 1rem 0; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td, th {{ border: 1px solid #d1d5db; padding: 0.5rem; text-align: left; }}
  </style>
</head>
<body>
  <h1>Reliability Report</h1>
  <div class="status" style="background:{context['summary']['status_color']}">
    Status: {context['summary']['status'].upper()}
  </div>
  <div class="card">
    <h2>Coverage</h2>
    <table>
      <tr><th>Target Coverage</th><td>{_fmt(context['summary']['target_coverage'])}</td></tr>
      <tr><th>Empirical Coverage</th><td>{_fmt(context['summary']['empirical_coverage'])}</td></tr>
      <tr><th>Average Set Size</th><td>{_fmt(context['conformal'].get('avg_set_size'))}</td></tr>
    </table>
  </div>
  <div class="card">
    {img_block}
    <p>ECE (calibration): {_fmt(context['calibration'].get('ece'))}</p>
  </div>
  <div class="card">
    <h2>Drift</h2>
    <p>Drift detected: {context['drift'].get('drift_detected')}</p>
    <p>MMD statistic: {_fmt(context['drift']['mmd'].get('statistic'))}</p>
    <p>Top drifted features: {top_features_txt}</p>
  </div>
  <div class="card">
    <h2>Deficiency Certificate</h2>
    <p>delta: {_fmt(context['deficiency'].get('delta'))}</p>
    <p>95% CI: [{_fmt(context['deficiency'].get('ci_lower'))}, {_fmt(context['deficiency'].get('ci_upper'))}]</p>
    <p>Coverage bound: {_fmt(context['deficiency'].get('coverage_bound'))}</p>
    <p>Clipped coverage bound: {_fmt(context['deficiency'].get('coverage_bound_clipped'))}</p>
    <p>Applied multiplier: {_fmt(context['deficiency'].get('c_multiplier'))}</p>
    <p>Operational action: {context['deficiency'].get('operational_action')}</p>
    <p>Message: {context['deficiency'].get('message', '')}</p>
  </div>
  <div class="card">
    <h2>Assumptions</h2>
    <p>Task: {context['metadata'].get('task')}</p>
    <p>Calibration/Test sizes: {context['metadata'].get('n_cal')} / {context['metadata'].get('n_test')}</p>
    <p>Deficiency method requested: class_conditional</p>
    <p>Deficiency multiplier: {_fmt(context['metadata'].get('deficiency_bound_multiplier'))}</p>
  </div>
</body>
</html>
"""

    def to_html(self, filepath: str) -> None:
        """
        Save reliability report as HTML.

        Parameters
        ----------
        filepath : output HTML path
        """
        if self._results is None:
            raise RuntimeError("Call compute() before to_html().")

        results = self.to_dict()
        fig = self._artifacts.get('reliability_figure')
        fig_b64 = None
        if fig is not None:
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
                fig_b64 = base64.b64encode(buffer.getvalue()).decode('ascii')

        drift_ks = results.get('drift', {}).get('ks', {})
        top_drifted_features = drift_ks.get('drifted_features', [])[:3]
        context = {
            'summary': results['summary'],
            'conformal': results['conformal'],
            'calibration': results['calibration'],
            'drift': results['drift'],
            'deficiency': results['deficiency'],
            'metadata': results['metadata'],
            'reliability_plot_b64': fig_b64,
            'top_drifted_features': top_drifted_features,
        }

        template = None
        try:
            from jinja2 import Template
            template = Template(
                """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Reliability Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; color: #1f2937; }
    .status { color: white; padding: 0.8rem 1rem; border-radius: 6px; margin-bottom: 1rem; }
    .card { border: 1px solid #d1d5db; border-radius: 8px; padding: 1rem; margin: 1rem 0; }
    table { border-collapse: collapse; width: 100%; }
    td, th { border: 1px solid #d1d5db; padding: 0.5rem; text-align: left; }
  </style>
</head>
<body>
  <h1>Reliability Report</h1>
  <div class="status" style="background:{{ summary.status_color }}">
    Status: {{ summary.status.upper() }}
  </div>

  <div class="card">
    <h2>Coverage</h2>
    <table>
      <tr><th>Target Coverage</th><td>{{ fmt(summary.target_coverage) }}</td></tr>
      <tr><th>Empirical Coverage</th><td>{{ fmt(summary.empirical_coverage) }}</td></tr>
      <tr><th>Average Set Size</th><td>{{ fmt(conformal.get('avg_set_size')) }}</td></tr>
      <tr><th>Coverage Gap</th><td>{{ fmt(summary.coverage_gap) }}</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Calibration</h2>
    <p>ECE (calibration): {{ fmt(calibration.get('ece')) }}</p>
    <p>ECE (test): {{ fmt(calibration.get('ece_test')) }}</p>
    {% if reliability_plot_b64 %}
    <img alt="Reliability Diagram" style="max-width: 560px; width: 100%;" src="data:image/png;base64,{{ reliability_plot_b64 }}" />
    {% else %}
    <p>No reliability diagram generated for this task.</p>
    {% endif %}
  </div>

  <div class="card">
    <h2>Drift</h2>
    <p>Drift detected: {{ drift.get('drift_detected') }}</p>
    <p>Severity: {{ drift.get('severity') }}</p>
    <p>MMD statistic: {{ fmt(drift.get('mmd', {}).get('statistic')) }}</p>
    <p>KS max statistic: {{ fmt(drift.get('ks', {}).get('max_statistic')) }}</p>
    <p>Top drifted features:
      {% if top_drifted_features %}{{ top_drifted_features|join(', ') }}{% else %}None{% endif %}
    </p>
  </div>

  <div class="card">
    <h2>Deficiency Certificate</h2>
    <p>delta: {{ fmt(deficiency.get('delta')) }}</p>
    <p>95% CI: [{{ fmt(deficiency.get('ci_lower')) }}, {{ fmt(deficiency.get('ci_upper')) }}]</p>
    <p>Coverage bound: {{ fmt(deficiency.get('coverage_bound')) }}</p>
    <p>Clipped coverage bound: {{ fmt(deficiency.get('coverage_bound_clipped')) }}</p>
    <p>Applied multiplier: {{ fmt(deficiency.get('c_multiplier')) }}</p>
    <p>Method used: {{ deficiency.get('method_used') }}</p>
    <p>Operational action: {{ deficiency.get('operational_action') }}</p>
    <p>Message: {{ deficiency.get('message', '') }}</p>
  </div>

  <div class="card">
    <h2>Assumptions</h2>
    <ul>
      <li>Task: {{ metadata.get('task') }}</li>
      <li>alpha: {{ fmt(metadata.get('alpha')) }}</li>
      <li>Calibration size: {{ metadata.get('n_cal') }}</li>
      <li>Test size: {{ metadata.get('n_test') }}</li>
      <li>Deficiency method requested: class_conditional</li>
      <li>Deficiency multiplier: {{ fmt(metadata.get('deficiency_bound_multiplier')) }}</li>
    </ul>
  </div>
</body>
</html>
"""
            )
        except Exception:
            template = None

        if template is not None:
            html_content = template.render(
                fmt=_fmt,
                summary=context['summary'],
                conformal=context['conformal'],
                calibration=context['calibration'],
                drift=context['drift'],
                deficiency=context['deficiency'],
                metadata=context['metadata'],
                reliability_plot_b64=context['reliability_plot_b64'],
                top_drifted_features=context['top_drifted_features'],
            )
        else:
            html_content = self._html_fallback(context)

        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding='utf-8')
