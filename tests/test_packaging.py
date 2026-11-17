"""Regression tests for packaging and benchmark-environment bugs (2026-10-28).

- pyproject.toml named a build backend that does not exist
  (``setuptools.backends.legacy:build``), so ``pip install .`` failed.
- experiments/wilds_benchmark.py passed ``multi_class`` to LogisticRegression,
  which scikit-learn 1.8 removed, so every benchmark run ended as "blocked".
"""
import importlib
import inspect
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def test_build_backend_is_importable():
    text = _read("pyproject.toml")
    m = re.search(r'build-backend\s*=\s*"([^"]+)"', text)
    assert m, "pyproject.toml has no build-backend"
    module, _, attr = m.group(1).partition(":")
    mod = importlib.import_module(module)
    if attr:
        assert hasattr(mod, attr)


def test_benchmark_models_construct_on_installed_sklearn():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "wilds_benchmark", os.path.join(ROOT, "experiments", "wilds_benchmark.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._make_model(0) is not None
    assert mod._make_text_model(0) is not None
    assert "multi_class" not in inspect.getsource(mod._make_model)
    assert "multi_class" not in inspect.getsource(mod._make_text_model)


def test_benchmark_rng_is_per_dataset():
    # A single generator shared across datasets made results depend on the
    # --datasets order (found 2026-10-28 when 20NG numbers drifted).
    with open(os.path.join(ROOT, "experiments", "wilds_benchmark.py")) as fh:
        src = fh.read()
    loop = src.index("for dataset_name in args.datasets:")
    assert "default_rng(args.seed)" in src[loop:loop + 400]
