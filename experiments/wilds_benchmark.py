"""
WILDS benchmark runner for ReliabilityLayer.

This script evaluates domain-level conformal reliability metrics on WILDS
Camelyon17 and Amazon datasets when dependencies and local data are available.
If execution is blocked (missing dependency/data/runtime issue), it records a
machine-readable blocker report instead of fabricating results.

Outputs (under --output-dir):
- wilds_<dataset>_domains.csv
- wilds_<dataset>_summary.json
- wilds_benchmark_summary.csv
- wilds_benchmark_summary.json
"""

import argparse
import csv
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.datasets import fetch_covtype
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reliability.calibration import compute_ece
from reliability.conformal import ConformalClassifier
from reliability.drift import DeficiencyMonitor


def _to_numpy(array_like: Any) -> np.ndarray:
    """Convert numpy/torch-like array to numpy ndarray."""
    if hasattr(array_like, "detach"):
        array_like = array_like.detach()
    if hasattr(array_like, "cpu"):
        array_like = array_like.cpu()
    if hasattr(array_like, "numpy"):
        return np.asarray(array_like.numpy())
    return np.asarray(array_like)


def _subsample_indices(indices: np.ndarray, max_n: Optional[int], rng: np.random.Generator) -> np.ndarray:
    """Randomly subsample indices without replacement (deterministic by seed)."""
    if max_n is None or max_n <= 0 or len(indices) <= max_n:
        return indices
    chosen = rng.choice(indices, size=max_n, replace=False)
    return np.sort(chosen)


def _pick_split_name(split_dict: Dict[str, int], candidates: Sequence[str]) -> Optional[str]:
    for name in candidates:
        if name in split_dict:
            return name
    return None


def _normalize_metadata(metadata_array: np.ndarray) -> np.ndarray:
    if metadata_array.ndim == 1:
        return metadata_array.reshape(-1, 1)
    return metadata_array


def _select_domain_field(metadata_fields: List[str]) -> Tuple[int, str]:
    preferred = [
        "hospital",
        "domain",
        "region",
        "site",
        "center",
        "year",
        "time",
        "location",
    ]
    field_to_idx = {name: i for i, name in enumerate(metadata_fields)}
    for name in preferred:
        if name in field_to_idx:
            return field_to_idx[name], name
    if metadata_fields:
        return len(metadata_fields) - 1, metadata_fields[-1]
    return 0, "metadata_0"


def _domain_value_to_name(dataset_obj: Any, domain_field_idx: int, raw_value: Any) -> str:
    """
    Convert raw domain value to a readable label when metadata_map is available.
    Fallback to the numeric value string.
    """
    try:
        metadata_map = getattr(dataset_obj, "metadata_map", None)
        if metadata_map is not None and len(metadata_map) > domain_field_idx:
            mapping = metadata_map[domain_field_idx]
            if isinstance(mapping, dict):
                if raw_value in mapping:
                    return str(mapping[raw_value])
                int_key = int(raw_value)
                if int_key in mapping:
                    return str(mapping[int_key])
            if isinstance(mapping, (list, tuple)):
                int_key = int(raw_value)
                if 0 <= int_key < len(mapping):
                    return str(mapping[int_key])
    except Exception:
        pass
    return str(raw_value)


def _safe_pearson(x: List[float], y: List[float]) -> Tuple[float, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if valid.sum() < 3:
        return float("nan"), float("nan")
    if np.isclose(np.std(x_arr[valid]), 0.0) or np.isclose(np.std(y_arr[valid]), 0.0):
        return float("nan"), float("nan")
    r, p = pearsonr(x_arr[valid], y_arr[valid])
    return float(r), float(p)


def _safe_spearman(x: List[float], y: List[float]) -> Tuple[float, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if valid.sum() < 3:
        return float("nan"), float("nan")
    r, p = spearmanr(x_arr[valid], y_arr[valid])
    return float(r), float(p)


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _effective_n_bins(args: argparse.Namespace) -> int:
    """Return n_bins to use for DeficiencyMonitor, respecting --n-bins-exact override."""
    if (
        getattr(args, "n_bins_exact", None) is not None
        and args.solver in ("exact", "lp", "scipy", "cvxpy")
    ):
        return args.n_bins_exact
    return args.n_bins


def _make_model(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )


def _make_text_model(seed: int) -> LogisticRegression:
    """Classifier for sparse text features."""
    return LogisticRegression(
        max_iter=700,
        random_state=seed,
        solver="lbfgs",  # multinomial for K>2 (the old multiclass kwarg is gone in sklearn 1.8)
    )


def _topic_prefixes(targets: np.ndarray, target_names: Sequence[str]) -> np.ndarray:
    """Map 20 Newsgroups labels to coarse topic prefixes (comp/rec/sci/talk/...)."""
    return np.asarray([target_names[int(y)].split(".")[0] for y in targets], dtype=object)


def _benchmark_20newsgroups(
    dataset_name: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Fallback benchmark when WILDS Amazon is unavailable:
    20 Newsgroups text classification with domain shift by topic prefix.
    """
    start_utc = datetime.utcnow().isoformat() + "Z"

    try:
        train_bundle = fetch_20newsgroups(
            subset="train",
            remove=("headers", "footers", "quotes"),
        )
        test_bundle = fetch_20newsgroups(
            subset="test",
            remove=("headers", "footers", "quotes"),
        )
    except Exception as exc:
        return {
            "dataset": dataset_name,
            "status": "blocked",
            "blocker_type": "data_or_init",
            "blocker_message": f"20 Newsgroups fetch failed: {exc}",
            "started_at_utc": start_utc,
            "finished_at_utc": datetime.utcnow().isoformat() + "Z",
            "domain_rows": [],
            "summary": {},
        }

    try:
        X_train_text_all = np.asarray(train_bundle.data, dtype=object)
        y_train_all = np.asarray(train_bundle.target, dtype=int)
        X_test_text_all = np.asarray(test_bundle.data, dtype=object)
        y_test_all = np.asarray(test_bundle.target, dtype=int)
        target_names = train_bundle.target_names

        train_domains = _topic_prefixes(y_train_all, target_names)
        test_domains = _topic_prefixes(y_test_all, target_names)

        cal_domain = args.news_cal_domain
        cal_pool = np.where(train_domains == cal_domain)[0]
        if len(cal_pool) == 0:
            unique_domains, counts = np.unique(train_domains, return_counts=True)
            fallback_domain = str(unique_domains[np.argmax(counts)])
            cal_domain = fallback_domain
            cal_pool = np.where(train_domains == cal_domain)[0]

        # Split cal_pool: half for calibration, half kept in training so the
        # model sees all label classes. Without this, comp labels (1-5) are
        # absent from training and the model produces garbage scores on them.
        n_cal_take = min(len(cal_pool) // 2, args.max_cal)
        cal_indices = _subsample_indices(cal_pool, n_cal_take, rng)
        model_train_pool = np.setdiff1d(np.arange(len(y_train_all)), cal_indices)
        train_indices = _subsample_indices(model_train_pool, args.max_train, rng)

        if len(train_indices) == 0 or len(cal_indices) == 0:
            return {
                "dataset": dataset_name,
                "status": "blocked",
                "blocker_type": "data",
                "blocker_message": (
                    f"Insufficient train/cal rows after split: "
                    f"train={len(train_indices)} cal={len(cal_indices)}"
                ),
                "started_at_utc": start_utc,
                "finished_at_utc": datetime.utcnow().isoformat() + "Z",
                "domain_rows": [],
                "summary": {},
            }

        vectorizer = TfidfVectorizer(
            max_features=args.news_max_features,
            stop_words="english",
            min_df=2,
        )

        X_train_sparse = vectorizer.fit_transform(X_train_text_all[train_indices].tolist())
        y_train = y_train_all[train_indices]
        X_cal_sparse = vectorizer.transform(X_train_text_all[cal_indices].tolist())
        y_cal = y_train_all[cal_indices]

        model = _make_text_model(args.seed)
        model.fit(X_train_sparse, y_train)

        conformal = ConformalClassifier(model=model, alpha=args.alpha, score="softmax")
        conformal.calibrate(X_cal_sparse, y_cal)

        probs_cal = model.predict_proba(X_cal_sparse)
        monitor = DeficiencyMonitor(
            n_bins=_effective_n_bins(args),
            solver=args.solver,
            n_bootstrap=args.n_bootstrap,
            method="class_conditional",
            bound_multiplier=args.bound_multiplier,
            hard_reject_threshold=args.hard_reject_threshold,
        )
        monitor.fit(probs_cal, y_ref=y_cal)

        domain_rows: List[Dict[str, Any]] = []
        eval_domains = np.unique(test_domains)

        for domain in eval_domains:
            domain_idx_all = np.where(test_domains == domain)[0]
            domain_idx_all = _subsample_indices(domain_idx_all, args.max_test_per_domain, rng)
            if len(domain_idx_all) < args.min_domain_samples:
                continue

            X_domain_sparse = vectorizer.transform(X_test_text_all[domain_idx_all].tolist())
            y_domain = y_test_all[domain_idx_all]
            probs_domain = model.predict_proba(X_domain_sparse)

            coverage = float(conformal.empirical_coverage(X_domain_sparse, y_domain))
            avg_set_size = float(conformal.average_set_size(X_domain_sparse))
            ece = float(compute_ece(probs_domain, y_domain, n_bins=15))

            deficiency = monitor.estimate(
                probs_domain,
                y_new=y_domain,
                alpha=args.alpha,
                warning_threshold=args.warning_threshold,
            )

            delta = deficiency.get("delta")
            ci_upper = deficiency.get("ci_upper")
            coverage_bound = deficiency.get("coverage_bound")
            coverage_bound_clipped = deficiency.get("coverage_bound_clipped")
            warning = deficiency.get("warning")
            hard_reject = deficiency.get("hard_reject")
            loss = float(max(0.0, (1.0 - args.alpha) - coverage))

            domain_rows.append(
                {
                    "dataset": dataset_name,
                    "domain_field": "topic_prefix",
                    "domain_value": str(domain),
                    "domain_name": str(domain),
                    "n_samples": int(len(domain_idx_all)),
                    "coverage": coverage,
                    "coverage_loss": loss,
                    "avg_set_size": avg_set_size,
                    "ece": ece,
                    "delta": None if delta is None else float(delta),
                    "delta_ci_upper": None if ci_upper is None else float(ci_upper),
                    "coverage_bound": None if coverage_bound is None else float(coverage_bound),
                    "coverage_bound_clipped": None if coverage_bound_clipped is None else float(coverage_bound_clipped),
                    "c_multiplier": deficiency.get("c_multiplier"),
                    "warning": bool(warning) if warning is not None else None,
                    "hard_reject": bool(hard_reject) if hard_reject is not None else None,
                    "operational_action": deficiency.get("operational_action"),
                    "method_used": deficiency.get("method_used"),
                    "fallback_reason": deficiency.get("fallback_reason"),
                }
            )

        if len(domain_rows) == 0:
            return {
                "dataset": dataset_name,
                "status": "blocked",
                "blocker_type": "data",
                "blocker_message": (
                    f"No eval domains with at least {args.min_domain_samples} samples."
                ),
                "started_at_utc": start_utc,
                "finished_at_utc": datetime.utcnow().isoformat() + "Z",
                "domain_rows": [],
                "summary": {
                    "train_split": "20ng_train_minus_cal",
                    "cal_split": f"20ng_train_domain={cal_domain}",
                    "eval_split": "20ng_test_by_topic_prefix",
                    "domain_field": "topic_prefix",
                },
            }

        losses = [float(row["coverage_loss"]) for row in domain_rows]
        deltas = [float(row["delta"]) if row["delta"] is not None else float("nan") for row in domain_rows]
        pearson_r, pearson_p = _safe_pearson(losses, deltas)
        spearman_r, spearman_p = _safe_spearman(losses, deltas)

        summary = {
            "dataset": dataset_name,
            "status": "success",
            "source_dataset": "sklearn_20newsgroups",
            "train_split": "20ng_train_minus_cal",
            "cal_split": f"20ng_train_domain={cal_domain}",
            "eval_split": "20ng_test_by_topic_prefix",
            "domain_field": "topic_prefix",
            "n_domains": len(domain_rows),
            "n_train": int(len(train_indices)),
            "n_cal": int(len(cal_indices)),
            "n_eval_total": int(len(y_test_all)),
            "coverage_mean": float(np.mean([row["coverage"] for row in domain_rows])),
            "coverage_loss_mean": float(np.mean([row["coverage_loss"] for row in domain_rows])),
            "ece_mean": float(np.mean([row["ece"] for row in domain_rows])),
            "delta_mean": float(np.nanmean(deltas)),
            "hard_reject_count": int(sum(bool(row.get("hard_reject")) for row in domain_rows)),
            "vacuous_bound_count": int(
                sum(
                    row.get("coverage_bound") is not None
                    and np.isfinite(float(row["coverage_bound"]))
                    and float(row["coverage_bound"]) < 0.0
                    for row in domain_rows
                )
            ),
            "pearson_r_loss_delta": pearson_r,
            "pearson_p_loss_delta": pearson_p,
            "spearman_r_loss_delta": spearman_r,
            "spearman_p_loss_delta": spearman_p,
        }

        return {
            "dataset": dataset_name,
            "status": "success",
            "blocker_type": None,
            "blocker_message": None,
            "started_at_utc": start_utc,
            "finished_at_utc": datetime.utcnow().isoformat() + "Z",
            "domain_rows": domain_rows,
            "summary": summary,
        }

    except Exception as exc:
        return {
            "dataset": dataset_name,
            "status": "blocked",
            "blocker_type": "runtime",
            "blocker_message": f"{exc}",
            "traceback": traceback.format_exc(),
            "started_at_utc": start_utc,
            "finished_at_utc": datetime.utcnow().isoformat() + "Z",
            "domain_rows": [],
            "summary": {},
        }


def _benchmark_covtype_wilderness(
    dataset_name: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Fallback benchmark for Camelyon-style domain shift:
    UCI Covertype with domain split by Wilderness_Area (4 groups).
    """
    start_utc = datetime.utcnow().isoformat() + "Z"

    try:
        X_all, y_all_raw = fetch_covtype(return_X_y=True)
    except Exception as exc:
        return {
            "dataset": dataset_name,
            "status": "blocked",
            "blocker_type": "data_or_init",
            "blocker_message": f"covtype fetch failed: {exc}",
            "started_at_utc": start_utc,
            "finished_at_utc": datetime.utcnow().isoformat() + "Z",
            "domain_rows": [],
            "summary": {},
        }

    try:
        X_all = np.asarray(X_all, dtype=float)
        y_all_raw = np.asarray(y_all_raw)
        if X_all.ndim != 2 or X_all.shape[1] < 14:
            return {
                "dataset": dataset_name,
                "status": "blocked",
                "blocker_type": "data",
                "blocker_message": (
                    f"Unexpected covtype shape {X_all.shape}; expected at least 14 features."
                ),
                "started_at_utc": start_utc,
                "finished_at_utc": datetime.utcnow().isoformat() + "Z",
                "domain_rows": [],
                "summary": {},
            }

        # Covertype label values are 1..7. Map to 0..6 for conformal indexing.
        label_encoder = LabelEncoder()
        y_all = label_encoder.fit_transform(y_all_raw)

        # Covertype schema: first 10 continuous, next 4 one-hot Wilderness_Area.
        wilderness_onehot = X_all[:, 10:14]
        domain_ids = np.argmax(wilderness_onehot, axis=1)

        feature_mask = np.ones(X_all.shape[1], dtype=bool)
        feature_mask[10:14] = False  # drop explicit domain indicators
        X_features = X_all[:, feature_mask]

        all_indices = np.arange(X_features.shape[0])
        train_indices, test_indices = train_test_split(
            all_indices,
            test_size=args.covtype_test_size,
            random_state=args.seed,
            stratify=y_all,
        )

        train_domains = domain_ids[train_indices]
        cal_domain = int(args.covtype_cal_domain)
        cal_pool = train_indices[train_domains == cal_domain]
        if len(cal_pool) == 0:
            unique_domains, counts = np.unique(train_domains, return_counts=True)
            cal_domain = int(unique_domains[np.argmax(counts)])
            cal_pool = train_indices[train_domains == cal_domain]

        cal_indices = _subsample_indices(cal_pool, args.max_cal, rng)
        model_train_pool = np.setdiff1d(train_indices, cal_indices)
        model_train_indices = _subsample_indices(model_train_pool, args.max_train, rng)

        if len(model_train_indices) == 0 or len(cal_indices) == 0:
            return {
                "dataset": dataset_name,
                "status": "blocked",
                "blocker_type": "data",
                "blocker_message": (
                    f"Insufficient covtype rows after split: "
                    f"train={len(model_train_indices)} cal={len(cal_indices)}"
                ),
                "started_at_utc": start_utc,
                "finished_at_utc": datetime.utcnow().isoformat() + "Z",
                "domain_rows": [],
                "summary": {},
            }

        X_train = X_features[model_train_indices]
        y_train = y_all[model_train_indices]
        X_cal = X_features[cal_indices]
        y_cal = y_all[cal_indices]

        unique_train = np.unique(y_train)
        unique_cal = np.unique(y_cal)
        if unique_train.size < 2 or unique_cal.size < 2:
            return {
                "dataset": dataset_name,
                "status": "blocked",
                "blocker_type": "label_support",
                "blocker_message": (
                    f"Insufficient class diversity: train_classes={unique_train.tolist()} "
                    f"cal_classes={unique_cal.tolist()}"
                ),
                "started_at_utc": start_utc,
                "finished_at_utc": datetime.utcnow().isoformat() + "Z",
                "domain_rows": [],
                "summary": {},
            }

        model = _make_model(args.seed)
        model.fit(X_train, y_train)

        conformal = ConformalClassifier(model=model, alpha=args.alpha, score="softmax")
        conformal.calibrate(X_cal, y_cal)

        monitor = DeficiencyMonitor(
            n_bins=_effective_n_bins(args),
            solver=args.solver,
            n_bootstrap=args.n_bootstrap,
            method="class_conditional",
            bound_multiplier=args.bound_multiplier,
            hard_reject_threshold=args.hard_reject_threshold,
        )
        monitor.fit(X_cal, y_ref=y_cal)

        test_domains = domain_ids[test_indices]
        eval_domains = np.unique(test_domains)
        domain_rows: List[Dict[str, Any]] = []

        for domain in eval_domains:
            domain_idx_all = test_indices[test_domains == domain]
            domain_idx_all = _subsample_indices(domain_idx_all, args.max_test_per_domain, rng)
            if len(domain_idx_all) < args.min_domain_samples:
                continue

            X_domain = X_features[domain_idx_all]
            y_domain = y_all[domain_idx_all]

            coverage = float(conformal.empirical_coverage(X_domain, y_domain))
            avg_set_size = float(conformal.average_set_size(X_domain))
            probs = model.predict_proba(X_domain)
            ece = float(compute_ece(probs, y_domain, n_bins=15))

            deficiency = monitor.estimate(
                X_domain,
                y_new=y_domain,
                alpha=args.alpha,
                warning_threshold=args.warning_threshold,
            )

            delta = deficiency.get("delta")
            ci_upper = deficiency.get("ci_upper")
            coverage_bound = deficiency.get("coverage_bound")
            coverage_bound_clipped = deficiency.get("coverage_bound_clipped")
            warning = deficiency.get("warning")
            hard_reject = deficiency.get("hard_reject")
            loss = float(max(0.0, (1.0 - args.alpha) - coverage))

            domain_rows.append(
                {
                    "dataset": dataset_name,
                    "domain_field": "wilderness_area",
                    "domain_value": str(int(domain)),
                    "domain_name": f"wilderness_{int(domain) + 1}",
                    "n_samples": int(len(domain_idx_all)),
                    "coverage": coverage,
                    "coverage_loss": loss,
                    "avg_set_size": avg_set_size,
                    "ece": ece,
                    "delta": None if delta is None else float(delta),
                    "delta_ci_upper": None if ci_upper is None else float(ci_upper),
                    "coverage_bound": None if coverage_bound is None else float(coverage_bound),
                    "coverage_bound_clipped": None if coverage_bound_clipped is None else float(coverage_bound_clipped),
                    "c_multiplier": deficiency.get("c_multiplier"),
                    "warning": bool(warning) if warning is not None else None,
                    "hard_reject": bool(hard_reject) if hard_reject is not None else None,
                    "operational_action": deficiency.get("operational_action"),
                    "method_used": deficiency.get("method_used"),
                    "fallback_reason": deficiency.get("fallback_reason"),
                }
            )

        if len(domain_rows) == 0:
            return {
                "dataset": dataset_name,
                "status": "blocked",
                "blocker_type": "data",
                "blocker_message": (
                    f"No wilderness domains with at least {args.min_domain_samples} samples."
                ),
                "started_at_utc": start_utc,
                "finished_at_utc": datetime.utcnow().isoformat() + "Z",
                "domain_rows": [],
                "summary": {
                    "train_split": "covtype_train_minus_cal",
                    "cal_split": f"covtype_train_wilderness={cal_domain}",
                    "eval_split": "covtype_test_by_wilderness",
                    "domain_field": "wilderness_area",
                },
            }

        losses = [float(row["coverage_loss"]) for row in domain_rows]
        deltas = [float(row["delta"]) if row["delta"] is not None else float("nan") for row in domain_rows]
        pearson_r, pearson_p = _safe_pearson(losses, deltas)
        spearman_r, spearman_p = _safe_spearman(losses, deltas)

        summary = {
            "dataset": dataset_name,
            "status": "success",
            "source_dataset": "sklearn_covtype",
            "train_split": "covtype_train_minus_cal",
            "cal_split": f"covtype_train_wilderness={cal_domain}",
            "eval_split": "covtype_test_by_wilderness",
            "domain_field": "wilderness_area",
            "n_domains": len(domain_rows),
            "n_train": int(len(model_train_indices)),
            "n_cal": int(len(cal_indices)),
            "n_eval_total": int(len(test_indices)),
            "coverage_mean": float(np.mean([row["coverage"] for row in domain_rows])),
            "coverage_loss_mean": float(np.mean([row["coverage_loss"] for row in domain_rows])),
            "ece_mean": float(np.mean([row["ece"] for row in domain_rows])),
            "delta_mean": float(np.nanmean(deltas)),
            "hard_reject_count": int(sum(bool(row.get("hard_reject")) for row in domain_rows)),
            "vacuous_bound_count": int(
                sum(
                    row.get("coverage_bound") is not None
                    and np.isfinite(float(row["coverage_bound"]))
                    and float(row["coverage_bound"]) < 0.0
                    for row in domain_rows
                )
            ),
            "pearson_r_loss_delta": pearson_r,
            "pearson_p_loss_delta": pearson_p,
            "spearman_r_loss_delta": spearman_r,
            "spearman_p_loss_delta": spearman_p,
        }

        return {
            "dataset": dataset_name,
            "status": "success",
            "blocker_type": None,
            "blocker_message": None,
            "started_at_utc": start_utc,
            "finished_at_utc": datetime.utcnow().isoformat() + "Z",
            "domain_rows": domain_rows,
            "summary": summary,
        }

    except Exception as exc:
        return {
            "dataset": dataset_name,
            "status": "blocked",
            "blocker_type": "runtime",
            "blocker_message": f"{exc}",
            "traceback": traceback.format_exc(),
            "started_at_utc": start_utc,
            "finished_at_utc": datetime.utcnow().isoformat() + "Z",
            "domain_rows": [],
            "summary": {},
        }


def _benchmark_dataset(
    dataset_name: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    if dataset_name in ("20newsgroups", "news20", "amazon_fallback"):
        return _benchmark_20newsgroups(dataset_name=dataset_name, args=args, rng=rng)
    if dataset_name in ("camelyon_fallback", "covtype_fallback", "covtype_wilderness"):
        return _benchmark_covtype_wilderness(dataset_name=dataset_name, args=args, rng=rng)

    start_utc = datetime.utcnow().isoformat() + "Z"

    try:
        import importlib

        wilds_spec = importlib.util.find_spec("wilds")
        torch_spec = importlib.util.find_spec("torch")
        if wilds_spec is None or torch_spec is None:
            missing = []
            if wilds_spec is None:
                missing.append("wilds")
            if torch_spec is None:
                missing.append("torch")
            return {
                "dataset": dataset_name,
                "status": "blocked",
                "blocker_type": "dependency",
                "blocker_message": f"Missing dependencies: {', '.join(missing)}",
                "started_at_utc": start_utc,
                "finished_at_utc": datetime.utcnow().isoformat() + "Z",
                "domain_rows": [],
                "summary": {},
            }

        from wilds import get_dataset

    except Exception as exc:
        return {
            "dataset": dataset_name,
            "status": "blocked",
            "blocker_type": "dependency",
            "blocker_message": f"Dependency import failure: {exc}",
            "started_at_utc": start_utc,
            "finished_at_utc": datetime.utcnow().isoformat() + "Z",
            "domain_rows": [],
            "summary": {},
        }

    try:
        dataset_obj = get_dataset(
            dataset=dataset_name,
            root_dir=str(Path(args.root_dir).resolve()),
            download=args.download,
        )
    except Exception as exc:
        return {
            "dataset": dataset_name,
            "status": "blocked",
            "blocker_type": "data_or_init",
            "blocker_message": str(exc),
            "started_at_utc": start_utc,
            "finished_at_utc": datetime.utcnow().isoformat() + "Z",
            "domain_rows": [],
            "summary": {},
        }

    try:
        y_all = _to_numpy(getattr(dataset_obj, "y_array"))
        metadata_all = _normalize_metadata(_to_numpy(getattr(dataset_obj, "metadata_array")))
        split_all = _to_numpy(getattr(dataset_obj, "split_array")).astype(int)
        split_dict = dict(getattr(dataset_obj, "split_dict"))
        metadata_fields = list(getattr(dataset_obj, "metadata_fields", []))

        if y_all.ndim > 1:
            y_all = y_all.reshape(-1)
        y_all = y_all.astype(int)

        if len(metadata_fields) == 0:
            metadata_fields = [f"metadata_{j}" for j in range(metadata_all.shape[1])]

        domain_idx, domain_field = _select_domain_field(metadata_fields)

        train_split = _pick_split_name(split_dict, ["train", "id_train"])
        cal_split = _pick_split_name(split_dict, ["id_val", "val", "validation", "id_test"])
        eval_split = _pick_split_name(split_dict, ["test", "ood_test", "id_test", "val"])

        if train_split is None or cal_split is None or eval_split is None:
            return {
                "dataset": dataset_name,
                "status": "blocked",
                "blocker_type": "split",
                "blocker_message": (
                    "Required splits not found. "
                    f"available={sorted(split_dict.keys())}, "
                    f"picked train={train_split}, cal={cal_split}, eval={eval_split}"
                ),
                "started_at_utc": start_utc,
                "finished_at_utc": datetime.utcnow().isoformat() + "Z",
                "domain_rows": [],
                "summary": {},
            }

        train_indices = np.where(split_all == split_dict[train_split])[0]
        cal_indices = np.where(split_all == split_dict[cal_split])[0]
        eval_indices = np.where(split_all == split_dict[eval_split])[0]

        # Remove unlabeled rows (-1) if present.
        train_indices = train_indices[y_all[train_indices] >= 0]
        cal_indices = cal_indices[y_all[cal_indices] >= 0]
        eval_indices = eval_indices[y_all[eval_indices] >= 0]

        if len(train_indices) == 0 or len(cal_indices) == 0 or len(eval_indices) == 0:
            return {
                "dataset": dataset_name,
                "status": "blocked",
                "blocker_type": "data",
                "blocker_message": (
                    f"Empty required split after label filtering: "
                    f"train={len(train_indices)} cal={len(cal_indices)} eval={len(eval_indices)}"
                ),
                "started_at_utc": start_utc,
                "finished_at_utc": datetime.utcnow().isoformat() + "Z",
                "domain_rows": [],
                "summary": {},
            }

        train_indices = _subsample_indices(train_indices, args.max_train, rng)
        cal_indices = _subsample_indices(cal_indices, args.max_cal, rng)

        feature_cols = [j for j in range(metadata_all.shape[1]) if j != domain_idx]
        if not feature_cols:
            feature_cols = [domain_idx]

        X_train = metadata_all[train_indices][:, feature_cols].astype(float)
        y_train = y_all[train_indices]
        X_cal = metadata_all[cal_indices][:, feature_cols].astype(float)
        y_cal = y_all[cal_indices]

        unique_train = np.unique(y_train)
        unique_cal = np.unique(y_cal)
        if unique_train.size < 2 or unique_cal.size < 2:
            return {
                "dataset": dataset_name,
                "status": "blocked",
                "blocker_type": "label_support",
                "blocker_message": (
                    f"Insufficient class diversity: train_classes={unique_train.tolist()} "
                    f"cal_classes={unique_cal.tolist()}"
                ),
                "started_at_utc": start_utc,
                "finished_at_utc": datetime.utcnow().isoformat() + "Z",
                "domain_rows": [],
                "summary": {},
            }

        model = _make_model(args.seed)
        model.fit(X_train, y_train)

        conformal = ConformalClassifier(model=model, alpha=args.alpha, score="softmax")
        conformal.calibrate(X_cal, y_cal)

        monitor = DeficiencyMonitor(
            n_bins=args.n_bins,
            solver=args.solver,
            n_bootstrap=args.n_bootstrap,
            method="class_conditional",
        )
        monitor.fit(X_cal, y_ref=y_cal)

        eval_domains = metadata_all[eval_indices, domain_idx]
        domain_values = np.unique(eval_domains)
        domain_rows: List[Dict[str, Any]] = []

        for raw_domain in domain_values:
            domain_mask = eval_domains == raw_domain
            domain_idx_all = eval_indices[domain_mask]
            domain_idx_all = _subsample_indices(domain_idx_all, args.max_test_per_domain, rng)

            if len(domain_idx_all) < args.min_domain_samples:
                continue

            X_domain = metadata_all[domain_idx_all][:, feature_cols].astype(float)
            y_domain = y_all[domain_idx_all]

            coverage = float(conformal.empirical_coverage(X_domain, y_domain))
            avg_set_size = float(conformal.average_set_size(X_domain))
            probs = model.predict_proba(X_domain)
            ece = float(compute_ece(probs, y_domain, n_bins=15))

            deficiency = monitor.estimate(
                X_domain,
                y_new=y_domain,
                alpha=args.alpha,
                warning_threshold=args.warning_threshold,
            )

            delta = deficiency.get("delta")
            ci_upper = deficiency.get("ci_upper")
            coverage_bound = deficiency.get("coverage_bound")
            coverage_bound_clipped = deficiency.get("coverage_bound_clipped")
            warning = deficiency.get("warning")
            hard_reject = deficiency.get("hard_reject")

            loss = float(max(0.0, (1.0 - args.alpha) - coverage))
            domain_name = _domain_value_to_name(dataset_obj, domain_idx, raw_domain)

            domain_rows.append(
                {
                    "dataset": dataset_name,
                    "domain_field": domain_field,
                    "domain_value": str(raw_domain),
                    "domain_name": domain_name,
                    "n_samples": int(len(domain_idx_all)),
                    "coverage": coverage,
                    "coverage_loss": loss,
                    "avg_set_size": avg_set_size,
                    "ece": ece,
                    "delta": None if delta is None else float(delta),
                    "delta_ci_upper": None if ci_upper is None else float(ci_upper),
                    "coverage_bound": None if coverage_bound is None else float(coverage_bound),
                    "coverage_bound_clipped": None if coverage_bound_clipped is None else float(coverage_bound_clipped),
                    "c_multiplier": deficiency.get("c_multiplier"),
                    "warning": bool(warning) if warning is not None else None,
                    "hard_reject": bool(hard_reject) if hard_reject is not None else None,
                    "operational_action": deficiency.get("operational_action"),
                    "method_used": deficiency.get("method_used"),
                    "fallback_reason": deficiency.get("fallback_reason"),
                }
            )

        if len(domain_rows) == 0:
            return {
                "dataset": dataset_name,
                "status": "blocked",
                "blocker_type": "data",
                "blocker_message": (
                    f"No eval domains with at least {args.min_domain_samples} samples "
                    f"in split '{eval_split}'."
                ),
                "started_at_utc": start_utc,
                "finished_at_utc": datetime.utcnow().isoformat() + "Z",
                "domain_rows": [],
                "summary": {
                    "train_split": train_split,
                    "cal_split": cal_split,
                    "eval_split": eval_split,
                    "domain_field": domain_field,
                },
            }

        losses = [float(row["coverage_loss"]) for row in domain_rows]
        deltas = [float(row["delta"]) if row["delta"] is not None else float("nan") for row in domain_rows]
        pearson_r, pearson_p = _safe_pearson(losses, deltas)
        spearman_r, spearman_p = _safe_spearman(losses, deltas)

        summary = {
            "dataset": dataset_name,
            "status": "success",
            "train_split": train_split,
            "cal_split": cal_split,
            "eval_split": eval_split,
            "domain_field": domain_field,
            "n_domains": len(domain_rows),
            "n_train": int(len(train_indices)),
            "n_cal": int(len(cal_indices)),
            "n_eval_total": int(len(eval_indices)),
            "coverage_mean": float(np.mean([row["coverage"] for row in domain_rows])),
            "coverage_loss_mean": float(np.mean([row["coverage_loss"] for row in domain_rows])),
            "ece_mean": float(np.mean([row["ece"] for row in domain_rows])),
            "delta_mean": float(np.nanmean(deltas)),
            "hard_reject_count": int(sum(bool(row.get("hard_reject")) for row in domain_rows)),
            "vacuous_bound_count": int(
                sum(
                    row.get("coverage_bound") is not None
                    and np.isfinite(float(row["coverage_bound"]))
                    and float(row["coverage_bound"]) < 0.0
                    for row in domain_rows
                )
            ),
            "pearson_r_loss_delta": pearson_r,
            "pearson_p_loss_delta": pearson_p,
            "spearman_r_loss_delta": spearman_r,
            "spearman_p_loss_delta": spearman_p,
        }

        return {
            "dataset": dataset_name,
            "status": "success",
            "blocker_type": None,
            "blocker_message": None,
            "started_at_utc": start_utc,
            "finished_at_utc": datetime.utcnow().isoformat() + "Z",
            "domain_rows": domain_rows,
            "summary": summary,
        }

    except Exception as exc:
        return {
            "dataset": dataset_name,
            "status": "blocked",
            "blocker_type": "runtime",
            "blocker_message": f"{exc}",
            "traceback": traceback.format_exc(),
            "started_at_utc": start_utc,
            "finished_at_utc": datetime.utcnow().isoformat() + "Z",
            "domain_rows": [],
            "summary": {},
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ReliabilityLayer WILDS benchmark.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["camelyon17", "amazon"],
        help=(
            "Datasets to evaluate. WILDS ids (e.g. camelyon17, amazon) are "
            "supported, plus fallback aliases: 20newsgroups/news20/amazon_fallback/"
            "camelyon_fallback/covtype_wilderness."
        ),
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=str(PROJECT_ROOT / "experiments" / "data"),
        help="Root directory for WILDS datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "experiments" / "results"),
        help="Directory for CSV/JSON outputs.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow WILDS to download datasets if missing (can be very large).",
    )
    parser.add_argument("--alpha", type=float, default=0.10, help="Conformal miscoverage rate.")
    parser.add_argument("--n-bins", type=int, default=12, help="Histogram bins for deficiency estimator.")
    parser.add_argument("--n-bootstrap", type=int, default=20, help="Bootstrap samples for deficiency CI.")
    parser.add_argument("--solver", type=str, default="auto", help="Deficiency solver backend.")
    parser.add_argument(
        "--n-bins-exact",
        type=int,
        default=None,
        help=(
            "Override --n-bins when --solver exact is used. "
            "Exact LP solve time scales as O(n_bins^3); use 3-5 for tractable bootstrap runs "
            "(e.g. --solver exact --n-bins-exact 5 finishes in ~13 min vs hours at n-bins=12)."
        ),
    )
    parser.add_argument("--warning-threshold", type=float, default=0.05, help="Delta warning threshold.")
    parser.add_argument("--hard-reject-threshold", type=float, default=0.30, help="Operational hard-reject threshold.")
    parser.add_argument("--bound-multiplier", type=float, default=2.0, help="Fixed operational multiplier for the deficiency certificate.")
    parser.add_argument("--max-train", type=int, default=12000, help="Max training samples per dataset.")
    parser.add_argument("--max-cal", type=int, default=8000, help="Max calibration samples per dataset.")
    parser.add_argument(
        "--max-test-per-domain",
        type=int,
        default=4000,
        help="Max evaluation samples per domain.",
    )
    parser.add_argument(
        "--min-domain-samples",
        type=int,
        default=100,
        help="Minimum samples required to report a domain row.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--news-cal-domain",
        type=str,
        default="comp",
        help="Calibration domain prefix for 20 Newsgroups fallback benchmark.",
    )
    parser.add_argument(
        "--news-max-features",
        type=int,
        default=20000,
        help="TF-IDF max_features for 20 Newsgroups fallback benchmark.",
    )
    parser.add_argument(
        "--covtype-cal-domain",
        type=int,
        default=0,
        help="Calibration wilderness domain id (0..3) for covtype fallback.",
    )
    parser.add_argument(
        "--covtype-test-size",
        type=float,
        default=0.3,
        help="Test fraction for covtype fallback split.",
    )
    parser.add_argument(
        "--allow-blockers",
        action="store_true",
        help="Exit 0 even when one or more datasets are blocked.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("ReliabilityLayer WILDS Benchmark")
    print("=" * 90)
    print(f"datasets={args.datasets}")
    print(f"root_dir={Path(args.root_dir).resolve()}")
    print(f"output_dir={output_dir}")
    print(f"download={args.download} | seed={args.seed} | alpha={args.alpha}")
    print(f"bound_multiplier={args.bound_multiplier} | hard_reject_threshold={args.hard_reject_threshold}")
    print()

    run_records: List[Dict[str, Any]] = []

    for dataset_name in args.datasets:
        print(f"[dataset={dataset_name}] starting...")
        # fresh generator per dataset: results must not depend on dataset order
        rng = np.random.default_rng(args.seed)
        run = _benchmark_dataset(dataset_name=dataset_name, args=args, rng=rng)
        run_records.append(run)

        domain_rows = run.get("domain_rows", [])
        domain_csv_path = output_dir / f"wilds_{dataset_name}_domains.csv"
        domain_fieldnames = [
            "dataset",
            "domain_field",
            "domain_value",
            "domain_name",
            "n_samples",
            "coverage",
            "coverage_loss",
            "avg_set_size",
            "ece",
            "delta",
            "delta_ci_upper",
            "coverage_bound",
            "coverage_bound_clipped",
            "c_multiplier",
            "warning",
            "hard_reject",
            "operational_action",
            "method_used",
            "fallback_reason",
        ]

        if domain_rows:
            _write_csv(domain_csv_path, domain_rows, domain_fieldnames)
            print(f"[dataset={dataset_name}] wrote domain metrics -> {domain_csv_path}")
        else:
            blocker_row = {
                "dataset": dataset_name,
                "domain_field": "",
                "domain_value": "",
                "domain_name": "",
                "n_samples": 0,
                "coverage": "",
                "coverage_loss": "",
                "avg_set_size": "",
                "ece": "",
                "delta": "",
                "delta_ci_upper": "",
                "coverage_bound": "",
                "coverage_bound_clipped": "",
                "c_multiplier": "",
                "warning": "",
                "hard_reject": "",
                "operational_action": "",
                "method_used": "",
                "fallback_reason": run.get("blocker_message", ""),
            }
            _write_csv(domain_csv_path, [blocker_row], domain_fieldnames)
            print(f"[dataset={dataset_name}] wrote blocker domain stub -> {domain_csv_path}")

        dataset_summary = {
            "dataset": dataset_name,
            "status": run.get("status"),
            "blocker_type": run.get("blocker_type"),
            "blocker_message": run.get("blocker_message"),
            "started_at_utc": run.get("started_at_utc"),
            "finished_at_utc": run.get("finished_at_utc"),
            "summary": run.get("summary", {}),
            "n_domain_rows": len(domain_rows),
        }
        dataset_json_path = output_dir / f"wilds_{dataset_name}_summary.json"
        _write_json(dataset_json_path, dataset_summary)
        print(f"[dataset={dataset_name}] wrote summary -> {dataset_json_path}")

        if run.get("status") == "success":
            summary = run.get("summary", {})
            print(
                f"[dataset={dataset_name}] status=success | "
                f"domains={summary.get('n_domains')} | "
                f"coverage_mean={summary.get('coverage_mean'):.4f} | "
                f"delta_mean={summary.get('delta_mean'):.4f}"
            )
        else:
            print(
                f"[dataset={dataset_name}] status=blocked | "
                f"type={run.get('blocker_type')} | "
                f"message={run.get('blocker_message')}"
            )
        print()

    summary_rows: List[Dict[str, Any]] = []
    for run in run_records:
        dataset_name = run.get("dataset")
        summary = run.get("summary", {})
        summary_rows.append(
            {
                "dataset": dataset_name,
                "status": run.get("status"),
                "blocker_type": run.get("blocker_type"),
                "blocker_message": run.get("blocker_message"),
                "n_domain_rows": len(run.get("domain_rows", [])),
                "domain_field": summary.get("domain_field"),
                "train_split": summary.get("train_split"),
                "cal_split": summary.get("cal_split"),
                "eval_split": summary.get("eval_split"),
                "n_domains": summary.get("n_domains"),
                "coverage_mean": summary.get("coverage_mean"),
                "coverage_loss_mean": summary.get("coverage_loss_mean"),
                "ece_mean": summary.get("ece_mean"),
                "delta_mean": summary.get("delta_mean"),
                "pearson_r_loss_delta": summary.get("pearson_r_loss_delta"),
                "pearson_p_loss_delta": summary.get("pearson_p_loss_delta"),
                "spearman_r_loss_delta": summary.get("spearman_r_loss_delta"),
                "spearman_p_loss_delta": summary.get("spearman_p_loss_delta"),
            }
        )

    aggregate_csv = output_dir / "wilds_benchmark_summary.csv"
    aggregate_fields = [
        "dataset",
        "status",
        "blocker_type",
        "blocker_message",
        "n_domain_rows",
        "domain_field",
        "train_split",
        "cal_split",
        "eval_split",
        "n_domains",
        "coverage_mean",
        "coverage_loss_mean",
        "ece_mean",
        "delta_mean",
        "pearson_r_loss_delta",
        "pearson_p_loss_delta",
        "spearman_r_loss_delta",
        "spearman_p_loss_delta",
    ]
    _write_csv(aggregate_csv, summary_rows, aggregate_fields)

    aggregate_json = output_dir / "wilds_benchmark_summary.json"
    aggregate_payload = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "args": {
            "datasets": args.datasets,
            "root_dir": str(Path(args.root_dir).resolve()),
            "output_dir": str(output_dir),
            "download": bool(args.download),
            "alpha": float(args.alpha),
            "n_bins": int(args.n_bins),
            "n_bootstrap": int(args.n_bootstrap),
            "solver": args.solver,
            "warning_threshold": float(args.warning_threshold),
            "max_train": int(args.max_train),
            "max_cal": int(args.max_cal),
            "max_test_per_domain": int(args.max_test_per_domain),
            "min_domain_samples": int(args.min_domain_samples),
            "news_cal_domain": args.news_cal_domain,
            "news_max_features": int(args.news_max_features),
            "covtype_cal_domain": int(args.covtype_cal_domain),
            "covtype_test_size": float(args.covtype_test_size),
            "seed": int(args.seed),
        },
        "datasets": run_records,
    }
    _write_json(aggregate_json, aggregate_payload)

    print("=" * 90)
    print("Benchmark summary files")
    print("=" * 90)
    print(aggregate_csv)
    print(aggregate_json)

    blocked = [run for run in run_records if run.get("status") != "success"]
    if blocked and not args.allow_blockers:
        print()
        print("One or more datasets are blocked. Re-run with --allow-blockers to force exit code 0.")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
