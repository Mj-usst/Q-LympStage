# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve


LOGGER = logging.getLogger("qlympstage.evaluation")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def normalize_stage_labels(values: pd.Series, context: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    observed = numeric.dropna()
    non_integer = observed[~np.isclose(observed, np.round(observed))]
    if not non_integer.empty:
        raise ValueError(f"{context} contains non-integer stage values: {sorted(non_integer.unique().tolist())[:10]}")
    n_stage_zero = int((numeric == 0).sum())
    if n_stage_zero:
        LOGGER.info("%s: mapped %d stage-0 observation(s) to stage I.", context, n_stage_zero)
    return numeric.replace(0, 1)


def read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input data must be CSV/XLS/XLSX.")


def extract_model(package: Any) -> Any:
    if isinstance(package, dict):
        for key in ("model", "best_model", "estimator", "pipeline"):
            if key in package:
                return extract_model(package[key])
    if hasattr(package, "predict_proba"):
        return package
    raise ValueError("Saved package does not contain a predict_proba estimator.")


def extract_features(package: Any, model: Any) -> list[str]:
    if isinstance(package, dict):
        for key in ("selected_features", "feature_columns", "features"):
            value = package.get(key)
            if isinstance(value, (list, tuple, np.ndarray, pd.Index)) and len(value):
                return [str(v) for v in value]
    if hasattr(model, "feature_names_in_"):
        return [str(v) for v in model.feature_names_in_]
    raise ValueError("Selected feature names are not stored in the model package.")


def resolve_labels(package: Any, requested: Sequence[int] | None) -> tuple[int, ...]:
    if requested is not None:
        labels = tuple(int(v) for v in requested)
    elif isinstance(package, dict) and package.get("labels") is not None:
        labels = tuple(int(v) for v in package["labels"])
    else:
        raise ValueError("Model labels are unavailable; provide --labels explicitly.")
    if len(labels) not in {2, 3} or len(set(labels)) != len(labels):
        raise ValueError("This script evaluates a binary (2-label) or three-class (3-label) model.")
    return labels


def model_classes(model: Any, n_columns: int) -> list[int]:
    candidates = [model]
    if hasattr(model, "named_steps"):
        candidates.extend(model.named_steps.values())
    for candidate in candidates:
        if hasattr(candidate, "classes_") and len(candidate.classes_) == n_columns:
            return [int(v) for v in candidate.classes_]
    return list(range(n_columns))


def predict_probabilities(package: Any, model: Any, x: pd.DataFrame, labels: Sequence[int]) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=float)
    classes = model_classes(model, raw.shape[1])
    package_labels = [int(v) for v in package.get("labels", labels)] if isinstance(package, dict) else list(labels)
    if set(classes) == set(labels):
        mapped = classes
    elif set(classes) == set(range(len(package_labels))):
        mapped = [package_labels[i] for i in classes]
    else:
        raise ValueError(f"Cannot align model classes {classes} with labels {list(labels)}")
    lookup = {label: i for i, label in enumerate(mapped)}
    return np.column_stack([raw[:, lookup[label]] for label in labels])


def safe_div_scalar(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def calibration_statistics(binary: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """Return calibration intercept and slope from outcome ~ logit(probability)."""
    binary = np.asarray(binary, dtype=int)
    if len(np.unique(binary)) < 2:
        return np.nan, np.nan
    score = np.clip(np.asarray(score, dtype=float), 1e-6, 1.0 - 1e-6)
    logit = np.log(score / (1.0 - score))
    design = sm.add_constant(logit, has_constant="add")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", PerfectSeparationWarning)
            warnings.simplefilter("error", ConvergenceWarning)
            fit = sm.GLM(binary, design, family=sm.families.Binomial()).fit(maxiter=100, disp=0)
    except Exception:
        return np.nan, np.nan
    if not getattr(fit, "converged", True):
        return np.nan, np.nan
    params = np.asarray(fit.params, dtype=float)
    if len(params) != 2 or not np.all(np.isfinite(params)):
        return np.nan, np.nan
    return float(params[0]), float(params[1])


def metric_dict(y: np.ndarray, probability: np.ndarray, labels: Sequence[int]) -> dict[str, float]:
    labels = tuple(int(v) for v in labels)
    pred = np.asarray(labels)[np.argmax(probability, axis=1)]
    out: dict[str, float] = {"ACC": float(accuracy_score(y, pred))}

    if len(labels) == 2:
        positive = labels[1]
        binary = (y == positive).astype(int)
        pred_binary = (pred == positive).astype(int)
        tn, fp, fn, tp = confusion_matrix(binary, pred_binary, labels=(0, 1)).ravel()
        intercept, slope = calibration_statistics(binary, probability[:, 1])
        out.update({
            "AUC": float(roc_auc_score(binary, probability[:, 1])) if len(np.unique(binary)) == 2 else np.nan,
            "SENS": safe_div_scalar(float(tp), float(tp + fn)),
            "SPEC": safe_div_scalar(float(tn), float(tn + fp)),
            "PPV": safe_div_scalar(float(tp), float(tp + fp)),
            "NPV": safe_div_scalar(float(tn), float(tn + fn)),
            "BRIER": float(np.mean((binary - probability[:, 1]) ** 2)),
            "CALIBRATION_INTERCEPT": intercept,
            "CALIBRATION_SLOPE": slope,
        })
        return out

    cm = confusion_matrix(y, pred, labels=labels)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)
    out.update({
        "SENS_macro": float(np.nanmean(np.divide(tp, tp + fn, out=np.full_like(tp, np.nan), where=(tp + fn) != 0))),
        "SPEC_macro": float(np.nanmean(np.divide(tn, tn + fp, out=np.full_like(tp, np.nan), where=(tn + fp) != 0))),
        "PPV_macro": float(np.nanmean(np.divide(tp, tp + fp, out=np.full_like(tp, np.nan), where=(tp + fp) != 0))),
        "NPV_macro": float(np.nanmean(np.divide(tn, tn + fn, out=np.full_like(tp, np.nan), where=(tn + fn) != 0))),
    })
    aucs: list[float] = []
    briers: list[float] = []
    for i, label in enumerate(labels):
        binary = (y == label).astype(int)
        auc = float(roc_auc_score(binary, probability[:, i])) if len(np.unique(binary)) == 2 else np.nan
        brier = float(np.mean((binary - probability[:, i]) ** 2))
        intercept, slope = calibration_statistics(binary, probability[:, i])
        out[f"AUC_stage_{label}"] = auc
        out[f"BRIER_stage_{label}"] = brier
        out[f"CALIBRATION_INTERCEPT_stage_{label}"] = intercept
        out[f"CALIBRATION_SLOPE_stage_{label}"] = slope
        if np.isfinite(auc):
            aucs.append(auc)
        briers.append(brier)
    out["AUC_macro_ovr"] = float(np.mean(aucs)) if aucs else np.nan
    out["BRIER_macro"] = float(np.mean(briers))
    return out


def clustered_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    raw = np.asarray(groups, dtype=object)
    if pd.isna(raw).any():
        raise ValueError("Patient identifiers contain missing values.")
    groups = raw.astype(str)
    unique = pd.unique(groups)
    if len(unique) < 2:
        raise ValueError("At least two patient clusters are required for bootstrap.")
    sampled = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([np.flatnonzero(groups == group) for group in sampled])


def bootstrap_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    labels: Sequence[int],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    point = metric_dict(y, probability, labels)
    samples = {key: [] for key in point}
    if repetitions > 0:
        rng = np.random.default_rng(seed)
        for _ in range(repetitions):
            idx = clustered_indices(groups, rng)
            try:
                values = metric_dict(y[idx], probability[idx], labels)
            except ValueError:
                continue
            for key, value in values.items():
                if np.isfinite(value):
                    samples[key].append(float(value))
    rows = []
    for key, value in point.items():
        vals = samples[key]
        rows.append({
            "metric": key,
            "value": value,
            "ci_low": float(np.percentile(vals, 2.5)) if vals else np.nan,
            "ci_high": float(np.percentile(vals, 97.5)) if vals else np.nan,
            "bootstrap_valid": len(vals),
        })
    return pd.DataFrame(rows)


def target_indices(labels: Sequence[int]) -> list[int]:
    """Use the second class as positive for binary models; use OVR for three classes."""
    return [1] if len(labels) == 2 else list(range(len(labels)))


def make_roc(y: np.ndarray, probability: np.ndarray, labels: Sequence[int], output: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    for i in target_indices(labels):
        label = labels[i]
        binary = (y == label).astype(int)
        if len(np.unique(binary)) < 2:
            continue
        fpr, tpr, thresholds = roc_curve(binary, probability[:, i])
        auc = roc_auc_score(binary, probability[:, i])
        ax.plot(fpr, tpr, label=f"Stage {label} (AUC={auc:.3f})")
        rows.extend(
            {"stage": int(label), "fpr": float(a), "tpr": float(b), "threshold": float(c)}
            for a, b, c in zip(fpr, tpr, thresholds)
        )
    ax.plot([0, 1], [0, 1], "--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve" if len(labels) == 2 else "One-vs-rest ROC curves")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "roc_ovr.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def make_calibration(
    y: np.ndarray,
    probability: np.ndarray,
    labels: Sequence[int],
    output: Path,
    bins: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    for i in target_indices(labels):
        label = labels[i]
        binary = (y == label).astype(int)
        frac, mean_pred = calibration_curve(binary, probability[:, i], n_bins=bins, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", label=f"Stage {label}")
        rows.extend(
            {"stage": int(label), "mean_predicted_probability": float(x), "observed_fraction": float(z)}
            for x, z in zip(mean_pred, frac)
        )
    ax.plot([0, 1], [0, 1], "--", linewidth=1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction")
    ax.set_title("Calibration curve" if len(labels) == 2 else "One-vs-rest calibration curves")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def net_benefit(binary: np.ndarray, score: np.ndarray, threshold: float) -> float:
    prediction = score >= threshold
    tp = np.sum(prediction & (binary == 1))
    fp = np.sum(prediction & (binary == 0))
    n = len(binary)
    return float(tp / n - fp / n * threshold / (1.0 - threshold))


def make_dca(y: np.ndarray, probability: np.ndarray, labels: Sequence[int], output: Path) -> pd.DataFrame:
    thresholds = np.linspace(0.01, 0.99, 99)
    rows: list[dict[str, float | int | str]] = []
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    for i in target_indices(labels):
        label = labels[i]
        binary = (y == label).astype(int)
        values = [net_benefit(binary, probability[:, i], threshold) for threshold in thresholds]
        ax.plot(thresholds, values, label=f"Stage {label} model")
        prevalence = float(binary.mean())
        treat_all_values = [prevalence - (1 - prevalence) * threshold / (1 - threshold) for threshold in thresholds]
        ax.plot(thresholds, treat_all_values, ":", linewidth=1, label=f"Stage {label} treat all")
        for threshold, value, treat_all in zip(thresholds, values, treat_all_values):
            rows.append({
                "stage": int(label),
                "threshold": float(threshold),
                "model_net_benefit": float(value),
                "treat_all": float(treat_all),
                "treat_none": 0.0,
            })
    ax.axhline(0.0, linestyle="--", linewidth=1, label="Treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve analysis" if len(labels) == 2 else "One-vs-rest decision curve analysis")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "decision_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def comparison_statistics(y: np.ndarray, probability: np.ndarray, labels: Sequence[int]) -> dict[str, float]:
    labels = tuple(int(v) for v in labels)
    prediction = np.asarray(labels)[np.argmax(probability, axis=1)]
    if len(labels) == 2:
        binary = (y == labels[1]).astype(int)
        auc = float(roc_auc_score(binary, probability[:, 1])) if len(np.unique(binary)) == 2 else np.nan
    else:
        aucs = []
        for i, label in enumerate(labels):
            binary = (y == label).astype(int)
            if len(np.unique(binary)) == 2:
                aucs.append(float(roc_auc_score(binary, probability[:, i])))
        auc = float(np.mean(aucs)) if aucs else np.nan
    return {"ACC": float(accuracy_score(y, prediction)), "AUC": auc}


def paired_model_comparison(
    y: np.ndarray,
    primary: np.ndarray,
    comparison: np.ndarray,
    groups: np.ndarray,
    labels: Sequence[int],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    primary_point = comparison_statistics(y, primary, labels)
    comparison_point = comparison_statistics(y, comparison, labels)
    point = {key: primary_point[key] - comparison_point[key] for key in primary_point}
    samples = {key: [] for key in point}
    if repetitions > 0:
        rng = np.random.default_rng(seed)
        for _ in range(repetitions):
            idx = clustered_indices(groups, rng)
            try:
                a = comparison_statistics(y[idx], primary[idx], labels)
                b = comparison_statistics(y[idx], comparison[idx], labels)
            except ValueError:
                continue
            for key in point:
                delta = a[key] - b[key]
                if np.isfinite(delta):
                    samples[key].append(float(delta))
    rows = []
    for key, delta in point.items():
        values = np.asarray(samples[key], dtype=float)
        if len(values):
            p_low = (np.sum(values <= 0) + 1) / (len(values) + 1)
            p_high = (np.sum(values >= 0) + 1) / (len(values) + 1)
            p_value = min(1.0, 2.0 * min(p_low, p_high))
        else:
            p_value = np.nan
        rows.append({
            "metric": key,
            "delta_primary_minus_comparison": delta,
            "ci_low": float(np.percentile(values, 2.5)) if len(values) else np.nan,
            "ci_high": float(np.percentile(values, 97.5)) if len(values) else np.nan,
            "p_value_two_sided": p_value,
            "bootstrap_valid": int(len(values)),
        })
    return pd.DataFrame(rows)


def parse_labels(value: str) -> tuple[int, ...]:
    labels = tuple(int(v.strip()) for v in value.split(",") if v.strip())
    if len(labels) not in {2, 3} or len(set(labels)) != len(labels):
        raise argparse.ArgumentTypeError("Labels must contain two or three distinct comma-separated integers.")
    return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--comparison-model", type=Path, default=None, help="Optional locked model for paired delta ACC/AUC.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="test")
    parser.add_argument("--label-column", default="标签")
    parser.add_argument("--id-column", default="序号")
    parser.add_argument("--center-column", default="对应中心")
    parser.add_argument("--center", default=None, help="Optional center filter.")
    parser.add_argument("--labels", type=parse_labels, default=None, help="Defaults to labels stored in the model package.")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=255)
    parser.add_argument("--calibration-bins", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    if args.bootstrap < 0:
        raise ValueError("--bootstrap must be >= 0")
    if args.calibration_bins < 2:
        raise ValueError("--calibration-bins must be >= 2")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    package = joblib.load(args.model.resolve())
    model = extract_model(package)
    labels = resolve_labels(package, args.labels)

    df = read_table(args.input.resolve())
    for column in (args.label_column, args.id_column):
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")
    if args.center is not None:
        if args.center_column not in df.columns:
            raise ValueError(f"Center filter requested but {args.center_column!r} is absent.")
        df = df[df[args.center_column].astype(str).str.strip() == str(args.center).strip()].copy()
    df[args.label_column] = normalize_stage_labels(df[args.label_column], args.dataset_name)
    df = df.dropna(subset=[args.label_column, args.id_column]).copy()
    df[args.label_column] = df[args.label_column].astype(int)
    df = df[df[args.label_column].isin(labels)].reset_index(drop=True)
    missing_labels = sorted(set(labels) - set(df[args.label_column].unique()))
    if missing_labels:
        raise ValueError(f"Evaluation data lack label(s): {missing_labels}")

    features = extract_features(package, model)
    if any("diagnostics" in feature.lower() for feature in features):
        raise ValueError("The saved model contains forbidden PyRadiomics diagnostics columns; retrain with script 04.")
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} selected model features: {missing[:20]}")
    x = df[features].apply(pd.to_numeric, errors="coerce")
    probability = predict_probabilities(package, model, x, labels)
    y = df[args.label_column].to_numpy(int)
    groups = df[args.id_column].astype(str).to_numpy()

    metrics = bootstrap_metrics(y, probability, groups, labels, args.bootstrap, args.random_state)
    metrics.insert(0, "dataset", args.dataset_name)
    metrics.insert(1, "analysis", "binary" if len(labels) == 2 else "three-class")
    metrics.insert(2, "positive_label", labels[1] if len(labels) == 2 else np.nan)
    metrics.to_csv(output / "performance_metrics.csv", index=False, encoding="utf-8-sig")
    make_roc(y, probability, labels, output).to_csv(output / "roc_points.csv", index=False)
    make_calibration(y, probability, labels, output, args.calibration_bins).to_csv(output / "calibration_points.csv", index=False)
    make_dca(y, probability, labels, output).to_csv(output / "decision_curve_points.csv", index=False)

    predictions = df[[column for column in (args.id_column, args.center_column, args.label_column) if column in df.columns]].copy()
    predictions["predicted_label"] = np.asarray(labels)[np.argmax(probability, axis=1)]
    for i, label in enumerate(labels):
        predictions[f"probability_stage_{label}"] = probability[:, i]
    predictions.to_csv(output / "case_predictions.csv", index=False, encoding="utf-8-sig")

    comparison_path = None
    if args.comparison_model is not None:
        comparison_path = args.comparison_model.resolve()
        comparison_package = joblib.load(comparison_path)
        comparison_estimator = extract_model(comparison_package)
        comparison_labels = resolve_labels(comparison_package, args.labels)
        if comparison_labels != labels:
            raise ValueError(f"Comparison model labels {comparison_labels} do not match primary labels {labels}.")
        comparison_features = extract_features(comparison_package, comparison_estimator)
        if any("diagnostics" in feature.lower() for feature in comparison_features):
            raise ValueError("The comparison model contains forbidden PyRadiomics diagnostics columns.")
        comparison_missing = [feature for feature in comparison_features if feature not in df.columns]
        if comparison_missing:
            raise ValueError(f"Missing comparison-model features: {comparison_missing[:20]}")
        comparison_probability = predict_probabilities(
            comparison_package,
            comparison_estimator,
            df[comparison_features].apply(pd.to_numeric, errors="coerce"),
            labels,
        )
        paired_model_comparison(
            y,
            probability,
            comparison_probability,
            groups,
            labels,
            args.bootstrap,
            args.random_state,
        ).to_csv(output / "paired_model_comparison.csv", index=False, encoding="utf-8-sig")

    (output / "run_info.json").write_text(json.dumps({
        "dataset": args.dataset_name,
        "analysis": "binary" if len(labels) == 2 else "three-class",
        "positive_label": labels[1] if len(labels) == 2 else None,
        "n_rows": len(df),
        "n_patients": int(pd.Series(groups).nunique()),
        "bootstrap": args.bootstrap,
        "bootstrap_unit": "patient cluster",
        "labels": list(labels),
        "model": str(args.model.resolve()),
        "comparison_model": str(comparison_path) if comparison_path is not None else None,
        "selected_features": features,
        "calibration_definition": "GLM outcome ~ intercept + slope * logit(predicted probability)",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Evaluation complete: %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
