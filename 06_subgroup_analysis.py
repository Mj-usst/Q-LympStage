# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "multiclass" / "subgroup_analysis"

DEFAULT_LABEL_COLUMN = "标签"
DEFAULT_ID_COLUMN = "序号"
DEFAULT_CENTER_COLUMN = "对应中心"
DEFAULT_ETIOLOGY_COLUMN = "病因"
DEFAULT_PAIR_COLUMN = "配对类别"
DEFAULT_EXTERNAL_CENTER = "中心4"
RANDOM_STATE = 255

CLASS_LABELS = (1, 2, 3)
ETIOLOGY_LEVELS = ("原发性", "继发性")
PAIR_LEVELS = ("双同", "双不同")

SUBGROUP_DISPLAY = {
    ("etiology", "原发性"): "Etiology: Primary",
    ("etiology", "继发性"): "Etiology: Secondary",
    ("pair_type", "双同"): "Bilateral stage concordance: Concordant",
    ("pair_type", "双不同"): "Bilateral stage concordance: Discordant",
}

METRICS = ("ACC", "AUC", "SENS", "SPEC", "PPV", "NPV")


@dataclass(frozen=True)
class Config:
    input_excel: Path
    model_path: Path
    output_dir: Path
    label_column: str
    id_column: str
    center_column: str
    external_center: str
    etiology_column: str
    pair_column: str
    labels: tuple[int, ...]
    bootstrap_repetitions: int
    random_state: int
    minimum_subgroup_size: int
    make_plots: bool


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def normalize_stage_labels(values: pd.Series, context: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    observed = numeric.dropna()
    non_integer = observed[~np.isclose(observed, np.round(observed))]
    if not non_integer.empty:
        raise ValueError(f"{context} contains non-integer stage values: {sorted(non_integer.unique().tolist())[:10]}")
    n_stage_zero = int((numeric == 0).sum())
    if n_stage_zero:
        logging.info("%s: mapped %d stage-0 observation(s) to stage I.", context, n_stage_zero)
    return numeric.replace(0, 1)


def clean_group_value(value: Any) -> str | float:
    """Remove ordinary/full-width whitespace and normalize missing strings."""
    if pd.isna(value):
        return np.nan
    text = "".join(str(value).replace("　", " ").split())
    if not text or text.lower() in {"nan", "none", "null"}:
        return np.nan
    return text


def normalize_etiology(value: Any) -> str | float:
    text = clean_group_value(value)
    if pd.isna(text):
        return np.nan
    if "原发" in text:
        return "原发性"
    if "继发" in text:
        return "继发性"
    return np.nan


def normalize_pair_type(value: Any) -> str | float:
    text = clean_group_value(value)
    if pd.isna(text):
        return np.nan
    # Test the most specific expression first to prevent partial matching.
    if "双不同" in text or ("双" in text and "不同" in text):
        return "双不同"
    if "双同" in text or ("双" in text and ("相同" in text or "同" in text)):
        return "双同"
    if "单" in text and "双" not in text:
        return "单"
    return np.nan


def require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")


def read_subgroup_data(cfg: Config) -> pd.DataFrame:
    if not cfg.input_excel.is_file():
        raise FileNotFoundError(f"Input workbook not found: {cfg.input_excel}")
    logging.info("Reading subgroup data: %s", cfg.input_excel)
    df = pd.read_excel(cfg.input_excel)
    df.columns = df.columns.astype(str).str.strip()
    required = [cfg.label_column, cfg.id_column, cfg.center_column, cfg.etiology_column, cfg.pair_column]
    require_columns(df, required)

    df = df.copy()
    df.insert(0, "source_row", np.arange(2, len(df) + 2))
    df[cfg.label_column] = normalize_stage_labels(df[cfg.label_column], "Independent external cohort")
    df[cfg.id_column] = df[cfg.id_column].astype(str).str.strip()
    df = df.dropna(subset=[cfg.label_column]).copy()
    df[cfg.label_column] = df[cfg.label_column].astype(int)
    df = df[df[cfg.label_column].isin(cfg.labels)].copy()

    df[cfg.center_column] = df[cfg.center_column].astype(str).str.strip()
    available = sorted(df[cfg.center_column].unique().tolist())
    if cfg.external_center not in available:
        raise ValueError(
            f"Independent external center {cfg.external_center!r} was not found in "
            f"{cfg.center_column!r}. Available values: {available}."
        )
    df = df[df[cfg.center_column] == cfg.external_center].copy()
    if df.empty:
        raise ValueError("No eligible external-test observations remain after filtering.")

    missing_labels = sorted(set(cfg.labels) - set(df[cfg.label_column].unique()))
    if missing_labels:
        logging.warning("The external cohort does not contain label(s): %s", missing_labels)
    logging.info("Eligible rows=%d; label counts=%s", len(df), df[cfg.label_column].value_counts().sort_index().to_dict())
    return df.reset_index(drop=True)


def load_model_package(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Model package not found: {path}")
    try:
        return joblib.load(path)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"The model requires the unavailable package {exc.name!r}. Install the same "
            "dependencies used for model training before running subgroup analysis."
        ) from exc


def extract_estimator(package: Any) -> Any:
    if isinstance(package, dict):
        for key in ("model", "best_model", "estimator", "clf", "classifier", "pipeline"):
            if key in package:
                return extract_estimator(package[key])
    if hasattr(package, "predict_proba"):
        return package
    raise ValueError("The model package does not contain an estimator with predict_proba().")


def require_median_imputation(model: Any) -> None:
    """Require the locked pipeline to implement the manuscript's median imputation."""
    candidates = [model, *getattr(model, "named_steps", {}).values()]
    imputers = [candidate for candidate in candidates if isinstance(candidate, SimpleImputer)]
    if not imputers:
        raise ValueError("The locked model pipeline does not contain the required median SimpleImputer.")
    if any(imputer.strategy != "median" for imputer in imputers):
        raise ValueError("All imputers in the locked model pipeline must use strategy='median'.")


FEATURE_KEYS = (
    "selected_features",
    "selected_cols",
    "feature_cols",
    "feature_columns",
    "features",
    "feature_names",
    "selected_feature_names",
    "input_features",
    "x_cols",
    "X_cols",
)


def clean_feature_list(value: Any) -> list[str] | None:
    if isinstance(value, (list, tuple, np.ndarray, pd.Index)):
        result = [str(item).strip() for item in value]
        return result or None
    return None


def recursively_find_features(value: Any, depth: int = 3) -> list[str] | None:
    if depth < 0 or not isinstance(value, dict):
        return None
    for key in FEATURE_KEYS:
        if key in value:
            features = clean_feature_list(value[key])
            if features:
                return features
    for nested in value.values():
        features = recursively_find_features(nested, depth - 1)
        if features:
            return features
    return None


def extract_feature_names(package: Any, model: Any) -> list[str]:
    features = recursively_find_features(package)
    if features:
        return features
    for candidate in (model, *getattr(model, "named_steps", {}).values()):
        if hasattr(candidate, "feature_names_in_"):
            features = clean_feature_list(candidate.feature_names_in_)
            if features:
                return features
    raise ValueError(
        "Feature names were not found in the model package. Re-export the model "
        "with selected_features/feature_cols metadata."
    )


def find_package_labels(package: Any) -> list[int] | None:
    if not isinstance(package, dict):
        return None
    for key in ("labels", "label_order", "class_labels", "classes", "labels_sorted"):
        if key in package:
            try:
                values = [int(value) for value in package[key]]
            except (TypeError, ValueError):
                continue
            if values:
                return values
    for nested in package.values():
        labels = find_package_labels(nested)
        if labels:
            return labels
    return None


def model_classes(model: Any, n_columns: int) -> list[int]:
    candidates = [model, *getattr(model, "named_steps", {}).values()]
    for candidate in candidates:
        if hasattr(candidate, "classes_"):
            values = list(candidate.classes_)
            if len(values) == n_columns:
                return [int(value) for value in values]
    return list(range(n_columns))


def predict_aligned_probabilities(
    package: Any,
    model: Any,
    x: pd.DataFrame,
    labels: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    raw_probability = np.asarray(model.predict_proba(x), dtype=float)
    raw_classes = model_classes(model, raw_probability.shape[1])
    package_labels = find_package_labels(package) or list(labels)

    if set(raw_classes) == set(labels):
        probability_labels = raw_classes
    elif set(raw_classes) == set(range(len(package_labels))):
        probability_labels = [package_labels[index] for index in raw_classes]
    else:
        raise ValueError(
            f"Cannot align model classes {raw_classes} with requested labels {list(labels)}."
        )

    class_to_column = {label: index for index, label in enumerate(probability_labels)}
    missing = [label for label in labels if label not in class_to_column]
    if missing:
        raise ValueError(f"The fitted model has no probability column for label(s): {missing}")
    probability = np.column_stack([
        raw_probability[:, class_to_column[label]] for label in labels
    ])
    prediction = np.asarray(labels)[np.argmax(probability, axis=1)]
    return probability, prediction


def build_subgroups(df: pd.DataFrame, cfg: Config) -> list[tuple[str, str, pd.DataFrame]]:
    etiology = df[cfg.etiology_column].map(normalize_etiology)
    pair_type = df[cfg.pair_column].map(normalize_pair_type)
    limb_count = df.groupby(cfg.id_column)[cfg.id_column].transform("size")
    bilateral = limb_count >= 2
    logging.info("Normalized etiology counts=%s", etiology.value_counts(dropna=False).to_dict())
    logging.info("Normalized pair-type counts=%s", pair_type.value_counts(dropna=False).to_dict())

    subgroups: list[tuple[str, str, pd.DataFrame]] = []
    for level in ETIOLOGY_LEVELS:
        subgroups.append(("etiology", level, df[etiology == level].copy()))
    for level in PAIR_LEVELS:
        subgroups.append(("pair_type", level, df[bilateral & (pair_type == level)].copy()))
    return subgroups


def macro_auc_ovr(y_true: np.ndarray, probability: np.ndarray, labels: Sequence[int]) -> float:
    values: list[float] = []
    for index, label in enumerate(labels):
        binary = (y_true == label).astype(int)
        if len(np.unique(binary)) < 2:
            continue
        try:
            values.append(float(roc_auc_score(binary, probability[:, index])))
        except ValueError:
            continue
    return float(np.mean(values)) if values else np.nan


def point_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
    labels: Sequence[int],
) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, prediction, labels=labels)
    total = matrix.sum()
    true_positive = np.diag(matrix).astype(float)
    false_negative = matrix.sum(axis=1).astype(float) - true_positive
    false_positive = matrix.sum(axis=0).astype(float) - true_positive
    true_negative = total - true_positive - false_negative - false_positive

    def divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        return np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=float),
            where=denominator != 0,
        )

    return {
        "ACC": float(accuracy_score(y_true, prediction)),
        "AUC": macro_auc_ovr(y_true, probability, labels),
        "SENS": float(np.nanmean(divide(true_positive, true_positive + false_negative))),
        "SPEC": float(np.nanmean(divide(true_negative, true_negative + false_positive))),
        "PPV": float(np.nanmean(divide(true_positive, true_positive + false_positive))),
        "NPV": float(np.nanmean(divide(true_negative, true_negative + false_negative))),
        "confusion_matrix": matrix,
    }


def cluster_bootstrap_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    raw = np.asarray(groups, dtype=object)
    if pd.isna(raw).any():
        raise ValueError("Patient identifiers contain missing values.")
    groups = raw.astype(str)
    unique = pd.unique(groups)
    if len(unique) < 2:
        raise ValueError("At least two patient clusters are required for clustered bootstrap.")
    sampled = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([np.flatnonzero(groups == group) for group in sampled])


def bootstrap_intervals(
    y_true: np.ndarray,
    probability: np.ndarray,
    labels: Sequence[int],
    groups: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Patient-clustered percentile bootstrap for subgroup performance."""
    if repetitions <= 0:
        return {metric: (np.nan, np.nan) for metric in METRICS}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for _ in range(repetitions):
        index = cluster_bootstrap_indices(groups, rng)
        sampled_probability = probability[index]
        sampled_prediction = np.asarray(labels)[np.argmax(sampled_probability, axis=1)]
        metrics = point_metrics(y_true[index], sampled_probability, sampled_prediction, labels)
        for metric in METRICS:
            value = metrics[metric]
            if np.isfinite(value):
                samples[metric].append(float(value))
    return {
        metric: (
            float(np.percentile(values, 2.5)) if values else np.nan,
            float(np.percentile(values, 97.5)) if values else np.nan,
        )
        for metric, values in samples.items()
    }


def format_interval(point: float, low: float, high: float) -> str:
    if not np.isfinite(point):
        return "NA"
    if not np.isfinite(low) or not np.isfinite(high):
        return f"{point:.3f}"
    return f"{point:.3f} ({low:.3f}-{high:.3f})"


def subgroup_skip_reason(df: pd.DataFrame, cfg: Config) -> str | None:
    if len(df) < cfg.minimum_subgroup_size:
        return f"fewer than {cfg.minimum_subgroup_size} observations"
    if df[cfg.label_column].nunique() < 2:
        return "only one observed class"
    return None


def plot_roc_curves(items: list[dict[str, Any]], output: Path) -> None:
    if not items:
        return
    figure, axis = plt.subplots(figsize=(10, 7), dpi=150)
    for item in items:
        axis.plot(
            item["fpr"],
            item["tpr"],
            linewidth=1.4,
            label=f"{item['display']} | class {item['label']} (AUC={item['auc']:.3f})",
        )
    axis.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    axis.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="Subgroup ROC curves (one-vs-rest)")
    axis.legend(fontsize=7, loc="lower right", frameon=False)
    figure.tight_layout()
    figure.savefig(output / "subgroup_roc_ovr.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrices(
    items: list[dict[str, Any]],
    labels: Sequence[int],
    output: Path,
) -> None:
    if not items:
        return
    columns = min(3, len(items))
    rows = math.ceil(len(items) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5.4 * columns, 4.5 * rows), squeeze=False)
    for axis, item in zip(axes.flat, items):
        sns.heatmap(
            item["matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=[f"Pred {label}" for label in labels],
            yticklabels=[f"True {label}" for label in labels],
            ax=axis,
        )
        axis.set_title(f"{item['display']}\nN={item['n']}")
        axis.set_xlabel("")
        axis.set_ylabel("")
    for axis in list(axes.flat)[len(items) :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output / "subgroup_confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def evaluate_subgroups(
    df: pd.DataFrame,
    package: Any,
    model: Any,
    features: Sequence[str],
    cfg: Config,
) -> dict[str, pd.DataFrame | list[dict[str, Any]]]:
    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        raise ValueError(
            f"Subgroup data are missing {len(missing_features)} model feature(s): "
            + ", ".join(missing_features[:20])
        )

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []
    roc_items: list[dict[str, Any]] = []
    matrix_items: list[dict[str, Any]] = []

    for variable, level, part in build_subgroups(df, cfg):
        display = SUBGROUP_DISPLAY[(variable, level)]
        reason = subgroup_skip_reason(part, cfg)
        if reason:
            skipped.append({
                "subgroup_variable": variable,
                "subgroup_level": level,
                "display": display,
                "n": len(part),
                "label_distribution": json.dumps(part[cfg.label_column].value_counts().sort_index().to_dict()),
                "reason": reason,
            })
            continue

        x = part[list(features)].apply(pd.to_numeric, errors="coerce")
        missing_mask = x.isna().any(axis=1)
        rows_with_missing = int(missing_mask.sum())

        y_true = part[cfg.label_column].astype(int).to_numpy()
        probability, prediction = predict_aligned_probabilities(package, model, x, cfg.labels)
        point = point_metrics(y_true, probability, prediction, cfg.labels)
        groups = part[cfg.id_column].astype(str).to_numpy()
        intervals = bootstrap_intervals(
            y_true,
            probability,
            cfg.labels,
            groups,
            cfg.bootstrap_repetitions,
            cfg.random_state,
        )
        row: dict[str, Any] = {
            "subgroup_variable": variable,
            "subgroup_level": level,
            "display": display,
            "n": len(part),
            "rows_median_imputed_by_model": rows_with_missing,
            "label_distribution": json.dumps(part[cfg.label_column].value_counts().sort_index().to_dict()),
        }
        for metric in METRICS:
            low, high = intervals[metric]
            row[metric] = point[metric]
            row[f"{metric}_CI_low"] = low
            row[f"{metric}_CI_high"] = high
            row[f"{metric}_with_95CI"] = format_interval(point[metric], low, high)
        for i, true_label in enumerate(cfg.labels):
            for j, predicted_label in enumerate(cfg.labels):
                row[f"CM_true_{true_label}_pred_{predicted_label}"] = int(point["confusion_matrix"][i, j])
        metric_rows.append(row)

        identity_columns = [
            column for column in (
                "source_row", "序号", "肢体", cfg.center_column,
                cfg.label_column, cfg.etiology_column, cfg.pair_column,
            ) if column in part.columns
        ]
        predictions = part[identity_columns].copy()
        predictions["subgroup_variable"] = variable
        predictions["subgroup_level"] = level
        predictions["predicted_label"] = prediction
        predictions["correct"] = y_true == prediction
        for index, label in enumerate(cfg.labels):
            predictions[f"probability_label_{label}"] = probability[:, index]
        prediction_frames.append(predictions)

        matrix_items.append({"display": display, "n": len(part), "matrix": point["confusion_matrix"]})
        for index, label in enumerate(cfg.labels):
            binary = (y_true == label).astype(int)
            if len(np.unique(binary)) < 2:
                continue
            fpr, tpr, _ = roc_curve(binary, probability[:, index])
            roc_items.append({
                "display": display,
                "label": label,
                "auc": float(roc_auc_score(binary, probability[:, index])),
                "fpr": fpr,
                "tpr": tpr,
            })

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    errors = predictions[~predictions["correct"]].copy() if not predictions.empty else pd.DataFrame()
    return {
        "metrics": pd.DataFrame(metric_rows),
        "predictions": predictions,
        "errors": errors,
        "skipped": pd.DataFrame(skipped),
        "roc_items": roc_items,
        "matrix_items": matrix_items,
    }


def save_results(
    results: dict[str, Any],
    cfg: Config,
    features: Sequence[str],
    package: Any,
) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    workbook = cfg.output_dir / "multiclass_subgroup_analysis.xlsx"
    run_info = pd.DataFrame([
        {"key": "generated_at", "value": datetime.now().isoformat(timespec="seconds")},
        {"key": "input_excel", "value": str(cfg.input_excel)},
        {"key": "model_path", "value": str(cfg.model_path)},
        {"key": "model_task", "value": package.get("task", "") if isinstance(package, dict) else ""},
        {"key": "model_name", "value": package.get("model_name", "") if isinstance(package, dict) else ""},
        {"key": "feature_set", "value": package.get("feature_set", "") if isinstance(package, dict) else ""},
        {"key": "n_model_features", "value": len(features)},
        {"key": "class_labels", "value": json.dumps(cfg.labels)},
        {"key": "patient_id_column", "value": cfg.id_column},
        {"key": "center_filter", "value": cfg.external_center},
        {"key": "subgroups", "value": "etiology (primary/secondary); bilateral stage concordance (concordant/discordant)"},
        {"key": "bootstrap_repetitions", "value": cfg.bootstrap_repetitions},
        {"key": "random_state", "value": cfg.random_state},
        {"key": "minimum_subgroup_size", "value": cfg.minimum_subgroup_size},
        {"key": "missing_policy", "value": "median imputation in locked model pipeline; no complete-case row dropping"},
        {"key": "python", "value": platform.python_version()},
        {"key": "pandas", "value": pd.__version__},
        {"key": "numpy", "value": np.__version__},
        {"key": "scikit_learn", "value": sklearn.__version__},
    ])
    feature_table = pd.DataFrame({"rank": np.arange(1, len(features) + 1), "feature": features})
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        results["metrics"].to_excel(writer, sheet_name="subgroup_metrics", index=False)
        results["predictions"].to_excel(writer, sheet_name="all_predictions", index=False)
        results["errors"].to_excel(writer, sheet_name="misclassified_cases", index=False)
        results["skipped"].to_excel(writer, sheet_name="skipped_subgroups", index=False)
        feature_table.to_excel(writer, sheet_name="model_features", index=False)
        run_info.to_excel(writer, sheet_name="run_info", index=False)
    if cfg.make_plots:
        plot_roc_curves(results["roc_items"], cfg.output_dir)
        plot_confusion_matrices(results["matrix_items"], cfg.labels, cfg.output_dir)
    return workbook


def parse_labels(text: str) -> tuple[int, ...]:
    try:
        labels = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Labels must be comma-separated integers.") from exc
    if len(labels) < 3 or len(set(labels)) != len(labels):
        raise argparse.ArgumentTypeError("Provide at least three distinct class labels.")
    return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-excel", type=Path, required=True, help="Workbook containing external-test cases and subgroup columns.")
    parser.add_argument("--model", type=Path, required=True, help="Trusted fitted three-class model package (.joblib).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--label-column", default=DEFAULT_LABEL_COLUMN)
    parser.add_argument("--id-column", default=DEFAULT_ID_COLUMN, help="Patient identifier used for clustered bootstrap.")
    parser.add_argument("--center-column", default=DEFAULT_CENTER_COLUMN)
    parser.add_argument(
        "--external-center",
        default=DEFAULT_EXTERNAL_CENTER,
        help="Independent external-test center to retain; default: 中心4.",
    )
    parser.add_argument("--etiology-column", default=DEFAULT_ETIOLOGY_COLUMN)
    parser.add_argument("--pair-column", default=DEFAULT_PAIR_COLUMN)
    parser.add_argument("--labels", type=parse_labels, default=CLASS_LABELS, help="Comma-separated ordered labels; default: 1,2,3.")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Patient-clustered bootstrap repetitions; use 0 to omit confidence intervals.")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--minimum-subgroup-size", type=int, default=10)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--check-data", action="store_true", help="Validate model metadata, features, and subgroup sizes without prediction.")
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    if args.bootstrap < 0:
        raise ValueError("--bootstrap must be zero or a positive integer.")
    if args.minimum_subgroup_size < 1:
        raise ValueError("--minimum-subgroup-size must be positive.")
    cfg = Config(
        input_excel=args.input_excel.resolve(),
        model_path=args.model.resolve(),
        output_dir=args.output_dir.resolve(),
        label_column=str(args.label_column).strip(),
        id_column=str(args.id_column).strip(),
        center_column=str(args.center_column).strip(),
        external_center=str(args.external_center).strip(),
        etiology_column=str(args.etiology_column).strip(),
        pair_column=str(args.pair_column).strip(),
        labels=tuple(args.labels),
        bootstrap_repetitions=args.bootstrap,
        random_state=args.random_state,
        minimum_subgroup_size=args.minimum_subgroup_size,
        make_plots=not args.no_plots,
    )
    df = read_subgroup_data(cfg)
    package = load_model_package(cfg.model_path)
    model = extract_estimator(package)
    require_median_imputation(model)
    features = extract_feature_names(package, model)
    if any("diagnostics" in feature.lower() for feature in features):
        raise ValueError("The locked model contains forbidden PyRadiomics diagnostics columns.")
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError(f"Data are missing {len(missing)} model feature(s): {missing[:20]}")
    logging.info("Model features=%d", len(features))

    if args.check_data:
        summary = [
            {
                "subgroup_variable": variable,
                "subgroup_level": level,
                "n": len(part),
                "label_distribution": part[cfg.label_column].value_counts().sort_index().to_dict(),
            }
            for variable, level, part in build_subgroups(df, cfg)
        ]
        logging.info("Subgroup data check:\n%s", pd.DataFrame(summary).to_string(index=False))
        logging.info("Data and model metadata checks completed; no output was written.")
        return 0

    results = evaluate_subgroups(df, package, model, features, cfg)
    workbook = save_results(results, cfg, features, package)
    logging.info("Subgroup metrics:\n%s", results["metrics"].to_string(index=False))
    logging.info("Saved subgroup analysis to %s", workbook)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
