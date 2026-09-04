# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOGGER = logging.getLogger("qlympstage.shap")
MANUSCRIPT_REPRESENTATIVE_FEATURES = (
    "wavelet-LLH_ngtdm_Coarseness",
    "underskin_bone_middle1of3_volume_ratio",
)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")


def normalize_stage_labels(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    observed = numeric.dropna()
    non_integer = observed[~np.isclose(observed, np.round(observed))]
    if not non_integer.empty:
        raise ValueError(f"Input contains non-integer stage values: {sorted(non_integer.unique().tolist())[:10]}")
    n_stage_zero = int((numeric == 0).sum())
    if n_stage_zero:
        LOGGER.info("Mapped %d stage-0 observation(s) to stage I.", n_stage_zero)
    return numeric.replace(0, 1)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input must be CSV/XLS/XLSX.")


def extract_model(package: Any) -> Any:
    if isinstance(package, dict):
        for key in ("model", "best_model", "pipeline", "estimator"):
            if key in package:
                return extract_model(package[key])
    if hasattr(package, "predict_proba"):
        return package
    raise ValueError("No probabilistic model was found in the saved package.")


def features_from_package(package: Any, model: Any) -> list[str]:
    if isinstance(package, dict):
        for key in ("selected_features", "feature_columns", "features"):
            value = package.get(key)
            if isinstance(value, (list, tuple, np.ndarray, pd.Index)) and len(value):
                return [str(v) for v in value]
    if hasattr(model, "feature_names_in_"):
        return [str(v) for v in model.feature_names_in_]
    raise ValueError("Selected feature names are not available.")


def prepare_estimator(model: Any, x: pd.DataFrame) -> tuple[Any, np.ndarray]:
    """Return final estimator and preprocessed numeric matrix."""
    if hasattr(model, "steps") and len(model.steps) >= 2:
        preprocess = model[:-1]
        estimator = model.steps[-1][1]
        values = preprocess.transform(x)
    else:
        estimator = model
        values = x.to_numpy(dtype=float)
    if hasattr(values, "toarray"):
        values = values.toarray()
    return estimator, np.asarray(values, dtype=float)


def normalize_shap(values: Any, n_features: int) -> np.ndarray:
    """Normalize SHAP values to shape (n_samples, n_features, n_outputs)."""
    if isinstance(values, list):
        arrays = [np.asarray(v, dtype=float) for v in values]
        return np.stack(arrays, axis=2)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2:
        if arr.shape[1] != n_features:
            raise ValueError(f"Unexpected SHAP shape: {arr.shape}")
        return arr[:, :, None]
    if arr.ndim == 3:
        if arr.shape[1] == n_features:
            return arr
        if arr.shape[2] == n_features:
            return np.transpose(arr, (0, 2, 1))
    raise ValueError(f"Unsupported SHAP value shape: {arr.shape}")


def stage_labels(package: Any, estimator: Any, n_outputs: int, requested: Sequence[int]) -> list[str]:
    if isinstance(package, dict) and package.get("labels") is not None:
        labels = [str(v) for v in package["labels"]]
    elif hasattr(estimator, "classes_"):
        labels = [str(v) for v in estimator.classes_]
    else:
        labels = [str(v) for v in requested]
    if n_outputs == 1:
        return ["overall"]
    if len(labels) == n_outputs:
        return labels
    return [f"output_{i}" for i in range(n_outputs)]


def compute_explanation(estimator: Any, x: np.ndarray, background: np.ndarray):
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is required. Install shap==0.50.0 to match the manuscript environment.") from exc

    name = estimator.__class__.__name__.lower()
    if "xgb" in name or "forest" in name or "tree" in name:
        explainer = shap.TreeExplainer(estimator)
        return explainer(x)
    if "logistic" in name or "linear" in name:
        explainer = shap.LinearExplainer(estimator, background)
        return explainer(x)
    # SVM and other probability models: use a model-agnostic probability explainer.
    explainer = shap.Explainer(estimator.predict_proba, background)
    return explainer(x)


def importance_tables(values: np.ndarray, features: list[str], labels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_values = np.mean(np.abs(values), axis=(0, 2))
    global_table = pd.DataFrame({"feature": features, "mean_abs_shap": global_values}).sort_values("mean_abs_shap", ascending=False)
    stage_rows = []
    for output_index, label in enumerate(labels):
        scores = np.mean(np.abs(values[:, :, output_index]), axis=0)
        for feature, score in zip(features, scores):
            stage_rows.append({"stage": label, "feature": feature, "mean_abs_shap": float(score)})
    stage_table = pd.DataFrame(stage_rows).sort_values(["stage", "mean_abs_shap"], ascending=[True, False])
    return global_table, stage_table


def plot_global(table: pd.DataFrame, output: Path, top_n: int) -> None:
    part = table.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(part) + 1)), dpi=160)
    ax.barh(part["feature"], part["mean_abs_shap"])
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("")
    ax.set_title("Global SHAP feature importance")
    fig.tight_layout()
    fig.savefig(output / "shap_global_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def filename_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def display_stage(value: object) -> str:
    return {"1": "I", "2": "II", "3": "III"}.get(str(value), str(value))


def plot_representative(df: pd.DataFrame, label_col: str, feature: str, output: Path) -> None:
    data = df[[label_col, feature]].copy()
    data[feature] = pd.to_numeric(data[feature], errors="coerce")
    data = data.dropna()
    groups = sorted(data[label_col].unique())
    arrays = [data.loc[data[label_col] == group, feature].to_numpy() for group in groups]
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    ax.boxplot(arrays, tick_labels=[display_stage(group) for group in groups], showfliers=False)
    rng = np.random.default_rng(255)
    for i, values in enumerate(arrays, start=1):
        jitter = rng.normal(i, 0.035, size=len(values))
        ax.scatter(jitter, values, s=9, alpha=0.55)
    ax.set_xlabel("ISL stage")
    ax.set_ylabel(feature)
    ax.set_title(f"{feature} by ISL stage")
    fig.tight_layout()
    fig.savefig(output / f"representative_{filename_token(feature)}_by_stage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_stage_summaries(
    values: np.ndarray,
    feature_values: np.ndarray,
    features: list[str],
    labels: list[str],
    output: Path,
    top_n: int,
) -> None:
    """Create the manuscript Figure 6D-F one-vs-rest SHAP beeswarm panels."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is required. Install shap==0.50.0 to match the manuscript environment.") from exc
    for output_index, label in enumerate(labels):
        plt.figure(figsize=(9, max(5, 0.34 * min(top_n, len(features)) + 1)), dpi=160)
        shap.summary_plot(
            values[:, :, output_index],
            feature_values,
            feature_names=features,
            max_display=top_n,
            plot_type="dot",
            show=False,
        )
        stage = display_stage(label)
        plt.title(f"ISL stage {stage}: one-vs-rest SHAP summary")
        plt.tight_layout()
        plt.savefig(output / f"shap_summary_stage_{filename_token(stage)}.png", dpi=300, bbox_inches="tight")
        plt.close()


def parse_labels(value: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-column", default="标签")
    parser.add_argument("--labels", type=parse_labels, default=(1, 2, 3))
    parser.add_argument("--center-column", default="对应中心")
    parser.add_argument("--center", default=None)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--background-samples", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--representative-feature", default=None)
    parser.add_argument("--random-state", type=int, default=255)
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    df = read_table(args.input.resolve())
    if args.label_column not in df.columns:
        raise ValueError(f"Missing label column: {args.label_column}")
    if args.center is not None:
        if args.center_column not in df.columns:
            raise ValueError(f"Missing center column: {args.center_column}")
        df = df[df[args.center_column].astype(str).str.strip() == str(args.center).strip()].copy()
    df[args.label_column] = normalize_stage_labels(df[args.label_column])
    df = df[df[args.label_column].isin(args.labels)].copy().reset_index(drop=True)

    package = joblib.load(args.model.resolve())
    model = extract_model(package)
    features = features_from_package(package, model)
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing selected model features: {missing[:20]}")
    x_frame = df[features].apply(pd.to_numeric, errors="coerce")
    estimator, x = prepare_estimator(model, x_frame)

    rng = np.random.default_rng(args.random_state)
    if len(x) > args.max_samples:
        eval_idx = np.sort(rng.choice(len(x), size=args.max_samples, replace=False))
    else:
        eval_idx = np.arange(len(x))
    if len(x) > args.background_samples:
        bg_idx = np.sort(rng.choice(len(x), size=args.background_samples, replace=False))
    else:
        bg_idx = np.arange(len(x))

    explanation = compute_explanation(estimator, x[eval_idx], x[bg_idx])
    values = normalize_shap(explanation.values, len(features))
    labels = stage_labels(package, estimator, values.shape[2], args.labels)
    global_table, stage_table = importance_tables(values, features, labels)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    global_table.to_csv(output / "shap_global_importance.csv", index=False, encoding="utf-8-sig")
    stage_table.to_csv(output / "shap_stage_specific_importance.csv", index=False, encoding="utf-8-sig")
    plot_global(global_table, output, args.top_n)
    plot_stage_summaries(values, x[eval_idx], features, labels, output, args.top_n)
    representatives = list(MANUSCRIPT_REPRESENTATIVE_FEATURES)
    if args.representative_feature and args.representative_feature not in representatives:
        representatives.append(args.representative_feature)
    missing_representatives = [feature for feature in representatives if feature not in df.columns]
    if missing_representatives:
        raise ValueError(f"Manuscript representative feature(s) not found: {missing_representatives}")
    for representative in representatives:
        plot_representative(df, args.label_column, representative, output)
    LOGGER.info("SHAP analysis complete: %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
