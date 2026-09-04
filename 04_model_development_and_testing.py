# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import logging
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "model_development_and_testing"

LABEL_COL = "标签"
ID_COL = "序号"
SIDE_COL = "肢体"
CENTER_COL = "对应中心"

# Manuscript cohort labels. The prospective temporal cohort is supplied separately.
DEFAULT_DEVELOPMENT_CENTERS = ("中心1", "中心2", "中心3")
DEFAULT_EXTERNAL_CENTER = "中心4"

RANDOM_STATE = 255
CV_FOLDS = 5
EXPECTED_RADIOMICS_FEATURES = 1116
FEATURE_SETS = ("Radiomics_only", "Morphology_only", "Combined")
BASE_MODELS = ("LR", "RF", "SVM")

MORPHOLOGY_FEATURES = (
    "leg_middle1of3_volume_mm3",
    "underskin_middle1of3_volume_mm3",
    "leg_max_csa_mm2",
    "underskin_area_at_leg_max_csa_mm2",
    "underskin_middle1of3_top_bottom_area_ratio",
    "underskin_muscle_middle1of3_volume_ratio",
    "underskin_bone_middle1of3_volume_ratio",
    "underskin_muscle_max_csa_ratio_at_legmax",
    "underskin_bone_max_csa_ratio_at_legmax",
)

RADIOMICS_KEYWORDS = (
    "original_",
    "wavelet",
    "log-sigma",
    "log_sigma",
    "log_",
    "square",
    "squareroot",
    "logarithm",
    "exponential",
    "gradient",
    "lbp",
    "firstorder",
    "glcm",
    "glrlm",
    "glszm",
    "gldm",
    "ngtdm",
)


@dataclass(frozen=True)
class AnalysisSpec:
    """Prespecified settings for each manuscript analysis."""

    kind: str
    task_name: str
    labels: tuple[int, ...]
    anova_top_k: int
    lasso_c: float
    lasso_solver: str
    lasso_max_iter: int
    bootstrap_repetitions: int
    second_split_seed: int


@dataclass(frozen=True)
class RunConfig:
    input_excel: Path
    output_dir: Path
    development_centers: tuple[str, ...]
    external_center: str
    prospective_excel: Path | None = None
    label_col: str = LABEL_COL
    id_col: str = ID_COL
    side_col: str = SIDE_COL
    center_col: str = CENTER_COL
    random_state: int = RANDOM_STATE
    cv_folds: int = CV_FOLDS
    bootstrap_override: int | None = None
    use_xgboost: bool = True


ANALYSES = (
    AnalysisSpec("binary", "stage_1_vs_2", (1, 2), 50, 0.05, "saga", 20000, 2000, 255),
    AnalysisSpec("binary", "stage_2_vs_3", (2, 3), 50, 0.05, "saga", 20000, 2000, 255),
    AnalysisSpec("multiclass", "stage_1_vs_2_vs_3", (1, 2, 3), 50, 0.05, "saga", 20000, 2000, 255),
)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def normalize_stage_labels(values: pd.Series, context: str) -> pd.Series:
    """Map manuscript stage 0 to stage I and reject non-integer stage values."""
    numeric = pd.to_numeric(values, errors="coerce")
    observed = numeric.dropna()
    non_integer = observed[~np.isclose(observed, np.round(observed))]
    if not non_integer.empty:
        raise ValueError(f"{context} contains non-integer stage values: {sorted(non_integer.unique().tolist())[:10]}")
    n_stage_zero = int((numeric == 0).sum())
    if n_stage_zero:
        logging.info("%s: mapped %d stage-0 observation(s) to stage I.", context, n_stage_zero)
    return numeric.replace(0, 1)


def is_shape_feature(column: str) -> bool:
    return re.search(r"(^|[_-])shape(?:2d)?([_-]|$)", str(column).lower()) is not None


def is_metadata_column(column: str, cfg: RunConfig) -> bool:
    if column in {cfg.label_col, cfg.id_col, cfg.side_col, cfg.center_col}:
        return True
    text = str(column).lower()
    keywords = (
        "label", "标签", "center", "中心", "patient", "患者", "姓名",
        "limb", "肢体", "side", "左右", "path", "路径", "file", "文件",
        "dicom", "dcm", "mask", "segmentation", "分割", "date", "日期",
    )
    return any(token in text for token in keywords)


def read_and_validate_data(cfg: RunConfig) -> pd.DataFrame:
    if not cfg.input_excel.is_file():
        raise FileNotFoundError(f"Input workbook not found: {cfg.input_excel}")
    logging.info("Reading %s", cfg.input_excel)
    df = pd.read_excel(cfg.input_excel)
    required = [cfg.label_col, cfg.center_col, cfg.id_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    df = df.copy()
    df[cfg.label_col] = normalize_stage_labels(df[cfg.label_col], "Retrospective cohort")
    df = df.dropna(subset=[cfg.label_col, cfg.center_col]).copy()
    df[cfg.label_col] = df[cfg.label_col].astype(int)
    df[cfg.center_col] = df[cfg.center_col].astype(str).str.strip()
    df[cfg.id_col] = df[cfg.id_col].astype(str).str.strip()

    centers = sorted(df[cfg.center_col].unique().tolist())
    requested_centers = (*cfg.development_centers, cfg.external_center)
    absent = [center for center in requested_centers if center not in centers]
    if absent:
        raise ValueError(
            "Requested center value(s) not found: "
            f"{absent}. Values present in {cfg.center_col!r}: {centers}. "
            "Update the workbook or pass --development-centers/--external-center."
        )
    if len(set(cfg.development_centers)) != len(cfg.development_centers):
        raise ValueError("Development center values must be unique.")
    if cfg.external_center in cfg.development_centers:
        raise ValueError("The external center cannot also be a development center.")

    logging.info("Rows=%d, columns=%d", len(df), df.shape[1])
    logging.info("Center x label counts:\n%s", pd.crosstab(df[cfg.center_col], df[cfg.label_col]))
    return df.reset_index(drop=True)


def read_additional_test_data(path: Path, cfg: RunConfig, name: str = "prospective_test") -> pd.DataFrame:
    """Read a locked-model test cohort from a separate workbook.

    The prospective temporal cohort is deliberately read separately so it can
    never enter feature selection, cross-validation, or model selection.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{name} workbook not found: {path}")
    df = pd.read_excel(path)
    required = [cfg.label_col, cfg.id_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required column(s): {missing}")
    df = df.copy()
    df[cfg.label_col] = normalize_stage_labels(df[cfg.label_col], name)
    df = df.dropna(subset=[cfg.label_col, cfg.id_col]).copy()
    df[cfg.label_col] = df[cfg.label_col].astype(int)
    df[cfg.id_col] = df[cfg.id_col].astype(str).str.strip()
    if cfg.center_col in df.columns:
        df[cfg.center_col] = df[cfg.center_col].astype(str).str.strip()
    logging.info("%s rows=%d; label counts=%s", name, len(df), df[cfg.label_col].value_counts().sort_index().to_dict())
    return df.reset_index(drop=True)


def detect_feature_pools(
    df: pd.DataFrame,
    spec: AnalysisSpec,
    cfg: RunConfig,
) -> dict[str, list[str]]:
    """Validate prespecified candidate predictors using development data only."""
    missing_morphology = [column for column in MORPHOLOGY_FEATURES if column not in df.columns]
    if missing_morphology:
        raise ValueError(f"Development data lack prespecified morphology feature(s): {missing_morphology}")
    morph = list(MORPHOLOGY_FEATURES)
    named_radiomics = [
        c for c in df.columns
        if not is_metadata_column(c, cfg)
        and c not in MORPHOLOGY_FEATURES
        and any(token in str(c).lower() for token in RADIOMICS_KEYWORDS)
        and "diagnostics" not in str(c).lower()
        and not is_shape_feature(c)
    ]
    radiomics: list[str] = []
    for column in named_radiomics:
        numeric = pd.to_numeric(df[column], errors="coerce")
        invalid = df[column].notna() & numeric.isna()
        if invalid.any():
            raise ValueError(f"Development radiomics feature {column!r} contains non-numeric values.")
        # Columns populated only in a held-out cohort remain all-missing in the
        # development slice and therefore cannot enter or alter this pool.
        if numeric.notna().any():
            radiomics.append(column)
    if len(radiomics) != EXPECTED_RADIOMICS_FEATURES:
        raise ValueError(
            f"Development data contain {len(radiomics)} eligible non-shape radiomics features; "
            f"the manuscript workflow requires exactly {EXPECTED_RADIOMICS_FEATURES}."
        )

    logging.info(
        "%s feature pools: radiomics=%d, morphology=%d",
        spec.task_name,
        len(radiomics),
        len(morph),
    )
    return {"radiomics": radiomics, "morphology": morph}


def group_strata(df: pd.DataFrame, spec: AnalysisSpec, cfg: RunConfig) -> pd.DataFrame:
    grouped = df.groupby(cfg.id_col, dropna=False)[cfg.label_col]
    table = grouped.agg(
        label_pattern=lambda values: "_".join(
            map(str, sorted(pd.Series(values).dropna().astype(int).unique().tolist()))
        ),
        dominant_label=lambda values: int(pd.Series(values).mode().iloc[0]),
    ).reset_index()

    if spec.kind == "binary":
        table["stratum"] = table["dominant_label"].astype(str)
    else:
        pattern_counts = table["label_pattern"].value_counts()
        if not pattern_counts.empty and int(pattern_counts.min()) >= 2:
            table["stratum"] = table["label_pattern"]
        else:
            dominant_counts = table["dominant_label"].value_counts()
            table["stratum"] = (
                table["dominant_label"].astype(str)
                if not dominant_counts.empty and int(dominant_counts.min()) >= 2
                else "all"
            )
    return table


def group_aware_holdout(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
    spec: AnalysisSpec,
    cfg: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = group_strata(df, spec, cfg)
    counts = table["stratum"].value_counts()
    stratify = table["stratum"] if len(counts) > 1 and int(counts.min()) >= 2 else None
    train_groups, test_groups = train_test_split(
        table[cfg.id_col].astype(str),
        test_size=test_size,
        stratify=stratify,
        random_state=seed,
    )
    train_set, test_set = set(train_groups), set(test_groups)
    if train_set & test_set:
        raise RuntimeError("Patient/group leakage occurred during holdout splitting.")
    keys = df[cfg.id_col].astype(str)
    return df[keys.isin(train_set)].copy(), df[keys.isin(test_set)].copy()


def split_cohorts(
    df: pd.DataFrame,
    spec: AnalysisSpec,
    cfg: RunConfig,
) -> dict[str, pd.DataFrame]:
    task_df = df[df[cfg.label_col].isin(spec.labels)].copy()
    development = task_df[task_df[cfg.center_col].isin(cfg.development_centers)].copy()
    external = task_df[task_df[cfg.center_col] == cfg.external_center].copy()
    if development.empty or external.empty:
        raise ValueError(f"{spec.task_name}: development or external cohort is empty.")
    for name, cohort in (("development", development), ("external", external)):
        absent = sorted(set(spec.labels) - set(cohort[cfg.label_col].unique()))
        if absent:
            raise ValueError(f"{spec.task_name}: {name} cohort lacks label(s) {absent}.")

    train, temporary = group_aware_holdout(
        development, 0.40, cfg.random_state, spec, cfg
    )
    validation, internal_test = group_aware_holdout(
        temporary, 0.50, spec.second_split_seed, spec, cfg
    )
    splits = {
        "train": train,
        "validation": validation,
        "internal_test": internal_test,
        "external_test": external,
    }
    for name, part in splits.items():
        part["__split__"] = name

    internal_names = ("train", "validation", "internal_test")
    group_sets = {name: set(splits[name][cfg.id_col].astype(str)) for name in internal_names}
    for i, left in enumerate(internal_names):
        for right in internal_names[i + 1 :]:
            overlap = group_sets[left] & group_sets[right]
            if overlap:
                raise RuntimeError(f"Group leakage between {left} and {right}: {len(overlap)}")

    logging.info(
        "%s split sizes: train=%d, validation=%d, internal_test=%d, external_test=%d",
        spec.task_name,
        *(len(splits[name]) for name in splits),
    )
    return {name: part.reset_index(drop=True) for name, part in splits.items()}


def numeric_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return df.reindex(columns=list(columns)).apply(pd.to_numeric, errors="coerce")


def impute_scale(df: pd.DataFrame, columns: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    x = numeric_frame(df, columns)
    valid = [c for c in x.columns if x[c].notna().any()]
    if not valid:
        return np.empty((len(df), 0)), []
    values = SimpleImputer(strategy="median").fit_transform(x[valid])
    return StandardScaler().fit_transform(values), valid


def pearson_filter_ordered(x: np.ndarray, columns: Sequence[str], threshold: float = 0.90) -> list[str]:
    columns = list(columns)
    if len(columns) <= 1:
        return columns
    corr = pd.DataFrame(x, columns=columns).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped = {c for c in upper.columns if bool((upper[c] > threshold).any())}
    return [c for c in columns if c not in dropped]


def pearson_filter_ranked(
    x: np.ndarray,
    columns: Sequence[str],
    scores: dict[str, float],
    threshold: float = 0.90,
) -> list[str]:
    columns = list(columns)
    if len(columns) <= 1:
        return columns
    frame = pd.DataFrame(x, columns=columns)
    corr = frame.corr().abs().fillna(0.0)
    ranked = sorted(columns, key=lambda c: (scores.get(c, 0.0), c), reverse=True)
    kept: list[str] = []
    for column in ranked:
        if not kept or float(corr.loc[column, kept].max()) <= threshold:
            kept.append(column)
    return kept


def record_selection(
    rows: list[dict[str, Any]],
    spec: AnalysisSpec,
    context: str,
    fold: int | None,
    feature_set: str,
    domain: str,
    step: str,
    columns: Iterable[str],
) -> None:
    columns = list(columns)
    for rank, column in enumerate(columns, start=1):
        rows.append({
            "task": spec.task_name,
            "context": context,
            "fold": fold,
            "feature_set": feature_set,
            "domain": domain,
            "step": step,
            "n_features": len(columns),
            "rank": rank,
            "feature": column,
        })


def select_radiomics(
    train_df: pd.DataFrame,
    y: np.ndarray,
    candidates: Sequence[str],
    spec: AnalysisSpec,
    records: list[dict[str, Any]],
    context: str,
    fold: int | None,
    feature_set: str,
) -> list[str]:
    scaled, columns = impute_scale(train_df, candidates)
    record_selection(records, spec, context, fold, feature_set, "radiomics", "00_candidates", columns)
    if not columns:
        return []

    variance = VarianceThreshold(1e-8)
    varied = variance.fit_transform(scaled)
    # NumPy indexing returns ``numpy.str_`` values.  Converting back to native
    # strings keeps scikit-learn feature-name validation happy when radiomics
    # and morphology columns are combined.
    varied_columns = [str(c) for c in np.asarray(columns)[variance.get_support()]]
    record_selection(records, spec, context, fold, feature_set, "radiomics", "01_variance", varied_columns)
    if not varied_columns:
        return []

    k = min(spec.anova_top_k, len(varied_columns))
    anova = SelectKBest(f_classif, k=k)
    anova_values = anova.fit_transform(varied, y)
    anova_columns = [str(c) for c in np.asarray(varied_columns)[anova.get_support()]]
    raw_scores = anova.scores_
    scores = {
        c: float(s) if np.isfinite(s) else 0.0
        for c, s in zip(varied_columns, raw_scores)
    }
    if spec.kind == "multiclass":
        anova_columns = sorted(anova_columns, key=lambda c: scores.get(c, 0.0), reverse=True)
        indices = [varied_columns.index(c) for c in anova_columns]
        anova_values = varied[:, indices]
        pearson_columns = pearson_filter_ranked(anova_values, anova_columns, scores)
    else:
        pearson_columns = pearson_filter_ordered(anova_values, anova_columns)
    record_selection(records, spec, context, fold, feature_set, "radiomics", "02_anova", anova_columns)
    record_selection(records, spec, context, fold, feature_set, "radiomics", "03_pearson", pearson_columns)
    if not pearson_columns:
        return []

    pearson_indices = [anova_columns.index(c) for c in pearson_columns]
    lasso_x = anova_values[:, pearson_indices]
    lasso_kwargs: dict[str, Any] = dict(
        penalty="l1",
        solver=spec.lasso_solver,
        C=spec.lasso_c,
        class_weight="balanced",
        max_iter=spec.lasso_max_iter,
        random_state=RANDOM_STATE,
    )
    lasso = LogisticRegression(**lasso_kwargs)
    lasso.fit(lasso_x, y)
    coefficients = np.asarray(lasso.coef_)
    if coefficients.ndim == 1:
        coefficients = coefficients.reshape(1, -1)
    keep = np.any(np.abs(coefficients) > 1e-8, axis=0)
    selected = [str(c) for c in np.asarray(pearson_columns)[keep]]
    if spec.kind == "multiclass":
        importance = np.mean(np.abs(coefficients), axis=0)
        weight = dict(zip(pearson_columns, importance))
        selected = sorted(selected, key=lambda c: weight.get(c, 0.0), reverse=True)
    if not selected:
        selected = pearson_columns[: min(10, len(pearson_columns))]
    record_selection(records, spec, context, fold, feature_set, "radiomics", "04_lasso", selected)
    return selected


def select_morphology(
    train_df: pd.DataFrame,
    y: np.ndarray,
    candidates: Sequence[str],
    spec: AnalysisSpec,
    records: list[dict[str, Any]],
    context: str,
    fold: int | None,
    feature_set: str,
) -> list[str]:
    scaled, columns = impute_scale(train_df, candidates)
    record_selection(records, spec, context, fold, feature_set, "morphology", "00_candidates", columns)
    if not columns:
        return []
    if spec.kind == "multiclass":
        raw_scores, _ = f_classif(scaled, y)
        scores = {c: float(s) if np.isfinite(s) else 0.0 for c, s in zip(columns, raw_scores)}
        selected = pearson_filter_ranked(scaled, columns, scores)
    else:
        selected = pearson_filter_ordered(scaled, columns)
    record_selection(records, spec, context, fold, feature_set, "morphology", "01_pearson", selected)
    return selected or columns


def select_features(
    train_df: pd.DataFrame,
    y: np.ndarray,
    pools: dict[str, list[str]],
    feature_set: str,
    spec: AnalysisSpec,
    records: list[dict[str, Any]],
    context: str,
    fold: int | None,
) -> list[str]:
    radiomics: list[str] = []
    morphology: list[str] = []
    if feature_set in {"Radiomics_only", "Combined"}:
        radiomics = select_radiomics(
            train_df, y, pools["radiomics"], spec, records, context, fold, feature_set
        )
    if feature_set in {"Morphology_only", "Combined"}:
        morphology = select_morphology(
            train_df, y, pools["morphology"], spec, records, context, fold, feature_set
        )
    selected = list(dict.fromkeys(radiomics + morphology))
    if not selected:
        raise RuntimeError(f"No features remained for {feature_set}.")
    return selected


def available_models(use_xgboost: bool) -> list[str]:
    models = list(BASE_MODELS)
    if use_xgboost:
        try:
            import xgboost  # noqa: F401
        except Exception as exc:
            logging.warning("XGBoost unavailable and will be skipped: %s", exc)
        else:
            models.append("XGB")
    return models


def build_model(name: str, spec: AnalysisSpec, seed: int) -> Any:
    if name == "LR":
        solver = "liblinear" if spec.kind == "binary" else "lbfgs"
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                penalty="l2",
                solver=solver,
                class_weight="balanced",
                max_iter=5000,
                random_state=seed,
            )),
        ])
    if name == "RF":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1 if spec.kind == "binary" else 2,
                class_weight="balanced" if spec.kind == "binary" else "balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            )),
        ])
    if name == "SVM":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=seed,
            )),
        ])
    if name == "XGB":
        from xgboost import XGBClassifier

        common = dict(
            max_depth=3,
            learning_rate=0.03,
            random_state=seed,
            n_jobs=-1,
        )
        if spec.kind == "binary":
            classifier = XGBClassifier(
                n_estimators=500,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                **common,
            )
        else:
            classifier = XGBClassifier(
                n_estimators=300,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                objective="multi:softprob",
                num_class=len(spec.labels),
                eval_metric="mlogloss",
                **common,
            )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ])
    raise ValueError(f"Unknown model: {name}")


def encode_labels(values: Sequence[int], labels: Sequence[int]) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(labels)}
    return np.asarray([mapping[int(value)] for value in values], dtype=int)


def predict_probabilities(model: Any, x: pd.DataFrame, n_classes: int) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=float)
    classes = np.asarray(model.classes_, dtype=int)
    aligned = np.zeros((len(x), n_classes), dtype=float)
    for source, class_index in enumerate(classes):
        aligned[:, int(class_index)] = raw[:, source]
    return aligned


def point_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probability: np.ndarray,
    spec: AnalysisSpec,
) -> dict[str, Any]:
    n_classes = len(spec.labels)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    result: dict[str, Any] = {
        "ACC": float(accuracy_score(y_true, y_pred)),
        "BAL_ACC": float(balanced_accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, average="binary" if n_classes == 2 else "macro", zero_division=0)),
        "CM": cm,
    }
    if n_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        result.update({
            "AUC": float(roc_auc_score(y_true, probability[:, 1])) if len(np.unique(y_true)) == 2 else np.nan,
            "SENS": float(tp / (tp + fn)) if tp + fn else np.nan,
            "SPEC": float(tn / (tn + fp)) if tn + fp else np.nan,
            "PPV": float(tp / (tp + fp)) if tp + fp else np.nan,
            "NPV": float(tn / (tn + fn)) if tn + fn else np.nan,
            "BRIER": float(np.mean((y_true - probability[:, 1]) ** 2)),
        })
    else:
        total = cm.sum()
        true_positive = np.diag(cm).astype(float)
        false_positive = cm.sum(axis=0) - true_positive
        false_negative = cm.sum(axis=1) - true_positive
        true_negative = total - true_positive - false_positive - false_negative
        divide = lambda a, b: np.divide(a, b, out=np.full_like(a, np.nan), where=b != 0)
        result.update({
            "AUC": float(roc_auc_score(
                label_binarize(y_true, classes=np.arange(n_classes)),
                probability,
                average="macro",
                multi_class="ovr",
            )) if len(np.unique(y_true)) == n_classes else np.nan,
            "SENS": float(np.nanmean(divide(true_positive, true_positive + false_negative))),
            "SPEC": float(np.nanmean(divide(true_negative, true_negative + false_positive))),
            "PPV": float(np.nanmean(divide(true_positive, true_positive + false_positive))),
            "NPV": float(np.nanmean(divide(true_negative, true_negative + false_negative))),
            "BRIER": float(np.mean(np.sum((label_binarize(y_true, classes=np.arange(n_classes)) - probability) ** 2, axis=1))),
        })
    return result


METRIC_NAMES = ("ACC", "AUC", "SENS", "SPEC", "PPV", "NPV", "F1", "BAL_ACC", "BRIER")


def cluster_bootstrap_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    raw = np.asarray(groups, dtype=object)
    if pd.isna(raw).any():
        raise ValueError("Patient/group identifiers contain missing values.")
    groups = raw.astype(str)
    unique = pd.unique(groups)
    if len(unique) < 2:
        raise ValueError("At least two patient clusters are required for clustered bootstrap.")
    sampled = rng.choice(unique, size=len(unique), replace=True)
    pieces = [np.flatnonzero(groups == group) for group in sampled]
    return np.concatenate(pieces)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    spec: AnalysisSpec,
    repetitions: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Patient-clustered percentile bootstrap confidence intervals."""
    if repetitions <= 0:
        return {metric: (np.nan, np.nan) for metric in METRIC_NAMES}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {metric: [] for metric in METRIC_NAMES}
    for _ in range(repetitions):
        index = cluster_bootstrap_indices(groups, rng)
        try:
            metrics = point_metrics(y_true[index], y_pred[index], probability[index], spec)
        except Exception:
            continue
        for metric in METRIC_NAMES:
            value = metrics.get(metric, np.nan)
            if np.isfinite(value):
                samples[metric].append(float(value))
    return {
        metric: (
            float(np.percentile(values, 2.5)) if values else np.nan,
            float(np.percentile(values, 97.5)) if values else np.nan,
        )
        for metric, values in samples.items()
    }


def cv_splits(dev: pd.DataFrame, y: np.ndarray, cfg: RunConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    minimum_class = int(pd.Series(y).value_counts().min())
    folds = min(cfg.cv_folds, minimum_class)
    if folds < 2:
        raise ValueError("At least two observations per class are required for cross-validation.")
    groups = dev[cfg.id_col].astype(str).to_numpy()
    if len(np.unique(groups)) < folds:
        raise ValueError(f"Only {len(np.unique(groups))} unique patients are available for {folds}-fold group CV.")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=cfg.random_state)
    splits = list(splitter.split(np.zeros((len(dev), 1)), y, groups))
    for fold, (train_index, valid_index) in enumerate(splits, start=1):
        train_groups = set(groups[train_index])
        valid_groups = set(groups[valid_index])
        overlap = train_groups & valid_groups
        if overlap:
            raise RuntimeError(f"Patient leakage in CV fold {fold}: {len(overlap)} overlapping patient(s).")
    return splits


def summarize_cv(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric in METRIC_NAMES:
        values = pd.to_numeric(pd.Series([r.get(metric) for r in rows]), errors="coerce").dropna()
        summary[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
        summary[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
    return summary


def cross_validate_candidate(
    dev: pd.DataFrame,
    pools: dict[str, list[str]],
    feature_set: str,
    model_name: str,
    spec: AnalysisSpec,
    cfg: RunConfig,
    selection_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y = encode_labels(dev[cfg.label_col], spec.labels)
    rows: list[dict[str, Any]] = []
    for fold, (train_index, valid_index) in enumerate(cv_splits(dev, y, cfg), start=1):
        fold_train = dev.iloc[train_index].copy()
        fold_valid = dev.iloc[valid_index].copy()
        y_train, y_valid = y[train_index], y[valid_index]
        selected = select_features(
            fold_train, y_train, pools, feature_set, spec, selection_records,
            "cv_training_fold_only", fold,
        )
        model = build_model(model_name, spec, cfg.random_state)
        model.fit(numeric_frame(fold_train, selected), y_train)
        probability = predict_probabilities(model, numeric_frame(fold_valid, selected), len(spec.labels))
        prediction = probability.argmax(axis=1)
        metrics = point_metrics(y_valid, prediction, probability, spec)
        rows.append({
            "task": spec.task_name,
            "feature_set": feature_set,
            "model": model_name,
            "fold": fold,
            "n_train": len(train_index),
            "n_validation": len(valid_index),
            "n_selected_features": len(selected),
            "selected_features": "; ".join(selected),
            **{metric: metrics[metric] for metric in METRIC_NAMES},
        })
    return {
        "task": spec.task_name,
        "analysis": spec.kind,
        "feature_set": feature_set,
        "model": model_name,
        "n_folds": len(rows),
        "mean_n_selected_features": float(np.mean([r["n_selected_features"] for r in rows])),
        **summarize_cv(rows),
    }, rows


def evaluate_model(
    model: Any,
    selected: Sequence[str],
    data: pd.DataFrame,
    split_name: str,
    feature_set: str,
    model_name: str,
    spec: AnalysisSpec,
    cfg: RunConfig,
    seed_offset: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y_true = encode_labels(data[cfg.label_col], spec.labels)
    probability = predict_probabilities(model, numeric_frame(data, selected), len(spec.labels))
    y_pred = probability.argmax(axis=1)
    point = point_metrics(y_true, y_pred, probability, spec)
    repetitions = cfg.bootstrap_override if cfg.bootstrap_override is not None else spec.bootstrap_repetitions
    groups = data[cfg.id_col].astype(str).to_numpy()
    interval = bootstrap_ci(
        y_true, y_pred, probability, groups, spec, repetitions, cfg.random_state + seed_offset
    )
    row: dict[str, Any] = {
        "task": spec.task_name,
        "analysis": spec.kind,
        "feature_set": feature_set,
        "model": model_name,
        "dataset": split_name,
        "n": len(data),
        "n_selected_features": len(selected),
    }
    for metric in METRIC_NAMES:
        row[metric] = point[metric]
        row[f"{metric}_CI_low"] = interval[metric][0]
        row[f"{metric}_CI_high"] = interval[metric][1]
    for i, true_label in enumerate(spec.labels):
        for j, predicted_label in enumerate(spec.labels):
            row[f"CM_true_{true_label}_pred_{predicted_label}"] = int(point["CM"][i, j])

    identity = [c for c in (cfg.id_col, cfg.side_col, cfg.center_col, cfg.label_col) if c in data]
    predictions = data[identity].copy()
    predictions["task"] = spec.task_name
    predictions["feature_set"] = feature_set
    predictions["model"] = model_name
    predictions["dataset"] = split_name
    predictions["predicted_label"] = [spec.labels[i] for i in y_pred]
    predictions["correct"] = y_true == y_pred
    for index, label in enumerate(spec.labels):
        predictions[f"probability_label_{label}"] = probability[:, index]
    return row, predictions


def choose_best(candidates: pd.DataFrame) -> pd.Series:
    ranked = candidates.copy()
    for column in ("AUC_mean", "ACC_mean", "PPV_mean"):
        ranked[column] = pd.to_numeric(ranked[column], errors="coerce").fillna(-np.inf)
    return ranked.sort_values(
        ["AUC_mean", "ACC_mean", "PPV_mean"], ascending=False, kind="stable"
    ).iloc[0]


def safe_sheet(writer: pd.ExcelWriter, frame: pd.DataFrame, name: str) -> None:
    frame.to_excel(writer, sheet_name=name[:31], index=False)


def run_analysis(
    df: pd.DataFrame,
    spec: AnalysisSpec,
    cfg: RunConfig,
    models: Sequence[str],
    check_only: bool,
    prospective_df: pd.DataFrame | None = None,
) -> dict[str, list[pd.DataFrame]]:
    logging.info("Starting %s", spec.task_name)
    splits = split_cohorts(df, spec, cfg)
    # Candidate-column eligibility is determined before CV from the modeling
    # development partition only (training + validation). Neither internal nor
    # external/prospective test values can affect the candidate feature pool.
    dev = pd.concat([splits["train"], splits["validation"]], ignore_index=True)
    pools = detect_feature_pools(dev, spec, cfg)
    if prospective_df is not None:
        prospective = prospective_df[prospective_df[cfg.label_col].isin(spec.labels)].copy()
        if prospective.empty:
            raise ValueError(f"{spec.task_name}: prospective temporal cohort is empty after label filtering.")
        missing_labels = sorted(set(spec.labels) - set(prospective[cfg.label_col].unique()))
        if missing_labels:
            raise ValueError(f"{spec.task_name}: prospective cohort lacks label(s) {missing_labels}.")
        prospective["__split__"] = "prospective_test"
        splits["prospective_test"] = prospective.reset_index(drop=True)
    split_table = pd.concat(splits.values(), ignore_index=True)
    split_table.insert(0, "task", spec.task_name)
    pool_table = pd.DataFrame(
        [{"task": spec.task_name, "domain": domain, "feature": feature}
         for domain, features in pools.items() for feature in features]
    )
    if check_only:
        counts = pd.crosstab(split_table["__split__"], split_table[cfg.label_col])
        logging.info("%s split x label:\n%s", spec.task_name, counts)
        return {"splits": [split_table], "pools": [pool_table]}

    selection_records: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for feature_set in FEATURE_SETS:
        for model_name in models:
            logging.info("CV: %s / %s / %s", spec.task_name, feature_set, model_name)
            try:
                summary, folds = cross_validate_candidate(
                    dev, pools, feature_set, model_name, spec, cfg, selection_records
                )
            except Exception as exc:
                logging.exception("Candidate failed: %s / %s", feature_set, model_name)
                candidate_rows.append({
                    "task": spec.task_name,
                    "analysis": spec.kind,
                    "feature_set": feature_set,
                    "model": model_name,
                    "error": str(exc),
                    "AUC_mean": np.nan,
                    "ACC_mean": np.nan,
                    "PPV_mean": np.nan,
                })
                continue
            candidate_rows.append(summary)
            fold_rows.extend(folds)

    candidates = pd.DataFrame(candidate_rows)
    successful = candidates[pd.to_numeric(candidates["AUC_mean"], errors="coerce").notna()].copy()
    if successful.empty:
        raise RuntimeError(f"All candidate models failed for {spec.task_name}.")
    global_best = choose_best(successful)
    best_by_set = pd.DataFrame([
        choose_best(successful[successful["feature_set"] == feature_set]).to_dict()
        for feature_set in FEATURE_SETS
        if not successful[successful["feature_set"] == feature_set].empty
    ])

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    model_rows: list[dict[str, Any]] = []
    used_feature_rows: list[dict[str, Any]] = []
    model_dir = cfg.output_dir / spec.kind / "models" / spec.task_name
    model_dir.mkdir(parents=True, exist_ok=True)

    for _, best in best_by_set.iterrows():
        feature_set = str(best["feature_set"])
        model_name = str(best["model"])
        y_dev = encode_labels(dev[cfg.label_col], spec.labels)
        selected = select_features(
            dev, y_dev, pools, feature_set, spec, selection_records,
            "final_training_plus_validation", None,
        )
        model = build_model(model_name, spec, cfg.random_state)
        model.fit(numeric_frame(dev, selected), y_dev)
        is_global = bool(
            feature_set == str(global_best["feature_set"])
            and model_name == str(global_best["model"])
        )
        for rank, feature in enumerate(selected, start=1):
            used_feature_rows.append({
                "task": spec.task_name,
                "feature_set": feature_set,
                "model": model_name,
                "is_global_best": is_global,
                "rank": rank,
                "domain": "morphology" if feature in pools["morphology"] else "radiomics",
                "feature": feature,
            })

        artifact = {
            "model": model,
            "selected_features": selected,
            "labels": list(spec.labels),
            "label_column": cfg.label_col,
            "id_column": cfg.id_col,
            "center_column": cfg.center_col,
            "development_centers": list(cfg.development_centers),
            "external_center": cfg.external_center,
            "task": spec.task_name,
            "analysis": spec.kind,
            "feature_set": feature_set,
            "model_name": model_name,
            "is_global_best": is_global,
            "selection_rule": "highest mean CV AUC, then mean ACC, then mean PPV",
        }
        artifact_path = model_dir / f"{feature_set}.joblib"
        joblib.dump(artifact, artifact_path)
        model_rows.append({
            "task": spec.task_name,
            "feature_set": feature_set,
            "model": model_name,
            "is_global_best": is_global,
            "n_features": len(selected),
            "model_path": str(artifact_path),
        })

        for offset, (split_name, part) in enumerate(splits.items(), start=1):
            metric_row, predictions = evaluate_model(
                model, selected, part, split_name, feature_set, model_name,
                spec, cfg, seed_offset=offset * 1000,
            )
            metric_row["is_global_best"] = is_global
            metric_row["cv_auc_mean"] = best["AUC_mean"]
            metric_row["cv_acc_mean"] = best["ACC_mean"]
            metric_row["cv_ppv_mean"] = best["PPV_mean"]
            metrics_rows.append(metric_row)
            prediction_frames.append(predictions)

    return {
        "candidates": [candidates],
        "folds": [pd.DataFrame(fold_rows)],
        "metrics": [pd.DataFrame(metrics_rows)],
        "predictions": prediction_frames,
        "selection": [pd.DataFrame(selection_records)],
        "used_features": [pd.DataFrame(used_feature_rows)],
        "models": [pd.DataFrame(model_rows)],
        "splits": [split_table],
        "pools": [pool_table],
    }


def concatenate(items: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [item for item in items if item is not None and not item.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def write_results(kind: str, results: dict[str, list[pd.DataFrame]], cfg: RunConfig) -> Path:
    output = cfg.output_dir / kind
    output.mkdir(parents=True, exist_ok=True)
    workbook = output / f"{kind}_model_development_and_testing.xlsx"
    run_info = pd.DataFrame([
        {"key": "generated_at", "value": datetime.now().isoformat(timespec="seconds")},
        {"key": "script", "value": str(Path(__file__).resolve())},
        {"key": "input_excel", "value": str(cfg.input_excel)},
        {"key": "development_centers", "value": json.dumps(cfg.development_centers, ensure_ascii=False)},
        {"key": "external_test_center", "value": cfg.external_center},
        {"key": "prospective_excel", "value": str(cfg.prospective_excel) if cfg.prospective_excel else "not supplied"},
        {"key": "split", "value": "60% train / 20% validation / 20% internal test"},
        {"key": "cv", "value": f"{cfg.cv_folds}-fold on Training + Validation"},
        {"key": "random_state", "value": cfg.random_state},
        {"key": "python", "value": platform.python_version()},
        {"key": "pandas", "value": pd.__version__},
        {"key": "numpy", "value": np.__version__},
        {"key": "scikit_learn", "value": sklearn.__version__},
    ])
    sheet_map = {
        "metrics": "test_metrics",
        "candidates": "candidate_cv_summary",
        "folds": "candidate_cv_folds",
        "used_features": "final_selected_features",
        "selection": "feature_selection_trace",
        "models": "saved_models",
        "predictions": "case_predictions",
        "splits": "cohort_assignments",
        "pools": "detected_feature_pools",
    }
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for key, sheet_name in sheet_map.items():
            safe_sheet(writer, concatenate(results.get(key, [])), sheet_name)
        safe_sheet(writer, run_info, "run_info")
    return workbook


def parse_centers(value: str) -> tuple[str, ...]:
    centers = tuple(item.strip() for item in value.split(",") if item.strip())
    if not centers or len(set(centers)) != len(centers):
        raise argparse.ArgumentTypeError("Development centers must be distinct comma-separated values.")
    return centers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("binary", "multiclass", "all"), default="all")
    parser.add_argument("--input-excel", type=Path, required=True, help="Retrospective workbook containing Centers 1-4.")
    parser.add_argument("--prospective-excel", type=Path, default=None, help="Optional prospective temporal test workbook. Never used for model selection.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--development-centers",
        type=parse_centers,
        default=DEFAULT_DEVELOPMENT_CENTERS,
        help="Comma-separated development centers; default: 中心1,中心2,中心3.",
    )
    parser.add_argument("--external-center", default=DEFAULT_EXTERNAL_CENTER)
    parser.add_argument("--bootstrap", type=int, default=None, help="Override bootstrap repetitions; use 0 to skip CIs.")
    parser.add_argument("--no-xgboost", action="store_true", help="Use LR, RF, and SVM only.")
    parser.add_argument("--check-data", action="store_true", help="Validate cohorts and splits without fitting models.")
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    if args.bootstrap is not None and args.bootstrap < 0:
        raise ValueError("--bootstrap must be zero or a positive integer.")
    cfg = RunConfig(
        input_excel=args.input_excel.resolve(),
        output_dir=args.output_dir.resolve(),
        development_centers=tuple(args.development_centers),
        external_center=str(args.external_center).strip(),
        prospective_excel=args.prospective_excel.resolve() if args.prospective_excel is not None else None,
        bootstrap_override=args.bootstrap,
        use_xgboost=not args.no_xgboost,
    )
    logging.info("Configuration:\n%s", json.dumps({
        "task": args.task,
        "input_excel": str(cfg.input_excel),
        "output_dir": str(cfg.output_dir),
        "development_centers": list(cfg.development_centers),
        "external_test_center": cfg.external_center,
        "prospective_excel": str(cfg.prospective_excel) if cfg.prospective_excel else None,
    }, ensure_ascii=False, indent=2))
    df = read_and_validate_data(cfg)
    prospective_df = read_additional_test_data(cfg.prospective_excel, cfg) if cfg.prospective_excel is not None else None
    models = available_models(cfg.use_xgboost)
    requested = [spec for spec in ANALYSES if args.task in {"all", spec.kind}]
    by_kind: dict[str, dict[str, list[pd.DataFrame]]] = {}
    for spec in requested:
        result = run_analysis(df, spec, cfg, models, args.check_data, prospective_df=prospective_df)
        destination = by_kind.setdefault(spec.kind, {})
        for key, frames in result.items():
            destination.setdefault(key, []).extend(frames)

    if args.check_data:
        logging.info("Data check completed; no model or result file was written.")
        return 0
    for kind, result in by_kind.items():
        workbook = write_results(kind, result, cfg)
        logging.info("Saved %s results to %s", kind, workbook)
    logging.info("Model development and testing completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
