# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("qlympstage.reproducibility")
ICC_MODEL = "ICC(2,1): two-way random-effects, absolute agreement, single measurement"
MORPHOLOGY_FEATURES = {
    "leg_middle1of3_volume_mm3",
    "underskin_middle1of3_volume_mm3",
    "leg_max_csa_mm2",
    "underskin_area_at_leg_max_csa_mm2",
    "underskin_middle1of3_top_bottom_area_ratio",
    "underskin_muscle_middle1of3_volume_ratio",
    "underskin_bone_middle1of3_volume_ratio",
    "underskin_muscle_max_csa_ratio_at_legmax",
    "underskin_bone_max_csa_ratio_at_legmax",
}


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Feature tables must be CSV/XLS/XLSX.")


def icc_two_way(values: np.ndarray) -> float:
    """ICC(2,1) for complete n x k repeated-measure data."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        return np.nan
    n, k = values.shape
    grand = values.mean()
    row_means = values.mean(axis=1)
    col_means = values.mean(axis=0)
    ss_rows = k * np.sum((row_means - grand) ** 2)
    ss_cols = n * np.sum((col_means - grand) ** 2)
    ss_total = np.sum((values - grand) ** 2)
    ss_error = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    return float((ms_rows - ms_error) / denominator) if denominator != 0 else np.nan


def feature_iccs(first: pd.DataFrame, second: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    merged = first.merge(second, on=keys, suffixes=("__r1", "__r2"), how="inner", validate="one_to_one")
    first_features = {c[:-4] for c in merged.columns if c.endswith("__r1")}
    second_features = {c[:-4] for c in merged.columns if c.endswith("__r2")}
    candidates = sorted(first_features & second_features)
    rows: list[dict[str, object]] = []
    for feature in candidates:
        if feature in keys:
            continue
        a = pd.to_numeric(merged[f"{feature}__r1"], errors="coerce")
        b = pd.to_numeric(merged[f"{feature}__r2"], errors="coerce")
        valid = a.notna() & b.notna()
        if valid.sum() < 2:
            continue
        value = icc_two_way(np.column_stack([a[valid].to_numpy(), b[valid].to_numpy()]))
        domain = "morphology" if feature in MORPHOLOGY_FEATURES else "radiomics"
        # Do not count obvious file/path/identifier columns as quantitative features.
        lower = feature.lower()
        if any(token in lower for token in ("file", "path", "case_id", "patient", "label", "center", "side", "diagnostics")):
            continue
        rows.append({"feature": feature, "domain": domain, "n_pairs": int(valid.sum()), "icc": value, "icc_model": ICC_MODEL})
    return pd.DataFrame(rows)


def nifti_stem(path: Path) -> str:
    lower = path.name.lower()
    if lower.endswith(".nii.gz"):
        return path.name[:-7]
    if lower.endswith(".nii"):
        return path.name[:-4]
    raise ValueError(path)


def iter_nifti(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.name.lower().endswith((".nii", ".nii.gz")):
            yield path


def mask_index(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in iter_nifti(root):
        key = nifti_stem(path).casefold()
        if key in result:
            raise ValueError(f"Duplicate segmentation stem: {key}")
        result[key] = path
    return result


def binary_image(path: Path):
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("SimpleITK==2.5.2 is required when segmentation-mask Dice is requested.") from exc
    return sitk.Cast(sitk.ReadImage(str(path)) > 0, sitk.sitkUInt8)


def same_geometry(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-6)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-6)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-6)
    )


def dice_score(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    denom = int(a.sum() + b.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def segmentation_dice(first_dir: Path, second_dir: Path) -> pd.DataFrame:
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("SimpleITK==2.5.2 is required when segmentation-mask Dice is requested.") from exc
    first = mask_index(first_dir)
    second = mask_index(second_dir)
    common = sorted(set(first) & set(second))
    rows = []
    for key in common:
        a = binary_image(first[key])
        b = binary_image(second[key])
        if not same_geometry(a, b):
            b = sitk.Resample(b, a, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        score = dice_score(sitk.GetArrayViewFromImage(a), sitk.GetArrayViewFromImage(b))
        rows.append({"case_id": key, "dice": score, "repeat1": str(first[key]), "repeat2": str(second[key])})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-repeat1", type=Path, required=True)
    parser.add_argument("--features-repeat2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--id-column", default="序号")
    parser.add_argument("--side-column", default="肢体")
    parser.add_argument("--masks-repeat1", type=Path, default=None)
    parser.add_argument("--masks-repeat2", type=Path, default=None)
    parser.add_argument("--icc-threshold", type=float, default=0.80)
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    first = read_table(args.features_repeat1.resolve())
    second = read_table(args.features_repeat2.resolve())
    keys = [args.id_column]
    if args.side_column in first.columns and args.side_column in second.columns:
        keys.append(args.side_column)
    for key in keys:
        if key not in first.columns or key not in second.columns:
            raise ValueError(f"Missing pairing key: {key}")
    for frame in (first, second):
        for key in keys:
            frame[key] = frame[key].astype(str).str.strip()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    iccs = feature_iccs(first, second, keys)
    iccs.to_csv(output / "feature_icc.csv", index=False, encoding="utf-8-sig")

    summary_rows: list[dict[str, object]] = []
    for domain, part in iccs.groupby("domain", dropna=False):
        vals = pd.to_numeric(part["icc"], errors="coerce").dropna()
        summary_rows.append({
            "domain": domain,
            "n_features": int(len(vals)),
            "mean_icc": float(vals.mean()) if len(vals) else np.nan,
            "sd_icc": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
            "n_above_threshold": int((vals > args.icc_threshold).sum()),
            "percent_above_threshold": float((vals > args.icc_threshold).mean() * 100) if len(vals) else np.nan,
            "threshold": args.icc_threshold,
            "icc_model": ICC_MODEL,
        })

    if (args.masks_repeat1 is None) ^ (args.masks_repeat2 is None):
        raise ValueError("Provide both --masks-repeat1 and --masks-repeat2, or neither.")
    if args.masks_repeat1 is not None:
        dice = segmentation_dice(args.masks_repeat1.resolve(), args.masks_repeat2.resolve())
        dice.to_csv(output / "segmentation_dice.csv", index=False, encoding="utf-8-sig")
        if not dice.empty:
            summary_rows.append({
                "domain": "segmentation",
                "n_features": int(len(dice)),
                "mean_icc": np.nan,
                "sd_icc": np.nan,
                "n_above_threshold": np.nan,
                "percent_above_threshold": np.nan,
                "threshold": np.nan,
                "icc_model": "not applicable",
                "mean_dice": float(dice["dice"].mean()),
                "sd_dice": float(dice["dice"].std(ddof=1)) if len(dice) > 1 else np.nan,
            })

    pd.DataFrame(summary_rows).to_csv(output / "reproducibility_summary.csv", index=False, encoding="utf-8-sig")
    LOGGER.info("Reproducibility analysis saved to %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
