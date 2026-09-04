# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import logging
import math
from pathlib import Path
from typing import Iterable

import SimpleITK as sitk


LOGGER = logging.getLogger("qlympstage.preprocess")
TARGET_SPACING = (1.0, 1.0, 1.0)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def nifti_stem(path: Path) -> str:
    name = path.name
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return name[:-7]
    if lower.endswith(".nii"):
        return name[:-4]
    raise ValueError(f"Not a NIfTI file: {path}")


def iter_nifti(root: Path, recursive: bool) -> Iterable[Path]:
    iterator = root.rglob("*") if recursive else root.iterdir()
    for path in iterator:
        if path.is_file() and path.name.lower().endswith((".nii", ".nii.gz")):
            yield path


def build_index(root: Path, recursive: bool) -> dict[str, Path]:
    if not root.is_dir():
        raise NotADirectoryError(root)
    index: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in sorted(iter_nifti(root, recursive), key=lambda p: str(p).lower()):
        key = nifti_stem(path).strip().casefold()
        if key in index:
            duplicates.append(key)
        else:
            index[key] = path
    if duplicates:
        raise ValueError(f"Duplicate NIfTI stems found: {sorted(set(duplicates))[:10]}")
    return index


def output_size(image: sitk.Image, spacing: tuple[float, float, float]) -> list[int]:
    old_size = image.GetSize()
    old_spacing = image.GetSpacing()
    return [
        max(1, int(round((old_size[i] - 1) * old_spacing[i] / spacing[i])) + 1)
        for i in range(3)
    ]


def resample(
    image: sitk.Image,
    spacing: tuple[float, float, float],
    interpolator: int,
    default_value: float,
    output_pixel_type: int | None = None,
) -> sitk.Image:
    filt = sitk.ResampleImageFilter()
    filt.SetOutputSpacing(spacing)
    filt.SetSize(output_size(image, spacing))
    filt.SetOutputOrigin(image.GetOrigin())
    filt.SetOutputDirection(image.GetDirection())
    filt.SetTransform(sitk.Transform())
    filt.SetInterpolator(interpolator)
    filt.SetDefaultPixelValue(float(default_value))
    if output_pixel_type is not None:
        filt.SetOutputPixelType(output_pixel_type)
    return filt.Execute(image)


def geometry_equal(a: sitk.Image, b: sitk.Image, tol: float = 1e-6) -> bool:
    if a.GetSize() != b.GetSize():
        return False
    for left, right in zip(a.GetSpacing(), b.GetSpacing()):
        if not math.isclose(left, right, rel_tol=0.0, abs_tol=tol):
            return False
    for left, right in zip(a.GetOrigin(), b.GetOrigin()):
        if not math.isclose(left, right, rel_tol=0.0, abs_tol=tol):
            return False
    for left, right in zip(a.GetDirection(), b.GetDirection()):
        if not math.isclose(left, right, rel_tol=0.0, abs_tol=tol):
            return False
    return True


def resample_mask_to_reference(mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
    return sitk.Resample(
        mask,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-image-dir", type=Path, required=True)
    parser.add_argument("--output-mask-dir", type=Path, required=True)
    parser.add_argument("--spacing", type=float, nargs=3, default=TARGET_SPACING, metavar=("SX", "SY", "SZ"))
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--strict-pairs", action="store_true", help="Fail if an image or mask is unmatched.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    spacing = tuple(float(v) for v in args.spacing)
    if any(v <= 0 for v in spacing):
        raise ValueError("Target spacing values must be positive.")

    image_index = build_index(args.image_dir.resolve(), args.recursive)
    mask_index = build_index(args.mask_dir.resolve(), args.recursive)
    common = sorted(set(image_index) & set(mask_index))
    unmatched = sorted(set(image_index) ^ set(mask_index))
    if args.strict_pairs and unmatched:
        raise ValueError(f"Unmatched image/mask stems: {unmatched[:20]}")
    if not common:
        raise ValueError("No matched image/mask pairs were found.")

    out_images = args.output_image_dir.resolve()
    out_masks = args.output_mask_dir.resolve()
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    report_rows: list[dict[str, object]] = []

    for key in common:
        image_path = image_index[key]
        mask_path = mask_index[key]
        output_image = out_images / f"{nifti_stem(image_path)}.nii.gz"
        output_mask = out_masks / f"{nifti_stem(mask_path)}.nii.gz"
        if not args.overwrite and output_image.exists() and output_mask.exists():
            LOGGER.info("Skip existing: %s", key)
            continue

        image = sitk.ReadImage(str(image_path))
        mask = sitk.ReadImage(str(mask_path))
        if image.GetDimension() != 3 or mask.GetDimension() != 3:
            raise ValueError(f"Only 3-D images are supported: {key}")

        # The mask is first harmonized to the original image geometry if needed.
        if not geometry_equal(image, mask):
            LOGGER.warning("Mask geometry differs from image; aligning mask to image: %s", key)
            mask = resample_mask_to_reference(mask, image)

        image_out = resample(
            sitk.Cast(image, sitk.sitkFloat32),
            spacing,
            sitk.sitkBSpline,
            0.0,
            sitk.sitkFloat32,
        )
        mask_out = sitk.Resample(
            mask,
            image_out,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
        # Enforce binary foreground values.
        mask_out = sitk.Cast(mask_out > 0, sitk.sitkUInt8)

        if not geometry_equal(image_out, mask_out):
            raise RuntimeError(f"Output image/mask geometry mismatch: {key}")

        sitk.WriteImage(image_out, str(output_image), useCompression=True)
        sitk.WriteImage(mask_out, str(output_mask), useCompression=True)
        report_rows.append(
            {
                "case_id": key,
                "input_image": str(image_path),
                "input_mask": str(mask_path),
                "output_image": str(output_image),
                "output_mask": str(output_mask),
                "input_spacing": "x".join(f"{v:.6g}" for v in image.GetSpacing()),
                "output_spacing": "x".join(f"{v:.6g}" for v in image_out.GetSpacing()),
                "output_size": "x".join(map(str, image_out.GetSize())),
                "foreground_voxels": int(sitk.GetArrayViewFromImage(mask_out).sum()),
            }
        )
        LOGGER.info("Processed %s", key)

    report = out_images.parent / "preprocessing_report.csv"
    if report_rows:
        with report.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]))
            writer.writeheader()
            writer.writerows(report_rows)
    LOGGER.info("Completed %d case(s). Report: %s", len(report_rows), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
