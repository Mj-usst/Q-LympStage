# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import SimpleITK as sitk


LOGGER = logging.getLogger("qlympstage.morphology")
FEATURE_NAMES = (
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


@dataclass(frozen=True)
class MaskSet:
    case_id: str
    bone: Path
    muscle: Path
    subcutaneous: Path


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def nifti_stem(path: Path) -> str:
    lower = path.name.lower()
    if lower.endswith(".nii.gz"):
        return path.name[:-7]
    if lower.endswith(".nii"):
        return path.name[:-4]
    raise ValueError(f"Not a NIfTI file: {path}")


def iter_nifti(root: Path, recursive: bool) -> Iterable[Path]:
    iterator = root.rglob("*") if recursive else root.iterdir()
    for path in iterator:
        if path.is_file() and path.name.lower().endswith((".nii", ".nii.gz")):
            yield path


def build_index(root: Path, recursive: bool) -> dict[str, Path]:
    if not root.is_dir():
        raise NotADirectoryError(root)
    result: dict[str, Path] = {}
    for path in sorted(iter_nifti(root, recursive), key=lambda p: str(p).lower()):
        key = nifti_stem(path).casefold()
        if key in result:
            raise ValueError(f"Duplicate case key {key!r} in {root}")
        result[key] = path
    return result


def geometry_signature(image: sitk.Image) -> tuple:
    return (
        tuple(image.GetSize()),
        tuple(round(v, 8) for v in image.GetSpacing()),
        tuple(round(v, 8) for v in image.GetOrigin()),
        tuple(round(v, 8) for v in image.GetDirection()),
    )


def read_binary(path: Path) -> sitk.Image:
    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3-D mask: {path}")
    return sitk.Cast(image > 0, sitk.sitkUInt8)


def align_to_reference(mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
    if geometry_signature(mask) == geometry_signature(reference):
        return mask
    return sitk.Resample(
        mask,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


def si_numpy_axis(image: sitk.Image) -> tuple[int, int]:
    """Return (numpy_axis, sign) most aligned with physical superior-inferior.

    SimpleITK direction columns correspond to image index axes x/y/z; the third
    physical row is treated as the superior-inferior direction. NumPy arrays
    returned by SimpleITK are ordered z/y/x, hence np_axis = 2-index_axis.
    sign=+1 means increasing array index moves toward increasing physical S/I.
    """
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    index_axis = int(np.argmax(np.abs(direction[2, :])))
    sign_index = 1 if direction[2, index_axis] >= 0 else -1
    np_axis = 2 - index_axis
    # Reversing x/y/z into z/y/x preserves the index sign for the corresponding axis.
    return np_axis, sign_index


def spacing_for_numpy_axes(image: sitk.Image) -> tuple[float, float, float]:
    sx, sy, sz = image.GetSpacing()
    return (sz, sy, sx)


def slice_counts(mask: np.ndarray, axis: int) -> np.ndarray:
    other_axes = tuple(i for i in range(3) if i != axis)
    return mask.sum(axis=other_axes).astype(float)


def middle_third_indices(bone: np.ndarray, axis: int) -> np.ndarray:
    counts = slice_counts(bone, axis)
    occupied = np.flatnonzero(counts > 0)
    if len(occupied) < 3:
        raise ValueError("Bone VOI does not span enough slices to define a middle third.")
    start, end = int(occupied.min()), int(occupied.max())
    length = end - start + 1
    middle_start = start + int(np.floor(length / 3.0))
    middle_end = start + int(np.ceil(2.0 * length / 3.0)) - 1
    middle_end = max(middle_start, min(end, middle_end))
    return np.arange(middle_start, middle_end + 1, dtype=int)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else np.nan


def feature_row(mask_set: MaskSet) -> dict[str, float | str]:
    bone_img = read_binary(mask_set.bone)
    muscle_img = align_to_reference(read_binary(mask_set.muscle), bone_img)
    sub_img = align_to_reference(read_binary(mask_set.subcutaneous), bone_img)

    bone = sitk.GetArrayFromImage(bone_img).astype(bool)
    muscle = sitk.GetArrayFromImage(muscle_img).astype(bool)
    sub = sitk.GetArrayFromImage(sub_img).astype(bool)
    whole = bone | muscle | sub
    if not bone.any() or not whole.any():
        raise ValueError(f"Empty bone/whole-calf VOI for {mask_set.case_id}")

    axis, si_sign = si_numpy_axis(bone_img)
    middle = middle_third_indices(bone, axis)
    spacing_np = spacing_for_numpy_axes(bone_img)
    voxel_volume = float(np.prod(spacing_np))
    cross_spacing = float(np.prod([spacing_np[i] for i in range(3) if i != axis]))

    whole_counts = slice_counts(whole, axis)
    sub_counts = slice_counts(sub, axis)
    muscle_counts = slice_counts(muscle, axis)
    bone_counts = slice_counts(bone, axis)

    whole_csa = whole_counts * cross_spacing
    sub_csa = sub_counts * cross_spacing
    muscle_csa = muscle_counts * cross_spacing
    bone_csa = bone_counts * cross_spacing

    max_index = int(np.argmax(whole_csa))
    leg_max = float(whole_csa[max_index])
    sub_at_max = float(sub_csa[max_index])
    muscle_at_max = float(muscle_csa[max_index])
    bone_at_max = float(bone_csa[max_index])

    # Volumes are measured only within the osseous middle-third reference region.
    leg_mid_volume = float(whole_counts[middle].sum() * voxel_volume)
    sub_mid_volume = float(sub_counts[middle].sum() * voxel_volume)
    muscle_mid_volume = float(muscle_counts[middle].sum() * voxel_volume)
    bone_mid_volume = float(bone_counts[middle].sum() * voxel_volume)

    # The manuscript's upper-to-lower ratio is fixed here as the subcutaneous
    # CSA at the superior boundary divided by the CSA at the inferior boundary
    # of the osseous middle third. Image direction determines superior/inferior.
    ordered_middle = middle if si_sign >= 0 else middle[::-1]
    superior_area = float(sub_csa[int(ordered_middle[-1])])
    inferior_area = float(sub_csa[int(ordered_middle[0])])

    return {
        "case_id": mask_set.case_id,
        "leg_middle1of3_volume_mm3": leg_mid_volume,
        "underskin_middle1of3_volume_mm3": sub_mid_volume,
        "leg_max_csa_mm2": leg_max,
        "underskin_area_at_leg_max_csa_mm2": sub_at_max,
        "underskin_middle1of3_top_bottom_area_ratio": safe_ratio(superior_area, inferior_area),
        "underskin_muscle_middle1of3_volume_ratio": safe_ratio(sub_mid_volume, muscle_mid_volume),
        "underskin_bone_middle1of3_volume_ratio": safe_ratio(sub_mid_volume, bone_mid_volume),
        "underskin_muscle_max_csa_ratio_at_legmax": safe_ratio(sub_at_max, muscle_at_max),
        "underskin_bone_max_csa_ratio_at_legmax": safe_ratio(sub_at_max, bone_at_max),
        "middle_third_start_slice": int(middle.min()),
        "middle_third_end_slice": int(middle.max()),
        "leg_max_csa_slice": max_index,
        "top_bottom_mode": "middle_third_boundary_superior_over_inferior",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bone-dir", type=Path, required=True)
    parser.add_argument("--muscle-dir", type=Path, required=True)
    parser.add_argument("--subcutaneous-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--strict-pairs", action="store_true")
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    bone = build_index(args.bone_dir.resolve(), args.recursive)
    muscle = build_index(args.muscle_dir.resolve(), args.recursive)
    sub = build_index(args.subcutaneous_dir.resolve(), args.recursive)
    common = sorted(set(bone) & set(muscle) & set(sub))
    union = set(bone) | set(muscle) | set(sub)
    unmatched = sorted(union - set(common))
    if args.strict_pairs and unmatched:
        raise ValueError(f"Cases missing one or more VOIs: {unmatched[:20]}")
    if not common:
        raise ValueError("No complete bone/muscle/subcutaneous mask triplets found.")

    rows: list[dict[str, float | str]] = []
    failures: list[dict[str, str]] = []
    for key in common:
        masks = MaskSet(key, bone[key], muscle[key], sub[key])
        try:
            rows.append(feature_row(masks))
            LOGGER.info("Extracted morphology: %s", key)
        except Exception as exc:
            LOGGER.exception("Failed morphology extraction: %s", key)
            failures.append({"case_id": key, "error": f"{type(exc).__name__}: {exc}"})

    output = args.output_csv.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    if failures:
        pd.DataFrame(failures).to_csv(output.with_name(output.stem + "_failures.csv"), index=False, encoding="utf-8-sig")
    LOGGER.info("Saved %d rows to %s; failures=%d", len(rows), output, len(failures))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
