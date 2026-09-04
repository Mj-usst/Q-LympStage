from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd
import radiomics
import SimpleITK as sitk
from radiomics import featureextractor


SCRIPT_VERSION = "1.0.0"
EXPECTED_RADIOMIC_FEATURES = 1_116
LOG_SIGMAS_MM = [1.0, 2.0, 3.0]
FEATURE_CLASSES = (
    "firstorder",
    "glcm",
    "glrlm",
    "glszm",
    "gldm",
    "ngtdm",
)
IDENTIFIER_COLUMNS = ("case_id", "image_file", "mask_file")

EXTRACTOR_SETTINGS: dict[str, Any] = {
    "binWidth": 25,
    "resampledPixelSpacing": None,
    "interpolator": "sitkBSpline",
    "normalize": True,
    "normalizeScale": 1,
    "removeOutliers": None,
    "force2D": False,
    "label": 1,
    "geometryTolerance": 1e-3,
}

LOGGER = logging.getLogger("radiomics_feature_extraction")
_WORKER_EXTRACTOR: featureextractor.RadiomicsFeatureExtractor | None = None
_WORKER_LABEL = 1


@dataclass(frozen=True)
class CasePair:
    """One matched image/mask pair."""

    case_id: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class ExtractionTask:
    """Serializable task passed to a worker process."""

    case_id: str
    image_path: str
    mask_path: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the 1,116 radiomic features used in the manuscript from "
            "matched NIfTI images and masks."
        )
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing resampled NIfTI images.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        required=True,
        help="Directory containing the corresponding binary NIfTI masks.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Final feature table. Parent directories are created automatically.",
    )
    parser.add_argument(
        "--key-regex",
        default=None,
        help=(
            "Optional regular expression used to derive the pairing key from "
            "both file stems. The named group 'key', first capture group, or "
            "full match is used in that order. Default: use the complete stem."
        ),
    )
    parser.add_argument(
        "--image-strip-prefix",
        default="",
        help="Fixed prefix removed from image stems before matching.",
    )
    parser.add_argument(
        "--mask-strip-prefix",
        default="",
        help="Fixed prefix removed from mask stems before matching.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input directories recursively.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, cpu_count() - 1)),
        help="Worker processes. Default: up to 4 to limit 3-D image memory use.",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=1,
        help="Foreground value in the binary mask. Default: 1.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Per-case checkpoint directory. Default: a directory named after "
            "the output CSV beside that file."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract cases even when compatible checkpoints already exist.",
    )
    parser.add_argument(
        "--strict-pairs",
        action="store_true",
        help="Stop if any image or mask cannot be paired.",
    )
    parser.add_argument(
        "--allow-feature-count-mismatch",
        action="store_true",
        help=(
            "Do not fail a case when PyRadiomics returns a number of features "
            "other than 1,116. Intended only for debugging version differences."
        ),
    )
    return parser.parse_args(argv)


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(processName)s | %(message)s"
    )
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    LOGGER.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    # PyRadiomics INFO output is very verbose for a large cohort.
    logging.getLogger("radiomics").setLevel(logging.WARNING)


def nifti_stem(path: Path) -> str:
    name = path.name
    if name.casefold().endswith(".nii.gz"):
        return name[:-7]
    if name.casefold().endswith(".nii"):
        return name[:-4]
    raise ValueError(f"Not a NIfTI filename: {path}")


def strip_prefix_case_insensitive(value: str, prefix: str) -> str:
    if prefix and value.casefold().startswith(prefix.casefold()):
        return value[len(prefix) :]
    return value


def pairing_key(path: Path, strip_prefix: str, key_regex: str | None) -> str:
    stem = strip_prefix_case_insensitive(nifti_stem(path), strip_prefix).strip()
    if not stem:
        raise ValueError(f"Empty pairing key after prefix removal: {path}")
    if key_regex is None:
        return stem

    match = re.search(key_regex, stem)
    if match is None:
        raise ValueError(
            f"Filename does not match --key-regex {key_regex!r}: {path.name}"
        )
    if "key" in match.groupdict():
        key = match.group("key")
    elif match.lastindex:
        key = match.group(1)
    else:
        key = match.group(0)
    key = key.strip()
    if not key:
        raise ValueError(f"Regular expression produced an empty key: {path.name}")
    return key


def iter_nifti_files(root: Path, recursive: bool) -> Iterator[Path]:
    iterator: Iterable[Path] = root.rglob("*") if recursive else root.iterdir()
    for path in iterator:
        if not path.is_file():
            continue
        name = path.name.casefold()
        if name.endswith(".nii") or name.endswith(".nii.gz"):
            yield path


def build_file_index(
    root: Path,
    strip_prefix: str,
    key_regex: str | None,
    recursive: bool,
) -> dict[str, tuple[str, Path]]:
    if not root.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {root}")

    index: dict[str, tuple[str, Path]] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in sorted(iter_nifti_files(root, recursive), key=lambda p: str(p).casefold()):
        display_key = pairing_key(path, strip_prefix, key_regex)
        normalized_key = display_key.casefold()
        if normalized_key in index:
            duplicates.setdefault(normalized_key, [index[normalized_key][1]]).append(path)
        else:
            index[normalized_key] = (display_key, path)

    if duplicates:
        details = "; ".join(
            f"{key}: {[str(path) for path in paths]}"
            for key, paths in list(duplicates.items())[:10]
        )
        raise ValueError(
            "Duplicate pairing keys were found. Use a more specific --key-regex "
            f"or full-stem matching. Examples: {details}"
        )
    if not index:
        raise FileNotFoundError(f"No .nii or .nii.gz files found in: {root}")
    return index


def pair_cases(args: argparse.Namespace) -> tuple[list[CasePair], pd.DataFrame]:
    image_index = build_file_index(
        args.image_dir,
        args.image_strip_prefix,
        args.key_regex,
        args.recursive,
    )
    mask_index = build_file_index(
        args.mask_dir,
        args.mask_strip_prefix,
        args.key_regex,
        args.recursive,
    )

    common_keys = sorted(set(image_index) & set(mask_index))
    pairs = [
        CasePair(
            case_id=image_index[key][0],
            image_path=image_index[key][1],
            mask_path=mask_index[key][1],
        )
        for key in common_keys
    ]

    unmatched_rows: list[dict[str, str]] = []
    for key in sorted(set(image_index) - set(mask_index)):
        unmatched_rows.append(
            {
                "case_id": image_index[key][0],
                "missing": "mask",
                "file": str(image_index[key][1]),
            }
        )
    for key in sorted(set(mask_index) - set(image_index)):
        unmatched_rows.append(
            {
                "case_id": mask_index[key][0],
                "missing": "image",
                "file": str(mask_index[key][1]),
            }
        )

    unmatched = pd.DataFrame(unmatched_rows, columns=["case_id", "missing", "file"])
    if not pairs:
        raise ValueError("No image/mask pairs were found with the selected matching rules.")
    return pairs, unmatched


def make_extractor(label: int) -> featureextractor.RadiomicsFeatureExtractor:
    settings = dict(EXTRACTOR_SETTINGS)
    settings["label"] = label
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    extractor.disableAllImageTypes()

    for feature_class in FEATURE_CLASSES:
        extractor.enableFeatureClassByName(feature_class)

    extractor.enableImageTypeByName("Original")
    extractor.enableImageTypeByName("Wavelet")
    extractor.enableImageTypeByName(
        "LoG",
        customArgs={"sigma": list(LOG_SIGMAS_MM)},
    )
    return extractor


def init_worker(label: int) -> None:
    global _WORKER_EXTRACTOR, _WORKER_LABEL
    _WORKER_LABEL = label
    _WORKER_EXTRACTOR = make_extractor(label)


def scalar_value(value: Any) -> Any:
    """Convert NumPy/SimpleITK scalar-like output to a CSV-safe Python value."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return value


def extract_one_case(
    task: ExtractionTask,
) -> tuple[str, str, str, dict[str, Any] | None, float, str | None]:
    started = time.perf_counter()
    try:
        if _WORKER_EXTRACTOR is None:
            init_worker(_WORKER_LABEL)
        assert _WORKER_EXTRACTOR is not None
        values = _WORKER_EXTRACTOR.execute(
            task.image_path,
            task.mask_path,
            label=_WORKER_LABEL,
        )
        features = {
            key: scalar_value(value)
            for key, value in values.items()
            if not key.startswith("diagnostics_")
        }
        elapsed = time.perf_counter() - started
        return (
            task.case_id,
            task.image_path,
            task.mask_path,
            features,
            elapsed,
            None,
        )
    except Exception as exc:  # Preserve the cohort run and report the case.
        elapsed = time.perf_counter() - started
        return (
            task.case_id,
            task.image_path,
            task.mask_path,
            None,
            elapsed,
            f"{type(exc).__name__}: {exc}",
        )


def extraction_signature(label: int) -> str:
    payload = {
        "script_version": SCRIPT_VERSION,
        "settings": {**EXTRACTOR_SETTINGS, "label": label},
        "feature_classes": FEATURE_CLASSES,
        "image_types": {
            "Original": {},
            "Wavelet": {},
            "LoG": {"sigma": LOG_SIGMAS_MM},
        },
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def safe_checkpoint_name(case_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", case_id).strip("._") or "case"
    digest = hashlib.sha1(case_id.encode("utf-8")).hexdigest()[:10]
    return f"{safe[:80]}_{digest}.csv"


def checkpoint_path(checkpoint_dir: Path, case_id: str) -> Path:
    return checkpoint_dir / safe_checkpoint_name(case_id)


def checkpoint_is_compatible(path: Path, signature: str) -> bool:
    if not path.is_file():
        return False
    try:
        row = pd.read_csv(
            path,
            nrows=1,
            dtype={"case_id": "string", "_extraction_signature": "string"},
        )
    except Exception:
        return False
    if row.empty or "_extraction_signature" not in row.columns:
        return False
    return str(row.loc[0, "_extraction_signature"]) == signature


def write_checkpoint(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    pd.DataFrame([row]).to_csv(temporary_path, index=False, encoding="utf-8-sig")
    os.replace(temporary_path, path)


def output_companion_path(output_csv: Path, suffix: str) -> Path:
    return output_csv.with_name(f"{output_csv.stem}{suffix}")


def save_unmatched_report(output_csv: Path, unmatched: pd.DataFrame) -> Path:
    path = output_companion_path(output_csv, "_unmatched_files.csv")
    unmatched.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def process_pairs(
    pairs: list[CasePair],
    args: argparse.Namespace,
    checkpoint_dir: Path,
    signature: str,
) -> tuple[list[Path], pd.DataFrame]:
    checkpoint_paths = [checkpoint_path(checkpoint_dir, pair.case_id) for pair in pairs]
    pending: list[ExtractionTask] = []
    for pair, path in zip(pairs, checkpoint_paths):
        if not args.overwrite and checkpoint_is_compatible(path, signature):
            continue
        pending.append(
            ExtractionTask(
                case_id=pair.case_id,
                image_path=str(pair.image_path),
                mask_path=str(pair.mask_path),
            )
        )

    resumed = len(pairs) - len(pending)
    LOGGER.info(
        "Matched %d cases; resuming %d; extracting %d with %d worker(s).",
        len(pairs),
        resumed,
        len(pending),
        args.workers,
    )

    failures: list[dict[str, Any]] = []
    if args.workers == 1:
        init_worker(args.label)
        results: Iterable[tuple[str, str, str, dict[str, Any] | None, float, str | None]] = (
            extract_one_case(task) for task in pending
        )
        pool = None
    else:
        pool = Pool(processes=args.workers, initializer=init_worker, initargs=(args.label,))
        results = pool.imap_unordered(extract_one_case, pending, chunksize=1)

    try:
        for completed, result in enumerate(results, start=1):
            case_id, image_path, mask_path, features, elapsed, error = result
            if error is not None or features is None:
                failures.append(
                    {
                        "case_id": case_id,
                        "image_file": image_path,
                        "mask_file": mask_path,
                        "elapsed_seconds": elapsed,
                        "error": error or "Unknown extraction error",
                    }
                )
                LOGGER.error("[%d/%d] %s failed: %s", completed, len(pending), case_id, error)
                continue

            feature_count = len(features)
            if (
                feature_count != EXPECTED_RADIOMIC_FEATURES
                and not args.allow_feature_count_mismatch
            ):
                error = (
                    f"Expected {EXPECTED_RADIOMIC_FEATURES} radiomic features but "
                    f"PyRadiomics returned {feature_count}."
                )
                failures.append(
                    {
                        "case_id": case_id,
                        "image_file": image_path,
                        "mask_file": mask_path,
                        "elapsed_seconds": elapsed,
                        "error": error,
                    }
                )
                LOGGER.error("[%d/%d] %s failed: %s", completed, len(pending), case_id, error)
                continue

            row = {
                "case_id": case_id,
                "image_file": Path(image_path).name,
                "mask_file": Path(mask_path).name,
                **features,
                "_extraction_signature": signature,
            }
            write_checkpoint(checkpoint_path(checkpoint_dir, case_id), row)
            LOGGER.info(
                "[%d/%d] %s: %d features in %.1f s",
                completed,
                len(pending),
                case_id,
                feature_count,
                elapsed,
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    failure_table = pd.DataFrame(
        failures,
        columns=[
            "case_id",
            "image_file",
            "mask_file",
            "elapsed_seconds",
            "error",
        ],
    )
    return checkpoint_paths, failure_table


def combine_checkpoints(
    checkpoint_paths: list[Path],
    signature: str,
    output_csv: Path,
) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    for path in checkpoint_paths:
        if not checkpoint_is_compatible(path, signature):
            continue
        frames.append(
            pd.read_csv(
                path,
                dtype={"case_id": "string", "_extraction_signature": "string"},
            )
        )
    if not frames:
        raise RuntimeError("No successful compatible checkpoints are available to combine.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.drop(columns=["_extraction_signature"], errors="ignore")
    combined = combined.sort_values("case_id", kind="stable").reset_index(drop=True)

    feature_columns = [
        column for column in combined.columns if column not in IDENTIFIER_COLUMNS
    ]
    ordered_columns = [*IDENTIFIER_COLUMNS, *feature_columns]
    combined = combined.loc[:, ordered_columns]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_csv.with_name(output_csv.name + ".tmp")
    combined.to_csv(temporary_path, index=False, encoding="utf-8-sig")
    os.replace(temporary_path, output_csv)
    return combined, feature_columns


def write_metadata(
    args: argparse.Namespace,
    output_csv: Path,
    checkpoint_dir: Path,
    signature: str,
    combined: pd.DataFrame,
    feature_columns: list[str],
    matched_count: int,
    unmatched_count: int,
    failure_count: int,
) -> Path:
    metadata_path = output_companion_path(output_csv, "_metadata.json")
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "extraction_signature": signature,
        "software": {
            "python": platform.python_version(),
            "pyradiomics": getattr(radiomics, "__version__", "unknown"),
            "simpleitk": sitk.Version_VersionString(),
            "pandas": pd.__version__,
        },
        "inputs": {
            "image_dir": str(args.image_dir.resolve()),
            "mask_dir": str(args.mask_dir.resolve()),
            "key_regex": args.key_regex,
            "image_strip_prefix": args.image_strip_prefix,
            "mask_strip_prefix": args.mask_strip_prefix,
            "recursive": args.recursive,
            "mask_label": args.label,
        },
        "output": {
            "feature_csv": str(output_csv.resolve()),
            "checkpoint_dir": str(checkpoint_dir.resolve()),
            "successful_cases": len(combined),
            "matched_cases": matched_count,
            "unmatched_files": unmatched_count,
            "failed_cases_this_run": failure_count,
            "identifier_columns": list(IDENTIFIER_COLUMNS),
            "radiomic_feature_count": len(feature_columns),
            "radiomic_feature_columns": feature_columns,
        },
        "pyradiomics_configuration": {
            "settings": {**EXTRACTOR_SETTINGS, "label": args.label},
            "feature_classes": list(FEATURE_CLASSES),
            "image_types": {
                "Original": {},
                "Wavelet": {},
                "LoG": {"sigma": LOG_SIGMAS_MM},
            },
            "expected_feature_count": EXPECTED_RADIOMIC_FEATURES,
            "diagnostics_included": False,
            "shape_included": False,
        },
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    return metadata_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.key_regex is not None:
        re.compile(args.key_regex)

    output_csv = args.output_csv.resolve()
    checkpoint_dir = (
        args.checkpoint_dir.resolve()
        if args.checkpoint_dir is not None
        else output_csv.with_name(f"{output_csv.stem}_checkpoints")
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_companion_path(output_csv, "_extraction.log")
    configure_logging(log_path)

    LOGGER.info("PyRadiomics manuscript feature extraction %s", SCRIPT_VERSION)
    LOGGER.info("Image directory: %s", args.image_dir)
    LOGGER.info("Mask directory: %s", args.mask_dir)

    pairs, unmatched = pair_cases(args)
    unmatched_path = save_unmatched_report(output_csv, unmatched)
    if not unmatched.empty:
        LOGGER.warning(
            "%d unmatched files were written to %s.",
            len(unmatched),
            unmatched_path,
        )
        if args.strict_pairs:
            raise ValueError(
                "Unmatched image/mask files were found and --strict-pairs was set."
            )

    signature = extraction_signature(args.label)
    checkpoint_paths, failures = process_pairs(
        pairs,
        args,
        checkpoint_dir,
        signature,
    )
    failure_path = output_companion_path(output_csv, "_failures.csv")
    failures.to_csv(failure_path, index=False, encoding="utf-8-sig")

    combined, feature_columns = combine_checkpoints(
        checkpoint_paths,
        signature,
        output_csv,
    )
    if (
        len(feature_columns) != EXPECTED_RADIOMIC_FEATURES
        and not args.allow_feature_count_mismatch
    ):
        raise RuntimeError(
            f"Combined output contains {len(feature_columns)} radiomic features; "
            f"expected {EXPECTED_RADIOMIC_FEATURES}."
        )

    metadata_path = write_metadata(
        args=args,
        output_csv=output_csv,
        checkpoint_dir=checkpoint_dir,
        signature=signature,
        combined=combined,
        feature_columns=feature_columns,
        matched_count=len(pairs),
        unmatched_count=len(unmatched),
        failure_count=len(failures),
    )

    LOGGER.info("Completed: %d successful cases.", len(combined))
    LOGGER.info("Radiomic feature columns: %d", len(feature_columns))
    LOGGER.info("Feature table: %s", output_csv)
    LOGGER.info("Metadata: %s", metadata_path)
    LOGGER.info("Failures: %s", failure_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user. Completed checkpoints are retained.")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.exception("Extraction stopped: %s", exc)
        raise SystemExit(1)
