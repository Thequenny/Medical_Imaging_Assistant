"""NIfTI volume analysis for one CT image or one patient.

This module implements Step 1 of the roadmap:
- load .nii and .nii.gz files with nibabel;
- read NIfTI metadata;
- compute basic CT intensity statistics for one image;
- return patient-level information as a dictionary/JSON structure.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import nibabel as nib
except ImportError as exc:  # pragma: no cover - depends on environment setup
    nib = None
    _NIBABEL_IMPORT_ERROR = exc
else:
    _NIBABEL_IMPORT_ERROR = None


NIFTI_EXTENSIONS = (".nii.gz", ".nii")
DICOM_EXTENSION = ".dcm"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "CT_data.json"
DEFAULT_PREPARED_DATASET_DIRNAME = "converted_nifti"
DEFAULT_CONVERSION_BUFFER_DIRNAME = ".conversion_buffer"
DEFAULT_WARNING_FILENAME = "Warning.txt"
DEFAULT_WARNING_PATH = (
    Path(__file__).resolve().parents[1] / "data" / DEFAULT_WARNING_FILENAME
)
MINIMUM_DICOM_CT_SLICES = 4


@dataclass
class ConversionIssue:
    """One DICOM conversion problem that blocks later analysis phases."""

    path: str
    reason: str


@dataclass
class DatasetNiftiPreparation:
    """Result of Phase 1.1 dataset preparation."""

    dataset_root: str
    prepared_dataset_root: str
    nifti_file_count: int
    dicom_file_count: int
    converted_file_count: int
    conversion_was_needed: bool
    warning_path: str | None
    issues: list[ConversionIssue]

    @property
    def can_continue(self) -> bool:
        return not self.issues


class DatasetNiftiPreparationError(RuntimeError):
    """Raised when DICOM conversion blocks later analysis phases."""

    def __init__(
        self,
        message: str,
        warning_path: Path,
        issues: list[ConversionIssue],
    ) -> None:
        super().__init__(message)
        self.warning_path = str(warning_path)
        self.issues = issues


@dataclass
class MemoryEstimate:
    """Estimated storage and in-memory size for a NIfTI volume."""

    file_size_bytes: int
    native_array_bytes: int
    float32_array_bytes: int
    float64_array_bytes: int


@dataclass
class NiftiMetadata:
    """Metadata read from a NIfTI file without computing intensity stats."""

    path: str
    dimensions: list[int]
    number_of_slices: int
    slice_thickness: float | None
    voxel_spacing: list[float]
    orientation: list[str]
    datatype: str
    affine: list[list[float]]
    voxel_count: int
    physical_voxel_size_mm3: float | None
    physical_size_mm: list[float]
    memory_estimate: MemoryEstimate


@dataclass
class IntensityInformation:
    """Basic intensity distribution for one CT volume."""

    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    percentiles: dict[str, float | None]
    finite_voxel_count: int
    non_finite_voxel_count: int


@dataclass
class NiftiVolumeAnalysis:
    """Full analysis of one NIfTI image volume."""

    metadata: NiftiMetadata
    intensity: IntensityInformation

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnnotationInformation:
    """Information about an optional patient annotation/label."""

    present: bool
    path: str | None
    metadata: NiftiMetadata | None
    message: str


@dataclass
class PatientAnalysis:
    """Patient-level structure expected by later dataset analysis steps."""

    patient_id: str
    image: NiftiVolumeAnalysis
    annotation: AnnotationInformation

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_nifti(path: str | Path) -> Any:
    """Load a .nii or .nii.gz file with nibabel."""

    _require_nibabel()
    nifti_path = _validate_nifti_path(path)
    return nib.load(str(nifti_path))


def read_nifti_metadata(path: str | Path) -> NiftiMetadata:
    """Read dimensions, spacing, orientation, datatype, and affine matrix."""

    nifti_path = _validate_nifti_path(path)
    image = load_nifti(nifti_path)
    shape = tuple(int(value) for value in image.shape)
    zooms = tuple(float(value) for value in image.header.get_zooms()[: len(shape)])
    datatype = str(image.get_data_dtype())

    voxel_count = _compute_voxel_count(shape)
    physical_voxel_size_mm3 = _compute_physical_voxel_size(zooms)
    physical_size_mm = [
        float(dimension * spacing)
        for dimension, spacing in zip(shape[:3], zooms[:3], strict=False)
    ]

    metadata = NiftiMetadata(
        path=str(nifti_path),
        dimensions=list(shape),
        number_of_slices=_number_of_slices(shape),
        slice_thickness=_slice_thickness(zooms),
        voxel_spacing=list(zooms),
        orientation=list(nib.orientations.aff2axcodes(image.affine)),
        datatype=datatype,
        affine=_matrix_to_list(image.affine),
        voxel_count=voxel_count,
        physical_voxel_size_mm3=physical_voxel_size_mm3,
        physical_size_mm=physical_size_mm,
        memory_estimate=_estimate_memory(nifti_path, shape, datatype),
    )
    image.uncache()
    return metadata


def analyze_nifti_volume(path: str | Path) -> NiftiVolumeAnalysis:
    """Analyze one CT NIfTI volume and compute intensity information."""

    nifti_path = _validate_nifti_path(path)
    metadata = read_nifti_metadata(nifti_path)
    image = load_nifti(nifti_path)
    data = image.get_fdata(dtype=np.float32)
    intensity = _compute_intensity_information(data)
    image.uncache()
    return NiftiVolumeAnalysis(metadata=metadata, intensity=intensity)


def analyze_patient(
    image_path: str | Path,
    label_path: str | Path | None = None,
) -> PatientAnalysis:
    """Analyze one patient image and optional annotation label."""

    nifti_image_path = _validate_nifti_path(image_path)
    image_analysis = analyze_nifti_volume(nifti_image_path)

    if label_path is None:
        annotation = AnnotationInformation(
            present=False,
            path=None,
            metadata=None,
            message="No annotation label was provided for this patient.",
        )
    else:
        nifti_label_path = _validate_nifti_path(label_path)
        # Labels are read as metadata only here. Dataset-level checks need their
        # dimensions, spacing, orientation, and affine, not their voxel values.
        annotation = AnnotationInformation(
            present=True,
            path=str(nifti_label_path),
            metadata=read_nifti_metadata(nifti_label_path),
            message="Annotation label is present.",
        )

    return PatientAnalysis(
        patient_id=_patient_id_from_path(nifti_image_path),
        image=image_analysis,
        annotation=annotation,
    )


def save_patient_analysis(
    analysis: PatientAnalysis,
    output_path: str | Path,
) -> None:
    """Save patient analysis as formatted JSON."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(analysis.to_dict(), file, indent=2)


def prepare_dataset_nifti_files(
    dataset_root: str | Path,
    output_dir: str | Path | None = None,
    warning_path: str | Path | None = None,
) -> DatasetNiftiPreparation:
    """Ensure a dataset can continue as .nii/.nii.gz files only.

    If DICOM files are found, they are converted through
    ``general_conversion.CTdcm_to_Nii`` into a prepared NIfTI folder. If any
    series cannot produce a coherent CT volume, Warning.txt is written to the
    project data folder, generated conversion folders are removed, and later
    phases are blocked.
    """

    root = _validate_dataset_root(dataset_root)
    prepared_root = _prepared_dataset_root(root, output_dir)
    conversion_buffer = root / DEFAULT_CONVERSION_BUFFER_DIRNAME
    warning_file = _warning_file_path(root, warning_path)
    _remove_file_if_exists(warning_file)

    files = list(
        _iter_dataset_files(
            root,
            excluded_dirs=[prepared_root, conversion_buffer],
        )
    )
    nifti_files = [path for path in files if _is_nifti_file(path)]
    dicom_files = [path for path in files if _is_dicom_file(path)]

    if not dicom_files:
        return DatasetNiftiPreparation(
            dataset_root=str(root),
            prepared_dataset_root=str(root),
            nifti_file_count=len(nifti_files),
            dicom_file_count=0,
            converted_file_count=0,
            conversion_was_needed=False,
            warning_path=None,
            issues=[],
        )

    _reset_preparation_directory(prepared_root, root)
    _reset_preparation_directory(conversion_buffer, root)
    _copy_nifti_files_to_prepared_root(nifti_files, root, prepared_root)

    converted_files, issues = _convert_dicom_files(
        dicom_files=dicom_files,
        dataset_root=root,
        output_buffer=conversion_buffer,
        prepared_root=prepared_root,
    )

    if issues:
        _write_conversion_warning(
            warning_file=warning_file,
            dicom_file_count=len(dicom_files),
            issues=issues,
        )
        _cleanup_preparation_directories(prepared_root, conversion_buffer, root)
        raise DatasetNiftiPreparationError(
            (
                "DICOM conversion failed during Phase 1.1. "
                f"Details were written to: {warning_file}"
            ),
            warning_file,
            issues,
        )

    return DatasetNiftiPreparation(
        dataset_root=str(root),
        prepared_dataset_root=str(prepared_root),
        nifti_file_count=len(nifti_files) + len(converted_files),
        dicom_file_count=len(dicom_files),
        converted_file_count=len(converted_files),
        conversion_was_needed=True,
        warning_path=None,
        issues=[],
    )


def _compute_intensity_information(data: np.ndarray) -> IntensityInformation:
    # CT statistics should ignore NaN and infinite values, while still counting
    # them so the dataset report can warn about invalid voxels.
    finite_mask = np.isfinite(data)
    finite_voxel_count = int(np.count_nonzero(finite_mask))
    non_finite_voxel_count = int(data.size - finite_voxel_count)

    if finite_voxel_count == 0:
        return IntensityInformation(
            min=None,
            max=None,
            mean=None,
            std=None,
            percentiles=_empty_percentiles(),
            finite_voxel_count=0,
            non_finite_voxel_count=non_finite_voxel_count,
        )

    finite_data = data[finite_mask]
    return IntensityInformation(
        min=float(np.min(finite_data)),
        max=float(np.max(finite_data)),
        mean=float(np.mean(finite_data, dtype=np.float64)),
        std=float(np.std(finite_data, dtype=np.float64)),
        percentiles=_compute_percentiles(finite_data),
        finite_voxel_count=finite_voxel_count,
        non_finite_voxel_count=non_finite_voxel_count,
    )


def _compute_percentiles(data: np.ndarray) -> dict[str, float | None]:
    percentiles = [0.5, 1, 5, 25, 50, 75, 95, 99, 99.5]
    values = np.percentile(data, percentiles)
    return {
        _percentile_key(percentile): float(value)
        for percentile, value in zip(percentiles, values, strict=True)
    }


def _empty_percentiles() -> dict[str, float | None]:
    return {
        _percentile_key(percentile): None
        for percentile in [0.5, 1, 5, 25, 50, 75, 95, 99, 99.5]
    }


def _percentile_key(percentile: float) -> str:
    label = str(percentile).replace(".", "_")
    return f"p{label}"


def _estimate_memory(
    path: Path,
    shape: tuple[int, ...],
    datatype: str,
) -> MemoryEstimate:
    voxel_count = _compute_voxel_count(shape)
    try:
        native_dtype_size = np.dtype(datatype).itemsize
    except TypeError:
        native_dtype_size = 0

    return MemoryEstimate(
        file_size_bytes=int(path.stat().st_size),
        native_array_bytes=int(voxel_count * native_dtype_size),
        float32_array_bytes=int(voxel_count * np.dtype(np.float32).itemsize),
        float64_array_bytes=int(voxel_count * np.dtype(np.float64).itemsize),
    )


def _compute_voxel_count(shape: tuple[int, ...]) -> int:
    if not shape:
        return 0
    return int(np.prod(shape, dtype=np.int64))


def _compute_physical_voxel_size(zooms: tuple[float, ...]) -> float | None:
    spatial_zooms = zooms[:3]
    if len(spatial_zooms) < 3:
        return None
    return float(np.prod(spatial_zooms, dtype=np.float64))


def _number_of_slices(shape: tuple[int, ...]) -> int:
    if len(shape) >= 3:
        return int(shape[2])
    if len(shape) >= 2:
        return 1
    return 0


def _slice_thickness(zooms: tuple[float, ...]) -> float | None:
    if len(zooms) < 3:
        return None
    return float(zooms[2])


def _matrix_to_list(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix.tolist()]


def _patient_id_from_path(path: Path) -> str:
    name = path.name
    for extension in NIFTI_EXTENSIONS:
        if name.lower().endswith(extension):
            return name[: -len(extension)]
    return path.stem


def _validate_dataset_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")
    return root


def _prepared_dataset_root(root: Path, output_dir: str | Path | None) -> Path:
    if output_dir is None:
        return (root / DEFAULT_PREPARED_DATASET_DIRNAME).resolve()

    prepared_root = Path(output_dir).expanduser().resolve()
    if prepared_root == root:
        raise ValueError(
            "The prepared NIfTI output directory cannot be the dataset root "
            "because the original DICOM files must remain separated."
        )
    return prepared_root


def _warning_file_path(_root: Path, warning_path: str | Path | None) -> Path:
    if warning_path is None:
        return DEFAULT_WARNING_PATH.resolve()
    return Path(warning_path).expanduser().resolve()


def _iter_dataset_files(
    root: Path,
    excluded_dirs: list[Path],
) -> list[Path]:
    excluded_roots = [path.resolve() for path in excluded_dirs]
    files: list[Path] = []
    for path in root.rglob("*"):
        resolved = path.resolve()
        if any(_is_same_or_inside(resolved, excluded) for excluded in excluded_roots):
            continue
        if resolved.is_file():
            files.append(resolved)
    return files


def _reset_preparation_directory(path: Path, dataset_root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = dataset_root.resolve()
    if resolved_path == resolved_root or not _is_inside(resolved_path, resolved_root):
        raise ValueError(
            "For safety, generated conversion directories must be inside the "
            f"dataset root. Got: {resolved_path}"
        )

    if resolved_path.exists():
        shutil.rmtree(resolved_path)
    resolved_path.mkdir(parents=True, exist_ok=True)


def _cleanup_preparation_directories(
    prepared_root: Path,
    conversion_buffer: Path,
    dataset_root: Path,
) -> None:
    for path in [prepared_root, conversion_buffer]:
        resolved_path = path.resolve()
        resolved_root = dataset_root.resolve()
        if resolved_path == resolved_root or not _is_inside(resolved_path, resolved_root):
            raise ValueError(
                "For safety, generated conversion directories must be inside the "
                f"dataset root. Got: {resolved_path}"
            )
        if resolved_path.exists():
            shutil.rmtree(resolved_path)


def _remove_file_if_exists(path: Path) -> None:
    if path.exists() and path.is_file():
        path.unlink()


def _copy_nifti_files_to_prepared_root(
    nifti_files: list[Path],
    dataset_root: Path,
    prepared_root: Path,
) -> None:
    for source in nifti_files:
        relative_path = source.relative_to(dataset_root)
        destination = prepared_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _convert_dicom_files(
    dicom_files: list[Path],
    dataset_root: Path,
    output_buffer: Path,
    prepared_root: Path,
) -> tuple[list[Path], list[ConversionIssue]]:
    converter, import_issue = _load_general_conversion_function()
    if converter is None:
        return [], [
            ConversionIssue(
                path=str(dicom_dir),
                reason=import_issue or "DICOM conversion function is unavailable.",
            )
            for dicom_dir, _series_files in _group_dicom_files_by_directory(dicom_files)
        ]

    converted_files: list[Path] = []
    issues: list[ConversionIssue] = []

    for series_index, (dicom_dir, series_files) in enumerate(
        _group_dicom_files_by_directory(dicom_files),
        start=1,
    ):
        skip_reason = _dicom_series_skip_reason(series_files)
        if skip_reason is not None:
            issues.append(ConversionIssue(path=str(dicom_dir), reason=skip_reason))
            continue

        staged_dicom_dir = _stage_dicom_series(
            dicom_dir=dicom_dir,
            dicom_files=series_files,
            output_buffer=output_buffer,
            series_index=series_index,
        )
        series_prepared_root = prepared_root / dicom_dir.relative_to(dataset_root)
        series_prepared_root.mkdir(parents=True, exist_ok=True)
        before_conversion = set(_find_nifti_files(prepared_root))
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                converter(staged_dicom_dir, output_buffer, series_prepared_root)
        except Exception as exc:
            issues.append(
                ConversionIssue(
                    path=str(dicom_dir),
                    reason=(
                        "general_conversion.CTdcm_to_Nii raised "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            )
            continue

        after_conversion = set(_find_nifti_files(prepared_root))
        new_files = sorted(after_conversion - before_conversion)
        if not new_files:
            issues.append(
                ConversionIssue(
                    path=str(dicom_dir),
                    reason=(
                        "No .nii or .nii.gz file was produced. The DICOM "
                        "series may not form a coherent CT stack."
                    ),
                )
            )
            continue

        converted_files.extend(new_files)

    return converted_files, issues


def _load_general_conversion_function() -> tuple[
    Callable[[Path, Path, Path], Any] | None,
    str | None,
]:
    try:
        module = importlib.import_module("general_conversion")
    except Exception as exc:
        return (
            None,
            (
                "Could not import general_conversion.CTdcm_to_Nii. "
                f"Reason: {type(exc).__name__}: {exc}"
            ),
        )

    converter = getattr(module, "CTdcm_to_Nii", None)
    if not callable(converter):
        return None, "general_conversion.CTdcm_to_Nii is missing or not callable."
    return converter, None


def _group_dicom_files_by_directory(
    dicom_files: list[Path],
) -> list[tuple[Path, list[Path]]]:
    grouped: dict[Path, list[Path]] = {}
    for path in sorted(dicom_files):
        grouped.setdefault(path.parent, []).append(path)
    return sorted(grouped.items(), key=lambda item: str(item[0]))


def _stage_dicom_series(
    dicom_dir: Path,
    dicom_files: list[Path],
    output_buffer: Path,
    series_index: int,
) -> Path:
    staged_root = output_buffer / "_dicom_input"
    staged_dir = staged_root / _safe_folder_name(
        f"{series_index:04d}_{dicom_dir.name}"
    )
    if staged_dir.exists():
        shutil.rmtree(staged_dir)
    staged_dir.mkdir(parents=True, exist_ok=True)

    for source in dicom_files:
        shutil.copy2(source, staged_dir / source.name)

    return staged_dir


def _safe_folder_name(value: str) -> str:
    safe_characters = [
        character if character.isalnum() or character in "-_." else "_"
        for character in value
    ]
    safe_name = "".join(safe_characters).strip("._")
    return safe_name or "dicom_series"


def _dicom_series_skip_reason(dicom_files: list[Path]) -> str | None:
    if len(dicom_files) < MINIMUM_DICOM_CT_SLICES:
        return (
            f"Only {len(dicom_files)} DICOM slice(s) were found. A coherent CT "
            f"stack requires at least {MINIMUM_DICOM_CT_SLICES} slices."
        )

    try:
        from pydicom import dcmread
    except ImportError:
        return None

    try:
        first_dicom = dcmread(
            str(dicom_files[0]),
            stop_before_pixels=True,
            force=True,
        )
    except Exception as exc:
        return f"The first DICOM header could not be read: {type(exc).__name__}: {exc}"

    modality = str(getattr(first_dicom, "Modality", "")).upper()
    if modality and modality != "CT":
        return f"The series modality is '{modality}', not CT."

    series_description = str(
        getattr(first_dicom, "SeriesDescription", "")
    ).lower()
    image_type = _dicom_image_type_as_text(getattr(first_dicom, "ImageType", []))
    if "scout" in series_description or "localizer" in image_type:
        return "Scout/localizer DICOM series are not coherent CT stacks."

    return None


def _dicom_image_type_as_text(image_type: Any) -> str:
    if isinstance(image_type, str):
        return image_type.lower()
    try:
        return " ".join(str(value).lower() for value in image_type)
    except TypeError:
        return str(image_type).lower()


def _find_nifti_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and _is_nifti_file(path)
    )


def _write_conversion_warning(
    warning_file: Path,
    dicom_file_count: int,
    issues: list[ConversionIssue],
) -> None:
    warning_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "WARNING: DICOM to NIfTI conversion failed",
        "",
        "The analyzer stopped at Phase 1.1.",
        (
            "The HTML report can't be generated because at least one DICOM "
            "series could not be converted."
        ),
        f"DICOM files detected: {dicom_file_count}",
        "",
        "Conversion problems:",
    ]
    lines.extend(_format_conversion_issue(issue) for issue in issues)
    warning_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_conversion_issue(issue: ConversionIssue) -> str:
    series_name = Path(issue.path).name or "unknown series"
    return f"- {series_name}: {issue.reason}"


def _is_dicom_file(path: Path) -> bool:
    return path.suffix.lower() == DICOM_EXTENSION


def _is_same_or_inside(path: Path, parent: Path) -> bool:
    return path == parent or _is_inside(path, parent)


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_nifti_path(path: str | Path) -> Path:
    nifti_path = Path(path).expanduser().resolve()
    if not nifti_path.exists():
        raise FileNotFoundError(f"NIfTI file does not exist: {nifti_path}")
    if not nifti_path.is_file():
        raise FileNotFoundError(f"NIfTI path is not a file: {nifti_path}")
    if not _is_nifti_file(nifti_path):
        raise ValueError(f"Expected a .nii or .nii.gz file: {nifti_path}")
    return nifti_path


def _is_nifti_file(path: Path) -> bool:
    lower_name = path.name.lower()
    return any(lower_name.endswith(extension) for extension in NIFTI_EXTENSIONS)


def _require_nibabel() -> None:
    if nib is None:
        raise ImportError(
            "nibabel is required to load NIfTI files. Install it with "
            "`pip install nibabel`."
        ) from _NIBABEL_IMPORT_ERROR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze one NIfTI CT volume.")
    parser.add_argument(
        "image",
        help="Path to the patient CT image (.nii or .nii.gz).",
    )
    parser.add_argument(
        "--label",
        help="Optional path to the patient annotation/label (.nii or .nii.gz).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="JSON output path. Defaults to data\\CT_data.json.",
    )
    args = parser.parse_args(argv)

    analysis = analyze_patient(args.image, args.label)
    save_patient_analysis(analysis, args.output)
    print(f"Patient analysis written to: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
