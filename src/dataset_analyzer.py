"""Dataset-level NIfTI CT analysis.

This module implements Step 2 of the roadmap:
- recover all patient image/label relationships from a dataset folder;
- run the single-patient NIfTI analyzer for each detected image;
- compute dataset-level consistency and resource-use summaries.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

try:
    from .check_structure_dataset import (
        DatasetStructure,
        identify_dataset_structure,
    )
    from .nifti_analyzer import (
        DatasetNiftiPreparationError,
        NiftiMetadata,
        PatientAnalysis,
        analyze_patient,
        prepare_dataset_nifti_files,
    )
except ImportError:
    from check_structure_dataset import (
        DatasetStructure,
        identify_dataset_structure,
    )
    from nifti_analyzer import (
        DatasetNiftiPreparationError,
        NiftiMetadata,
        PatientAnalysis,
        analyze_patient,
        prepare_dataset_nifti_files,
    )


SPATIAL_ROUNDING_DIGITS = 3
ALIGNMENT_TOLERANCE = 1e-3
PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "dataset_analysis" / "analyse_dataset.json"


@dataclass
class PatientWorkItem:
    """One image and optional label scheduled for patient-level analysis."""

    subject_id: str
    split: str
    image_path: str
    label_path: str | None
    source: str


@dataclass
class PatientDatasetEntry:
    """Successful patient-level analysis stored with dataset context."""

    subject_id: str
    split: str
    image_path: str
    label_path: str | None
    source: str
    analysis: PatientAnalysis


@dataclass
class FailedPatientAnalysis:
    """Patient-level analysis failure that should not stop the full dataset."""

    subject_id: str
    split: str
    image_path: str
    label_path: str | None
    error_type: str
    error_message: str


@dataclass
class DatasetCounts:
    """Basic dataset recovery counts."""

    patients_detected: int
    patients_analyzed: int
    patients_failed: int
    image_files_detected: int
    label_files_detected: int
    image_label_pairs_detected: int
    annotations_present: int
    annotations_missing: int


@dataclass
class MissingDataItem:
    """Missing or unavailable patient data for future report tables."""

    patient_id: str
    split: str
    image_path: str | None
    label_path: str | None
    missing_fields: list[str]
    reason: str


@dataclass
class ValueSummary:
    """Generic numeric summary for non-byte values."""

    count: int
    total: float
    min: float | None
    max: float | None
    mean: float | None
    median: float | None


@dataclass
class NumericSummary:
    """Summary for storage and memory values."""

    count: int
    total: float
    total_readable: str
    min: float | None
    min_readable: str | None
    max: float | None
    max_readable: str | None
    mean: float | None
    mean_readable: str | None
    median: float | None
    median_readable: str | None


@dataclass
class FrequencyItem:
    """Frequency of one metadata value such as spacing or dimensions."""

    value: Any
    display_value: str
    count: int
    percentage: float


@dataclass
class SliceThicknessFrequencyItem:
    """Frequency of one slice thickness value for report tables."""

    thickness_mm: float | None
    count: int
    percentage: float


@dataclass
class PercentileSummaryItem:
    """Report-friendly intensity percentile summary."""

    percentile: str
    count: int
    min: float | None
    max: float | None
    mean: float | None
    median: float | None


@dataclass
class IntensityStatisticItem:
    """Report-friendly intensity statistic summary table row."""

    statistic: str
    unit: str
    count: int
    min: float | None
    max: float | None
    mean: float | None
    median: float | None


@dataclass
class VoxelValiditySummary:
    """Readable summary of valid and invalid voxel counts."""

    total_voxels: int
    total_voxels_readable: str
    valid_voxels: int
    valid_voxels_readable: str
    valid_voxel_percentage: float
    non_finite_voxels: int
    non_finite_voxels_readable: str
    non_finite_voxel_percentage: float
    explanation: str


@dataclass
class PatientIntensityScaleItem:
    """Patient-level intensity scale classification."""

    patient_id: str
    split: str
    image_path: str
    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    scale: str
    reason: str


@dataclass
class IntensityScaleEvaluation:
    """Detect whether CT images look like raw HU or normalized arrays."""

    raw_hu_like_patient_count: int
    normalized_patient_count: int
    unknown_patient_count: int
    normalized_patient_ids: list[str]
    normalized_patients: list[PatientIntensityScaleItem]
    unknown_patients: list[PatientIntensityScaleItem]
    explanation: str


@dataclass
class LabelAlignmentIssue:
    """One image/label pair that is not aligned on the same voxel grid."""

    patient_id: str
    split: str
    image_path: str
    label_path: str
    issues: list[str]
    image_dimensions: list[int]
    label_dimensions: list[int]
    image_voxel_spacing: list[float]
    label_voxel_spacing: list[float]
    image_orientation: list[str]
    label_orientation: list[str]
    affine_max_absolute_difference: float | None


@dataclass
class LabelAlignmentEvaluation:
    """Dataset-level image/label alignment summary."""

    checked_pairs: int
    aligned_pairs: int
    misaligned_pairs: int
    alignment_percentage: float
    misaligned_patients: list[LabelAlignmentIssue]
    explanation: str


@dataclass
class StorageEvaluation:
    """Estimated dataset storage size from detected image and label files."""

    total_file_size_bytes: int
    total_file_size_readable: str
    image_file_size_bytes: int
    image_file_size_readable: str
    label_file_size_bytes: int
    label_file_size_readable: str
    image_file_size_summary_bytes: NumericSummary
    label_file_size_summary_bytes: NumericSummary


@dataclass
class MemoryEvaluation:
    """Estimated memory requirement for loading one image volume."""

    minimum_required_memory_bytes: int | None
    minimum_required_memory_readable: str | None
    minimum_required_memory_basis: str
    native_array_bytes: NumericSummary
    float32_array_bytes: NumericSummary
    float64_array_bytes: NumericSummary
    largest_native_volume_bytes: int | None
    largest_native_volume_readable: str | None
    largest_float32_volume_bytes: int | None
    largest_float32_volume_readable: str | None
    largest_float64_volume_bytes: int | None
    largest_float64_volume_readable: str | None


@dataclass
class ConsistencyEvaluation:
    """Resolution, thickness, spacing, and dimension consistency checks."""

    most_common_orientation: list[str] | None
    percentage_same_orientation: float
    orientation_frequencies: list[FrequencyItem]
    most_common_dimensionality: str | None
    dimensionality_frequencies: list[FrequencyItem]
    most_common_slice_count: int | None
    percentage_same_slice_count: float
    slice_count_frequencies: list[FrequencyItem]
    most_common_slice_count_and_thickness: list[Any] | None
    slice_count_thickness_frequencies: list[FrequencyItem]
    most_common_in_plane_resolution: list[float] | None
    percentage_same_resolution: float
    in_plane_resolution_frequencies: list[FrequencyItem]
    most_common_slice_thickness: float | None
    percentage_same_thickness: float
    slice_thickness_frequencies: list[SliceThicknessFrequencyItem]
    most_common_voxel_spacing: list[float] | None
    percentage_same_voxel_spacing: float
    percentage_different_voxel_spacing: float
    voxel_spacing_frequencies: list[FrequencyItem]
    voxel_volume_mm3_summary: ValueSummary
    physical_size_frequencies: list[FrequencyItem]
    most_common_dimensions: list[int] | None
    percentage_same_dimensions: float
    dimensions_are_consistent: bool
    dimension_frequencies: list[FrequencyItem]


@dataclass
class IntensityEvaluation:
    """Dataset-level CT intensity statistics."""

    global_min: float | None
    global_max: float | None
    patient_min_summary: ValueSummary
    patient_max_summary: ValueSummary
    patient_mean_summary: ValueSummary
    patient_std_summary: ValueSummary
    patient_intensity_statistics_table: list[IntensityStatisticItem]
    patient_percentile_summaries: dict[str, ValueSummary]
    patient_percentile_table: list[PercentileSummaryItem]
    voxel_validity: VoxelValiditySummary
    intensity_scale: IntensityScaleEvaluation
    finite_voxel_count: int
    non_finite_voxel_count: int
    non_finite_voxel_percentage: float


@dataclass
class PreprocessingRecommendation:
    """Recommendation that can be displayed later in the report."""

    category: str
    severity: str
    message: str
    reason: str


@dataclass
class DatasetEvaluation:
    """All dataset-level evaluations requested in Step 2."""

    storage: StorageEvaluation
    memory: MemoryEvaluation
    consistency: ConsistencyEvaluation
    alignment: LabelAlignmentEvaluation
    intensity: IntensityEvaluation
    missing_data: list[MissingDataItem]
    warnings: list[str]
    preprocessing_recommendations: list[PreprocessingRecommendation]


@dataclass
class ReportOverview:
    """Report-ready overview values for Step 3."""

    dataset_name: str
    dataset_root: str
    split_filter: str
    task_type: str
    task_reason: str
    patients_detected: int
    patients_analyzed: int
    patients_failed: int
    analysis_success_percentage: float
    annotation_coverage_percentage: float
    total_storage: str
    minimum_memory_needed: str | None


@dataclass
class ReportTaskDetection:
    """Task information formatted for the future report."""

    detected_task: str
    reason: str
    classification_classes: list[str]
    labels_found: bool
    label_files_detected: int
    image_label_pairs_detected: int


@dataclass
class ReportPreparation:
    """Data prepared by Step 2 for future HTML/PDF report generation."""

    overview: ReportOverview
    task_detection: ReportTaskDetection
    missing_data: list[MissingDataItem]
    warnings: list[str]
    preprocessing_recommendations: list[PreprocessingRecommendation]
    sections_ready: list[str]


@dataclass
class DatasetAnalysis:
    """Full Step 2 result, ready to be serialized as JSON."""

    dataset_root: str
    split_filter: str | None
    structure: DatasetStructure
    counts: DatasetCounts
    patients: list[PatientDatasetEntry]
    failed_patients: list[FailedPatientAnalysis]
    evaluation: DatasetEvaluation
    report_preparation: ReportPreparation

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_dataset(
    dataset_root: str | Path,
    split: str | None = "train",
) -> DatasetAnalysis:
    """Analyze detected patients in a dataset folder.

    The default split is ``train`` because preprocessing checks are meant to
    validate the data used for model training. Pass ``split=None`` to include
    every detected split.
    """

    root = Path(dataset_root).expanduser().resolve()
    preparation = prepare_dataset_nifti_files(root)
    root = Path(preparation.prepared_dataset_root)
    structure = identify_dataset_structure(root)
    # Build the patient list before reading voxel data so split filtering and
    # missing label handling stay consistent across the JSON and report.
    work_items = _build_patient_work_items(root, structure, split)

    patients: list[PatientDatasetEntry] = []
    failed_patients: list[FailedPatientAnalysis] = []

    for item in work_items:
        image_path = root / item.image_path
        label_path = root / item.label_path if item.label_path else None
        try:
            patient_analysis = analyze_patient(image_path, label_path)
        except Exception as exc:  # Keep dataset analysis useful if one file is bad.
            failed_patients.append(
                FailedPatientAnalysis(
                    subject_id=item.subject_id,
                    split=item.split,
                    image_path=item.image_path,
                    label_path=item.label_path,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            continue

        patients.append(
            PatientDatasetEntry(
                subject_id=item.subject_id,
                split=item.split,
                image_path=item.image_path,
                label_path=item.label_path,
                source=item.source,
                analysis=patient_analysis,
            )
        )

    counts = _build_counts(structure, work_items, patients, failed_patients, split)
    evaluation = evaluate_dataset(patients, failed_patients)
    report_preparation = _build_report_preparation(
        root=root,
        split=split,
        structure=structure,
        counts=counts,
        evaluation=evaluation,
    )

    return DatasetAnalysis(
        dataset_root=str(root),
        split_filter=split,
        structure=structure,
        counts=counts,
        patients=patients,
        failed_patients=failed_patients,
        evaluation=evaluation,
        report_preparation=report_preparation,
    )


def evaluate_dataset(
    patients: list[PatientDatasetEntry],
    failed_patients: list[FailedPatientAnalysis] | None = None,
) -> DatasetEvaluation:
    """Compute storage, memory, and metadata consistency summaries."""

    failed_patients = failed_patients or []
    image_metadata = [patient.analysis.image.metadata for patient in patients]
    label_metadata = [
        patient.analysis.annotation.metadata
        for patient in patients
        if patient.analysis.annotation.metadata is not None
    ]

    storage = _evaluate_storage(image_metadata, label_metadata)
    memory = _evaluate_memory(image_metadata)
    consistency = _evaluate_consistency(image_metadata)
    alignment = _evaluate_label_alignment(patients)
    intensity = _evaluate_intensity(patients)
    missing_data = _find_missing_data(patients, failed_patients)
    warnings = _build_evaluation_warnings(
        patient_count=len(patients),
        failed_count=len(failed_patients),
        missing_data_count=len(missing_data),
        consistency=consistency,
        alignment=alignment,
        intensity=intensity,
    )
    preprocessing_recommendations = _build_preprocessing_recommendations(
        patients=patients,
        failed_patients=failed_patients,
        memory=memory,
        consistency=consistency,
        alignment=alignment,
        intensity=intensity,
    )

    return DatasetEvaluation(
        storage=storage,
        memory=memory,
        consistency=consistency,
        alignment=alignment,
        intensity=intensity,
        missing_data=missing_data,
        warnings=warnings,
        preprocessing_recommendations=preprocessing_recommendations,
    )


def save_dataset_analysis(
    analysis: DatasetAnalysis,
    output_path: str | Path,
) -> None:
    """Save dataset analysis as formatted JSON."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(analysis.to_dict(), file, indent=2)


def _build_patient_work_items(
    root: Path,
    structure: DatasetStructure,
    split: str | None = None,
) -> list[PatientWorkItem]:
    items_by_image: dict[str, PatientWorkItem] = {}

    # Start from detected image/label pairs. This keeps labels attached to the
    # correct image when the structure detector found a reliable relationship.
    for pair in structure.image_label_pairs:
        items_by_image[pair.image] = PatientWorkItem(
            subject_id=pair.subject_id,
            split=pair.split,
            image_path=pair.image,
            label_path=pair.label,
            source=pair.source,
        )

    # Add remaining image files without labels so missing annotations can be
    # reported instead of silently ignored.
    for image in structure.image_files:
        if image.path in items_by_image:
            continue
        items_by_image[image.path] = PatientWorkItem(
            subject_id=image.subject_id,
            split=image.split,
            image_path=image.path,
            label_path=None,
            source="image_file",
        )

    work_items = list(items_by_image.values())
    if split is not None:
        # The command-line default is train; use --split all to pass None and
        # include test/validation/unknown splits in the analysis.
        work_items = [item for item in work_items if item.split == split]

    return sorted(
        work_items,
        key=lambda item: (item.split, item.subject_id, item.image_path),
    )


def _build_counts(
    structure: DatasetStructure,
    work_items: list[PatientWorkItem],
    patients: list[PatientDatasetEntry],
    failed_patients: list[FailedPatientAnalysis],
    split: str | None,
) -> DatasetCounts:
    annotations_present = sum(1 for patient in patients if patient.label_path is not None)
    annotations_missing = len(patients) - annotations_present
    image_files = _filter_split_items(structure.image_files, split)
    label_files = _filter_split_items(structure.label_files, split)
    image_label_pairs = _filter_split_items(structure.image_label_pairs, split)

    return DatasetCounts(
        patients_detected=len(work_items),
        patients_analyzed=len(patients),
        patients_failed=len(failed_patients),
        image_files_detected=len(image_files),
        label_files_detected=len(label_files),
        image_label_pairs_detected=len(image_label_pairs),
        annotations_present=annotations_present,
        annotations_missing=annotations_missing,
    )


def _filter_split_items(items: Iterable[Any], split: str | None) -> list[Any]:
    if split is None:
        return list(items)
    return [item for item in items if getattr(item, "split", None) == split]


def _evaluate_storage(
    image_metadata: list[NiftiMetadata],
    label_metadata: list[NiftiMetadata],
) -> StorageEvaluation:
    image_sizes = [
        metadata.memory_estimate.file_size_bytes for metadata in image_metadata
    ]
    label_sizes = [
        metadata.memory_estimate.file_size_bytes for metadata in label_metadata
    ]

    return StorageEvaluation(
        total_file_size_bytes=int(sum(image_sizes) + sum(label_sizes)),
        total_file_size_readable=_format_bytes(sum(image_sizes) + sum(label_sizes)),
        image_file_size_bytes=int(sum(image_sizes)),
        image_file_size_readable=_format_bytes(sum(image_sizes)),
        label_file_size_bytes=int(sum(label_sizes)),
        label_file_size_readable=_format_bytes(sum(label_sizes)),
        image_file_size_summary_bytes=_numeric_summary(image_sizes),
        label_file_size_summary_bytes=_numeric_summary(label_sizes),
    )


def _evaluate_memory(image_metadata: list[NiftiMetadata]) -> MemoryEvaluation:
    native_values = [
        metadata.memory_estimate.native_array_bytes for metadata in image_metadata
    ]
    float32_values = [
        metadata.memory_estimate.float32_array_bytes for metadata in image_metadata
    ]
    float64_values = [
        metadata.memory_estimate.float64_array_bytes for metadata in image_metadata
    ]
    minimum_required_memory = max(float32_values) if float32_values else None

    return MemoryEvaluation(
        minimum_required_memory_bytes=minimum_required_memory,
        minimum_required_memory_readable=_format_bytes(minimum_required_memory),
        minimum_required_memory_basis=(
            "Largest image loaded as float32, which is how analyze_nifti_volume "
            "reads CT voxel data."
        ),
        native_array_bytes=_numeric_summary(native_values),
        float32_array_bytes=_numeric_summary(float32_values),
        float64_array_bytes=_numeric_summary(float64_values),
        largest_native_volume_bytes=max(native_values) if native_values else None,
        largest_native_volume_readable=_format_bytes(max(native_values))
        if native_values
        else None,
        largest_float32_volume_bytes=max(float32_values) if float32_values else None,
        largest_float32_volume_readable=_format_bytes(max(float32_values))
        if float32_values
        else None,
        largest_float64_volume_bytes=max(float64_values) if float64_values else None,
        largest_float64_volume_readable=_format_bytes(max(float64_values))
        if float64_values
        else None,
    )


def _evaluate_consistency(image_metadata: list[NiftiMetadata]) -> ConsistencyEvaluation:
    orientations = [tuple(metadata.orientation) for metadata in image_metadata]
    dimensionalities = [
        _image_dimensionality(metadata.dimensions) for metadata in image_metadata
    ]
    slice_counts = [_slice_count(metadata.dimensions) for metadata in image_metadata]
    slice_count_thicknesses = [
        (_slice_count(metadata.dimensions), _slice_thickness(metadata))
        for metadata in image_metadata
    ]
    in_plane_resolutions = [
        _rounded_tuple(metadata.voxel_spacing[:2]) for metadata in image_metadata
    ]
    slice_thicknesses = [
        _rounded_float(metadata.voxel_spacing[2])
        for metadata in image_metadata
        if len(metadata.voxel_spacing) >= 3
    ]
    voxel_spacings = [
        _rounded_tuple(metadata.voxel_spacing[:3])
        for metadata in image_metadata
        if len(metadata.voxel_spacing) >= 3
    ]
    voxel_volumes = [
        metadata.physical_voxel_size_mm3
        for metadata in image_metadata
        if metadata.physical_voxel_size_mm3 is not None
    ]
    physical_sizes = [
        _rounded_tuple(metadata.physical_size_mm[:3])
        for metadata in image_metadata
        if len(metadata.physical_size_mm) >= 3
    ]
    dimensions = [tuple(metadata.dimensions) for metadata in image_metadata]

    orientation_frequencies = _frequency_items(orientations)
    dimensionality_frequencies = _frequency_items(dimensionalities)
    slice_count_frequencies = _frequency_items(slice_counts)
    slice_count_thickness_frequencies = _frequency_items(slice_count_thicknesses)
    resolution_frequencies = _frequency_items(in_plane_resolutions)
    thickness_frequencies = _slice_thickness_frequency_items(slice_thicknesses)
    spacing_frequencies = _frequency_items(voxel_spacings)
    physical_size_frequencies = _frequency_items(physical_sizes)
    dimension_frequencies = _frequency_items(dimensions)

    percentage_same_spacing = _top_percentage(spacing_frequencies)
    percentage_different_spacing = (
        round(100.0 - percentage_same_spacing, 3) if spacing_frequencies else 0.0
    )

    return ConsistencyEvaluation(
        most_common_orientation=_first_frequency_value(orientation_frequencies),
        percentage_same_orientation=_top_percentage(orientation_frequencies),
        orientation_frequencies=orientation_frequencies,
        most_common_dimensionality=_first_frequency_value(dimensionality_frequencies),
        dimensionality_frequencies=dimensionality_frequencies,
        most_common_slice_count=_first_frequency_value(slice_count_frequencies),
        percentage_same_slice_count=_top_percentage(slice_count_frequencies),
        slice_count_frequencies=slice_count_frequencies,
        most_common_slice_count_and_thickness=_first_frequency_value(
            slice_count_thickness_frequencies
        ),
        slice_count_thickness_frequencies=slice_count_thickness_frequencies,
        most_common_in_plane_resolution=_first_frequency_value(resolution_frequencies),
        percentage_same_resolution=_top_percentage(resolution_frequencies),
        in_plane_resolution_frequencies=resolution_frequencies,
        most_common_slice_thickness=_first_slice_thickness_value(thickness_frequencies),
        percentage_same_thickness=_top_percentage(thickness_frequencies),
        slice_thickness_frequencies=thickness_frequencies,
        most_common_voxel_spacing=_first_frequency_value(spacing_frequencies),
        percentage_same_voxel_spacing=percentage_same_spacing,
        percentage_different_voxel_spacing=percentage_different_spacing,
        voxel_spacing_frequencies=spacing_frequencies,
        voxel_volume_mm3_summary=_value_summary(voxel_volumes),
        physical_size_frequencies=physical_size_frequencies,
        most_common_dimensions=_first_frequency_value(dimension_frequencies),
        percentage_same_dimensions=_top_percentage(dimension_frequencies),
        dimensions_are_consistent=len(dimension_frequencies) <= 1,
        dimension_frequencies=dimension_frequencies,
    )


def _evaluate_label_alignment(
    patients: list[PatientDatasetEntry],
) -> LabelAlignmentEvaluation:
    checked_pairs = 0
    misaligned_patients: list[LabelAlignmentIssue] = []

    for patient in patients:
        label_metadata = patient.analysis.annotation.metadata
        if label_metadata is None:
            continue

        # Alignment is evaluated on metadata only. Reading label voxel values is
        # unnecessary because the spatial grid is fully described by metadata.
        checked_pairs += 1
        issue = _label_alignment_issue(
            patient=patient,
            image_metadata=patient.analysis.image.metadata,
            label_metadata=label_metadata,
        )
        if issue is not None:
            misaligned_patients.append(issue)

    aligned_pairs = checked_pairs - len(misaligned_patients)
    return LabelAlignmentEvaluation(
        checked_pairs=checked_pairs,
        aligned_pairs=aligned_pairs,
        misaligned_pairs=len(misaligned_patients),
        alignment_percentage=_percentage(aligned_pairs, checked_pairs),
        misaligned_patients=misaligned_patients,
        explanation=(
            "Image and label are considered aligned when dimensions, voxel spacing, "
            "orientation, and affine matrix match within a small numerical tolerance."
        ),
    )


def _label_alignment_issue(
    patient: PatientDatasetEntry,
    image_metadata: NiftiMetadata,
    label_metadata: NiftiMetadata,
) -> LabelAlignmentIssue | None:
    issues: list[str] = []

    # A usable segmentation label must live on the same voxel grid as the image.
    # Small affine/spacing differences are allowed to avoid false positives from
    # floating-point header precision.
    if image_metadata.dimensions != label_metadata.dimensions:
        issues.append("dimensions")

    if not _float_lists_close(
        image_metadata.voxel_spacing,
        label_metadata.voxel_spacing,
        tolerance=ALIGNMENT_TOLERANCE,
    ):
        issues.append("voxel_spacing")

    if image_metadata.orientation != label_metadata.orientation:
        issues.append("orientation")

    affine_difference = _matrix_max_absolute_difference(
        image_metadata.affine,
        label_metadata.affine,
    )
    if affine_difference is None or affine_difference > ALIGNMENT_TOLERANCE:
        issues.append("affine")

    if not issues:
        return None

    return LabelAlignmentIssue(
        patient_id=patient.subject_id,
        split=patient.split,
        image_path=patient.image_path,
        label_path=patient.label_path or "",
        issues=issues,
        image_dimensions=image_metadata.dimensions,
        label_dimensions=label_metadata.dimensions,
        image_voxel_spacing=image_metadata.voxel_spacing,
        label_voxel_spacing=label_metadata.voxel_spacing,
        image_orientation=image_metadata.orientation,
        label_orientation=label_metadata.orientation,
        affine_max_absolute_difference=affine_difference,
    )


def _evaluate_intensity(patients: list[PatientDatasetEntry]) -> IntensityEvaluation:
    intensities = [patient.analysis.image.intensity for patient in patients]
    finite_voxel_count = sum(item.finite_voxel_count for item in intensities)
    non_finite_voxel_count = sum(item.non_finite_voxel_count for item in intensities)
    total_voxel_count = finite_voxel_count + non_finite_voxel_count

    finite_min_values = [item.min for item in intensities if item.min is not None]
    finite_max_values = [item.max for item in intensities if item.max is not None]
    patient_means = [item.mean for item in intensities if item.mean is not None]
    patient_stds = [item.std for item in intensities if item.std is not None]
    patient_min_summary = _value_summary(finite_min_values)
    patient_max_summary = _value_summary(finite_max_values)
    patient_mean_summary = _value_summary(patient_means)
    patient_std_summary = _value_summary(patient_stds)
    percentile_summaries = _summarize_patient_percentiles(intensities)

    return IntensityEvaluation(
        global_min=min(finite_min_values) if finite_min_values else None,
        global_max=max(finite_max_values) if finite_max_values else None,
        patient_min_summary=patient_min_summary,
        patient_max_summary=patient_max_summary,
        patient_mean_summary=patient_mean_summary,
        patient_std_summary=patient_std_summary,
        patient_intensity_statistics_table=_intensity_statistics_table(
            patient_min_summary=patient_min_summary,
            patient_max_summary=patient_max_summary,
            patient_mean_summary=patient_mean_summary,
            patient_std_summary=patient_std_summary,
        ),
        patient_percentile_summaries=percentile_summaries,
        patient_percentile_table=_percentile_table(percentile_summaries),
        voxel_validity=_voxel_validity_summary(
            finite_voxel_count=finite_voxel_count,
            non_finite_voxel_count=non_finite_voxel_count,
        ),
        intensity_scale=_evaluate_intensity_scale(patients),
        finite_voxel_count=finite_voxel_count,
        non_finite_voxel_count=non_finite_voxel_count,
        non_finite_voxel_percentage=_percentage(
            non_finite_voxel_count,
            total_voxel_count,
        ),
    )


def _find_missing_data(
    patients: list[PatientDatasetEntry],
    failed_patients: list[FailedPatientAnalysis],
) -> list[MissingDataItem]:
    missing_data: list[MissingDataItem] = []

    for patient in patients:
        missing_fields: list[str] = []
        if patient.label_path is None:
            missing_fields.append("annotation_label")
        if patient.analysis.annotation.present and patient.analysis.annotation.metadata is None:
            missing_fields.append("annotation_metadata")

        if missing_fields:
            missing_data.append(
                MissingDataItem(
                    patient_id=patient.subject_id,
                    split=patient.split,
                    image_path=patient.image_path,
                    label_path=patient.label_path,
                    missing_fields=missing_fields,
                    reason="Required patient data is absent from the detected dataset structure.",
                )
            )

    for patient in failed_patients:
        missing_data.append(
            MissingDataItem(
                patient_id=patient.subject_id,
                split=patient.split,
                image_path=patient.image_path,
                label_path=patient.label_path,
                missing_fields=["analysis_result"],
                reason=f"{patient.error_type}: {patient.error_message}",
            )
        )

    return sorted(
        missing_data,
        key=lambda item: (item.split, item.patient_id, ",".join(item.missing_fields)),
    )


def _summarize_patient_percentiles(
    intensities: Iterable[Any],
) -> dict[str, ValueSummary]:
    values_by_percentile: dict[str, list[float]] = {}
    for intensity in intensities:
        for percentile, value in intensity.percentiles.items():
            if value is None:
                continue
            values_by_percentile.setdefault(percentile, []).append(value)

    return {
        percentile: _value_summary(values)
        for percentile, values in sorted(values_by_percentile.items())
    }


def _percentile_table(
    percentile_summaries: dict[str, ValueSummary],
) -> list[PercentileSummaryItem]:
    return [
        PercentileSummaryItem(
            percentile=_format_percentile_label(percentile),
            count=summary.count,
            min=summary.min,
            max=summary.max,
            mean=summary.mean,
            median=summary.median,
        )
        for percentile, summary in sorted(
            percentile_summaries.items(),
            key=lambda item: _percentile_sort_value(item[0]),
        )
    ]


def _intensity_statistics_table(
    patient_min_summary: ValueSummary,
    patient_max_summary: ValueSummary,
    patient_mean_summary: ValueSummary,
    patient_std_summary: ValueSummary,
) -> list[IntensityStatisticItem]:
    return [
        _intensity_statistic_item("Minimum intensity", patient_min_summary),
        _intensity_statistic_item("Maximum intensity", patient_max_summary),
        _intensity_statistic_item("Mean intensity", patient_mean_summary),
        _intensity_statistic_item("Standard deviation", patient_std_summary),
    ]


def _intensity_statistic_item(
    statistic: str,
    summary: ValueSummary,
) -> IntensityStatisticItem:
    return IntensityStatisticItem(
        statistic=statistic,
        unit="HU",
        count=summary.count,
        min=summary.min,
        max=summary.max,
        mean=summary.mean,
        median=summary.median,
    )


def _voxel_validity_summary(
    finite_voxel_count: int,
    non_finite_voxel_count: int,
) -> VoxelValiditySummary:
    total_voxel_count = finite_voxel_count + non_finite_voxel_count
    return VoxelValiditySummary(
        total_voxels=total_voxel_count,
        total_voxels_readable=_format_count(total_voxel_count, "voxels"),
        valid_voxels=finite_voxel_count,
        valid_voxels_readable=_format_count(finite_voxel_count, "voxels"),
        valid_voxel_percentage=_percentage(finite_voxel_count, total_voxel_count),
        non_finite_voxels=non_finite_voxel_count,
        non_finite_voxels_readable=_format_count(non_finite_voxel_count, "voxels"),
        non_finite_voxel_percentage=_percentage(
            non_finite_voxel_count,
            total_voxel_count,
        ),
        explanation=(
            "Finite voxels are valid numeric CT intensity values. Non-finite "
            "voxels are NaN or infinite values that should be cleaned before "
            "training."
        ),
    )


def _evaluate_intensity_scale(
    patients: list[PatientDatasetEntry],
) -> IntensityScaleEvaluation:
    items = [_patient_intensity_scale(patient) for patient in patients]
    normalized_patients = [item for item in items if item.scale == "normalized"]
    unknown_patients = [item for item in items if item.scale == "unknown"]
    raw_hu_like_patient_count = sum(1 for item in items if item.scale == "raw_hu")

    return IntensityScaleEvaluation(
        raw_hu_like_patient_count=raw_hu_like_patient_count,
        normalized_patient_count=len(normalized_patients),
        unknown_patient_count=len(unknown_patients),
        normalized_patient_ids=[item.patient_id for item in normalized_patients],
        normalized_patients=normalized_patients,
        unknown_patients=unknown_patients,
        explanation=(
            "This is a heuristic check. Raw CT is expected to look compatible with "
            "Hounsfield units. Volumes close to 0..1, -1..1, or centered around "
            "0 with standard deviation near 1 are reported as suspected normalized."
        ),
    )


def _patient_intensity_scale(patient: PatientDatasetEntry) -> PatientIntensityScaleItem:
    intensity = patient.analysis.image.intensity
    min_value = intensity.min
    max_value = intensity.max
    mean_value = intensity.mean
    std_value = intensity.std

    scale = "unknown"
    reason = "Intensity range is not clearly raw HU or normalized."
    # This classification is intentionally conservative: only very common
    # normalized ranges are marked as normalized; ambiguous volumes remain
    # unknown so they can be reviewed manually.
    if (
        min_value is None
        or max_value is None
        or mean_value is None
        or std_value is None
    ):
        reason = "No finite intensity values were available."
    elif _looks_like_normalized_intensity(min_value, max_value, mean_value, std_value):
        scale = "normalized"
        reason = "Intensity values look already normalized."
    elif _looks_like_raw_ct_hu(min_value, max_value):
        scale = "raw_hu"
        reason = "Intensity range is compatible with raw CT Hounsfield units."

    return PatientIntensityScaleItem(
        patient_id=patient.subject_id,
        split=patient.split,
        image_path=patient.image_path,
        min=min_value,
        max=max_value,
        mean=mean_value,
        std=std_value,
        scale=scale,
        reason=reason,
    )


def _looks_like_normalized_intensity(
    min_value: float,
    max_value: float,
    mean_value: float,
    std_value: float,
) -> bool:
    value_range = max_value - min_value
    if value_range <= 0.01:
        return False

    # Common preprocessing outputs: min-max normalization to 0..1, symmetric
    # scaling to -1..1, or z-score-like data centered around 0 with std near 1.
    if min_value >= -0.05 and max_value <= 1.05 and value_range >= 0.5:
        return True
    if min_value >= -1.1 and max_value <= 1.1 and value_range >= 1.0:
        return True
    return (
        min_value >= -10.0
        and max_value <= 10.0
        and abs(mean_value) <= 0.25
        and 0.5 <= std_value <= 2.0
        and value_range >= 3.0
    )


def _looks_like_raw_ct_hu(min_value: float, max_value: float) -> bool:
    value_range = max_value - min_value
    # Raw CT Hounsfield units usually include air near -1000 HU and/or high
    # positive values for bone/contrast. The thresholds are broad because
    # cropped or organ-focused datasets can have narrower HU ranges.
    if min_value <= -100.0 and max_value >= 100.0:
        return True
    if min_value <= -500.0 or max_value >= 1000.0:
        return True
    return value_range >= 500.0 and max_value > 100.0


def _build_evaluation_warnings(
    patient_count: int,
    failed_count: int,
    missing_data_count: int,
    consistency: ConsistencyEvaluation,
    alignment: LabelAlignmentEvaluation,
    intensity: IntensityEvaluation,
) -> list[str]:
    warnings: list[str] = []
    if patient_count == 0:
        warnings.append("No patient could be analyzed.")
    if failed_count:
        warnings.append(f"{failed_count} patient(s) failed during analysis.")
    if missing_data_count:
        warnings.append(f"{missing_data_count} patient(s) have missing data.")
    if patient_count > 0 and consistency.percentage_same_voxel_spacing < 100.0:
        warnings.append("Voxel spacing varies across CT images.")
    if patient_count > 0 and consistency.percentage_same_thickness < 100.0:
        warnings.append("Slice thickness varies across CT images.")
    if patient_count > 0 and not consistency.dimensions_are_consistent:
        warnings.append("Volume dimensions are not consistent across CT images.")
    if patient_count > 0 and consistency.percentage_same_orientation < 100.0:
        warnings.append("Image orientation varies across CT images.")
    if alignment.misaligned_pairs:
        warnings.append(
            f"{alignment.misaligned_pairs} image/label pair(s) are not aligned."
        )
    if patient_count > 0 and intensity.non_finite_voxel_count > 0:
        warnings.append("Some CT images contain non-finite intensity values.")
    if intensity.intensity_scale.normalized_patient_count:
        warnings.append(
            "Some CT images appear already normalized instead of raw HU: "
            + ", ".join(intensity.intensity_scale.normalized_patient_ids)
        )
    return warnings


def _build_preprocessing_recommendations(
    patients: list[PatientDatasetEntry],
    failed_patients: list[FailedPatientAnalysis],
    memory: MemoryEvaluation,
    consistency: ConsistencyEvaluation,
    alignment: LabelAlignmentEvaluation,
    intensity: IntensityEvaluation,
) -> list[PreprocessingRecommendation]:
    recommendations: list[PreprocessingRecommendation] = []

    if failed_patients:
        recommendations.append(
            PreprocessingRecommendation(
                category="data_integrity",
                severity="high",
                message="Inspect files that failed during analysis before training.",
                reason=f"{len(failed_patients)} patient(s) could not be analyzed.",
            )
        )

    missing_annotations = sum(1 for patient in patients if patient.label_path is None)
    if missing_annotations:
        recommendations.append(
            PreprocessingRecommendation(
                category="annotations",
                severity="high",
                message="Check missing labels or separate unlabeled test data from training data.",
                reason=f"{missing_annotations} analyzed patient(s) have no annotation label.",
            )
        )

    if patients and consistency.percentage_same_voxel_spacing < 100.0:
        recommendations.append(
            PreprocessingRecommendation(
                category="spatial_resampling",
                severity="high",
                message=(
                    "Resample CT images and labels to a common voxel spacing before "
                    "AI training."
                ),
                reason=(
                    "Only "
                    f"{consistency.percentage_same_voxel_spacing}% of patients share "
                    "the most common voxel spacing."
                ),
            )
        )

    if patients and consistency.percentage_same_thickness < 100.0:
        recommendations.append(
            PreprocessingRecommendation(
                category="slice_thickness",
                severity="medium",
                message="Standardize slice thickness or use preprocessing that handles anisotropy.",
                reason=(
                    "Slice thickness differs across patients; the most common thickness "
                    f"covers {consistency.percentage_same_thickness}% of patients."
                ),
            )
        )

    if patients and not consistency.dimensions_are_consistent:
        recommendations.append(
            PreprocessingRecommendation(
                category="shape_standardization",
                severity="medium",
                message="Crop, pad, resize, or patch volumes to a consistent training shape.",
                reason=(
                    "Volume dimensions differ; the most common dimensions cover "
                    f"{consistency.percentage_same_dimensions}% of patients."
                ),
            )
        )

    if patients and consistency.percentage_same_orientation < 100.0:
        recommendations.append(
            PreprocessingRecommendation(
                category="orientation",
                severity="medium",
                message="Reorient all images and labels to a common anatomical orientation.",
                reason=(
                    "Image orientations differ; the most common orientation covers "
                    f"{consistency.percentage_same_orientation}% of patients."
                ),
            )
        )

    if alignment.misaligned_pairs:
        patient_ids = ", ".join(
            item.patient_id for item in alignment.misaligned_patients
        )
        recommendations.append(
            PreprocessingRecommendation(
                category="label_alignment",
                severity="high",
                message="Fix image/label alignment before model training.",
                reason=f"Misaligned patient(s): {patient_ids}.",
            )
        )

    if intensity.non_finite_voxel_count:
        recommendations.append(
            PreprocessingRecommendation(
                category="intensity_cleanup",
                severity="high",
                message="Replace or remove non-finite CT intensity values before training.",
                reason=(
                    f"{intensity.non_finite_voxel_count} non-finite voxel(s) were detected."
                ),
            )
        )

    if intensity.intensity_scale.normalized_patient_count:
        recommendations.append(
            PreprocessingRecommendation(
                category="intensity_scale",
                severity="medium",
                message="Confirm intensity scale before applying CT HU preprocessing.",
                reason=(
                    "Suspected normalized patient(s): "
                    + ", ".join(intensity.intensity_scale.normalized_patient_ids)
                    + "."
                ),
            )
        )

    if patients and not intensity.intensity_scale.normalized_patient_count:
        recommendations.append(
            PreprocessingRecommendation(
                category="intensity_normalization",
                severity="medium",
                message="Apply CT intensity clipping and normalization before model training.",
                reason=(
                    "CT images are commonly normalized to reduce scanner and protocol "
                    "differences."
                ),
            )
        )

    if memory.minimum_required_memory_bytes is not None:
        recommendations.append(
            PreprocessingRecommendation(
                category="memory",
                severity="info",
                message=(
                    "Plan at least the estimated minimum memory per loaded CT volume, "
                    "plus overhead for batching and preprocessing."
                ),
                reason=(
                    "Estimated minimum memory for this analyzer is "
                    f"{memory.minimum_required_memory_readable}."
                ),
            )
        )

    return recommendations


def _build_report_preparation(
    root: Path,
    split: str | None,
    structure: DatasetStructure,
    counts: DatasetCounts,
    evaluation: DatasetEvaluation,
) -> ReportPreparation:
    overview = ReportOverview(
        dataset_name=root.name,
        dataset_root=str(root),
        split_filter=split or "all",
        task_type=structure.task_type,
        task_reason=structure.task_reason,
        patients_detected=counts.patients_detected,
        patients_analyzed=counts.patients_analyzed,
        patients_failed=counts.patients_failed,
        analysis_success_percentage=_percentage(
            counts.patients_analyzed,
            counts.patients_detected,
        ),
        annotation_coverage_percentage=_percentage(
            counts.annotations_present,
            counts.patients_analyzed,
        ),
        total_storage=evaluation.storage.total_file_size_readable,
        minimum_memory_needed=evaluation.memory.minimum_required_memory_readable,
    )
    task_detection = ReportTaskDetection(
        detected_task=structure.task_type,
        reason=structure.task_reason,
        classification_classes=structure.classification_classes,
        labels_found=counts.label_files_detected > 0,
        label_files_detected=counts.label_files_detected,
        image_label_pairs_detected=counts.image_label_pairs_detected,
    )

    return ReportPreparation(
        overview=overview,
        task_detection=task_detection,
        missing_data=evaluation.missing_data,
        warnings=evaluation.warnings,
        preprocessing_recommendations=evaluation.preprocessing_recommendations,
        sections_ready=[
            "dataset_overview",
            "patient_counts",
            "missing_data",
            "task_detection",
            "image_label_alignment",
            "memory_requirements",
            "slice_count_and_thickness",
            "voxel_spacing",
            "physical_volume",
            "resolution",
            "dimensions",
            "intensity_statistics",
            "intensity_statistics_table",
            "intensity_percentiles",
            "intensity_scale",
            "voxel_validity",
            "warnings",
            "preprocessing_recommendations",
        ],
    )


def _image_dimensionality(dimensions: list[int]) -> str:
    if len(dimensions) >= 3 and dimensions[2] > 1:
        return "3D"
    if len(dimensions) >= 2:
        return "2D"
    return "unknown"


def _slice_count(dimensions: list[int]) -> int:
    if len(dimensions) >= 3:
        return int(dimensions[2])
    if len(dimensions) >= 2:
        return 1
    return 0


def _slice_thickness(metadata: NiftiMetadata) -> float | None:
    if len(metadata.voxel_spacing) < 3:
        return None
    return _rounded_float(metadata.voxel_spacing[2])


def _percentage(part: int | float, total: int | float) -> float:
    if total == 0:
        return 0.0
    return round((float(part) / float(total)) * 100.0, 3)


def _value_summary(values: Iterable[int | float | None]) -> ValueSummary:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return ValueSummary(
            count=0,
            total=0.0,
            min=None,
            max=None,
            mean=None,
            median=None,
        )

    return ValueSummary(
        count=len(numeric_values),
        total=float(sum(numeric_values)),
        min=float(min(numeric_values)),
        max=float(max(numeric_values)),
        mean=float(mean(numeric_values)),
        median=float(median(numeric_values)),
    )


def _numeric_summary(values: Iterable[int | float]) -> NumericSummary:
    numeric_values = [float(value) for value in values]
    if not numeric_values:
        return NumericSummary(
            count=0,
            total=0.0,
            total_readable="0 B",
            min=None,
            min_readable=None,
            max=None,
            max_readable=None,
            mean=None,
            mean_readable=None,
            median=None,
            median_readable=None,
        )

    min_value = float(min(numeric_values))
    max_value = float(max(numeric_values))
    mean_value = float(mean(numeric_values))
    median_value = float(median(numeric_values))
    total_value = float(sum(numeric_values))

    return NumericSummary(
        count=len(numeric_values),
        total=total_value,
        total_readable=_format_bytes(total_value),
        min=min_value,
        min_readable=_format_bytes(min_value),
        max=max_value,
        max_readable=_format_bytes(max_value),
        mean=mean_value,
        mean_readable=_format_bytes(mean_value),
        median=median_value,
        median_readable=_format_bytes(median_value),
    )


def _frequency_items(values: Iterable[Any]) -> list[FrequencyItem]:
    values_list = list(values)
    if not values_list:
        return []

    counter = Counter(values_list)
    percentages = _frequency_percentages(list(counter.values()))
    items = []
    for (value, count), percentage in zip(counter.items(), percentages, strict=True):
        json_value = _json_value(value)
        items.append(
            FrequencyItem(
                value=json_value,
                display_value=_format_frequency_value(json_value),
                count=count,
                percentage=percentage,
            )
        )

    return sorted(items, key=lambda item: (-item.count, str(item.value)))


def _slice_thickness_frequency_items(
    values: Iterable[float | None],
) -> list[SliceThicknessFrequencyItem]:
    values_list = list(values)
    if not values_list:
        return []

    counter = Counter(values_list)
    percentages = _frequency_percentages(list(counter.values()))
    items = [
        SliceThicknessFrequencyItem(
            thickness_mm=value,
            count=count,
            percentage=percentage,
        )
        for (value, count), percentage in zip(counter.items(), percentages, strict=True)
    ]
    return sorted(items, key=lambda item: (-item.count, item.thickness_mm or 0.0))


def _frequency_percentages(counts: list[int]) -> list[float]:
    """Return three-decimal percentages that add up to exactly 100.000."""

    total = sum(counts)
    if total == 0:
        return [0.0 for _ in counts]

    raw_units = [count * 100000 for count in counts]
    units = [value // total for value in raw_units]
    remainders = [value % total for value in raw_units]
    missing_units = 100000 - sum(units)

    for index in sorted(
        range(len(counts)),
        key=lambda item: (-remainders[item], -counts[item], item),
    )[:missing_units]:
        units[index] += 1

    return [unit / 1000.0 for unit in units]


def _top_percentage(
    frequencies: list[FrequencyItem] | list[SliceThicknessFrequencyItem],
) -> float:
    if not frequencies:
        return 0.0
    return frequencies[0].percentage


def _first_frequency_value(frequencies: list[FrequencyItem]) -> Any:
    if not frequencies:
        return None
    return frequencies[0].value


def _first_slice_thickness_value(
    frequencies: list[SliceThicknessFrequencyItem],
) -> float | None:
    if not frequencies:
        return None
    return frequencies[0].thickness_mm


def _rounded_tuple(
    values: Iterable[float],
    digits: int = SPATIAL_ROUNDING_DIGITS,
) -> tuple[float, ...]:
    return tuple(_rounded_float(value, digits) for value in values)


def _rounded_float(value: float, digits: int = SPATIAL_ROUNDING_DIGITS) -> float:
    return round(float(value), digits)


def _float_lists_close(
    first: list[float],
    second: list[float],
    tolerance: float,
) -> bool:
    if len(first) != len(second):
        return False
    return all(abs(float(left) - float(right)) <= tolerance for left, right in zip(first, second, strict=True))


def _matrix_max_absolute_difference(
    first: list[list[float]],
    second: list[list[float]],
) -> float | None:
    if len(first) != len(second):
        return None

    max_difference = 0.0
    for first_row, second_row in zip(first, second, strict=True):
        if len(first_row) != len(second_row):
            return None
        for first_value, second_value in zip(first_row, second_row, strict=True):
            difference = abs(float(first_value) - float(second_value))
            if difference > max_difference:
                max_difference = difference
    return max_difference


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _format_bytes(value: int | float | None) -> str | None:
    if value is None:
        return None

    size = float(value)
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    while abs(size) >= 1000.0 and unit_index < len(units) - 1:
        size /= 1000.0
        unit_index += 1

    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    return f"{size:.2f} {units[unit_index]}"


def _format_count(value: int, unit: str) -> str:
    return f"{value:,} {unit}"


def _format_summary(analysis: DatasetAnalysis) -> str:
    counts = analysis.counts
    consistency = analysis.evaluation.consistency
    alignment = analysis.evaluation.alignment
    storage = analysis.evaluation.storage
    memory = analysis.evaluation.memory
    intensity = analysis.evaluation.intensity

    lines = [
        f"Dataset root: {analysis.dataset_root}",
        f"Split filter: {analysis.split_filter or 'all'}",
        f"Task type: {analysis.structure.task_type}",
        f"Patients detected: {counts.patients_detected}",
        f"Patients analyzed: {counts.patients_analyzed}",
        f"Patients failed: {counts.patients_failed}",
        f"Patients with missing data: {len(analysis.evaluation.missing_data)}",
        f"Total storage: {storage.total_file_size_readable}",
        f"- Images storage: {storage.image_file_size_readable}",
        f"- Labels storage: {storage.label_file_size_readable}",
        f"Estimated minimum memory needed: {memory.minimum_required_memory_readable}",
        f"Largest native volume memory: {memory.largest_native_volume_readable}",
        f"Intensity range: {intensity.global_min} to {intensity.global_max}",
        f"Valid voxels: {intensity.voxel_validity.valid_voxels_readable} "
        f"({intensity.voxel_validity.valid_voxel_percentage}%)",
        f"Non-finite voxels: {intensity.voxel_validity.non_finite_voxels_readable} "
        f"({intensity.voxel_validity.non_finite_voxel_percentage}%)",
        f"Same in-plane resolution: {consistency.percentage_same_resolution}%",
        f"Same slice count: {consistency.percentage_same_slice_count}%",
        f"Same slice thickness: {consistency.percentage_same_thickness}%",
        f"Same voxel spacing: {consistency.percentage_same_voxel_spacing}%",
        f"Same dimensions: {consistency.percentage_same_dimensions}%",
        f"Same orientation: {consistency.percentage_same_orientation}%",
        f"Aligned image/label pairs: {alignment.aligned_pairs}/{alignment.checked_pairs} "
        f"({alignment.alignment_percentage}%)",
        "Suspected normalized CT patients: "
        f"{intensity.intensity_scale.normalized_patient_count}",
        "",
        "Image dimensionality:",
        *_format_frequency_lines(consistency.dimensionality_frequencies),
        "",
        "Orientations:",
        *_format_frequency_lines(consistency.orientation_frequencies),
        "",
        "Slice counts:",
        *_format_frequency_lines(consistency.slice_count_frequencies),
        "",
        "Slice count + thickness:",
        *_format_frequency_lines(consistency.slice_count_thickness_frequencies),
        "",
        "In-plane resolutions (X x Y mm):",
        *_format_frequency_lines(consistency.in_plane_resolution_frequencies),
        "",
        "Slice thicknesses (Z mm):",
        *_format_slice_thickness_lines(consistency.slice_thickness_frequencies),
        "",
        "Voxel spacings (X x Y x Z mm):",
        *_format_frequency_lines(consistency.voxel_spacing_frequencies),
        "",
        "Physical sizes (X x Y x Z mm):",
        *_format_frequency_lines(consistency.physical_size_frequencies),
        "",
        "Dimensions (voxels):",
        *_format_frequency_lines(consistency.dimension_frequencies),
        "",
        "Intensity statistics:",
        *_format_intensity_statistics_lines(intensity.patient_intensity_statistics_table),
        "",
        "Intensity scale:",
        f"- Raw HU-like patients: {intensity.intensity_scale.raw_hu_like_patient_count}",
        f"- Suspected normalized patients: {intensity.intensity_scale.normalized_patient_count}",
        f"- Unknown intensity scale: {intensity.intensity_scale.unknown_patient_count}",
        "",
        "Intensity percentiles:",
        *_format_percentile_summary_lines(intensity.patient_percentile_summaries),
    ]
    if alignment.misaligned_patients:
        lines.append("Misaligned image/label pairs:")
        lines.extend(
            f"- {item.patient_id}: {', '.join(item.issues)}"
            for item in alignment.misaligned_patients
        )
    if intensity.intensity_scale.normalized_patients:
        lines.append("Patients with suspected normalized intensity:")
        lines.extend(
            f"- {item.patient_id}: {item.reason}"
            for item in intensity.intensity_scale.normalized_patients
        )
    if analysis.evaluation.missing_data:
        lines.append("Missing data:")
        lines.extend(
            f"- {item.patient_id}: {', '.join(item.missing_fields)}"
            for item in analysis.evaluation.missing_data
        )
    if analysis.evaluation.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in analysis.evaluation.warnings)
    if analysis.evaluation.preprocessing_recommendations:
        lines.append("Preprocessing recommendations:")
        lines.extend(
            f"- [{recommendation.severity}] {recommendation.message}"
            for recommendation in analysis.evaluation.preprocessing_recommendations
        )
    return "\n".join(lines)


def _format_frequency_lines(frequencies: list[FrequencyItem]) -> list[str]:
    if not frequencies:
        return ["- none"]
    return [
        f"- {item.display_value}: "
        f"{item.count} patient(s), {item.percentage}%"
        for item in frequencies
    ]


def _format_slice_thickness_lines(
    frequencies: list[SliceThicknessFrequencyItem],
) -> list[str]:
    if not frequencies:
        return ["- none"]
    return [
        f"- {_format_report_value(item.thickness_mm)} mm: "
        f"{item.count} patient(s), {item.percentage}%"
        for item in frequencies
    ]


def _format_intensity_statistics_lines(
    statistics: list[IntensityStatisticItem],
) -> list[str]:
    if not statistics:
        return ["- none"]
    return [
        f"- {item.statistic} ({item.unit}): min {item.min}, max {item.max}, "
        f"mean {item.mean}, median {item.median}"
        for item in statistics
    ]


def _format_frequency_value(value: Any) -> str:
    if isinstance(value, list):
        return " x ".join(_format_report_value(item) for item in value)
    return _format_report_value(value)


def _format_report_value(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _format_percentile_summary_lines(
    percentile_summaries: dict[str, ValueSummary],
) -> list[str]:
    if not percentile_summaries:
        return ["- none"]
    return [
        f"- {item.percentile}: median {item.median}, mean {item.mean}"
        for item in _percentile_table(percentile_summaries)
    ]


def _format_percentile_label(percentile_key: str) -> str:
    return percentile_key.upper().replace("_", ".")


def _percentile_sort_value(percentile_key: str) -> float:
    return float(percentile_key.removeprefix("p").replace("_", "."))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze all detected NIfTI CT patients in a dataset."
    )
    parser.add_argument("dataset_root", help="Path to the dataset root.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=(
            "JSON output path. Defaults to "
            "data/dataset_analysis/analyse_dataset.json."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "validation", "unknown", "all"],
        default="train",
        help=(
            "Split filter. Defaults to train. Use --split all to include every "
            "detected split."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full analysis as JSON instead of a short summary.",
    )
    args = parser.parse_args(argv)

    selected_split = None if args.split == "all" else args.split
    try:
        analysis = analyze_dataset(args.dataset_root, split=selected_split)
    except DatasetNiftiPreparationError as exc:
        print(str(exc))
        print(f"Warning file: {exc.warning_path}")
        return 1

    if args.json:
        print(json.dumps(analysis.to_dict(), indent=2))
    else:
        save_dataset_analysis(analysis, args.output)
        print(f"Dataset analysis written to: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
