"""Dataset structure discovery for NIfTI medical imaging datasets.

This module implements Step 0 of the roadmap:
- detect image and label folders;
- infer image/label relationships;
- infer whether the dataset is likely segmentation or classification.

It intentionally does not read NIfTI headers or voxel data. That work belongs
to the later NIfTI analysis steps.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


NIFTI_EXTENSIONS = (".nii.gz", ".nii")

IMAGE_FOLDER_HINTS = {
    "ct",
    "cts",
    "image",
    "images",
    "imagestr",
    "imagests",
    "imagesval",
    "img",
    "imgs",
    "scan",
    "scans",
    "volume",
    "volumes",
}

LABEL_FOLDER_HINTS = {
    "annotation",
    "annotations",
    "groundtruth",
    "gt",
    "label",
    "labels",
    "labelstr",
    "labelsts",
    "labelsval",
    "mask",
    "masks",
    "seg",
    "segs",
    "segmentation",
    "segmentations",
}

IMAGE_FOLDER_SUBSTRINGS = {"image", "serie", "series"}
LABEL_FOLDER_SUBSTRINGS = {"label"}

IMAGE_NAME_HINTS = {
    "ct",
    "image",
    "img",
    "scan",
    "serie",
    "series",
    "volume",
}

LABEL_NAME_HINTS = {
    "annotation",
    "gt",
    "label",
    "mask",
    "seg",
    "segmentation",
}

SPLIT_HINTS = {
    "train": "train",
    "training": "train",
    "tr": "train",
    "test": "test",
    "testing": "test",
    "ts": "test",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
}

CLASSIFICATION_COLUMN_HINTS = {
    "category",
    "class",
    "diagnosis",
    "group",
    "label",
    "labels",
    "target",
}

CLASS_FOLDER_HINTS = {
    "abnormal",
    "benign",
    "cancer",
    "control",
    "disease",
    "healthy",
    "malignant",
    "negative",
    "normal",
    "positive",
    "tumor",
}

PATIENT_FOLDER_HINTS = {
    "case",
    "id",
    "patient",
    "study",
    "sub",
    "subject",
}

METADATA_EXTENSIONS = {".csv", ".json", ".tsv", ".xlsx"}
IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}


@dataclass
class FolderSummary:
    """Summary of one folder containing NIfTI files."""

    path: str
    role: str
    split: str
    nifti_count: int


@dataclass
class NiftiFileSummary:
    """File-level structure information without opening the NIfTI content."""

    path: str
    folder: str
    role: str
    split: str
    subject_id: str


@dataclass
class ImageLabelPair:
    """Detected relationship between one image and an optional label."""

    subject_id: str
    image: str
    label: str | None
    split: str
    source: str


@dataclass
class DatasetStructure:
    """Serializable result of Step 0 dataset discovery."""

    root: str
    task_type: str
    task_reason: str
    training_folders: list[str]
    test_folders: list[str]
    classification_classes: list[str]
    image_folders: list[str]
    label_folders: list[str]
    unknown_folders: list[str]
    folder_summaries: list[FolderSummary]
    image_files: list[NiftiFileSummary]
    label_files: list[NiftiFileSummary]
    unknown_files: list[NiftiFileSummary]
    metadata_files: list[str]
    image_label_pairs: list[ImageLabelPair]
    unmatched_images: list[str]
    unmatched_labels: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return asdict(self)


def identify_dataset_structure(dataset_root: str | Path) -> DatasetStructure:
    """Identify folders, file roles, image/label pairs, and likely task type.

    Parameters
    ----------
    dataset_root:
        Root directory of the dataset to inspect.

    Returns
    -------
    DatasetStructure
        Step 0 discovery result. Paths are relative to ``dataset_root``.
    """

    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    warnings: list[str] = []
    # Prefer dataset.json when it exists because it gives explicit image/label
    # relationships. Folder and filename heuristics are used as a fallback for
    # datasets that do not follow the Decathlon manifest format.
    manifest = _load_dataset_manifest(root, warnings)
    manifest_roles = _manifest_file_roles(manifest)
    manifest_pairs = _pairs_from_manifest(root, manifest)

    nifti_paths = _find_nifti_files(root)
    metadata_files = _find_metadata_files(root)
    folder_roles = _infer_folder_roles(root, nifti_paths, manifest_roles)
    file_summaries = [
        _summarize_nifti_file(root, path, folder_roles, manifest_roles)
        for path in nifti_paths
    ]

    image_files = [item for item in file_summaries if item.role == "image"]
    label_files = [item for item in file_summaries if item.role == "label"]
    unknown_files = [item for item in file_summaries if item.role == "unknown"]

    heuristic_pairs = _pair_images_and_labels(image_files, label_files)
    image_label_pairs = _merge_pairs(manifest_pairs, heuristic_pairs)
    unmatched_images, unmatched_labels = _find_unmatched_files(
        image_files,
        label_files,
        image_label_pairs,
    )

    folder_summaries = _build_folder_summaries(root, nifti_paths, folder_roles)
    training_folders = sorted(
        item.path for item in folder_summaries if item.split == "train"
    )
    test_folders = sorted(
        item.path for item in folder_summaries if item.split == "test"
    )
    image_folders = sorted(item.path for item in folder_summaries if item.role == "image")
    label_folders = sorted(item.path for item in folder_summaries if item.role == "label")
    unknown_folders = sorted(
        item.path for item in folder_summaries if item.role == "unknown"
    )

    task_type, task_reason = _infer_task_type(
        root=root,
        manifest=manifest,
        metadata_files=metadata_files,
        image_files=image_files,
        label_files=label_files,
        pairs=image_label_pairs,
    )
    classification_classes = _classification_classes(root, image_files, task_type)

    if unknown_files:
        warnings.append(
            f"{len(unknown_files)} NIfTI file(s) could not be classified as image or label."
        )
    if label_files and not image_label_pairs:
        warnings.append("Label files were detected, but no image/label pair was found.")

    return DatasetStructure(
        root=str(root),
        task_type=task_type,
        task_reason=task_reason,
        training_folders=training_folders,
        test_folders=test_folders,
        classification_classes=classification_classes,
        image_folders=image_folders,
        label_folders=label_folders,
        unknown_folders=unknown_folders,
        folder_summaries=folder_summaries,
        image_files=image_files,
        label_files=label_files,
        unknown_files=unknown_files,
        metadata_files=sorted(_relative_path(root, path) for path in metadata_files),
        image_label_pairs=image_label_pairs,
        unmatched_images=unmatched_images,
        unmatched_labels=unmatched_labels,
        warnings=warnings,
    )


def _load_dataset_manifest(root: Path, warnings: list[str]) -> dict[str, Any]:
    manifest_path = root / "dataset.json"
    if not manifest_path.exists():
        return {}

    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        warnings.append(f"dataset.json is not valid JSON: {exc}")
        return {}

    if not isinstance(data, dict):
        warnings.append("dataset.json was ignored because it is not a JSON object.")
        return {}
    return data


def _manifest_file_roles(manifest: dict[str, Any]) -> dict[str, str]:
    roles: dict[str, str] = {}
    training_items = manifest.get("training")
    test_items = manifest.get("test")

    if isinstance(training_items, list):
        for item in training_items:
            if isinstance(item, dict):
                image = item.get("image")
                label = item.get("label")
                if isinstance(image, str):
                    roles[_normalize_manifest_path(image)] = "image"
                if isinstance(label, str):
                    roles[_normalize_manifest_path(label)] = "label"
            elif isinstance(item, str):
                roles[_normalize_manifest_path(item)] = "image"

    if isinstance(test_items, list):
        for item in test_items:
            if isinstance(item, str):
                roles[_normalize_manifest_path(item)] = "image"
            elif isinstance(item, dict):
                image = item.get("image")
                if isinstance(image, str):
                    roles[_normalize_manifest_path(image)] = "image"

    return roles


def _pairs_from_manifest(root: Path, manifest: dict[str, Any]) -> list[ImageLabelPair]:
    pairs: list[ImageLabelPair] = []
    training_items = manifest.get("training")
    if not isinstance(training_items, list):
        return pairs

    for item in training_items:
        if not isinstance(item, dict):
            continue
        image = item.get("image")
        label = item.get("label")
        if not isinstance(image, str) or not isinstance(label, str):
            continue

        image_path = _normalize_manifest_path(image)
        label_path = _normalize_manifest_path(label)
        pairs.append(
            ImageLabelPair(
                subject_id=_subject_id_from_name(Path(image_path).name),
                image=image_path,
                label=label_path,
                split=_infer_split_from_relative_path(image_path),
                source="dataset.json",
            )
        )

    return pairs


def _find_nifti_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in IGNORED_DIRS and not dirname.startswith(".")
        ]
        for filename in filenames:
            if _is_nifti_filename(filename):
                files.append(Path(current_root) / filename)
    return sorted(files, key=lambda path: _relative_path(root, path))


def _find_metadata_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in IGNORED_DIRS and not dirname.startswith(".")
        ]
        for filename in filenames:
            if filename.startswith("._"):
                continue
            path = Path(current_root) / filename
            if path.suffix.lower() in METADATA_EXTENSIONS:
                files.append(path)
    return sorted(files, key=lambda path: _relative_path(root, path))


def _infer_folder_roles(
    root: Path,
    nifti_paths: Iterable[Path],
    manifest_roles: dict[str, str],
) -> dict[str, str]:
    folder_role_votes: dict[str, list[str]] = defaultdict(list)
    folders = {_relative_path(root, path.parent) for path in nifti_paths}

    # Roles from dataset.json are trusted first; they can identify folders even
    # when the folder names do not contain obvious words such as image or label.
    for relative_path, role in manifest_roles.items():
        folder = str(Path(relative_path).parent).replace("\\", "/")
        if folder == ".":
            folder = ""
        if role in {"image", "label"}:
            folder_role_votes[folder].append(role)
            folders.add(folder)

    roles: dict[str, str] = {}
    for folder in folders:
        votes = folder_role_votes.get(folder, [])
        if votes:
            roles[folder] = _majority_vote(votes)
            continue

        # If no manifest role exists, infer the role from folder tokens. This
        # intentionally supports names like imagesTr, labelsTr, raw_image, or
        # manual_label_maps without requiring exact folder names.
        tokens = _path_tokens(folder)
        label_score = _role_score(
            tokens,
            exact_hints=LABEL_FOLDER_HINTS,
            substring_hints=LABEL_FOLDER_SUBSTRINGS,
        )
        image_score = _role_score(
            tokens,
            exact_hints=IMAGE_FOLDER_HINTS,
            substring_hints=IMAGE_FOLDER_SUBSTRINGS,
        )
        if label_score > image_score:
            roles[folder] = "label"
        elif image_score > label_score:
            roles[folder] = "image"
        else:
            roles[folder] = "unknown"

    _promote_class_folder_roles(roles)
    return roles


def _promote_class_folder_roles(roles: dict[str, str]) -> None:
    """Treat class-organized folders as image folders when no labels exist."""

    if any(role == "label" for role in roles.values()):
        return

    candidate_keys = {
        key
        for folder, role in roles.items()
        if role == "unknown"
        for key in [_class_folder_key(folder)]
        if key is not None
    }
    if not _looks_like_class_folder_set(candidate_keys):
        return

    for folder, role in list(roles.items()):
        if role != "unknown":
            continue
        if _class_folder_key(folder) in candidate_keys:
            roles[folder] = "image"


def _summarize_nifti_file(
    root: Path,
    path: Path,
    folder_roles: dict[str, str],
    manifest_roles: dict[str, str],
) -> NiftiFileSummary:
    relative_path = _relative_path(root, path)
    folder = _relative_path(root, path.parent)
    role = manifest_roles.get(relative_path)
    if role is None:
        role = folder_roles.get(folder, "unknown")
    if role == "unknown":
        role = _infer_role_from_filename(path.name)

    return NiftiFileSummary(
        path=relative_path,
        folder=folder,
        role=role,
        split=_infer_split_from_relative_path(relative_path),
        subject_id=_subject_id_from_name(path.name),
    )


def _pair_images_and_labels(
    image_files: list[NiftiFileSummary],
    label_files: list[NiftiFileSummary],
) -> list[ImageLabelPair]:
    labels_by_id: dict[str, list[NiftiFileSummary]] = defaultdict(list)
    for label in label_files:
        labels_by_id[label.subject_id].append(label)

    pairs: list[ImageLabelPair] = []
    for image in image_files:
        candidates = labels_by_id.get(image.subject_id, [])
        if not candidates:
            continue

        label = _best_label_candidate(image, candidates)
        pairs.append(
            ImageLabelPair(
                subject_id=image.subject_id,
                image=image.path,
                label=label.path,
                split=image.split,
                source="filename",
            )
        )

    return sorted(pairs, key=lambda pair: (pair.subject_id, pair.image, pair.label or ""))


def _merge_pairs(
    manifest_pairs: list[ImageLabelPair],
    heuristic_pairs: list[ImageLabelPair],
) -> list[ImageLabelPair]:
    merged: dict[tuple[str, str | None], ImageLabelPair] = {}
    for pair in heuristic_pairs:
        merged[(pair.image, pair.label)] = pair
    for pair in manifest_pairs:
        merged[(pair.image, pair.label)] = pair
    return sorted(
        merged.values(),
        key=lambda pair: (pair.split, pair.subject_id, pair.image, pair.label or ""),
    )


def _find_unmatched_files(
    image_files: list[NiftiFileSummary],
    label_files: list[NiftiFileSummary],
    pairs: list[ImageLabelPair],
) -> tuple[list[str], list[str]]:
    paired_images = {pair.image for pair in pairs}
    paired_labels = {pair.label for pair in pairs if pair.label is not None}

    unmatched_images = sorted(
        image.path
        for image in image_files
        if image.path not in paired_images and image.split != "test"
    )
    unmatched_labels = sorted(label.path for label in label_files if label.path not in paired_labels)
    return unmatched_images, unmatched_labels


def _build_folder_summaries(
    root: Path,
    nifti_paths: Iterable[Path],
    folder_roles: dict[str, str],
) -> list[FolderSummary]:
    counts: dict[str, int] = defaultdict(int)
    for path in nifti_paths:
        counts[_relative_path(root, path.parent)] += 1

    summaries = [
        FolderSummary(
            path=folder,
            role=folder_roles.get(folder, "unknown"),
            split=_infer_split_from_relative_path(folder),
            nifti_count=count,
        )
        for folder, count in counts.items()
    ]
    return sorted(summaries, key=lambda item: (item.role, item.path))


def _infer_task_type(
    root: Path,
    manifest: dict[str, Any],
    metadata_files: list[Path],
    image_files: list[NiftiFileSummary],
    label_files: list[NiftiFileSummary],
    pairs: list[ImageLabelPair],
) -> tuple[str, str]:
    if _manifest_describes_segmentation(manifest):
        return "segmentation", "dataset.json contains image/label training pairs."

    if label_files and pairs:
        return "segmentation", "Matching image and label NIfTI files were detected."

    metadata_reason = _classification_metadata_reason(metadata_files)
    if image_files and metadata_reason:
        return "classification", metadata_reason

    class_folders = _class_folder_candidates(root, image_files)
    if len(class_folders) >= 2:
        folders = ", ".join(sorted(class_folders))
        return "classification", f"Images are grouped in class-like folders: {folders}."

    if image_files and not label_files:
        return "classification", "No label NIfTI files were detected."

    if label_files and not pairs:
        return "unknown", "Label NIfTI files exist but could not be linked to images."

    return "unknown", "No usable NIfTI image structure was detected."


def _classification_classes(
    root: Path,
    image_files: list[NiftiFileSummary],
    task_type: str,
) -> list[str]:
    if task_type != "classification":
        return []
    return sorted(_class_folder_candidates(root, image_files))


def _manifest_describes_segmentation(manifest: dict[str, Any]) -> bool:
    training_items = manifest.get("training")
    if not isinstance(training_items, list):
        return False

    return any(
        isinstance(item, dict)
        and isinstance(item.get("image"), str)
        and isinstance(item.get("label"), str)
        for item in training_items
    )


def _classification_metadata_reason(metadata_files: Iterable[Path]) -> str | None:
    for path in metadata_files:
        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            columns = _read_table_header(path, delimiter)
            if CLASSIFICATION_COLUMN_HINTS.intersection(columns):
                return f"{path.name} contains classification-like columns."
        elif suffix == ".json" and path.name != "dataset.json":
            if _json_contains_classification_keys(path):
                return f"{path.name} contains classification-like keys."
    return None


def _read_table_header(path: Path, delimiter: str) -> set[str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file, delimiter=delimiter)
            header = next(reader, [])
    except (OSError, UnicodeDecodeError):
        return set()
    return {_normalize_token(column) for column in header}


def _json_contains_classification_keys(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False

    keys: set[str] = set()
    if isinstance(data, dict):
        keys.update(_normalize_token(key) for key in data)
        items = data.values()
    elif isinstance(data, list):
        items = data
    else:
        return False

    for item in items:
        if isinstance(item, dict):
            keys.update(_normalize_token(key) for key in item)
    return bool(CLASSIFICATION_COLUMN_HINTS.intersection(keys))


def _class_folder_candidates(
    root: Path,
    image_files: list[NiftiFileSummary],
) -> set[str]:
    candidates: set[str] = set()
    for image in image_files:
        key = _class_folder_key(str(Path(image.path).parent))
        if key is None:
            continue
        if (root / key).exists() or (root / image.folder).exists():
            candidates.add(key)
    return candidates


def _class_folder_key(folder: str) -> str | None:
    for part in folder.replace("\\", "/").split("/"):
        if not part:
            continue
        tokens = _path_tokens(part)
        if any(token in SPLIT_HINTS for token in tokens):
            continue
        if _role_score(
            tokens,
            exact_hints=IMAGE_FOLDER_HINTS,
            substring_hints=IMAGE_FOLDER_SUBSTRINGS,
        ):
            return None
        if _role_score(
            tokens,
            exact_hints=LABEL_FOLDER_HINTS,
            substring_hints=LABEL_FOLDER_SUBSTRINGS,
        ):
            return None
        return part
    return None


def _looks_like_class_folder_set(candidate_keys: set[str]) -> bool:
    if len(candidate_keys) < 2:
        return False

    normalized_keys = {_normalize_token(key) for key in candidate_keys}
    if CLASS_FOLDER_HINTS.intersection(normalized_keys):
        return True

    if len(candidate_keys) > 20:
        return False

    return not any(_looks_like_patient_folder(key) for key in candidate_keys)


def _looks_like_patient_folder(folder: str) -> bool:
    tokens = _filename_tokens(folder)
    return any(token in PATIENT_FOLDER_HINTS or token.isdigit() for token in tokens)


def _best_label_candidate(
    image: NiftiFileSummary,
    candidates: list[NiftiFileSummary],
) -> NiftiFileSummary:
    same_split = [label for label in candidates if label.split == image.split]
    if same_split:
        return sorted(same_split, key=lambda label: label.path)[0]
    return sorted(candidates, key=lambda label: label.path)[0]


def _infer_role_from_filename(filename: str) -> str:
    tokens = _filename_tokens(filename)
    label_score = sum(token in LABEL_NAME_HINTS for token in tokens)
    image_score = sum(token in IMAGE_NAME_HINTS for token in tokens)
    if label_score > image_score:
        return "label"
    if image_score > label_score:
        return "image"
    return "unknown"


def _role_score(
    tokens: Iterable[str],
    exact_hints: set[str],
    substring_hints: set[str],
) -> int:
    score = 0
    for token in tokens:
        if token in exact_hints:
            score += 2
        if any(hint in token for hint in substring_hints):
            score += 1
    return score


def _infer_split_from_relative_path(relative_path: str) -> str:
    for token in _path_tokens(relative_path):
        if token in SPLIT_HINTS:
            return SPLIT_HINTS[token]

        for suffix, split in SPLIT_HINTS.items():
            if token.endswith(suffix) and len(token) > len(suffix):
                return split
    return "unknown"


def _subject_id_from_name(filename: str) -> str:
    stem = _strip_nifti_extension(filename)
    tokens = _filename_tokens(stem)
    # Remove role and split words so image and label filenames can collapse to
    # the same patient id, for example patient_01_ct and patient_01_mask.
    tokens = [
        token
        for token in tokens
        if token not in IMAGE_NAME_HINTS
        and token not in LABEL_NAME_HINTS
        and token not in SPLIT_HINTS
    ]
    if tokens and tokens[-1].isdigit() and len(tokens[-1]) == 4:
        tokens = tokens[:-1]
    if not tokens:
        return _normalize_token(stem)
    return "_".join(tokens)


def _filename_tokens(filename: str) -> list[str]:
    stem = _strip_nifti_extension(filename)
    return [token for token in re.split(r"[^a-z0-9]+", stem.lower()) if token]


def _path_tokens(path: str) -> list[str]:
    tokens: list[str] = []
    for part in path.replace("\\", "/").split("/"):
        normalized = _normalize_token(part)
        if not normalized:
            continue
        tokens.append(normalized)
        tokens.extend(_split_known_suffix(normalized))
    return tokens


def _split_known_suffix(token: str) -> list[str]:
    split_tokens: list[str] = []
    for suffix in sorted(SPLIT_HINTS, key=len, reverse=True):
        if token.endswith(suffix) and len(token) > len(suffix):
            split_tokens.append(token[: -len(suffix)])
            split_tokens.append(suffix)
            break
    return split_tokens


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _strip_nifti_extension(filename: str) -> str:
    lower = filename.lower()
    for extension in NIFTI_EXTENSIONS:
        if lower.endswith(extension):
            return filename[: -len(extension)]
    return Path(filename).stem


def _is_nifti_filename(filename: str) -> bool:
    if filename.startswith("._"):
        return False
    lower = filename.lower()
    return any(lower.endswith(extension) for extension in NIFTI_EXTENSIONS)


def _normalize_manifest_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _relative_path(root: Path, path: Path) -> str:
    relative = path.resolve().relative_to(root).as_posix()
    return "" if relative == "." else relative


def _majority_vote(values: list[str]) -> str:
    return max(set(values), key=lambda value: (values.count(value), value))


def _format_summary(structure: DatasetStructure) -> str:
    lines = [
        f"Dataset root: {structure.root}",
        f"Task type: {structure.task_type} ({structure.task_reason})",
        f"Training folders: {len(structure.training_folders)}",
        f"Test folders: {len(structure.test_folders)}",
        f"Classification classes: {', '.join(structure.classification_classes) or 'none'}",
        f"Image folders: {len(structure.image_folders)}",
        f"Label folders: {len(structure.label_folders)}",
        f"Image files: {len(structure.image_files)}",
        f"Label files: {len(structure.label_files)}",
        f"Image/label pairs: {len(structure.image_label_pairs)}",
        f"Unmatched images: {len(structure.unmatched_images)}",
        f"Unmatched labels: {len(structure.unmatched_labels)}",
    ]
    if structure.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in structure.warnings)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Identify the structure of a NIfTI medical imaging dataset."
    )
    parser.add_argument("dataset_root", help="Path to the dataset root.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full structure as JSON.",
    )
    args = parser.parse_args(argv)

    structure = identify_dataset_structure(args.dataset_root)
    if args.json:
        print(json.dumps(structure.to_dict(), indent=2))
    else:
        print(_format_summary(structure))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
