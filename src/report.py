"""Generate an HTML report from data/dataset_analysis/analyse_dataset.json.

The report is intentionally dependency-free and focuses on medical dataset
quality information useful before AI training.
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from textwrap import wrap
from typing import Any, Iterable


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_DIR / "data" / "dataset_analysis"
DEFAULT_INPUT = REPORT_DIR / "analyse_dataset.json"
DEFAULT_HTML_OUTPUT = REPORT_DIR / "report.html"
DEFAULT_PDF_OUTPUT = REPORT_DIR / "report.pdf"


def generate_reports(
    input_path: str | Path = DEFAULT_INPUT,
    html_output_path: str | Path = DEFAULT_HTML_OUTPUT,
    pdf_output_path: str | Path | None = None,
) -> None:
    analysis = _load_json(input_path)
    html_report = _build_html_report(analysis)

    html_destination = Path(html_output_path).expanduser().resolve()
    html_destination.parent.mkdir(parents=True, exist_ok=True)
    html_destination.write_text(html_report, encoding="utf-8")

    if pdf_output_path is not None:
        pdf_lines = _build_text_report(analysis)
        pdf_destination = Path(pdf_output_path).expanduser().resolve()
        pdf_destination.parent.mkdir(parents=True, exist_ok=True)
        _write_pdf(pdf_destination, pdf_lines)


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as file:
        return json.load(file)


def _build_html_report(analysis: dict[str, Any]) -> str:
    overview = analysis["report_preparation"]["overview"]
    task_detection = analysis["report_preparation"]["task_detection"]
    counts = analysis["counts"]
    evaluation = analysis["evaluation"]
    consistency = evaluation["consistency"]
    alignment = evaluation["alignment"]
    intensity = evaluation["intensity"]
    intensity_scale = intensity["intensity_scale"]

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    sections = [
        _html_header(overview, generated_at),
        _section(
            "Dataset Overview",
            _table(
                ["Field", "Value"],
                [
                    ["Dataset name", overview["dataset_name"]],
                    ["Split", overview["split_filter"]],
                    ["Task type", overview["task_type"]],
                    ["Task reason", overview["task_reason"]],
                    ["Minimum memory needed", overview["minimum_memory_needed"]],
                ],
                class_name="kv",
            ),
        ),
        _section(
            "Patients Information",
            _table(
                ["Metric", "Value"],
                [
                    ["Number of patients", counts["patients_detected"]],
                    ["Number of analysed patients", counts["patients_analyzed"]],
                    ["Number of failed analyses", counts["patients_failed"]],
                    [
                        "Number of patients with missing data",
                        len(evaluation["missing_data"]),
                    ],
                    ["Annotations present", counts["annotations_present"]],
                    ["Annotations missing", counts["annotations_missing"]],
                    [
                        "Analysis success",
                        _format_percentage(overview["analysis_success_percentage"]),
                    ],
                    [
                        "Annotation coverage",
                        _format_percentage(
                            overview["annotation_coverage_percentage"]
                        ),
                    ],
                ],
                class_name="kv",
            ),
        ),
        _section(
            "Task Detection",
            _table(
                ["Field", "Value"],
                [
                    ["Detected task", task_detection["detected_task"]],
                    ["Reason", task_detection["reason"]],
                    [
                        "Classification classes",
                        _format_text_list(task_detection.get("classification_classes", [])),
                    ],
                    ["Labels found", _yes_no(task_detection["labels_found"])],
                    ["Label files detected", task_detection["label_files_detected"]],
                    [
                        "Image/label pairs detected",
                        task_detection["image_label_pairs_detected"],
                    ],
                ],
                class_name="kv",
            ),
        ),
        _section("Missing Data", _missing_data_table(evaluation["missing_data"])),
        _section(
            "Image/Label Alignment",
            _dict_table(
                [
                    ["Checked image/label pairs", alignment["checked_pairs"]],
                    ["Aligned pairs", alignment["aligned_pairs"]],
                    ["Misaligned pairs", alignment["misaligned_pairs"]],
                    [
                        "Alignment",
                        _format_percentage(alignment["alignment_percentage"]),
                    ],
                    ["Explanation", alignment["explanation"]],
                ]
            )
            + _subsection(
                "Misaligned Patients",
                _misaligned_patients_table(alignment["misaligned_patients"]),
            ),
        ),
        _section(
            "Slice Count and Thickness",
            _subsection("Dimensionality", _frequency_table(consistency["dimensionality_frequencies"]))
            + _subsection(
                "Slices data",
                _slice_number_thickness_table(
                    consistency["slice_count_thickness_frequencies"]
                ),
            )
        ),
        _section(
            "Voxel Size",
            _subsection(
                "Voxel Size (XxYxZ mm)",
                _frequency_table(consistency["voxel_spacing_frequencies"]),
            ),
        ),
        _section(
            "Intensity Statistics",
            _subsection(
                "Patient Intensity Summary",
                _intensity_summary_table(intensity),
            )
            + _subsection(
                "Intensity Scale Check",
                _dict_table(
                    [
                        [
                            "Raw HU-like patients",
                            intensity_scale["raw_hu_like_patient_count"],
                        ],
                        [
                            "Suspected normalized patients",
                            intensity_scale["normalized_patient_count"],
                        ],
                        [
                            "Unknown intensity scale",
                            intensity_scale["unknown_patient_count"],
                        ],
                        [
                            "Suspected normalized patient IDs",
                            _format_patient_ids(
                                intensity_scale["normalized_patient_ids"]
                            ),
                        ],
                        ["Explanation", intensity_scale["explanation"]],
                    ]
                )
                + _normalized_patients_table(
                    intensity_scale["normalized_patients"]
                ),
            )
            + _subsection(
                "Voxel Validity",
                _dict_table(
                    [
                        [
                            "Total voxels",
                            _format_voxel_count_text(
                                intensity["voxel_validity"]["total_voxels_readable"]
                            ),
                        ],
                        [
                            "Valid voxels",
                            f"{_format_voxel_count_text(intensity['voxel_validity']['valid_voxels_readable'])} "
                            f"({_format_percentage(intensity['voxel_validity']['valid_voxel_percentage'])})",
                        ],
                        [
                            "Non-finite voxels",
                            f"{_format_voxel_count_text(intensity['voxel_validity']['non_finite_voxels_readable'])} "
                            f"({_format_percentage(intensity['voxel_validity']['non_finite_voxel_percentage'])})",
                        ],
                        ["Explanation", intensity["voxel_validity"]["explanation"]],
                    ]
                ),
            ),
        ),
        _section(
            "Statistics",
            _table(
                ["Metric", "Rate"],
                [
                    [
                        "Same in-plane resolution",
                        _format_percentage(consistency["percentage_same_resolution"]),
                    ],
                    [
                        "Same slice count",
                        _format_percentage(consistency["percentage_same_slice_count"]),
                    ],
                    [
                        "Same slice thickness",
                        _format_percentage(consistency["percentage_same_thickness"]),
                    ],
                    [
                        "Same voxel spacing",
                        _format_percentage(
                            consistency["percentage_same_voxel_spacing"]
                        ),
                    ],
                    [
                        "Different voxel spacing",
                        _format_percentage(
                            consistency["percentage_different_voxel_spacing"]
                        ),
                    ],
                    [
                        "Same dimensions",
                        _format_percentage(consistency["percentage_same_dimensions"]),
                    ],
                    [
                        "Same orientation",
                        _format_percentage(consistency["percentage_same_orientation"]),
                    ],
                ],
                class_name="kv",
            )
            + _subsection(
                "Consistency Status",
                _table(
                    ["Metric", "Status"],
                    [
                        [
                            "Dimensions are consistent",
                            _yes_no(consistency["dimensions_are_consistent"]),
                        ]
                    ],
                    class_name="kv",
                ),
            ),
        ),
        _section("Warnings", _bullet_list(evaluation["warnings"])),
        _section(
            "Preprocessing Recommendations",
            _recommendations_table(evaluation["preprocessing_recommendations"]),
        ),
    ]

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en" tkinterweb-overflow-x="auto">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>Medical Imaging Dataset Analysis Report</title>",
            _css(),
            "</head>",
            "<body>",
            *sections,
            "</body>",
            "</html>",
        ]
    )


def _html_header(overview: dict[str, Any], generated_at: str) -> str:
    return f"""
<header class="report-header">
  <div class="eyebrow">Medical Imaging Dataset Analysis</div>
  <h1 class="report-title">{_escape(overview["dataset_name"])}</h1>
  <div class="report-date">Generated on {_escape(generated_at)}</div>
</header>
"""


def _css() -> str:
    return """
<style>
:root {
  --text: #172026;
  --muted: #5f6b73;
  --line: #d8dee3;
  --panel: #f6f8fa;
  --accent: #1f6feb;
}
* { box-sizing: border-box; }
html {
  max-width: 100%;
  overflow-x: hidden;
}
body {
  margin: 0;
  padding: 32px;
  max-width: 100%;
  overflow-x: hidden;
  color: var(--text);
  font-family: Arial, Helvetica, sans-serif;
  font-size: 14px;
  background: white;
  line-height: 1.5;
}
.report-header {
  border-bottom: 2px solid var(--line);
  margin-bottom: 28px;
  padding-bottom: 18px;
}
.eyebrow {
  color: var(--accent);
  font-size: 10px;
  font-weight: 700;
  line-height: 1.6;
  letter-spacing: .08em;
  margin: 0;
  padding: 0 0 22px;
  text-transform: uppercase;
}
.report-title {
  font-size: 27px;
  line-height: 1.25;
  margin: 0;
  padding: 0 0 14px;
}
.report-date {
  font-size: 13px;
  line-height: 1.5;
  margin: 0;
}
h2 {
  border-bottom: 1px solid var(--line);
  font-size: 19px;
  line-height: 1.35;
  margin: 34px 0 14px;
  padding-bottom: 6px;
}
h3 {
  color: #26333b;
  font-size: 15px;
  line-height: 1.4;
  margin: 20px 0 8px;
}
table {
  border-collapse: collapse;
  table-layout: auto;
  margin: 8px 0 18px;
  min-width: 700px;
  width: 100%;
}
.kv, .frequency-table, .slice-table, .intensity-table,
.recommendations-table, .wide-table {
  table-layout: fixed;
}
th, td {
  border: 1px solid var(--line);
  font-size: 13px;
  line-height: 1.8;
  overflow-wrap: anywhere;
  padding: 7px 9px;
  text-align: left;
  vertical-align: top;
  white-space: normal;
  word-break: normal;
  word-wrap: break-word;
}
th {
  background: var(--panel);
  font-weight: 700;
}
.mobile-label { display: none; }
.cell-value { display: inline; }
.kv td:first-child {
  color: var(--muted);
  font-weight: 700;
}
ul {
  margin: 8px 0 20px;
  padding-left: 28px;
}
li {
  line-height: 1.7;
  margin: 0 0 6px;
}
.empty {
  color: var(--muted);
  font-style: italic;
}
@media (max-width: 800px) {
  body { padding: 16px; }
  h1 { font-size: 23px; }
  h2 { font-size: 17px; }
  h3 { font-size: 13px; }
  table, tbody, tr, td {
    display: block;
    width: 100%;
  }
  table {
    border-collapse: separate;
    border-spacing: 0;
  }
  thead { display: none; }
  tr {
    border: 1px solid var(--line);
    display: block;
    margin: 0 0 12px;
  }
  td, .kv td:first-child {
    border: 0;
    border-bottom: 1px solid var(--line);
    display: block;
    font-size: 14px;
    line-height: 1.55;
    padding: 9px 10px;
    width: 100%;
  }
  td:last-child { border-bottom: 0; }
  .mobile-label {
    color: var(--muted);
    display: block;
    font-size: 10px;
    font-weight: 700;
    line-height: 1.4;
    margin: 0 0 4px;
    text-transform: uppercase;
  }
  .cell-value {
    display: block;
    line-height: 1.55;
  }
}
@media print {
  body { padding: 18px; }
  h2 { break-after: avoid; }
  table { break-inside: avoid; }
}
</style>
"""


def _section(title: str, content: str) -> str:
    return f"<section><h2>{_escape(title)}</h2>\n{content}</section>"


def _subsection(title: str, content: str) -> str:
    return f"<h3>{_escape(title)}</h3>\n{content}"


def _table(
    headers: list[str],
    rows: Iterable[Iterable[Any]],
    class_name: str = "",
    column_widths: Iterable[int] | None = None,
    min_width: int | None = None,
) -> str:
    row_list = list(rows)
    if not row_list:
        return '<p class="empty">No data available.</p>'

    if column_widths is None and "kv" in class_name.split():
        column_widths = (42, 58)

    class_attr = f' class="{_escape(class_name)}"' if class_name else ""
    style_attr = f' style="min-width: {min_width}px"' if min_width else ""
    colgroup_html = ""
    widths = None
    if column_widths is not None:
        widths = list(column_widths)
        if len(widths) != len(headers):
            raise ValueError("Column width count must match the table headers")
        colgroup_html = "<colgroup>" + "".join(
            f'<col style="width: {width}%">' for width in widths
        ) + "</colgroup>"

    header_html = "".join(
        (
            f'<th width="{widths[index]}%">{_escape(header)}</th>'
            if widths is not None
            else f"<th>{_escape(header)}</th>"
        )
        for index, header in enumerate(headers)
    )
    rows_html = []
    for row in row_list:
        cells = "".join(
            (
                f'<td{f" width={widths[index]}%" if widths is not None else ""}>'
                f'<span class="mobile-label">{_escape(headers[index])}</span>'
                f'<span class="cell-value">{_escape(_format_value(cell))}</span></td>'
            )
            for index, cell in enumerate(row)
        )
        rows_html.append(f"<tr>{cells}</tr>")
    return (
        f"<table{class_attr}{style_attr}>{colgroup_html}"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>"
    )


def _dict_table(rows: Iterable[Iterable[Any]]) -> str:
    return _table(["Field", "Value"], rows, class_name="kv")


def _value_summary_table(summary: dict[str, Any]) -> str:
    return _table(
        ["Metric", "Value"],
        [
            ["Count", summary["count"]],
            ["Minimum", summary["min"]],
            ["Maximum", summary["max"]],
            ["Mean", summary["mean"]],
            ["Median", summary["median"]],
        ],
        class_name="kv",
    )


def _frequency_table(items: list[dict[str, Any]]) -> str:
    return _table(
        ["Value", "Number of patients", "Percentage"],
        (
            [
                _frequency_display_value(item),
                item["count"],
                _format_percentage(item["percentage"]),
            ]
            for item in items
        ),
        class_name="frequency-table",
        column_widths=(34, 41, 25),
    )


def _slice_thickness_table(items: list[dict[str, Any]]) -> str:
    return _table(
        ["Thickness (mm)", "Number of patients", "Percentage"],
        (
            [
                item["thickness_mm"],
                item["count"],
                _format_percentage(item["percentage"]),
            ]
            for item in items
        ),
        class_name="frequency-table",
        column_widths=(34, 41, 25),
    )


def _slice_number_thickness_table(items: list[dict[str, Any]]) -> str:
    return _table(
        ["Number of slices", "Slice thickness (mm)", "Number of patients", "Percentage"],
        (
            [
                item["value"][0],
                item["value"][1],
                item["count"],
                _format_percentage(item["percentage"]),
            ]
            for item in items
        ),
        class_name="slice-table",
        column_widths=(22, 28, 28, 22),
    )


def _missing_data_table(items: list[dict[str, Any]]) -> str:
    return _table(
        ["Patient ID", "Split", "Missing data", "Image", "Label", "Reason"],
        (
            [
                item["patient_id"],
                item["split"],
                _format_missing_fields(item["missing_fields"]),
                item["image_path"],
                item["label_path"] or "Not available",
                item["reason"],
            ]
            for item in items
        ),
        class_name="wide-table",
        column_widths=(15, 8, 15, 19, 19, 24),
        min_width=1200,
    )


def _misaligned_patients_table(items: list[dict[str, Any]]) -> str:
    return _table(
        [
            "Patient ID",
            "Split",
            "Issues",
            "Image dimensions",
            "Label dimensions",
            "Image voxel spacing",
            "Label voxel spacing",
            "Affine max difference",
        ],
        (
            [
                item["patient_id"],
                item["split"],
                _format_label_list(item["issues"]),
                item["image_dimensions"],
                item["label_dimensions"],
                item["image_voxel_spacing"],
                item["label_voxel_spacing"],
                item["affine_max_absolute_difference"],
            ]
            for item in items
        ),
        class_name="wide-table",
        column_widths=(12, 6, 16, 12, 12, 14, 14, 14),
        min_width=1400,
    )


def _intensity_summary_table(intensity: dict[str, Any]) -> str:
    return _table(
        ["Statistic", "Unit", "Value", "Meaning"],
        _intensity_summary_rows(intensity),
        class_name="intensity-table",
        column_widths=(24, 9, 15, 52),
    )


def _normalized_patients_table(items: list[dict[str, Any]]) -> str:
    return _table(
        ["Patient ID", "Split", "Minimum", "Maximum", "Mean", "Std", "Reason"],
        (
            [
                item["patient_id"],
                item["split"],
                item["min"],
                item["max"],
                item["mean"],
                item["std"],
                item["reason"],
            ]
            for item in items
        ),
        class_name="wide-table",
        column_widths=(14, 8, 10, 10, 10, 10, 38),
        min_width=1100,
    )


def _percentile_table(items: list[dict[str, Any]]) -> str:
    return _table(
        ["Percentile", "Number of patients", "Minimum", "Maximum", "Mean", "Median"],
        (
            [
                item["percentile"],
                item["count"],
                item["min"],
                item["max"],
                item["mean"],
                item["median"],
            ]
            for item in items
        ),
        class_name="wide-table",
        column_widths=(16, 24, 15, 15, 15, 15),
        min_width=900,
    )


def _recommendations_table(items: list[dict[str, Any]]) -> str:
    return _table(
        ["Severity", "Category", "Recommendation", "Reason"],
        (
            [
                item["severity"],
                _format_label(item["category"]),
                item["message"],
                item["reason"],
            ]
            for item in items
        ),
        class_name="recommendations-table",
        column_widths=(12, 20, 30, 38),
        min_width=900,
    )


def _bullet_list(items: list[str]) -> str:
    if not items:
        return '<p class="empty">No warnings.</p>'
    return "<ul>" + "".join(f"<li>{_escape(item)}</li>" for item in items) + "</ul>"


def _build_text_report(analysis: dict[str, Any]) -> list[str]:
    overview = analysis["report_preparation"]["overview"]
    task_detection = analysis["report_preparation"]["task_detection"]
    counts = analysis["counts"]
    evaluation = analysis["evaluation"]
    consistency = evaluation["consistency"]
    alignment = evaluation["alignment"]
    intensity = evaluation["intensity"]
    intensity_scale = intensity["intensity_scale"]

    lines: list[str] = []
    _add_title(lines, "Medical Imaging Dataset Analysis Report")
    _add_lines(
        lines,
        [
            f"Dataset: {overview['dataset_name']}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Task: {overview['task_type']} ({overview['task_reason']})",
        ],
    )

    _add_title(lines, "Dataset Overview")
    _add_key_values(
        lines,
        [
            ("Split", overview["split_filter"]),
            ("Number of patients", counts["patients_detected"]),
            ("Number of analysed patients", counts["patients_analyzed"]),
            ("Number of failed analyses", counts["patients_failed"]),
            (
                "Number of patients with missing data",
                len(evaluation["missing_data"]),
            ),
            ("Minimum memory", overview["minimum_memory_needed"]),
        ],
    )

    _add_title(lines, "Task Detection")
    _add_key_values(
        lines,
        [
            ("Detected task", task_detection["detected_task"]),
            ("Reason", task_detection["reason"]),
            (
                "Classification classes",
                _format_text_list(task_detection.get("classification_classes", [])),
            ),
            ("Labels found", _yes_no(task_detection["labels_found"])),
            ("Label files detected", task_detection["label_files_detected"]),
            ("Image/label pairs", task_detection["image_label_pairs_detected"]),
        ],
    )

    _add_title(lines, "Missing Data")
    _add_table_lines(
        lines,
        ["Patient ID", "Split", "Missing data"],
        (
            [
                item["patient_id"],
                item["split"],
                _format_missing_fields(item["missing_fields"]),
            ]
            for item in evaluation["missing_data"]
        ),
    )

    _add_title(lines, "Image/Label Alignment")
    _add_key_values(
        lines,
        [
            ("Checked image/label pairs", alignment["checked_pairs"]),
            ("Aligned pairs", alignment["aligned_pairs"]),
            ("Misaligned pairs", alignment["misaligned_pairs"]),
            ("Alignment", _format_percentage(alignment["alignment_percentage"])),
        ],
    )
    _add_table_lines(
        lines,
        ["Patient ID", "Split", "Issues"],
        (
            [
                item["patient_id"],
                item["split"],
                _format_label_list(item["issues"]),
            ]
            for item in alignment["misaligned_patients"]
        ),
    )

    _add_title(lines, "Slices data")
    _add_table_lines(
        lines,
        ["Dimensionality", "Number of patients", "%"],
        _frequency_text_rows(consistency["dimensionality_frequencies"]),
    )
    _add_table_lines(
        lines,
        ["Number of slices", "Slice thickness (mm)", "Number of patients", "%"],
        (
            [
                item["value"][0],
                item["value"][1],
                item["count"],
                _format_percentage(item["percentage"]),
            ]
            for item in consistency["slice_count_thickness_frequencies"]
        ),
    )

    _add_title(lines, "Voxel Size")
    _add_table_lines(
        lines,
        ["Voxel size XxYxZ (mm)", "Number of patients", "%"],
        _frequency_text_rows(consistency["voxel_spacing_frequencies"]),
    )

    _add_title(lines, "Intensity")
    _add_key_values(
        lines,
        [
            ("Raw HU-like patients", intensity_scale["raw_hu_like_patient_count"]),
            (
                "Suspected normalized patients",
                intensity_scale["normalized_patient_count"],
            ),
            ("Unknown intensity scale", intensity_scale["unknown_patient_count"]),
            (
                "Suspected normalized patient IDs",
                _format_patient_ids(intensity_scale["normalized_patient_ids"]),
            ),
        ],
    )
    _add_table_lines(
        lines,
        ["Patient ID", "Split", "Min", "Max", "Mean", "Std"],
        (
            [
                item["patient_id"],
                item["split"],
                item["min"],
                item["max"],
                item["mean"],
                item["std"],
            ]
            for item in intensity_scale["normalized_patients"]
        ),
    )
    _add_table_lines(
        lines,
        ["Statistic", "Unit", "Value", "Meaning"],
        _intensity_summary_rows(intensity),
    )
    _add_key_values(
        lines,
        [
            (
                "Total voxels",
                _format_voxel_count_text(
                    intensity["voxel_validity"]["total_voxels_readable"]
                ),
            ),
            (
                "Valid voxels",
                _format_voxel_count_text(
                    intensity["voxel_validity"]["valid_voxels_readable"]
                ),
            ),
            (
                "Non-finite voxels",
                _format_voxel_count_text(
                    intensity["voxel_validity"]["non_finite_voxels_readable"]
                ),
            ),
        ],
    )

    _add_title(lines, "Warnings")
    _add_lines(lines, evaluation["warnings"] or ["No warnings."])

    _add_title(lines, "Preprocessing Recommendations")
    _add_table_lines(
        lines,
        ["Severity", "Category", "Recommendation"],
        (
            [item["severity"], _format_label(item["category"]), item["message"]]
            for item in evaluation["preprocessing_recommendations"]
        ),
    )

    return lines


def _frequency_text_rows(items: list[dict[str, Any]]) -> Iterable[list[Any]]:
    for item in items:
        yield [
            _frequency_display_value(item),
            item["count"],
            _format_percentage(item["percentage"]),
        ]


def _intensity_summary_rows(intensity: dict[str, Any]) -> list[list[Any]]:
    return [
        [
            "Minimum intensity",
            "HU",
            intensity["global_min"],
            "Lowest voxel intensity in the dataset.",
        ],
        [
            "Maximum intensity",
            "HU",
            intensity["global_max"],
            "Highest voxel intensity in the dataset.",
        ],
        [
            "Mean intensity",
            "HU",
            intensity["patient_mean_summary"]["mean"],
            "Average of the patient mean intensities.",
        ],
        [
            "Standard deviation",
            "HU",
            intensity["patient_std_summary"]["mean"],
            "Average of patient intensity standard deviations.",
        ],
    ]


def _frequency_display_value(item: dict[str, Any]) -> Any:
    value = item.get("value")
    if isinstance(value, list):
        return value
    return item.get("display_value", value)


def _format_missing_fields(fields: list[str]) -> str:
    return ", ".join(field.replace("_", " ") for field in fields)


def _format_label_list(values: list[str]) -> str:
    return ", ".join(_format_label(value) for value in values)


def _format_patient_ids(values: list[str]) -> str:
    return _format_text_list(values)


def _format_text_list(values: list[str]) -> str:
    if not values:
        return "None"
    return ", ".join(values)


def _format_label(value: str) -> str:
    return value.replace("_", " ")


def _format_voxel_count_text(value: str) -> str:
    return value.replace(",", ".")


def _format_percentage(value: float | int | None) -> str:
    if value is None:
        return "Not available"
    formatted = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return f"{formatted}%"


def _add_title(lines: list[str], title: str) -> None:
    lines.extend(["", title, "=" * len(title)])


def _add_lines(lines: list[str], values: Iterable[Any]) -> None:
    for value in values:
        lines.append(str(value))


def _add_key_values(lines: list[str], rows: Iterable[tuple[str, Any]]) -> None:
    for key, value in rows:
        lines.append(f"{key}: {_format_value(value)}")


def _add_table_lines(
    lines: list[str],
    headers: list[str],
    rows: Iterable[Iterable[Any]],
) -> None:
    row_list = [[_format_value(cell) for cell in row] for row in rows]
    if not row_list:
        lines.append("No data available.")
        return

    normalized_headers = [_format_value(header) for header in headers]
    widths = [
        min(
            max(
                len(normalized_headers[index]),
                *(len(row[index]) for row in row_list),
            ),
            34,
        )
        for index in range(len(normalized_headers))
    ]
    lines.append(_format_table_row(normalized_headers, widths))
    lines.append(_format_table_row(["-" * width for width in widths], widths))
    for row in row_list:
        lines.append(_format_table_row(row, widths))


def _format_table_row(row: list[str], widths: list[int]) -> str:
    cells = []
    for value, width in zip(row, widths, strict=True):
        if len(value) > width:
            value = value[: width - 1] + "."
        cells.append(value.ljust(width))
    return " | ".join(cells)


def _write_pdf(path: Path, lines: list[str]) -> None:
    pages = _paginate_pdf_lines(lines)
    objects: list[bytes] = []

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"")  # Pages object filled after page objects are known.
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_object_numbers: list[int] = []
    for page_lines in pages:
        page_object_number = len(objects) + 1
        content_object_number = page_object_number + 1
        page_object_numbers.append(page_object_number)

        content = _pdf_page_content(page_lines)
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                f"/Resources << /Font << /F1 3 0 R >> >> "
                f"/Contents {content_object_number} 0 R >>"
            ).encode("latin-1")
        )
        objects.append(
            b"<< /Length "
            + str(len(content)).encode("ascii")
            + b" >>\nstream\n"
            + content
            + b"\nendstream"
        )

    kids = " ".join(f"{number} 0 R" for number in page_object_numbers)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(
        "latin-1"
    )

    _write_pdf_objects(path, objects)


def _paginate_pdf_lines(lines: list[str], max_chars: int = 98, max_lines: int = 58) -> list[list[str]]:
    wrapped_lines: list[str] = []
    for line in lines:
        if not line:
            wrapped_lines.append("")
            continue
        wrapped = wrap(line, width=max_chars, replace_whitespace=False) or [""]
        wrapped_lines.extend(wrapped)

    return [
        wrapped_lines[index : index + max_lines]
        for index in range(0, len(wrapped_lines), max_lines)
    ] or [["No report content."]]


def _pdf_page_content(lines: list[str]) -> bytes:
    commands = ["BT", "/F1 9 Tf", "40 805 Td", "12 TL"]
    for line in lines:
        commands.append(f"({_pdf_escape(line)}) Tj")
        commands.append("T*")
    commands.append("ET")
    return "\n".join(commands).encode("latin-1", errors="replace")


def _write_pdf_objects(path: Path, objects: list[bytes]) -> None:
    content = bytearray()
    content.extend(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{index} 0 obj\n".encode("ascii"))
        content.extend(obj)
        content.extend(b"\nendobj\n")

    xref_offset = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    content.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    content.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(content)


def _pdf_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _format_value(value: Any) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, bool):
        return _yes_no(value)
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if isinstance(value, list):
        return "x".join(_format_value(item) for item in value)
    return str(value)


def _yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def _escape(value: Any) -> str:
    return html.escape(_format_value(value))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate an HTML report from analyse_dataset.json."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input dataset analysis JSON path.",
    )
    parser.add_argument(
        "--html",
        default=str(DEFAULT_HTML_OUTPUT),
        help="Output HTML report path.",
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help="Optional output PDF report path.",
    )
    args = parser.parse_args(argv)

    generate_reports(args.input, args.html, args.pdf)
    print(f"HTML report written to: {Path(args.html).resolve()}")
    if args.pdf is not None:
        print(f"PDF report written to: {Path(args.pdf).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
