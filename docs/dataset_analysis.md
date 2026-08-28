# Check_MedicalImaging_data

`check_data` is a Python tool for checking CT medical imaging datasets before
AI training. It inspects NIfTI datasets, detects image/label relationships,
computes patient-level and dataset-level statistics, and generates a readable
HTML report.

The project was tested with Medical Segmentation Decathlon datasets, including
`Task02_Heart` and `Task09_Spleen`.

## What It Does

- Detects dataset structure, training/test folders, image folders, label
  folders, metadata files, and image/label pairs.
- Supports standard layouts such as `imagesTr`, `labelsTr`, and `imagesTs`.
- Supports non-standard folder names when they contain image/label hints, such as `raw_image_volumes` or `manual_label_maps`.
- Infers whether the dataset is likely a segmentation, classification, or
  unknown task.
- Reads `.nii` and `.nii.gz` files with `nibabel`.
- Extracts dimensions, number of slices, voxel spacing, slice thickness,
  orientation, datatype, affine matrix, physical size, file size, and memory estimates.
- Compares patients for consistency of dimensions, resolution, voxel spacing, slice thickness, slice count, physical size, and computes statistical data.
- Writes an HTML report.


## Requirements

Use Python 3.10 or newer.

Required for NIfTI analysis:

```powershell
pip install numpy nibabel
```

Required only when the input dataset contains DICOM files:

```powershell
pip install pydicom dicom2nifti
```

The HTML and PDF report generator does not require extra Python packages.


## Supported Inputs

The analysis itself only uses CT files in NIfTI format:

- `.nii`
- `.nii.gz`

If a dataset contains DICOM files (`.dcm`), no extra command is required. The program detects the file type automatically, converts valid CT DICOM series to NIfTI first, then continues the analysis on the converted files.

```text
converted_nifti/
```

If a conversion problem occurs, the program stops and generates:

```text
data/dataset_analysis/Warning.txt
```

In that case, the HTML report should not be generated because the dataset
analysis is incomplete.

## Usage

Run commands from the repository root.

Launch the desktop interface with:

```powershell
python -m src.interface
```

### Step 0: Check Dataset Structure

This step does not read voxel data. It only detects folders, files,splits, image/label pairs, task type, and structure warnings.

```powershell
python -m src.check_structure_dataset dataset/Task09_Spleen
```

Print the full detected structure as JSON:

```powershell
python -m src.check_structure_dataset dataset/Task09_Spleen --json
```


### Step 1: Analyze One Patient

For the analyze one CT image without a label:

```powershell
python -m src.nifti_analyzer dataset/Task09_Spleen/imagesTr/spleen_10.nii.gz
```

Analyze one CT image with its label:

```powershell
python -m src.nifti_analyzer dataset/Task09_Spleen/imagesTr/spleen_10.nii.gz --label dataset/Task09_Spleen/labelsTr/spleen_10.nii.gz
```

By default, the output is written to:

```text
data/dataset_analysis/CT_data.json
```

Choose a custom output path:

```powershell
python -m src.nifti_analyzer dataset/Task09_Spleen/imagesTr/spleen_10.nii.gz --label dataset/Task09_Spleen/labelsTr/spleen_10.nii.gz --output data/dataset_analysis/CT_data.json
```


### Step 2: Analyze a Full Dataset

```powershell
python -m src.dataset_analyzer dataset/Task09_Spleen
```

By default, the output is written to:

```text
data/dataset_analysis/analyse_dataset.json
```

The default split is `train`, so test cases are not counted as missing training
labels.

Choose a split:

```powershell
python -m src.dataset_analyzer dataset/Task09_Spleen --split train --output data/dataset_analysis/analyse_dataset.json
```

Analyze all detected splits:

```powershell
python -m src.dataset_analyzer dataset/Task09_Spleen --split all --output data/dataset_analysis/analyse_dataset.json
```

Print the full dataset analysis JSON to the terminal:

```powershell
python -m src.dataset_analyzer dataset/Task09_Spleen --json
```


### Step 3: Generate the Report

Generate the HTML report from the default dataset analysis JSON:

```powershell
python -m src.report
```

Open the report on Windows:

```powershell
start data\dataset_analysis\report.html
```

The report includes:

- Dataset overview
- Patient counts and missing data
- Task detection
- Slice count and slice thickness
- Voxel size
- Intensity statistics
- Consistency statistics
- preprocessing recommendations


## Example Workflow

```powershell
python -m src.check_structure_dataset dataset/Task02_Heart
python -m src.dataset_analyzer dataset/Task02_Heart
python -m src.report
start data\dataset_analysis\report.html
```


## Tests

Run all tests:

```powershell
python -m unittest discover -s tests -v
```

The tests create small temporary NIfTI files automatically, so no real medical dataset is required to run them.
