# Medical_Imaging_Assistant

A desktop application for inspecting medical imaging datasets, generating
quality reports, and discussing selected slices with Qwen.

Detailed documentation for NIfTI/DICOM analysis and command-line workflows is
available in [docs/dataset_analysis.md](docs/dataset_analysis.md).

## Prerequisites

- Python 3.10 or newer;
- Tkinter for the desktop interface.

On Ubuntu, install Tkinter with:

```bash
sudo apt install python3-tk
```

## Installation

From the project root, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/interface.txt
```

On Windows:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements/interface.txt
```

## Qwen configuration

This step is optional for local dataset analysis, but required to send slices
to the model from the interface:

```bash
cp .env.example .env
```

Then configure `.env`:

```dotenv
QWEN_API_KEY=replace-with-your-api-key
QWEN_BASE_URL=https://example.com/v1
QWEN_MODEL=qwen-model-name
```

The `.env` file is ignored by Git. Never publish a real API key.

## Launching the interface

Activate the virtual environment, move to the project root, and run:

```bash
python -m src.interface
```

In the interface:

1. choose the split to analyze (`train`, `test`, `validation`, or `all`);
2. click **Select and analyze folder** and select the dataset directory;
3. open the generated report with **Open generated report**;
4. use the exploration workspace to display NIfTI slices and, when Qwen is
   configured, discuss them with the model.

Interface reports are written to:

```text
data/dataset_analysis/
```

## Supported data

The interface supports `.nii` and `.nii.gz` volumes as well as `.dcm` DICOM
series. Details about dataset detection, DICOM conversion, and report contents
are available in the
[dataset analysis documentation](docs/dataset_analysis.md).

## Privacy

Dataset analysis runs locally. Only slices explicitly submitted to Qwen leave
the machine and are sent to the server configured through `QWEN_BASE_URL`.
Review the applicable privacy requirements before sending medical data.

## Documentation

- [NIfTI/DICOM dataset analysis](docs/dataset_analysis.md)
- [Dependency groups](requirements/README.md)
- [Data directory organization](data/README.md)
