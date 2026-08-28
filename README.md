# Medical_Imaging_Assistant

A tool for inspecting medical imaging datasets before they are used to train AI models. It generates summary reports and analyzes selected slices with a vision-language model (VLM). This project uses the Qwen model via an Ollama server.

Detailed documentation for NIfTI/DICOM analysis and command-line workflows is available in [docs/dataset_analysis.md](docs/dataset_analysis.md).

## Prerequisites

- Python 3.10 or newer;
- Tkinter for the interface.

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
.\.venv\Scripts\Activate.ps1
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

From the project root, run the command for your operating system.

On Linux:

```bash
./.venv/bin/python -m src.interface
```

On Windows (PowerShell):

```powershell
.\.venv\Scripts\python.exe -m src.interface
```

In the interface:

1. choose the split to analyze (`train`, `test`, `validation`, or `all`);
2. click **Select and analyze folder** and select the dataset directory;
3. open the generated report with **Open generated report**;
4. use the exploration workspace to display NIfTI slices and discuss them with the model.

In the slice gallery:

- right-click a slice and choose **View larger** to visualize it or choose **Send to chat** to open a Qwen conversation with that slice attached;
- left-click one or more slices (or use **Select all slices**), then click **Chat with Qwen** in the workspace sidebar. The selected slices are
  automatically attached to the chat and sent to the model with your next message.

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
