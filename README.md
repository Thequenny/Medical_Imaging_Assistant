# LLM_project

## Linux setup

Tkinter is a system package on Ubuntu and must be installed once:

```bash
sudo apt install python3.14-tk
```

Activate the project environment before running a script:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the CHAOS MRI sample check with:

```bash
python src/true_data.py
```

To download the complete CHAOS MRI dataset (about 102 MiB), use:

```bash
python src/true_data.py --download-all
```
