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

Run the CHAOS MRI check on the first 10 examinations with:

```bash
python src/true_data.py
```

By default, only the image and mask files required for those 10 examinations
are downloaded. To calculate the score from existing predictions without a
download or a new Qwen request, use:

```bash
python src/true_data.py --score-only
```

To download the complete CHAOS MRI dataset (about 102 MiB), use:

```bash
python src/true_data.py --download-all
```
