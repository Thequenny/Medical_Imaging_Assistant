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

Configure the Qwen API once for every script:

```bash
cp .env.example .env
```

Then edit `.env` and replace `replace-with-your-api-key` with a newly generated
API key. The `.env` file is ignored by Git; never commit a real key.

Run the CHAOS MRI check on five distinct T1 patients and five other distinct
T2 patients with:

```bash
python src/true_data.py
```

By default, only the ten required images are downloaded. Each patient is sent
to the model three times, always in a fresh discussion with the same prompt.
The independent answers are saved without pixel counts in
`src/predictions_run_1.jsonl`, `src/predictions_run_2.jsonl` and
`src/predictions_run_3.jsonl`. Each file contains the same ten patients, and
each line contains the patient ID, detected organs and detected MRI sequence.
To calculate the organ and sequence scores from existing predictions without a
download or a new Qwen request, use:

```bash
python src/true_data.py --score-only
```

The sequence evaluation includes one T1/T2 confusion matrix per response and a
matrix accumulated over the three responses. Matrix rows are the actual
sequence, columns are the predicted sequence, and invalid model answers are
counted in a separate `invalide` column. The sequence score is the percentage
of predictions on the T1/T2 diagonal; it is not combined with the organ score.

To download the complete CHAOS MRI dataset (about 102 MiB), use:

```bash
python src/true_data.py --download-all
```
