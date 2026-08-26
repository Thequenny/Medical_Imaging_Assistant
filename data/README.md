# Data directories

- `dataset_analysis/`: generated JSON and HTML reports for dataset inspection.
- `evaluation/references/`: archived references not used by the current
  evaluation script.
- `evaluation/`: archived evaluation outputs. The active `train.jsonl`,
  `dice.jsonl` and prediction files are stored in
  `src/test_accuracy_evaluation/`.
- `interface/`: files from the earlier interface workspace, kept together for
  compatibility and archival purposes.
- `nii/`: locally downloaded CHAOS MRI files (ignored by Git).
- `slices/`: temporary PNG slices generated for VLM requests (ignored by Git).
