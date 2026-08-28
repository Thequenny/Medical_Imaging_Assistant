# Dependency groups

- `core.txt`: medical imaging, conversion, and report dependencies.
- `api.txt`: Qwen/OpenAI API client dependencies.
- `interface.txt`: everything required by the desktop interface.
- `evaluation.txt`: API and dataset-download dependencies for
  `src/vlm_evaluation.py`.
- `local-model.txt`: optional PyTorch and Transformers stack.
- `ai.txt`: complete AI stack combining API, evaluation, and local model tools.

Install only the group needed for a task, for example:

```bash
python -m pip install -r requirements/interface.txt
```

For API-based workflows without the large local model stack, install
`interface.txt` or `evaluation.txt`. The root `requirements.txt` remains the
complete installation entry point.
