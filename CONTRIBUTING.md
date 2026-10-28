# Practice and contribution guide

This repository is deliberately small so you can trace the full path from CSV to an answer. You do not need a GPU, external API key, or model download.

## Set up locally

Follow the Python 3.14 commands in [README.md](README.md). Keep dependencies inside `.venv`; do not install project packages into Homebrew's managed Python. Run `train` once before `ask` because the model checkpoint is created locally and ignored by Git.

## Where to start reading

| Topic | Code location |
| --- | --- |
| CSV validation and prompt templates | `micro_lm/dataset.py` |
| Tokenization and ingredient aliases | `micro_lm/text.py` |
| Model sizes and default paths | `micro_lm/config.py` |
| TF-IDF and cosine search | `micro_lm/retrieval.py` |
| Causal attention, Transformer and next-token model | `micro_lm/model.py` |
| Training and checkpoint | `micro_lm/training.py` |
| Query handling and output grounding | `micro_lm/inference.py` |
| Command-line flags | `micro_lm/cli.py` |

## Suggested portfolio exercises

- Add a held-out split and report a metric separately from training loss.
- Let the CLI print the top retrieved recipe and cosine score for learning/debugging.
- Add a small, documented set of ingredient synonyms and verify the supported vocabulary.
- Compare the current word-level vocabulary with a character or subword tokenizer.
- Experiment with model size, training steps, or sampling temperature; record the effect instead of claiming an improvement from one example.
- Add automated tests for dataset validation, cosine ranking, and out-of-domain fallback as a separate learning task.

## Change guidelines

- Keep the toy model small and explain new concepts in `docs/LEARNING_GUIDE.md`.
- Preserve the dataset as supplied; document any cleaning or derived data.
- Do not commit `.venv`, cache files, or generated weights. A fresh clone should recreate the checkpoint with `python micro_language_model.py train`.
- If you discover the dataset's source or license, update `docs/DATASET.md` with a link and the exact attribution/terms.
