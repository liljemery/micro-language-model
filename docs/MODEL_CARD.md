# Model card

## Summary

Micro Language Model de recetas is a local learning prototype that combines a tiny causal Transformer with cosine-based retrieval over a recipe CSV. It has no remote API dependency and is not intended to compete with general-purpose language models.

## Intended use

- Learn the pieces of a minimal language-model training and inference pipeline.
- Explore causal masking, next-token prediction, attention, lexical retrieval, and checkpoint loading.
- Demonstrate a reproducible command-line project using a small supplied dataset.

## Out of scope

- General question answering, factual research, or translation.
- Food-safety, allergy, nutrition, medical, or dietary advice.
- Reliable recipe recommendations beyond ingredient combinations represented in the CSV.
- Treating generated or retrieved recipe steps as kitchen-tested instructions.

## Model details

- Architecture: decoder-only causal Transformer, implemented in `micro_language_model.py` with PyTorch.
- Configuration: 2 blocks, 4 attention heads, 64-dimensional embeddings, 192-token context, and 0.1 dropout.
- Attention: normalized query/key vectors with cosine scores, causal mask, softmax, and value aggregation.
- Tokenization: word-level with punctuation and special tokens; unknown input words map to `<unk>`.
- Training: next-token cross-entropy on prompt/recipe sequences generated from the included CSV.
- Inference: TF-IDF-style cosine retrieval supplies one CSV row as context; the model samples a continuation and the CLI accepts it only if its normalized word sequence matches the source answer. Otherwise it returns the source row exactly.
- Runtime: CPU by default for a small, predictable local example.

## Data

The training data is `data/generated_recipe_dataset.csv`, with 1,000 rows in English and 18 ingredient terms in the `input` column. The supplied ZIP contained no source, author, or license information. See [DATASET.md](DATASET.md) for counts, schema, and provenance caveats.

## Known limitations and risks

- The model is extremely small and is trained only on a narrow recipe corpus.
- It does not have a held-out evaluation set; the training loss is not a measure of generalization.
- Retrieval is lexical TF-IDF, so phrasing with unknown synonyms may not find a recipe.
- Spanish support is a limited ingredient alias table; the answer remains in English.
- The dataset contains repeated, template-like recipe steps and has not been validated as safe or accurate cooking guidance.
- Soup is absent from the CSV, and the CLI explicitly declines soup requests.
- Exact-source fallback improves faithfulness to the CSV but can make generated output rarely visible.

## Reproducibility

Use Python 3.14 and install the pinned dependencies from `requirements.txt`. The verified environment is macOS on Apple Silicon; other operating systems and architectures are not yet verified. Training defaults to 1,000 steps, batch size 16, and seed 42. The checkpoint is generated locally as `micro_language_model.pt` and is intentionally excluded from version control. Hardware affects training duration.

## Licensing

No license was supplied for the dataset, and this repository does not assign one to it. No software license is currently declared either. Confirm ownership and licensing before reusing or redistributing repository contents.
