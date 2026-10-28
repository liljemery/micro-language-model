# Learning guide: from CSV to a micro language model

This guide maps the ideas in the project to the small modules in `micro_lm/`. It is an educational walkthrough, not a claim that this model has the scale or general knowledge of a commercial LLM.

## 1. Prepare examples and tokens

In `micro_lm/dataset.py`, `read_dataset` checks that the CSV has the `input`, `recipe_name`, and `output` columns and keeps complete rows. `make_training_sequences` builds 12 English question templates for every recipe. A sequence looks like:

```text
<user> what can i cook with carrot and peas?
<context> Peas Fry. Step 1: ...
<assistant> Peas Fry. Step 1: ... <eos>
```

In `micro_lm/text.py`, `tokenize` splits text into word-level tokens and punctuation. The vocabulary is built from the training sequences, with special tokens for roles, padding, unknown words, and end of response. The development training run with this dataset produced a vocabulary of 109 token IDs.

## 2. Learn next-token prediction

In `micro_lm/training.py`, `train` turns each sequence into input and target token arrays shifted by one position. For example, the target for `"heat oil"` is the sequence beginning at `"oil"`. The model learns with cross-entropy loss, AdamW, a learning rate of `0.0003`, gradient clipping, a default batch size of 16, and 1,000 steps. The seed is fixed to 42. Padding positions are ignored by the loss.

The training loop saves a checkpoint named `micro_language_model.pt` containing the vocabulary, model configuration, retrieval rows, retrieval weights, and learned parameters. It is a generated artifact and is not committed; rerun `python micro_language_model.py train` to create it. With `--retrieval-data`, the extra CSV rows go into the retrieval index but are excluded from next-token training. Shared paths and model sizes live in `micro_lm/config.py`.

## 3. Understand the Transformer

The small model has a token embedding, learned positional embedding, two `TransformerBlock`s, a final normalization layer, and a language-model output head. Each block applies:

1. Layer normalization, then self-attention, added back to the block input.
2. Layer normalization, then a two-layer feed-forward network, also added back.

In `micro_lm/model.py`, `CosineSelfAttention` projects each token into query (Q), key (K), and value (V) vectors. It L2-normalizes Q and K, then calculates their cosine scores:

```text
score(q, k) = normalize(q) · normalize(k)
```

The scores are scaled, future-token positions are set to a very small value using a lower-triangular causal mask, and softmax turns the remaining scores into attention weights. The weights mix the V vectors. This is attention inside the language model.

## 4. Keep retrieval separate from attention

The project uses cosine similarity in two different places:

| Mechanism | What is compared? | Why? |
| --- | --- | --- |
| Recipe retrieval (`micro_lm/retrieval.py`) | Query ingredient terms and CSV ingredient lists | Select a recipe row to provide context |
| Self-attention (`micro_lm/model.py`) | Q and K vectors for tokens in the current sequence | Let each generated token use earlier relevant tokens |

Retrieval is a TF-IDF-style lexical search, not a learned semantic embedding. The query must mention known ingredients, a recognizable recipe title, or a supported alias. “Natural language” here means flexible wording around those terms, not general language understanding.

## 5. Generate and stream an answer

In `micro_lm/inference.py`, `MicroLanguageModel.stream_answer` normalizes known ingredient aliases and may add a candidate recipe from cosine retrieval to the prompt. If no recipe matches, it still sends the user's prompt to the model without retrieved context. `MicroGPT.generate` in `micro_lm/model.py` yields each sampled token as soon as it is produced; the CLI writes and flushes that token immediately. It does not compare the output with a CSV row or replace it with a fixed response.

This project trains its own small checkpoint from random initialization; it does not download or use an external pretrained LLM. Ask about a topic such as soup that is absent from the CSV to see that generation still runs without a canned refusal. The response can be nonsensical because the model is tiny, trained only on recipe examples, and word-tokenized.

## 6. Suggested experiments

Try these small changes one at a time and compare their effects:

1. Add a query phrase to the template list in `micro_lm/dataset.py`; retrain and see whether that phrasing retrieves the expected context.
2. Add an ingredient alias to `INGREDIENT_ALIASES` in `micro_lm/text.py`; ask with its plural or Spanish equivalent.
3. Change `n_embd`, `n_head`, `n_layer`, or `block_size` in `CONFIG` in `micro_lm/config.py`; retrain because checkpoint shapes depend on them.
4. Change the retrieval weighting in `micro_lm/retrieval.py` or print the top candidate scores to inspect why a query selected a row.
5. Replace word-level tokenization with subword tokenization and compare vocabulary size, unknown terms, training speed, and output quality.
6. Use `python micro_language_model.py split` to create a recipe-title holdout, train with `--retrieval-data`, and evaluate repeated generations with `python micro_language_model.py evaluate`. Read [EVALUATION.md](EVALUATION.md) before interpreting the scores.

## 7. Limits to keep in mind

- The vocabulary and training corpus are tiny and recipe-specific.
- The data is English and its provenance and culinary quality are unknown.
- The retriever uses words, not embeddings; synonyms outside the alias table may not match.
- The default training command uses all rows, so its loss is not a held-out quality measure. Use the split workflow for an evaluation checkpoint.
- Generated text can repeat, contradict itself, or invent recipe details; the tiny model has no factuality guarantee.
