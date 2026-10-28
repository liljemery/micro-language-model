# Answer reliability evaluation

The `evaluate` command separates one automatic measure (recipe retrieval Hit@1) from sampled answers that need human review. It runs each query several times with reproducible seeds and writes a CSV containing the retrieved recipe, cosine score, generated text, and empty review columns.

## Quick smoke evaluation

The example queries check that the command and output format work:

```bash
.venv/bin/python micro_language_model.py evaluate \
  --queries data/evaluation_queries.example.csv \
  --runs 5
```

These examples use recipes already in the training dataset. They are a smoke check, **not** an independent reliability score.

## Build a held-out evaluation

1. Split the source CSV by `recipe_name`. This keeps every row with the same recipe title on one side and avoids separating variants of the same recipe at random:

   ```bash
   .venv/bin/python micro_language_model.py split --fraction 0.2 --seed 42
   ```

   By default this creates `artifacts/train_recipes.csv`, `artifacts/heldout_recipes.csv`, and `artifacts/heldout_queries.csv`. Generated queries are limited to ingredient combinations that map to one recipe title in the original CSV; ambiguous combinations are omitted from automatic Hit@1 scoring.
2. Train the model weights on the training side, and provide the held-out recipe rows as a separate retrieval catalog:

   ```bash
   .venv/bin/python micro_language_model.py train \
     --data artifacts/train_recipes.csv \
     --retrieval-data artifacts/heldout_recipes.csv
   ```

   `--retrieval-data` rows are indexed for context retrieval but are not used as next-token training examples. The CLI rejects overlapping recipe titles between the files. The held-out recipe's title and steps are therefore supplied through retrieval at evaluation time, not memorized as training targets.
3. Add custom paraphrases, Spanish aliases, ambiguous prompts, and out-of-domain questions to `artifacts/heldout_queries.csv`. Keep the columns:

   | Column | Meaning |
   | --- | --- |
   | `case_id` | Optional stable identifier; one is generated if omitted. |
   | `query` | Question for the held-out recipe. Add separate wording variants not used by training to measure paraphrase handling. |
   | `expected_recipe_name` | Expected retrieved title; leave blank for an unlabelled or out-of-domain case. |
   | `category` | Such as `known_ingredients`, `paraphrase`, `spanish_alias`, `ambiguous`, or `out_of_domain`. |

   Leave `expected_recipe_name` blank for unknown and out-of-domain cases so they do not affect Hit@1. `data/evaluation_queries.example.csv` is only a quick smoke sample; its recipes are part of the standard training CSV.
4. Run repeated generations and save the results:

   ```bash
   .venv/bin/python micro_language_model.py evaluate \
     --queries path/to/evaluation_queries.csv \
     --runs 5 \
     --seed 42 \
     --output artifacts/evaluation_results.csv
   ```

   The default output is `artifacts/evaluation_results.csv`, which is ignored by Git. Each run has a distinct, recorded seed; rerunning with the same inputs and settings reproduces the samples.

## Read the results

- **Retrieval Hit@1** is the fraction of cases with an expected title for which the top retrieved recipe has that title. It measures retrieval, not answer quality. The generated split avoids ingredient combinations associated with multiple recipe titles so the automatic labels are not ambiguous.
- **No context** counts questions for which retrieval selected no row. For unknown and out-of-domain prompts, inspect whether a retrieved row should have been absent.
- Review each generated answer against its retrieved recipe context. Score relevance as 0 (off topic), 1 (partially answers), or 2 (answers the question). Score source consistency as 0 (contradicts the context or adds unsupported details), 1 (mostly supported with some unsupported details), or 2 (supported by the context). Score completeness as 0 (not usable), 1 (partly useful but missing key steps), or 2 (includes the expected recipe and its essential steps). Record the number of unsupported claims in `unsupported_claims`, then add detail in `review_notes`.
- Compare the five runs for the same case. Repetition, contradiction, or changing unsupported details indicate sampling instability.

Report the number of queries and generations alongside the metrics. The example CSV is not a benchmark, and results from recipes included in the weights are not evidence of generalization. This dataset is small, repetitive, English-only, and not validated as cooking guidance. Human review remains necessary; these metrics do not establish food safety or general-language reliability.
