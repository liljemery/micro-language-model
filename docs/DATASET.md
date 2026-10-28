# Dataset notes

## Source and contents

The training file is `data/generated_recipe_dataset.csv`, copied from the single file inside the provided archive `SLM Dataset.zip`. The archive contained no README, source URL, author name, or license file. The CSV is included here because the project owner requested publishing the training data; its original redistribution rights have not been verified.

The CSV has 1,000 complete rows and three columns:

| Column | Meaning | Example |
| --- | --- | --- |
| `input` | Space-separated ingredient terms used for retrieval | `butter carrot onion peas` |
| `recipe_name` | Short recipe title | `Homestyle Butter Dish` |
| `output` | Recipe text shown as the training target and answer context | `Step 1: Heat oil in a pan...` |

The input column contains 18 distinct ingredient words: `butter`, `capsicum`, `carrot`, `cheese`, `chicken`, `egg`, `flour`, `garlic`, `ginger`, `milk`, `mushroom`, `onion`, `paneer`, `peas`, `potato`, `rice`, `spinach`, and `tomato`. There are 866 distinct input strings and 90 distinct recipe titles; the same ingredient combination can occur in more than one row. Each row has non-empty values in all three required columns.

## Processing

- Training expands each row into 12 English prompt variants. The CSV itself is not modified.
- Retrieval builds inverse-document-frequency weights from the `input` column, vectorizes a query, then ranks rows by cosine similarity.
- The model sees the recipe title and output as context. If its generated word sequence differs from the retrieved answer, the CLI returns the original row's title and output.
- Queries can use a small hand-written set of English plurals and Spanish ingredient aliases. This mapping is in `INGREDIENT_ALIASES` in `micro_language_model.py`; it does not translate the recipe text.

## Quality and appropriate use

The sample recipes use repetitive, template-like steps and the archive does not state how they were created or reviewed. Treat them as toy text for learning machine-learning mechanics, not as tested cooking guidance. The dataset does not contain a soup recipe. The model's intended domain is limited to these rows and the listed ingredient vocabulary.

## Licensing and attribution

No dataset license or attribution was present in the supplied archive. Public availability in this repository does not establish permission for other uses. Verify the dataset's original source and terms before redistributing it elsewhere or using it beyond this educational project. No license is assigned to the dataset here.
