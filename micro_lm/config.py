"""Configuración compartida y rutas predeterminadas del proyecto."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "generated_recipe_dataset.csv"
DEFAULT_MODEL = ROOT / "micro_language_model.pt"
DEFAULT_EVALUATION_OUTPUT = ROOT / "artifacts" / "evaluation_results.csv"
DEFAULT_SPLIT_TRAIN = ROOT / "artifacts" / "train_recipes.csv"
DEFAULT_SPLIT_HELDOUT = ROOT / "artifacts" / "heldout_recipes.csv"
DEFAULT_SPLIT_QUERIES = ROOT / "artifacts" / "heldout_queries.csv"

SPECIAL_TOKENS = ["<pad>", "<unk>", "<eos>", "<user>", "<context>", "<assistant>"]

CONFIG = {
    "block_size": 192,
    "n_embd": 64,
    "n_head": 4,
    "n_layer": 2,
    "dropout": 0.1,
}
