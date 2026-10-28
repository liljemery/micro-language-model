#!/usr/bin/env python3
"""Micro Language Model causal con atención de similitud coseno."""

import argparse
import csv
import math
import re
import sys
from collections import Counter
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print(
        "Falta una dependencia en este Python. En macOS/Linux instala con: "
        ".venv/bin/python -m pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "generated_recipe_dataset.csv"
DEFAULT_MODEL = ROOT / "micro_language_model.pt"
SPECIAL_TOKENS = ["<pad>", "<unk>", "<eos>", "<user>", "<context>", "<assistant>"]
TOKEN_RE = re.compile(
    r"<pad>|<unk>|<eos>|<user>|<context>|<assistant>|[^\W_]+(?:['’][^\W_]+)*|[^\w\s]",
    re.UNICODE,
)
WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
INGREDIENT_ALIASES = {
    "carrots": "carrot",
    "potatoes": "potato",
    "tomatoes": "tomato",
    "mushrooms": "mushroom",
    "onions": "onion",
    "eggs": "egg",
    "capsicums": "capsicum",
    "zanahoria": "carrot",
    "zanahorias": "carrot",
    "arveja": "peas",
    "arvejas": "peas",
    "guisante": "peas",
    "guisantes": "peas",
    "chicharo": "peas",
    "chicharos": "peas",
    "chícharo": "peas",
    "chícharos": "peas",
    "papa": "potato",
    "papas": "potato",
    "patata": "potato",
    "patatas": "potato",
    "tomate": "tomato",
    "tomates": "tomato",
    "cebolla": "onion",
    "cebollas": "onion",
    "ajo": "garlic",
    "pollo": "chicken",
    "huevo": "egg",
    "huevos": "egg",
    "harina": "flour",
    "leche": "milk",
    "queso": "cheese",
    "espinaca": "spinach",
    "espinacas": "spinach",
    "arroz": "rice",
    "champiñón": "mushroom",
    "champiñones": "mushroom",
    "pimiento": "capsicum",
    "pimientos": "capsicum",
    "mantequilla": "butter",
    "jengibre": "ginger",
}
CONFIG = {
    "block_size": 192,
    "n_embd": 64,
    "n_head": 4,
    "n_layer": 2,
    "dropout": 0.1,
}


def tokenize(text):
    return TOKEN_RE.findall(text)


def words(text):
    return WORD_RE.findall(text.lower())


def normalize_ingredient_terms(text):
    return {INGREDIENT_ALIASES.get(token, token) for token in words(text)}


def read_dataset(path):
    records = []
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"input", "recipe_name", "output"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("El CSV debe tener las columnas input, recipe_name y output.")
        for row in reader:
            record = {key: (row.get(key) or "").strip() for key in required}
            if all(record.values()):
                records.append(record)
    if not records:
        raise ValueError("El CSV no contiene recetas completas para entrenar.")
    return records


def make_training_sequences(records):
    templates = (
        "what's the recipe for {ingredients}?",
        "whats the recipe for {ingredients}?",
        "what is the recipe for {ingredients_natural}?",
        "what recipe can i make with {ingredients}?",
        "what's the recipe for {ingredients_natural}?",
        "what can i cook with {ingredients_natural}?",
        "how do i cook {ingredients}?",
        "could you give me a recipe with {ingredients_natural}?",
        "i have {ingredients_natural}. what can i make?",
        "what's the recipe for {name}?",
        "whats the recipe for {name}?",
        "how do i make {name}?",
    )
    sequences = []
    ingredient_terms = set()
    recipe_terms = set()
    for record in records:
        ingredients = record["input"].lower()
        ingredient_parts = ingredients.split()
        if len(ingredient_parts) > 2:
            ingredients_natural = ", ".join(ingredient_parts[:-1]) + ", and " + ingredient_parts[-1]
        elif len(ingredient_parts) == 2:
            ingredients_natural = " and ".join(ingredient_parts)
        else:
            ingredients_natural = ingredients
        name = record["recipe_name"]
        answer = f"{name}. {record['output']}"
        ingredient_terms.update(words(ingredients))
        recipe_terms.update(words(name))
        for template in templates:
            prompt = template.format(
                ingredients=ingredients,
                ingredients_natural=ingredients_natural,
                name=name.lower(),
            )
            sequences.append(
                f"<user> {prompt} <context> {answer} <assistant> {answer}<eos>"
            )
    return sequences, sorted(ingredient_terms), sorted(recipe_terms)


def build_idf(records):
    document_frequency = Counter()
    for record in records:
        document_frequency.update(set(words(record["input"])))
    total = len(records)
    return {
        token: math.log((total + 1) / (frequency + 1)) + 1.0
        for token, frequency in document_frequency.items()
    }


def vectorize(tokens, idf):
    counts = Counter(token for token in tokens if token in idf)
    return {
        token: (1.0 + math.log(count)) * idf[token]
        for token, count in counts.items()
    }


def vector_norm(vector):
    return math.sqrt(sum(value * value for value in vector.values()))


def cosine_similarity(left, right):
    denominator = vector_norm(left) * vector_norm(right)
    if not denominator:
        return 0.0
    return sum(value * right.get(token, 0.0) for token, value in left.items()) / denominator


class CosineSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        if n_embd % n_head:
            raise ValueError("n_embd debe ser divisible por n_head.")
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.projection = nn.Linear(n_embd, n_embd)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x):
        batch, length, channels = x.shape
        qkv = self.qkv(x).view(
            batch, length, 3, self.n_head, self.head_size
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query = F.normalize(query, p=2, dim=-1)
        key = F.normalize(key, p=2, dim=-1)

        # Q y K normalizados hacen que cada puntaje sea similitud de coseno.
        scores = (query @ key.transpose(-2, -1)) * 5.0
        scores = scores.masked_fill(
            ~self.causal_mask[:length, :length],
            torch.finfo(scores.dtype).min,
        )
        attention = self.attention_dropout(F.softmax(scores, dim=-1))
        attended = attention @ value
        attended = attended.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.residual_dropout(self.projection(attended))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config["n_embd"])
        self.attention = CosineSelfAttention(
            config["n_embd"],
            config["n_head"],
            config["block_size"],
            config["dropout"],
        )
        self.norm2 = nn.LayerNorm(config["n_embd"])
        self.feed_forward = nn.Sequential(
            nn.Linear(config["n_embd"], 4 * config["n_embd"]),
            nn.GELU(),
            nn.Linear(4 * config["n_embd"], config["n_embd"]),
            nn.Dropout(config["dropout"]),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class MicroGPT(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.block_size = config["block_size"]
        self.token_embedding = nn.Embedding(vocab_size, config["n_embd"])
        self.position_embedding = nn.Embedding(config["block_size"], config["n_embd"])
        self.dropout = nn.Dropout(config["dropout"])
        self.blocks = nn.Sequential(
            *(TransformerBlock(config) for _ in range(config["n_layer"]))
        )
        self.final_norm = nn.LayerNorm(config["n_embd"])
        self.language_head = nn.Linear(config["n_embd"], vocab_size, bias=False)
        self.language_head.weight = self.token_embedding.weight

    def forward(self, indices):
        _, length = indices.shape
        positions = torch.arange(length, device=indices.device)
        x = self.dropout(self.token_embedding(indices) + self.position_embedding(positions))
        x = self.final_norm(self.blocks(x))
        return self.language_head(x)

    @torch.no_grad()
    def generate(self, prompt_ids, itos, max_new_tokens, temperature):
        device = next(self.parameters()).device
        indices = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        blocked = {
            SPECIAL_TOKENS.index(token)
            for token in ("<pad>", "<unk>", "<user>", "<context>", "<assistant>")
        }
        for _ in range(max_new_tokens):
            context = indices[:, -self.block_size :]
            logits = self(context)[:, -1, :] / temperature
            for token_id in blocked:
                logits[0, token_id] = float("-inf")
            top_values, top_ids = torch.topk(logits, k=min(12, logits.size(-1)))
            probabilities = F.softmax(top_values, dim=-1)
            next_id = top_ids.gather(
                -1, torch.multinomial(probabilities, num_samples=1)
            )
            token_id = next_id.item()
            if itos[token_id] == "<eos>":
                break
            indices = torch.cat((indices, next_id), dim=1)
        return indices[0, len(prompt_ids) :].tolist()


def encode(text, stoi):
    unknown_id = stoi["<unk>"]
    return [stoi.get(token, unknown_id) for token in tokenize(text)]


def decode(token_ids, itos):
    hidden = set(SPECIAL_TOKENS)
    closing = {".", ",", ":", ";", "!", "?", ")", "]", "}"}
    output = []
    for token_id in token_ids:
        token = itos[token_id]
        if token in hidden:
            continue
        if token in closing and output:
            output[-1] += token
        else:
            output.append(token)
    return " ".join(output)


def choose_device():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    return torch.device("cpu")


def train(data_path, model_path, steps, batch_size):
    records = read_dataset(data_path)
    sequences, ingredient_terms, recipe_terms = make_training_sequences(records)
    vocabulary = sorted(
        {token for sequence in sequences for token in tokenize(sequence)}
        - set(SPECIAL_TOKENS)
    )
    itos = SPECIAL_TOKENS + vocabulary
    stoi = {token: index for index, token in enumerate(itos)}
    pad_id = stoi["<pad>"]
    training_rows = []
    for sequence in sequences:
        token_ids = encode(sequence, stoi)
        if len(token_ids) > CONFIG["block_size"] + 1:
            token_ids = token_ids[: CONFIG["block_size"]] + [stoi["<eos>"]]
        token_ids += [pad_id] * (CONFIG["block_size"] + 1 - len(token_ids))
        training_rows.append(token_ids)

    torch.manual_seed(42)
    device = choose_device()
    data = torch.tensor(training_rows, dtype=torch.long, device=device)
    model = MicroGPT(len(itos), CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()

    for step in range(1, steps + 1):
        row_ids = torch.randint(0, len(data), (batch_size,), device=device)
        batch = data[row_ids]
        x = batch[:, :-1]
        y = batch[:, 1:]
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, len(itos)),
            y.reshape(-1),
            ignore_index=pad_id,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == steps:
            print(f"Paso {step}/{steps} — pérdida {loss.item():.3f}", flush=True)

    model.to("cpu")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "vocabulary": itos,
            "records": records,
            "idf": build_idf(records),
            "ingredient_terms": ingredient_terms,
            "recipe_terms": recipe_terms,
            "config": CONFIG,
            "steps": steps,
            "state_dict": model.state_dict(),
        },
        model_path,
    )
    return len(records), len(itos), device.type


class MicroLanguageModel:
    def __init__(self, checkpoint, device=None):
        if checkpoint.get("format_version") != 2:
            raise ValueError("Versión de modelo no compatible.")
        self.itos = checkpoint["vocabulary"]
        self.stoi = {token: index for index, token in enumerate(self.itos)}
        self.records = checkpoint["records"]
        self.idf = checkpoint["idf"]
        self.ingredient_terms = set(checkpoint["ingredient_terms"])
        self.recipe_terms = set(checkpoint["recipe_terms"])
        self.record_vectors = [
            vectorize(words(record["input"]), self.idf) for record in self.records
        ]
        self.config = checkpoint["config"]
        self.device = device or choose_device()
        self.model = MicroGPT(len(self.itos), self.config)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        return cls(checkpoint)

    def answer(self, query, max_new_tokens=100, temperature=0.35):
        query_terms = normalize_ingredient_terms(query)
        if {"soup", "soups", "sopa", "sopas"}.intersection(query_terms):
            return (
                "El dataset no contiene recetas de sopa. Dime qué ingredientes "
                "tienes y te sugiero una receta basada en los datos."
            )
        query_vector = vectorize(query_terms, self.idf)
        if query_vector:
            scored = [
                (cosine_similarity(query_vector, vector), index)
                for index, vector in enumerate(self.record_vectors)
            ]
            best_score, best_index = max(scored, key=lambda item: item[0])
            record = self.records[best_index] if best_score > 0 else None
        else:
            title_matches = [
                record
                for record in self.records
                if query_terms.intersection(
                    normalize_ingredient_terms(record["recipe_name"])
                )
            ]
            record = title_matches[0] if title_matches else None

        if record is None:
            return (
                "Solo conozco las recetas del dataset. Menciona algún ingrediente "
                "o nombre de receta, por ejemplo: carrot, peas o milk."
            )

        context = f"{record['recipe_name']}. {record['output']}"
        prompt = f"<user> {query.lower()} <context> {context} <assistant> "
        prompt_ids = encode(prompt, self.stoi)[-self.config["block_size"] :]
        generated = self.model.generate(
            prompt_ids,
            self.itos,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        generated_text = decode(generated, self.itos).strip()
        if self._matches_retrieved_recipe(generated_text, record):
            return generated_text
        return context

    def _matches_retrieved_recipe(self, answer, record):
        """Accept generated text only when it exactly preserves the source row."""
        expected = f"{record['recipe_name']}. {record['output']}"
        return words(answer) == words(expected)


def main():
    parser = argparse.ArgumentParser(
        description="Micro Language Model de recetas con atención por coseno."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser(
        "train", help="Entrena Micro Language Model con el CSV."
    )
    train_parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    train_parser.add_argument("--steps", type=int, default=1000)
    train_parser.add_argument("--batch-size", type=int, default=16)

    ask_parser = commands.add_parser("ask", help="Pregunta al modelo en lenguaje natural.")
    ask_parser.add_argument("query", nargs="*", help="Pregunta libre sobre recetas.")
    ask_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ask_parser.add_argument("--max-new-tokens", type=int, default=100)
    ask_parser.add_argument("--temperature", type=float, default=0.35)

    args = parser.parse_args()
    try:
        if args.command == "train":
            if args.steps < 1 or args.batch_size < 1:
                raise ValueError("--steps y --batch-size deben ser positivos.")
            rows, vocabulary, device = train(
                args.data, args.model, args.steps, args.batch_size
            )
            print(
                f"Entrenado con {rows} recetas; vocabulario de {vocabulary} tokens "
                f"en {device}. Modelo: {args.model}"
            )
            return 0

        if args.temperature <= 0:
            raise ValueError("--temperature debe ser mayor que cero.")
        model = MicroLanguageModel.load(args.model)
        query = " ".join(args.query).strip()
        if query:
            print(model.answer(query, args.max_new_tokens, args.temperature))
            return 0

        print("Pregunta en lenguaje natural; escribe /salir para terminar.")
        while True:
            try:
                query = input("Tú> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if query.lower() in {"/salir", "/exit", "salir", "exit"}:
                break
            if query:
                print(
                    f"MicroLM> {model.answer(query, args.max_new_tokens, args.temperature)}"
                )
        return 0
    except (OSError, csv.Error, KeyError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
