"""Preparación de lotes, entrenamiento y guardado del checkpoint."""

from pathlib import Path

import torch
import torch.nn.functional as F

from .config import CONFIG, SPECIAL_TOKENS
from .dataset import make_training_sequences, read_dataset
from .model import MicroGPT, choose_device
from .retrieval import build_idf
from .text import encode, tokenize


def train(data_path, model_path, steps, batch_size):
    """Entrena el modelo desde el CSV y guarda vocabulario, datos y pesos."""
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
    model_path = Path(model_path)
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
