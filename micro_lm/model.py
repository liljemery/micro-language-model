"""Capas del Transformer causal pequeño con atención por similitud coseno."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SPECIAL_TOKENS


def choose_device():
    """Usa CPU para mantener el ejemplo compatible con equipos comunes."""
    torch.set_num_threads(min(4, torch.get_num_threads()))
    return torch.device("cpu")


class CosineSelfAttention(nn.Module):
    """Atención causal multi-cabeza que calcula similitud coseno entre Q y K."""

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
    """Bloque residual de autoatención seguido por una red feed-forward."""

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
    """Modelo causal basado en embeddings y bloques Transformer pequeños."""

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
