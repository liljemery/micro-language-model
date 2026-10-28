"""Carga del checkpoint, recuperación de contexto y respuesta a consultas."""

import torch

from .model import MicroGPT, choose_device
from .retrieval import cosine_similarity, vectorize
from .text import decode, encode, normalize_ingredient_terms, words


class MicroLanguageModel:
    """Combina búsqueda de recetas y generación causal condicionada."""

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
        """Busca una receta apropiada y genera una respuesta con ese contexto."""
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
        """Acepta generación solo si conserva exactamente el texto fuente."""
        expected = f"{record['recipe_name']}. {record['output']}"
        return words(answer) == words(expected)
