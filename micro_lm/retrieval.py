"""Búsqueda de recetas con vectores dispersos TF-IDF y similitud coseno."""

import math
from collections import Counter

from .text import words


def build_idf(records):
    """Calcula pesos IDF sobre los ingredientes descritos en cada fila."""
    document_frequency = Counter()
    for record in records:
        document_frequency.update(set(words(record["input"])))
    total = len(records)
    return {
        token: math.log((total + 1) / (frequency + 1)) + 1.0
        for token, frequency in document_frequency.items()
    }


def vectorize(tokens, idf):
    """Crea un vector disperso TF-IDF para los términos conocidos."""
    counts = Counter(token for token in tokens if token in idf)
    return {
        token: (1.0 + math.log(count)) * idf[token]
        for token, count in counts.items()
    }


def vector_norm(vector):
    """Calcula la norma L2 de un vector disperso."""
    return math.sqrt(sum(value * value for value in vector.values()))


def cosine_similarity(left, right):
    """Calcula la similitud coseno entre dos vectores dispersos."""
    denominator = vector_norm(left) * vector_norm(right)
    if not denominator:
        return 0.0
    return sum(value * right.get(token, 0.0) for token, value in left.items()) / denominator
