"""Tokenización y normalización de términos."""

import re

from .config import SPECIAL_TOKENS


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


def tokenize(text):
    """Divide texto en tokens de palabra, signos y tokens especiales."""
    return TOKEN_RE.findall(text)


def words(text):
    """Extrae palabras en minúsculas para comparar texto y recetas."""
    return WORD_RE.findall(text.lower())


def normalize_ingredient_terms(text):
    """Normaliza alias conocidos, como «zanahorias» a «carrot»."""
    return {INGREDIENT_ALIASES.get(token, token) for token in words(text)}


def encode(text, stoi):
    """Convierte texto a IDs, usando el token desconocido como alternativa."""
    unknown_id = stoi["<unk>"]
    return [stoi.get(token, unknown_id) for token in tokenize(text)]


def decode(token_ids, itos):
    """Convierte IDs a texto y oculta los tokens internos del modelo."""
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


def decode_token(token, first=False):
    """Format one generated token without buffering the rest of the answer."""
    if token in SPECIAL_TOKENS:
        return ""
    if token in {".", ",", ":", ";", "!", "?", ")", "]", "}"}:
        return token
    return ("" if first else " ") + token
