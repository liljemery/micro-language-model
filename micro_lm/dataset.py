"""Lectura del CSV, ejemplos de entrenamiento y búsqueda por coseno."""

import csv

from .text import words


def read_dataset(path):
    """Lee recetas completas y valida las columnas necesarias del CSV."""
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
    """Crea preguntas de ejemplo para que el modelo practique continuaciones."""
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
