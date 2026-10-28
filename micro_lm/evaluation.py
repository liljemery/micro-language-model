"""Evaluación repetible de recuperación y generaciones del modelo."""

import csv
import random
from pathlib import Path

import torch

from .dataset import read_dataset


DATA_COLUMNS = ("input", "recipe_name", "output")


OUTPUT_COLUMNS = (
    "case_id",
    "category",
    "query",
    "expected_recipe_name",
    "retrieved_recipe_name",
    "retrieval_method",
    "cosine_score",
    "retrieval_hit",
    "run",
    "seed",
    "temperature",
    "answer",
    "review_relevance_0_2",
    "review_source_consistency_0_2",
    "review_completeness_0_2",
    "unsupported_claims",
    "review_notes",
)


def read_queries(path):
    """Lee casos etiquetados; el nombre esperado puede quedar vacío."""
    required = {"query", "expected_recipe_name", "category"}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "El CSV de evaluación debe incluir query, "
                "expected_recipe_name y category."
            )
        cases = []
        for row_number, row in enumerate(reader, start=2):
            query = (row.get("query") or "").strip()
            if not query:
                raise ValueError(f"Falta query en la fila {row_number}.")
            cases.append(
                {
                    "case_id": (
                        row.get("case_id") or f"case-{row_number - 1:03d}"
                    ).strip(),
                    "category": (row.get("category") or "unspecified").strip(),
                    "query": query,
                    "expected_recipe_name": (
                        row.get("expected_recipe_name") or ""
                    ).strip(),
                }
            )
    if not cases:
        raise ValueError("El CSV de evaluación no contiene preguntas.")
    return cases


def split_dataset(
    data_path, train_path, heldout_path, queries_path, fraction, seed
):
    """Divide por título de receta y crea consultas para filas no ambiguas."""
    if not 0 < fraction < 1:
        raise ValueError("--fraction debe estar entre 0 y 1.")
    paths = [
        Path(data_path),
        Path(train_path),
        Path(heldout_path),
        Path(queries_path),
    ]
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError(
            "El CSV de entrada y los tres archivos de salida deben ser distintos."
        )

    records = read_dataset(data_path)
    grouped = {}
    input_titles = {}
    for record in records:
        title_key = _normalize_title(record["recipe_name"])
        input_key = _normalize_ingredients(record["input"])
        grouped.setdefault(title_key, []).append(record)
        input_titles.setdefault(input_key, set()).add(title_key)

    titles = sorted(grouped)
    if len(titles) < 2:
        raise ValueError("Se necesitan al menos dos títulos de receta para dividir.")
    random.Random(seed).shuffle(titles)
    heldout_count = min(len(titles) - 1, max(1, round(len(titles) * fraction)))
    heldout_titles = set(titles[:heldout_count])
    training_records = [
        record
        for record in records
        if _normalize_title(record["recipe_name"]) not in heldout_titles
    ]
    heldout_records = [
        record
        for record in records
        if _normalize_title(record["recipe_name"]) in heldout_titles
    ]

    query_records = []
    seen = set()
    for record in heldout_records:
        input_key = _normalize_ingredients(record["input"])
        if len(input_titles[input_key]) != 1:
            continue
        key = (input_key, _normalize_title(record["recipe_name"]))
        if key in seen:
            continue
        seen.add(key)
        query_records.append(
            {
                "case_id": f"heldout-{len(query_records) + 1:04d}",
                "query": _training_style_query(record["input"]),
                "expected_recipe_name": record["recipe_name"],
                "category": "heldout_recipe",
            }
        )
    if not query_records:
        raise ValueError(
            "La división no produjo consultas no ambiguas; prueba otra --seed o --fraction."
        )

    _write_csv(train_path, DATA_COLUMNS, training_records)
    _write_csv(heldout_path, DATA_COLUMNS, heldout_records)
    _write_csv(
        queries_path,
        ("case_id", "query", "expected_recipe_name", "category"),
        query_records,
    )
    return {
        "train_rows": len(training_records),
        "train_titles": len(
            {_normalize_title(row["recipe_name"]) for row in training_records}
        ),
        "heldout_rows": len(heldout_records),
        "heldout_titles": len(heldout_titles),
        "queries": len(query_records),
        "train_path": Path(train_path),
        "heldout_path": Path(heldout_path),
        "queries_path": Path(queries_path),
    }


def evaluate(
    model, queries_path, output_path, runs, seed, max_new_tokens, temperature
):
    """Guarda generaciones y métricas de recuperación para revisión humana."""
    cases = read_queries(queries_path)
    output_path = Path(output_path)
    if output_path.resolve() == Path(queries_path).resolve():
        raise ValueError("--output no puede sobrescribir el CSV de consultas.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hits = []
    no_context = 0

    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for case_index, case in enumerate(cases):
            record, cosine_score, retrieval_method = model.retrieve(case["query"])
            retrieved_name = record["recipe_name"] if record else ""
            expected_name = case["expected_recipe_name"]
            retrieval_hit = ""
            if expected_name:
                matched = _normalize_title(retrieved_name) == _normalize_title(
                    expected_name
                )
                hits.append(matched)
                retrieval_hit = str(matched).lower()
            if record is None:
                no_context += 1

            for run_index in range(runs):
                sample_seed = seed + case_index * runs + run_index
                torch.manual_seed(sample_seed)
                answer = "".join(
                    model.stream_answer(
                        case["query"],
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                )
                writer.writerow(
                    {
                        **case,
                        "retrieved_recipe_name": retrieved_name,
                        "retrieval_method": retrieval_method,
                        "cosine_score": (
                            "" if cosine_score is None else f"{cosine_score:.6f}"
                        ),
                        "retrieval_hit": retrieval_hit,
                        "run": run_index + 1,
                        "seed": sample_seed,
                        "temperature": temperature,
                        "answer": answer,
                        "review_relevance_0_2": "",
                        "review_source_consistency_0_2": "",
                        "review_completeness_0_2": "",
                        "unsupported_claims": "",
                        "review_notes": "",
                    }
                )

    return {
        "queries": len(cases),
        "generations": len(cases) * runs,
        "retrieval_labeled": len(hits),
        "retrieval_hits": sum(hits),
        "retrieval_hit_rate": sum(hits) / len(hits) if hits else None,
        "no_context": no_context,
        "output_path": output_path,
    }


def _normalize_title(title):
    return " ".join(title.casefold().split())


def _normalize_ingredients(ingredients):
    return " ".join(sorted(ingredients.casefold().split()))


def _training_style_query(ingredients):
    parts = ingredients.split()
    if len(parts) > 2:
        natural = ", ".join(parts[:-1]) + ", and " + parts[-1]
    elif len(parts) == 2:
        natural = " and ".join(parts)
    else:
        natural = ingredients
    return f"could you give me a recipe with {natural}?"


def _write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
