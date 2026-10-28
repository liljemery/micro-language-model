"""Interfaz de línea de comandos para entrenar y consultar el modelo."""

import argparse
import csv
import sys
from pathlib import Path

from .config import (
    DEFAULT_DATA,
    DEFAULT_EVALUATION_OUTPUT,
    DEFAULT_MODEL,
    DEFAULT_SPLIT_HELDOUT,
    DEFAULT_SPLIT_QUERIES,
    DEFAULT_SPLIT_TRAIN,
)
from .evaluation import evaluate, split_dataset
from .inference import MicroLanguageModel
from .training import train


def main():
    parser = argparse.ArgumentParser(
        description="Micro Language Model de recetas con atención por coseno."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser(
        "train", help="Entrena Micro Language Model con el CSV."
    )
    train_parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    train_parser.add_argument(
        "--retrieval-data",
        type=Path,
        help="CSV adicional para indexar, sin usar sus filas para entrenar pesos.",
    )
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    train_parser.add_argument("--steps", type=int, default=1000)
    train_parser.add_argument("--batch-size", type=int, default=16)

    split_parser = commands.add_parser(
        "split", help="Divide el CSV por título y prepara una evaluación inicial."
    )
    split_parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    split_parser.add_argument(
        "--train-output", type=Path, default=DEFAULT_SPLIT_TRAIN
    )
    split_parser.add_argument(
        "--heldout-output", type=Path, default=DEFAULT_SPLIT_HELDOUT
    )
    split_parser.add_argument(
        "--queries-output", type=Path, default=DEFAULT_SPLIT_QUERIES
    )
    split_parser.add_argument("--fraction", type=float, default=0.2)
    split_parser.add_argument("--seed", type=int, default=42)

    ask_parser = commands.add_parser("ask", help="Pregunta al modelo en lenguaje natural.")
    ask_parser.add_argument("query", nargs="*", help="Pregunta libre sobre recetas.")
    ask_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ask_parser.add_argument("--max-new-tokens", type=int, default=100)
    ask_parser.add_argument("--temperature", type=float, default=0.35)

    evaluate_parser = commands.add_parser(
        "evaluate", help="Evalúa recuperación y genera muestras para revisión."
    )
    evaluate_parser.add_argument("--queries", type=Path, required=True)
    evaluate_parser.add_argument(
        "--output", type=Path, default=DEFAULT_EVALUATION_OUTPUT
    )
    evaluate_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    evaluate_parser.add_argument("--runs", type=int, default=5)
    evaluate_parser.add_argument("--seed", type=int, default=42)
    evaluate_parser.add_argument("--max-new-tokens", type=int, default=100)
    evaluate_parser.add_argument("--temperature", type=float, default=0.35)

    args = parser.parse_args()
    try:
        if args.command == "split":
            summary = split_dataset(
                args.data,
                args.train_output,
                args.heldout_output,
                args.queries_output,
                args.fraction,
                args.seed,
            )
            print(
                f"Entrenamiento: {summary['train_rows']} filas, "
                f"{summary['train_titles']} títulos → {summary['train_path']}"
            )
            print(
                f"Reserva: {summary['heldout_rows']} filas, "
                f"{summary['heldout_titles']} títulos → {summary['heldout_path']}"
            )
            print(
                f"Consultas no ambiguas: {summary['queries']} → "
                f"{summary['queries_path']}"
            )
            print("Añade casos de paráfrasis, español y fuera de dominio a ese CSV.")
            return 0

        if args.command == "train":
            if args.steps < 1 or args.batch_size < 1:
                raise ValueError("--steps y --batch-size deben ser positivos.")
            rows, vocabulary, device = train(
                args.data,
                args.model,
                args.steps,
                args.batch_size,
                args.retrieval_data,
            )
            print(
                f"Entrenado con {rows} recetas; vocabulario de {vocabulary} tokens "
                f"en {device}. Modelo: {args.model}"
            )
            if args.retrieval_data:
                print(
                    "El CSV adicional se indexó para recuperación sin usarlo "
                    "como objetivo de entrenamiento."
                )
            return 0

        if args.temperature <= 0:
            raise ValueError("--temperature debe ser mayor que cero.")
        if args.max_new_tokens < 1:
            raise ValueError("--max-new-tokens debe ser positivo.")
        if args.command == "evaluate" and args.runs < 1:
            raise ValueError("--runs debe ser positivo.")
        if args.command == "evaluate" and args.output.resolve() == args.model.resolve():
            raise ValueError("--output no puede sobrescribir el checkpoint del modelo.")
        model = MicroLanguageModel.load(args.model)
        if args.command == "evaluate":
            summary = evaluate(
                model,
                args.queries,
                args.output,
                args.runs,
                args.seed,
                args.max_new_tokens,
                args.temperature,
            )
            print(f"Preguntas: {summary['queries']}")
            print(f"Generaciones: {summary['generations']}")
            print(f"Sin contexto recuperado: {summary['no_context']}")
            if summary["retrieval_hit_rate"] is None:
                print("Hit@1: sin casos etiquetados con receta esperada")
            else:
                print(
                    "Recuperación Hit@1: "
                    f"{summary['retrieval_hits']}/{summary['retrieval_labeled']} "
                    f"({summary['retrieval_hit_rate']:.1%})"
                )
            print(f"Resultados para revisión: {summary['output_path']}")
            return 0

        query = " ".join(args.query).strip()
        if query:
            _print_stream(model, query, args.max_new_tokens, args.temperature)
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
                sys.stdout.write("MicroLM> ")
                _write_stream(model, query, args.max_new_tokens, args.temperature)
                sys.stdout.write("\n")
                sys.stdout.flush()
        return 0
    except (OSError, csv.Error, KeyError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


def _write_stream(model, query, max_new_tokens, temperature):
    for chunk in model.stream_answer(query, max_new_tokens, temperature):
        sys.stdout.write(chunk)
        sys.stdout.flush()


def _print_stream(model, query, max_new_tokens, temperature):
    _write_stream(model, query, max_new_tokens, temperature)
    sys.stdout.write("\n")
