"""Interfaz de línea de comandos para entrenar y consultar el modelo."""

import argparse
import csv
import sys
from pathlib import Path

from .config import DEFAULT_DATA, DEFAULT_MODEL
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
