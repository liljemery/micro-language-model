#!/usr/bin/env python3
"""Punto de entrada compatible con los comandos del README."""

import sys

try:
    from micro_lm.cli import main
except ModuleNotFoundError as error:
    if error.name in {"numpy", "torch"}:
        print(
            "Falta una dependencia. Instálala en el entorno virtual del proyecto con: "
            ".venv/bin/python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    raise


if __name__ == "__main__":
    raise SystemExit(main())
