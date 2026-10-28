# Micro Language Model · Modelo de lenguaje pequeño

**Proyecto educativo de un modelo de lenguaje pequeño, entrenado desde cero con recetas.** Está pensado para practicar tokenización, generación autoregresiva, atención causal y recuperación por similitud de coseno. No es un asistente general ni un recetario validado.

**Educational tiny language model trained from scratch on a recipe dataset.** It demonstrates word-level tokenization, autoregressive generation, causal attention, and cosine-similarity retrieval. It is not a general assistant or a validated cookbook.

## Qué incluye / What this project includes

- Un GPT causal pequeño implementado en PyTorch: 2 bloques, 4 cabezas de atención, dimensión 64 y contexto máximo de 192 tokens.
- Atención propia: normaliza Q y K y calcula sus puntajes con similitud de coseno; una máscara causal impide mirar tokens futuros.
- Recuperación TF-IDF por coseno sobre los ingredientes del CSV, usada como contexto para responder preguntas.
- CLI para entrenar, preguntar una vez o conversar en modo interactivo.
- Dataset de entrenamiento incluido: 1.000 filas en `data/generated_recipe_dataset.csv`.

## Inicio rápido / Quick start

Requiere Python 3.14. En macOS y Linux, ejecuta desde la carpeta del proyecto:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python micro_language_model.py train
.venv/bin/python micro_language_model.py ask "I have carrots and peas, what can I make?"
```

En Windows PowerShell (los comandos son equivalentes; esa plataforma no se verificó en el equipo de desarrollo):

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe micro_language_model.py train
.venv\Scripts\python.exe micro_language_model.py ask "I have carrots and peas, what can I make?"
```

Usa el intérprete dentro de `.venv` para que `pip` instale PyTorch en ese entorno y no en el Python administrado por Homebrew. El entrenamiento tarda alrededor de un minuto en el equipo de desarrollo, pero el tiempo depende del hardware. Crea `micro_language_model.pt` localmente; ese artefacto se ignora en Git y se puede regenerar con `train`.

### Más formas de usarlo / More ways to use it

```bash
# Pregunta sin ingredientes conocidos en el CSV
.venv/bin/python micro_language_model.py ask "whats the recipe for soup"

# Consulta natural con plurales o alias en español
.venv/bin/python micro_language_model.py ask "Tengo zanahorias y guisantes, ¿qué puedo preparar?"

# Modo interactivo; escribe /salir para terminar
.venv/bin/python micro_language_model.py ask

# Opciones de entrenamiento y consulta
.venv/bin/python micro_language_model.py train --help
.venv/bin/python micro_language_model.py ask --help
```

Las respuestas generadas están en inglés, porque el dataset está en inglés. El prototipo reconoce algunos nombres de ingredientes, plurales y alias en español. Cada token de la respuesta se genera desde los pesos del checkpoint y se imprime en cuanto está disponible. La búsqueda coseno puede añadir una receta del CSV como contexto; sin una coincidencia, el modelo intenta responder sin contexto recuperado. Como el vocabulario y los datos son pequeños, esa salida puede ser incoherente o inventar pasos.

## Arquitectura / Architecture

```mermaid
flowchart TD
    cli["Interfaz<br/>micro_language_model.py<br/>micro_lm/cli.py"]
    training["Entrenamiento<br/>training.py"]
    inference["Consulta<br/>inference.py"]
    components["Componentes de micro_lm/<br/>dataset.py · retrieval.py<br/>model.py · text.py · config.py"]

    cli --> training
    cli --> inference
    training --> components
    inference --> components
```

El lanzador conserva los comandos existentes y delega en la CLI. Al consultar, `inference.py` puede recuperar una receta por coseno como contexto y pasa el prompt al Transformer causal. `model.py` genera un token cada vez y la CLI lo imprime inmediatamente; no hay respuestas fijas ni sustitución por la fila original del CSV. Consulta la [guía de aprendizaje](docs/LEARNING_GUIDE.md) y la [tarjeta del modelo](docs/MODEL_CARD.md) para ver detalles y límites.

## Estructura del repositorio / Repository layout

```text
micro-language-model/
├── data/generated_recipe_dataset.csv  # Datos usados para entrenar y recuperar
├── micro_lm/
│   ├── __init__.py                    # Paquete del modelo
│   ├── cli.py                         # Argumentos, comandos y modo interactivo
│   ├── config.py                      # Rutas, tokens especiales y tamaño del modelo
│   ├── dataset.py                     # CSV y ejemplos de entrenamiento
│   ├── inference.py                   # Carga, recuperación y respuestas
│   ├── model.py                       # Atención y bloques Transformer en PyTorch
│   ├── retrieval.py                   # Búsqueda TF-IDF y similitud coseno
│   ├── text.py                        # Tokenización, alias y conversión texto/IDs
│   └── training.py                    # Lotes, entrenamiento y checkpoint
├── docs/
│   ├── DATASET.md                     # Esquema, estadísticas y procedencia
│   ├── LEARNING_GUIDE.md              # Recorrido técnico paso a paso
│   └── MODEL_CARD.md                  # Uso previsto y limitaciones
├── micro_language_model.py            # Punto de entrada compatible
├── requirements.txt                   # Dependencias fijadas
└── CONTRIBUTING.md                    # Guía y ejercicios de práctica
```

Cada módulo tiene una responsabilidad y sus propios imports. `requirements.txt` sigue siendo el único lugar para instalar dependencias externas; actualmente contiene PyTorch y NumPy. La CLI pública permanece en `micro_language_model.py`, que delega en `micro_lm.cli` para conservar los comandos del inicio rápido.

## Contenido educativo / Learning path

1. Sigue `read_dataset` y `make_training_sequences` en `micro_lm/dataset.py`; estudia `build_idf` y `cosine_similarity` en `micro_lm/retrieval.py`; luego `train` en `micro_lm/training.py`.
2. Estudia `CosineSelfAttention`, `TransformerBlock` y `MicroGPT` en `micro_lm/model.py`.
3. Compara `cosine_similarity` en `micro_lm/retrieval.py` con Q/K por coseno en `micro_lm/model.py`.
4. Usa los ejercicios sugeridos en [CONTRIBUTING.md](CONTRIBUTING.md) para extender el prototipo.

## Datos y licencia / Data and licensing

El dataset proviene del archivo `SLM Dataset.zip` proporcionado para este proyecto. El ZIP solo contenía el CSV, sin autor, fuente ni licencia. El proyecto lo publica para el uso educativo solicitado; su procedencia y derechos de redistribución no están verificados. Consulta [docs/DATASET.md](docs/DATASET.md) antes de reutilizarlo fuera de este contexto. No se asigna una licencia de software o datos en este repositorio.

---

## English summary

Clone the repository, create a Python 3.14 virtual environment, install `requirements.txt`, then run `micro_language_model.py train` before asking questions. The CSV is included; generated weights are intentionally not versioned. This is a domain-limited learning prototype: it streams tokens from a tiny causal Transformer and can add cosine-retrieved recipe context. It is not a general-purpose or externally pretrained LLM. See the linked docs for setup, data provenance, architecture, and limitations.
