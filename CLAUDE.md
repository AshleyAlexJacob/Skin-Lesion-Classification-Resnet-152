# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Skin lesion image classification using a ResNet-152 backbone (PyTorch/torchvision). The codebase follows a modular ML pipeline pattern with separate concerns for data, model, and pipeline orchestration. Most source files are currently empty scaffolds awaiting implementation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
pip install -r requirements.txt
```

## Common Commands

**Lint:**
```bash
ruff check src/ app/
ruff format src/ app/
```

**Run the Streamlit app:**
```bash
streamlit run app/streamlit.py
```

**Run the FastAPI server:**
```bash
uvicorn app.main:app --reload
```

---

## Code Style — PEP 8

All source files must comply with PEP 8. `ruff` is the enforcer; CI will fail on lint errors.

### Naming
- Modules and packages: `snake_case`
- Classes: `PascalCase`
- Functions and variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private helpers: prefix with a single underscore

### Docstrings
Every public function, method, and class must have a Google-style docstring. Every docstring must include `Args`, `Returns`, and `Raises` sections where applicable. Every parameter in `Args` must have a description, not just a type. The `Raises` section must list every exception the function can raise intentionally.

### Type Annotations
All function signatures must include type annotations for every parameter and the return type. Use `from __future__ import annotations` at the top of each file.

### Line Length
Maximum line length is 88 characters. Break long function signatures across multiple lines with one argument per line.

### File / Module Size
Keep each source file between 300 and 500 lines. If a file grows beyond 500 lines, split it into focused sub-modules and re-export the public API from the package `__init__.py`.

### Imports
Order imports in three blocks separated by a blank line: standard library, third-party packages, then internal project modules.

### Other Rules
- Prefer `pathlib.Path` over raw string paths.
- Never use bare `except:`; always catch a specific exception type.
- Avoid mutable default arguments; use `None` and assign inside the function body.
- Use f-strings for string formatting; avoid `%` and `.format()`.

---

## FastAPI Best Practices

### Project Structure
Keep all API concerns inside `app/`. Organise into `routers/` (one file per domain), `schemas/` (separate files for request and response models), `dependencies.py` (shared `Depends()` providers), and `middleware.py` (CORS, logging, error handlers).

### Application Factory
Define the app via a `create_app()` factory function rather than a bare module-level instance. This keeps the app testable and configurable. Register all routers and middleware inside the factory.

### Pydantic Schemas
Define explicit Pydantic models for every request and response body; never use raw `dict`. Every field must have a type annotation and a `Field(description=...)` value so that the auto-generated OpenAPI docs are meaningful.

### Dependency Injection
Load heavy resources such as the model and label encoder once at startup using `Depends()` with `@lru_cache`. Never instantiate or load models inside a route handler.

### Route Handlers
Route handlers must be thin: validate input, call a service function from `src/`, and return a schema. Business logic must live in `src/`, not in `app/routers/`. Every route must declare a `summary`, `response_model`, and explicit `status_code`.

### Error Handling
Register a global exception handler in `middleware.py`. Do not wrap individual route handlers in try/except blocks for general errors.

### Async vs Sync
Use `async def` for I/O-bound routes. Use plain `def` for CPU-bound routes so FastAPI runs them in a thread pool. Never call blocking code directly inside an `async def` without offloading to an executor.

### Versioning
Prefix all routes with `/v1/` from day one. Add a `/v2/` router for breaking changes rather than modifying existing routes.

---

## Architecture

### `src/` — Core library
- **`src/data/loader.py`** — Dataset loading; wraps PyTorch `Dataset`/`DataLoader` for raw images from `data/raw/`
- **`src/data/encoder.py`** — Label encoding for skin lesion classes
- **`src/model/loader.py`** — Loads/instantiates ResNet-152; handles pretrained weights and checkpoint saving/loading
- **`src/model/train.py`** — Training loop (forward pass, loss, optimizer step)
- **`src/model/evaluation.py`** — Metrics computation (accuracy, confusion matrix, etc.)
- **`src/model/prediction.py`** — Inference on single images or batches
- **`src/utils.py`** — Shared utilities (device selection, path helpers)

### `src/pipelines/` — Orchestration scripts
- **`data_preprocessing.py`** — Reads `data/raw/`, applies transforms, writes to `data/processed/`
- **`model_training.py`** — Runs full training: data loading → model setup → train loop → checkpoint save
- **`model_evaluation.py`** — Loads a checkpoint and evaluates on a test split

### `app/`
- **`main.py`** — Application factory and entry point
- **`routers/`** — Route handlers grouped by domain
- **`schemas/`** — Pydantic request/response models
- **`dependencies.py`** — Shared `Depends()` providers (model, encoder)
- **`middleware.py`** — CORS, logging, global error handlers
- **`streamlit.py`** — Interactive demo; uploads an image and returns the predicted lesion class

### `data/`
- `data/raw/` — Original dataset (not tracked by git)
- `data/processed/` — Preprocessed/augmented data ready for training (not tracked by git)

### `notebooks/`
Jupyter notebooks for exploratory work; not part of the production pipeline.

---

## Key Dependencies

| Package | Role |
|---|---|
| `torch` / `torchvision` | ResNet-152 model and image transforms |
| `pandas` | Metadata/label CSV handling |
| `matplotlib` | Visualisation in notebooks |
| `streamlit` | Web demo |
| `fastapi` + `uvicorn` | REST API serving |
| `pydantic` | Request/response schema validation |
| `ruff` | Linting and formatting |