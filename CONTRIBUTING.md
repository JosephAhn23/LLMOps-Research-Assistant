# Contributing

## Setup

```bash
git clone https://github.com/JosephAhn23/LLMOps-Research-Assistant
cd LLMOps-Research-Assistant
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # pytest, ruff, mypy
```

## Code Standards

**Type hints** -- all public functions must have full type annotations.

```python
# Good
def retrieve(query: str, top_k: int = 10) -> list[RetrievedChunk]:
    ...

# Bad
def retrieve(query, top_k=10):
    ...
```

**Structured logging** -- use `logging.getLogger(__name__)`, never `print()`.

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Retrieved %d chunks in %.1f ms", len(chunks), latency_ms)
```

**Pydantic for config** -- all configurable classes use `pydantic.BaseModel` or `dataclasses.dataclass`.

**No magic numbers** -- extract constants with descriptive names.

## Testing

```bash
pytest                          # run all tests
pytest tests/ -v --tb=short     # verbose with short tracebacks
pytest tests/test_multi_agent.py -v  # specific module
```

New code requires tests. Target >80% coverage on new modules.

```bash
pytest --cov=agents --cov-report=term-missing
```

## Linting

```bash
ruff check .          # lint
ruff format .         # format
mypy agents/ --strict # type check
```

## Adding a New Module

1. Create `your_module/` with `__init__.py` that exports public classes.
2. Add an entry to the `Project Structure` table in `README.md`.
3. Add a row to the relevant section in `README.resume.md`.
4. Write tests in `tests/test_your_module.py`.
5. Add any new dependencies to `requirements.txt` with a comment explaining why.

## Pull Request Checklist

- [ ] Type hints on all public functions
- [ ] Structured logging (no print statements)
- [ ] Tests written and passing (`pytest`)
- [ ] `README.md` project structure updated if new directory added
- [ ] No hardcoded secrets or API keys
- [ ] Docstrings on public classes and non-obvious functions

## Architecture Principles

**Dependency injection over global state** -- pass dependencies as constructor arguments, not module-level singletons.

**Fail loudly in development, degrade gracefully in production** -- raise exceptions during testing; return degraded responses with logged warnings in production.

**Optional heavy dependencies** -- if a module requires `torch`, `mlflow`, or other large packages, use lazy imports so the module is importable without them:

```python
def __getattr__(name: str):
    if name == "HeavyClass":
        from heavy_module import HeavyClass
        return HeavyClass
    raise AttributeError(f"module has no attribute {name!r}")
```

**Measure before optimizing** -- include a `benchmark()` function or script for any performance-sensitive module.
