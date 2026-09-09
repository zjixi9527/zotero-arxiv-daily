# Copilot Instructions

> **Project overview, commands, architecture, plugin systems, configuration, data
> classes, and git workflow are maintained in [`CLAUDE.md`](../CLAUDE.md)**. Read that
> file first — it is the single source of truth for this repository.

The sections below cover conventions specific to this repo that Copilot should follow
when writing code. Everything else (pipeline structure, retriever/reranker plugin
registration, Hydra + OmegaConf config, `Paper`/`CorpusPaper` data classes) is
documented in [`CLAUDE.md`](../CLAUDE.md).

## Tooling

- Linting and formatting are handled by **ruff** (configured in `pyproject.toml`).
  Run `uv run ruff check .` and `uv run ruff format .`; both are enforced in CI.
- The `local` reranker needs heavy ML deps — install them with `uv sync --extra local`.

## Testing Conventions

- Tests use **pytest monkeypatch + `SimpleNamespace`** for stubs — not `unittest.mock`.
- A session-scoped Hydra config in `tests/conftest.py` is deep-copied per test via the `config` fixture.
- Canned response factories live in `tests/canned_responses.py` (e.g., `make_stub_openai_client()`, `make_stub_zotero_client()`).
- Tests marked `@pytest.mark.slow` require heavy dependencies (model downloads) and are excluded by default (`addopts = "-m 'not slow'"` in pyproject.toml).
- Monkeypatching targets the module-level import path (e.g., `"zotero_arxiv_daily.executor.zotero.Zotero"`).

## Coding Conventions

- **Logging:** `loguru.logger` throughout — never `print()` or stdlib `logging`.
- **Type hints:** Modern Python 3.13+ syntax (`list[Paper]`, `str | None`).
- **Constants:** Module-level `UPPER_SNAKE_CASE`.
- **Private methods:** Prefixed with `_` (e.g., `_retrieve_raw_papers`).
- **Error handling:** Graceful degradation with try/except and fallback logic; log warnings rather than raising.
- **Config injection:** All major components receive `DictConfig` at init and store it as `self.config`.
