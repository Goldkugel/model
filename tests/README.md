# Tests

The pytest suite of the `model` package. Run it from the repository root:

```bash
pytest -v
```

| File | Content |
|---|---|
| `conftest.py` | Registers stand-ins for `vllm` and `torch` when they are not installed, points the logger at a temporary folder, and defines `REPO_ROOT` |
| `ModelTest.py` | Tests of `Model`, `ModelConfig`, and `ModelUtils` |

The tests need neither a GPU nor a downloaded model: tests that touch the engine replace `torch.cuda` and the vLLM classes with fakes (`FakeLLM`, `FakeTokenizer`). `pytest` collects every `*Test.py` file in this folder (see `[tool.pytest.ini_options]` in `pyproject.toml`), so a new test file only needs that name to run locally and in CI.
