# Model

A thin wrapper around [vLLM](https://github.com/vllm-project/vllm) for prompting a local LLM: manages multiple independent conversation histories, builds Gemma-style chat prompts, runs batched generation, and logs prompts to disk.

## Features

- **Pydantic-validated configuration** — model selection and sampling parameters are loaded and type-checked via `ModelConfig`.
- **Multi-history chat state** — track several independent conversations at once; add messages by broadcasting to all histories, one-per-history, or initializing new histories from scratch.
- **Gemma-style prompt formatting** — conversation histories are rendered using `<start_of_turn>`/`<end_of_turn>` markers; system messages are folded into the following user turn, since Gemma has no dedicated system-turn type.
- **Batched generation via vLLM** — all pending histories are sent to the model in a single `generate()` call.
- **GPU-aware initialization** — automatically detects available GPUs and configures `tensor_parallel_size` accordingly; enables expert parallelism specifically for `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`.
- **Prompt logging** — write every message across every history to a semicolon-delimited log file for later inspection.

> **Note:** `generate()` currently only builds prompts in Gemma's turn format. `ModelUtils.py` also defines the special tokens for Llama-style formatting (`start_header_id`, `end_header_id`, `eot_id`, etc.), but nothing in this repository currently uses them — support for other model families appears to be anticipated but not yet implemented.

## Repository Structure

```
.
├── config/
│   └── config.yaml          # Configuration file; must contain an "llm" section
├── prompts/                 # (currently empty/placeholder)
├── src/
│   ├── __init__py           # Note: missing the leading dot — not recognized by
│   │                         # Python as __init__.py, so this directory isn't
│   │                         # currently importable as a package.
│   ├── ModelConfig.py       # Pydantic configuration schema
│   ├── Model.py             # Core Model class (chat state, prompting, generation, cleanup)
│   ├── ModelUtils.py        # Roles, special tokens, and prompt-log writer
│   └── ModelTest.py         # Pytest suite (stubs vLLM/CUDA, no GPU required)
├── LICENSE
└── pyproject.toml
```

> `src/__init__py` is missing the `.` before `py` — as named, Python won't treat `src/` as a package or run this file's imports automatically. If that's not intentional, renaming it to `__init__.py` would fix it.

## Requirements & Installation

This project depends on the [`logger`](https://github.com/Goldkugel/logger) package (installed directly from Git) plus vLLM and its dependencies. Only `logger` is currently declared in `pyproject.toml`; the rest need to be installed separately:

```bash
pip install vllm torch pydantic pyyaml
pip install "logger @ git+https://github.com/Goldkugel/logger.git@v1.0.5"
```

vLLM requires a CUDA-capable GPU to actually load and run a model; the test suite does not (see [Running Unit Tests](#running-unit-tests)).

For running the test suite, also install:

```bash
pip install pytest
```

## Configuration

`config.yaml` must contain an `llm` section:

```yaml
llm:
  model_id:
    - "google/medgemma-27b-text-it"
  seed: 2898231092
  temperature: 0.01
  top_p: 0.95
  max_model_len: 8192
  max_num_batched_tokens: 16384
  max_tokens: 2048
  prompt_log_folder: "../data/logs/"
  prompt_log_file: "prompts.log"
```

| Field | Type | Default | Description |
|---|---|---|---|
| `model_id` | `list` | `[]` | Candidate model identifiers. `Model(index=N)` selects which entry to load. |
| `seed` | `int` | `2898231092` | Random seed for reproducible generation. |
| `temperature` | `float` | `0.01` | Sampling temperature (0 = deterministic/greedy). |
| `top_p` | `float` | `0.95` | Nucleus sampling probability mass. |
| `max_model_len` | `int` | `8192` | Maximum context length (prompt + generated tokens combined). |
| `max_num_batched_tokens` | `int` | `16384` | vLLM throughput/memory tuning parameter for concurrent requests. |
| `max_tokens` | `int` | `2048` | Maximum tokens generated per output sequence. |
| `prompt_log_folder` | `str` | `"../data/logs/"` | Directory `logPrompts()` writes to. |
| `prompt_log_file` | `str` | `"prompts.log"` | Filename `logPrompts()` writes to, within `prompt_log_folder`. |

`ModelConfig` rejects unknown keys (`extra = "forbid"`), so typos in the config file surface as validation errors rather than being silently ignored.

## Usage

```python
from Model import Model

# Loads config["llm"]["model_id"][index] into vLLM. Defaults to
# "../config/config.yaml" relative to src/, and index=0.
model = Model("../config/config.yaml", index=0)

# Start two independent conversations at once - one message per history.
model.addPrompt(role="system", message=["You are a concise medical assistant."])
model.addPrompt(role="user", message=[
    "What is microcephaly?",
    "What is renal dysplasia?",
])

# Generate the next assistant turn for every history in one batched call.
model.generate()

for history in model.getMessageHistories():
    print(history[-1])  # the assistant's response for that history

# Write every prompt/response across every history to the configured log file.
model.logPrompts()

# Clear all conversation state.
model.reset()
```

Adding a follow-up message to every ongoing history (broadcast form — a single-item list is applied to all histories at once):

```python
model.addPrompt(role="user", message=["And how is it typically treated?"])
model.generate()
```

## Running Unit Tests

`src/ModelTest.py` stubs out `vllm` and `vllm.distributed` and forces `torch.cuda.device_count()` to `0` before importing `Model`, so the suite runs without a real vLLM installation or GPU. It covers configuration loading, `addPrompt()`'s three cases, `generate()`'s prompt construction (including system-message folding), output handling, `logPrompts()`, and safe cleanup in `__del__`.

Run from the `src/` directory or the project root:

```bash
pytest src/ModelTest.py -v
```

## License

Distributed under the GNU Affero General Public License v3.0 (AGPL-3.0). See [`LICENSE`](./LICENSE) for the full text.
