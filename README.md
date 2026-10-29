# Model

A thin wrapper around [vLLM](https://github.com/vllm-project/vllm) for prompting a local LLM in batches: manages many independent conversation histories, renders them with the model's own chat template, runs chunked batched generation with on-disk checkpoints, and logs prompts to disk.

## Features

- **Pydantic-validated configuration** — model selection, sampling parameters and file locations are loaded from YAML and type-checked via `ModelConfig`.
- **Multi-history chat state** — track many independent conversations at once; add messages by broadcasting to all histories, one per history, or by initializing new histories.
- **Prompt templates** — load prompt files from `prompts/` and fill their `{placeholder}` fields per history.
- **Native chat templates** — histories are formatted with the tokenizer's own chat template (`apply_chat_template`), so any chat model supported by vLLM works.
- **Chunked generation with checkpoints** — histories are generated in chunks of `chunk_size`; after every chunk all histories are written atomically to a checkpoint file, from which an interrupted run (e.g. a killed Slurm job) can resume.
- **Multi-GPU** — all visible GPUs are used via tensor parallelism; Mixture-of-Experts models listed in `expert_parallel_models` additionally get expert parallelism.
- **JSON extraction** — `toJSON()` parses the JSON object or array from a model answer, with or without a Markdown code block around it.
- **Prompt logging** — write every message of every history to a semicolon-delimited log file.

> **CPU is not supported yet.** `Model` requires at least one CUDA GPU and raises a `RuntimeError` otherwise. The `device` config field accepts `"cpu"`, but this currently has no effect.

## Repository Structure

```
.
├── .github/workflows/
│   └── tests.yml        # CI: runs the tests on every push (without vllm and torch)
├── config/
│   └── config.yaml      # Configuration; Model reads the "llm" section, Logger the "logger" section
├── data/
│   ├── logs/            # Log output (Logger log and prompt log)
│   ├── slurmlogs/       # Output of Slurm jobs
│   └── tmp/             # Generation checkpoints
├── prompts/
│   └── mapping.md       # Prompt template (HPO ↔ SNOMED CT mapping)
├── src/                 # Installed as the `model` package
│   ├── __init__.py
│   ├── Model.py         # Model class (chat state, prompting, generation, checkpoints, cleanup)
│   ├── ModelConfig.py   # Pydantic configuration schema
│   ├── ModelUtils.py    # Role/device constants, prompt log and checkpoint writers
│   ├── ModelTest.py     # Pytest suite (+ a manual GPU diagnostic, see below)
│   └── runTest.sh       # Slurm job running the GPU diagnostic
├── conftest.py          # Registers stand-ins for vllm and torch when they are not installed
├── LICENSE
└── pyproject.toml
```

## Requirements & Installation

- Python 3.10+
- A CUDA-capable GPU
- [`vllm`](https://github.com/vllm-project/vllm), [`torch`](https://pytorch.org/), [`pandas`](https://pandas.pydata.org/), [`pydantic`](https://docs.pydantic.dev/) 2.x, [`PyYAML`](https://pyyaml.org/) and [`logger`](https://github.com/Goldkugel/logger)

All dependencies are declared in `pyproject.toml` and installed with the package:

```bash
pip install git+https://github.com/Goldkugel/model.git
```

To run the tests, install the `test` extra from a local checkout:

```bash
pip install -e ".[test]"
```

The tests do not need `vllm` and `torch`. To run them on a machine without a GPU, skip both by installing the package without dependencies and the remaining dependencies by hand (as the CI does):

```bash
pip install --no-deps -e .
pip install "logger @ git+https://github.com/Goldkugel/logger.git@v1.1.0" pandas "pydantic>=2" pyyaml pytest
```

## Configuration

`config.yaml` must contain an `llm` section (and a `logger` section for the [`logger`](https://github.com/Goldkugel/logger) package):

```yaml
llm:
  model_id:
    - "google/medgemma-4b-it"
  device: "cuda"
  expert_parallel_models:
    - "Qwen3-30B"
  seed: 2898231092
  temperature: 0.01
  top_p: 0.95
  max_model_len: 8192
  max_num_batched_tokens: 16384
  max_tokens: 2048
  chunk_size: 500
  prompt_tmp_folder: "./data/tmp/"
  prompt_tmp_file: "prompts_raw.tmp"
  prompt_log_folder: "./data/logs/"
  prompt_log_file: "prompts.log"
  prompt_folder: "./prompts/"
  prompt_files:
    - "mapping.md"
```

| Field | Type | Default | Description |
|---|---|---|---|
| `model_id` | `list[str]` | `[]` | HuggingFace or local model identifiers. `Model(index=N)` selects which entry to load. |
| `seed` | `int` | `2898231092` | Seed for vLLM and every sampling request, for reproducible generation. |
| `temperature` | `float` | `0.01` | Sampling temperature (`0.0`–`2.0`; 0 = greedy). |
| `top_p` | `float` | `0.95` | Nucleus sampling probability mass (`>0`–`1.0`). |
| `max_model_len` | `int` | `8192` | Maximum context length (prompt + generated tokens). |
| `max_num_batched_tokens` | `int` | `16384` | Maximum number of tokens vLLM batches together per step. |
| `max_tokens` | `int` | `2048` | Maximum number of generated tokens per answer. |
| `chunk_size` | `int` | `500` | Number of histories generated before a checkpoint is written. |
| `prompt_tmp_folder` | `str` | `"./data/tmp/"` | Directory of the checkpoint file. |
| `prompt_tmp_file` | `str` | `"raw.tmp"` | Checkpoint file name. |
| `prompt_log_folder` | `str` | `"./data/logs/"` | Directory `logPrompts()` writes to. |
| `prompt_log_file` | `str` | `"prompts.log"` | File name `logPrompts()` writes to. |
| `prompt_folder` | `str` | `"./prompts/"` | Directory of the prompt template files. |
| `prompt_files` | `list[str]` | `[]` | Prompt template files, in prompting order. |
| `expert_parallel_models` | `list[str]` | `["Qwen3-30B"]` | Substrings of `model_id` that enable vLLM expert parallelism (MoE models). |
| `device` | `"cuda"` \| `"cpu"` | `"cuda"` | Only `"cuda"` is supported; see the note on CPU support above. |

`ModelConfig` rejects unknown keys (`extra="forbid"`), so typos in the config file surface as validation errors rather than being silently ignored.

See [`config/README.md`](./config/README.md) for the other sections of the shared config file, and [`data/README.md`](./data/README.md) for the files written to `data/`.

## Usage

```python
from model import Model

# Loads config["llm"]["model_id"][index] into vLLM on all visible GPUs.
with Model("./config/config.yaml", index=0) as model:

    # Create one history per item from the first prompt template ...
    concepts = [
        # One dict per history, with a value for every {placeholder}
        # of the template (hpo_label, hpo_synonyms, ..., sct_children).
        {"hpo_label": "Microcephaly", "sct_term": "Microcephaly"},
        {"hpo_label": "Renal dysplasia", "sct_term": "Renal dysplasia"},
    ]
    model.addPromptFromFile(index=0, amount=len(concepts))

    # ... and fill each history's {placeholders}.
    for i, values in enumerate(concepts):
        model.formatPromptFromFile(i, values)

    # Generate an answer for every history that has none yet.
    model.generate()

    for i in range(len(concepts)):
        print(model.toJSON(i))

    model.logPrompts()
```

Messages can also be added directly. A single-item list is broadcast to every history; a list with one item per history is added one-to-one; on empty histories, one new history is created per item:

```python
model.addPrompt(role="system", message=["You are a concise medical assistant."])
model.addPrompt(role="user", message=["What is microcephaly?"])
model.generate()
```

### Resuming an interrupted run

`generate()` writes all histories to `prompt_tmp_folder/prompt_tmp_file` after every chunk. To resume, load that checkpoint and call `generate()` again; histories that already end with an answer are skipped:

```python
model.importHistoriesFromFile()
model.generate()
```

The checkpoint is a semicolon-delimited CSV with one row per history: a `system` column for the system prompt, and one prompt/answer column pair per turn, named after the prompt file (e.g. `mapping` and `mapping_answer`), or `promptN` / `promptN_answer` beyond the configured files.

## Running Unit Tests

The test suite needs neither a GPU nor a downloaded model, nor `vllm` and `torch` installed: tests that touch the engine replace `torch.cuda` and the vLLM classes via `monkeypatch`, and [`conftest.py`](./conftest.py) registers stand-ins for `vllm` and `torch` if they are not installed. Run the tests from the repository root, since the logger reads `./config/config.yaml`:

```bash
pytest src/ModelTest.py -v
```

The same tests run on GitHub Actions for every push and pull request, on Python 3.10, 3.11 and 3.14 ([`.github/workflows/tests.yml`](./.github/workflows/tests.yml)).

`src/ModelTest.py` can also be run directly as a GPU diagnostic that loads the configured model; `src/runTest.sh` submits it as a Slurm job, writing its output to `data/slurmlogs/`:

```bash
cd src && sbatch runTest.sh
```

## License

Distributed under the GNU Affero General Public License v3.0 (AGPL-3.0). See [`LICENSE`](./LICENSE) for the full text.
