# Model

A thin wrapper around [vLLM](https://github.com/vllm-project/vllm) for prompting a local LLM in batches: manages many independent conversation histories, renders them with the model's own chat template, runs chunked batched generation with on-disk checkpoints, and logs prompts to disk.

## Features

- **Pydantic-validated configuration** — model selection, sampling parameters and file locations are loaded from YAML and type-checked via `ModelConfig`.
- **Multi-history chat state** — track many independent conversations at once; add messages by broadcasting to all histories, one per history, or by initializing new histories.
- **Prompt templates** — load prompt files from `prompts/` and fill their `{placeholder}` fields per history.
- **Native chat templates** — histories are sent to vLLM's chat interface (`LLM.chat()`), which formats them with the model's own chat template, so any chat model supported by vLLM works.
- **Chunked generation with checkpoints** — histories are generated in chunks of `chunk_size`; after every chunk all histories are written atomically to a checkpoint file, from which an interrupted run (e.g. a killed Slurm job) can resume.
- **Context-length check** — a history whose prompt does not fit into `max_model_len` is skipped (and listed in `skippedHistories`) instead of failing its whole chunk.
- **Progress logging** — after every chunk the log shows answers and tokens generated, tokens per second, the elapsed time and the estimated remaining time.
- **Run metadata** — package versions, the model's Hugging Face revision, GPUs, the full configuration, prompt file hashes and generation statistics are written to a JSON file for reproducibility.
- **Multi-GPU** — all visible GPUs are used via tensor parallelism; Mixture-of-Experts models listed in `expert_parallel_models` additionally get expert parallelism.
- **Thinking models** — the reasoning of models such as Qwen3 or DeepSeek-R1 is split off their answers and stored separately; thinking can be switched on or off for hybrid models, and answers cut off at `max_tokens` are listed in `truncatedHistories`.
- **JSON extraction** — `toJSON()` parses the JSON object or array from a model answer, with or without a Markdown code block around it.
- **Errors raise exceptions** — invalid arguments, missing files and mismatched message counts raise an exception instead of returning an error code.
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
│   ├── logs/            # Log output (Logger log, prompt log and run metadata)
│   ├── slurmlogs/       # Output of Slurm jobs
│   └── tmp/             # Generation checkpoints
├── prompts/
│   └── mapping.md       # Prompt template (HPO ↔ SNOMED CT mapping)
├── src/                 # Installed as the `model` package
│   ├── __init__.py
│   ├── Model.py         # Model class (chat state, prompting, generation, checkpoints, cleanup)
│   ├── ModelConfig.py   # Pydantic configuration schema
│   ├── ModelUtils.py    # Role/device constants, file writers and run metadata helpers
│   ├── ModelTest.py     # Pytest suite (+ a manual GPU diagnostic, see below)
│   └── runTest.sh       # Slurm job running the GPU diagnostic
├── conftest.py          # Registers stand-ins for vllm and torch when they are not installed
├── LICENSE
└── pyproject.toml
```

## Requirements & Installation

- Python 3.10+
- A CUDA-capable GPU
- [`vllm`](https://github.com/vllm-project/vllm) 0.9 or newer, [`torch`](https://pytorch.org/), [`pandas`](https://pandas.pydata.org/), [`pydantic`](https://docs.pydantic.dev/) 2.x, [`PyYAML`](https://pyyaml.org/) and [`logger`](https://github.com/Goldkugel/logger)

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
  enforce_eager: true
  gpu_memory_utilization: 0.9
  dtype: "auto"
  quantization: null
  enable_prefix_caching: true
  enable_thinking: null
  reasoning_start_tag: "<think>"
  reasoning_end_tag: "</think>"
  max_tokens: 2048
  chunk_size: 500
  prompt_tmp_folder: "./data/tmp/"
  prompt_tmp_file: "prompts_raw.tmp"
  prompt_log_folder: "./data/logs/"
  prompt_log_file: "prompts.log"
  metadata_folder: "./data/logs/"
  metadata_file: "run_metadata.json"
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
| `enforce_eager` | `bool` | `true` | Skip CUDA graph capture: faster start-up and less GPU memory; `false` generates faster. |
| `gpu_memory_utilization` | `float` | `0.9` | Fraction of each GPU's memory vLLM may use (`>0`–`1.0`). |
| `dtype` | `str` | `"auto"` | Data type of weights and activations (`"auto"`, `"bfloat16"`, `"float16"`, ...). |
| `quantization` | `str` \| `null` | `null` | vLLM quantization method (e.g. `"awq"`, `"gptq"`, `"fp8"`); `null` loads the weights as stored. |
| `enable_prefix_caching` | `bool` | `true` | Reuse the KV cache of shared prompt prefixes, e.g. a common prompt template. |
| `enable_thinking` | `bool` \| `null` | `null` | Passed to the chat template as `enable_thinking` to switch thinking on or off for hybrid models such as Qwen3; `null` keeps the template's default. See [Thinking models](#thinking-models). |
| `reasoning_start_tag` | `str` | `"<think>"` | Marker that opens a thinking model's reasoning. |
| `reasoning_end_tag` | `str` \| `null` | `"</think>"` | Marker that closes the reasoning; the answer is the text after its first occurrence. `null` stores the whole output as the answer. |
| `max_tokens` | `int` | `2048` | Maximum number of generated tokens per answer, including a thinking model's reasoning. |
| `chunk_size` | `int` | `500` | Number of histories generated before a checkpoint is written. |
| `prompt_tmp_folder` | `str` | `"./data/tmp/"` | Directory of the checkpoint file. |
| `prompt_tmp_file` | `str` | `"raw.tmp"` | Checkpoint file name. |
| `prompt_log_folder` | `str` | `"./data/logs/"` | Directory `logPrompts()` writes to. |
| `prompt_log_file` | `str` | `"prompts.log"` | File name `logPrompts()` writes to. |
| `metadata_folder` | `str` | `"./data/logs/"` | Directory of the run metadata file. |
| `metadata_file` | `str` | `"run_metadata.json"` | Run metadata file name (see [Run metadata](#run-metadata)). |
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

    # None for an answer without valid JSON, and for histories in
    # model.skippedHistories (prompt longer than max_model_len).
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

### Configuration sources

Besides a YAML path, `Model` accepts a dict or a `ModelConfig`:

```python
from model import Model, ModelConfig

# From a YAML file (default: "./config/config.yaml" relative to the
# working directory). A pathlib.Path works as well.
model = Model("./config/config.yaml", index=0)

# From a dict: either the whole configuration (with "llm" and optionally
# "logger" keys) or just the llm settings.
model = Model({"llm": {"model_id": ["google/medgemma-4b-it"]}, "logger": {"folder": "./data/logs/"}})
model = Model({"model_id": ["google/medgemma-4b-it"]})

# From a ModelConfig (a copy is stored).
model = Model(ModelConfig(model_id=["google/medgemma-4b-it"]))
```

If the logger singleton has not been configured yet, it is configured from the same source: the `logger` section of the YAML file, or the `logger` key of a full configuration dict. Otherwise (llm settings only, or a `ModelConfig`) configure it first with `Logger("path/to/config.yaml")`, or it falls back to `./config/config.yaml`.

### Thinking models

Thinking (reasoning) models such as Qwen3 or DeepSeek-R1 write their reasoning between `<think>` and `</think>` before the answer. `generate()` splits it off:

- The **answer** (the text after the first `reasoning_end_tag`) becomes the message text, so `toJSON()` never picks up JSON drafts from the reasoning.
- The **reasoning** is stored on the same message under the `"reasoning"` key (`messageReasoningElement`). It is not sent back to the model in later turns: only `role` and `content` are passed to vLLM and the chat template.
- Templates that put `<think>` into the prompt (DeepSeek-R1) are handled; their output only contains `</think>`. Models without these tags are unaffected.

```python
model.generate()
answer = model.getMessageHistories()[0][-1]
answer["content"]     # the answer
answer["reasoning"]   # the reasoning (key missing for non-thinking answers)
```

**Settings.** For hybrid models such as Qwen3, `enable_thinking: false` switches thinking off and `true` switches it on; the setting is passed both to `LLM.chat()` and to the context-length check. Reasoning counts towards `max_tokens`, so thinking models need a much larger value (e.g. 8192–32768) and a `max_model_len` that fits prompt plus output. Most model cards also recommend `temperature: 0.6` and `top_p: 0.95` instead of near-greedy decoding, which tends to make the reasoning repeat itself.

**Cut-off answers.** Every answer that stopped at `max_tokens` is listed in `model.truncatedHistories` and in the run metadata. A thinking model cut off during its reasoning gets an empty answer (`toJSON()` returns `None`); it is regenerated when the run is resumed from the checkpoint. With DeepSeek-R1-style templates, an output cut off before `</think>` contains no tag at all, so it is stored as the answer; it is still listed in `truncatedHistories`.

Models that use OpenAI's "harmony" format (gpt-oss) put their reasoning into a separate channel instead of `<think>` tags and are not supported yet.

### Resuming an interrupted run

`generate()` writes all histories to `prompt_tmp_folder/prompt_tmp_file` after every chunk. To resume, load that checkpoint and call `generate()` again; histories that already end with an answer are skipped. `importHistoriesFromFile()` raises `FileNotFoundError` if there is no checkpoint, i.e. no earlier run to resume:

```python
try:
    model.importHistoriesFromFile()
except FileNotFoundError:
    model.addPromptFromFile(index=0, amount=len(concepts))
    # ... format the prompts as above
model.generate()
```

The checkpoint is a semicolon-delimited CSV with one row per history: a `system` column for the system prompt, and one prompt/answer column pair per turn, named after the prompt file (e.g. `mapping` and `mapping_answer`), or `promptN` / `promptN_answer` beyond the configured files. The reasoning of thinking models is stored in an extra `mapping_reasoning` column (only present if some answer of that turn has reasoning) and restored on import. Since the column names are derived from `prompt_files`, resume with the same `prompt_files` the checkpoint was written with.

### Errors

All methods raise an exception on failure; none returns an error code:

| Method | Raises |
|---|---|
| `Model()` | `FileNotFoundError` for a missing config file, `KeyError` if it has no `llm` section, `pydantic.ValidationError` for unknown keys or invalid values, `ValueError` for an invalid model index, `RuntimeError` if no CUDA device is visible |
| `addPrompt()` | `ValueError` if the number of messages is neither 1, nor the number of histories, nor used to create new histories |
| `addPromptFromFile()` | `IndexError` for an invalid prompt index, `FileNotFoundError` for a missing prompt file, `ValueError` if `prompt_folder` is empty or `amount` is missing for empty histories |
| `formatPromptFromFile()` | `IndexError` for an invalid history index, `KeyError` for a placeholder without a value, `ValueError` for a malformed template (e.g. an unescaped `{`) |
| `exportHistoriesToFile()` | `ValueError` if there are no histories |
| `importHistoriesFromFile()` | `FileNotFoundError` if there is no checkpoint, `ValueError` if it is empty |
| `toJSON()` | `IndexError` for an invalid history index; returns `None` if the history has no answer or the answer contains no JSON |

`generate()` returns the number of answers it generated.

### Run metadata

`Model` writes a record of every run to `metadata_folder/metadata_file` when the model is loaded, and updates it after every `generate()` call. The file is a JSON list with one entry per `Model` instance, so resumed runs are appended rather than overwritten:

| Key | Content |
|---|---|
| `run_id`, `started`, `updated` | Unique id of the `Model` instance, its start time and the time of the last update |
| `host`, `slurm_job_id`, `python` | Host name, `SLURM_JOB_ID` (if any) and Python version |
| `packages` | Installed versions of `model`, `logger`, `vllm`, `torch` and `transformers` |
| `model` | `model_id`, its index in the config and the Hugging Face commit hash of the cached snapshot (`null` for a local directory) |
| `gpus` | Number and names of the GPUs and the CUDA version |
| `prompt_files` | SHA-256 of every configured prompt file (`null` if missing) |
| `config` | The complete `llm` configuration |
| `generations` | One entry per `generate()` call: start and end time, duration, number of histories, answers, skipped and truncated history indices, prompt and generated tokens, and the approximate number of reasoning tokens |

### Changes in 2.1

Version 2.1 is compatible with 2.0; existing code and configuration files keep working.

- `Model(config=...)` also accepts a dict or a `ModelConfig` instead of a file path.
- Thinking models: the reasoning is stored apart from the answer, and answers cut off at `max_tokens` are listed in `truncatedHistories` (see [Thinking models](#thinking-models)). New, optional config options: `enable_thinking`, `reasoning_start_tag`, `reasoning_end_tag`.
- Importing `model` no longer overwrites `CUDA_VISIBLE_DEVICES`, so the GPUs selected by Slurm or by hand are used.
- `vllm` 0.9 or newer is required.

### Migrating from 1.x

- Methods that returned `-1` now raise an exception (see [Errors](#errors)); replace checks such as `if model.importHistoriesFromFile() > 0:` by `try`/`except FileNotFoundError`.
- `addPrompt()` with a mismatched number of messages raises `ValueError` instead of logging and adding nothing.
- `toJSON()` raises `IndexError` for an invalid index instead of returning `None`.
- `generate()` returns the number of generated answers instead of `None`, and skips prompts longer than `max_model_len` (see `skippedHistories`).
- New config options: `enforce_eager`, `gpu_memory_utilization`, `dtype`, `quantization`, `enable_prefix_caching`, `metadata_folder`, `metadata_file`. All are optional: `enforce_eager` defaults to the formerly hard-coded `true`, the others to vLLM's usual defaults.

## Running Unit Tests

The test suite needs neither a GPU nor a downloaded model, nor `vllm` and `torch` installed: tests that touch the engine replace `torch.cuda` and the vLLM classes via `monkeypatch`, and [`conftest.py`](./conftest.py) registers stand-ins for `vllm` and `torch` if they are not installed. Run the tests from the repository root, since the logger reads `./config/config.yaml`:

```bash
pytest -v
```

`pytest` collects every `*Test.py` file in `src/` (configured in `pyproject.toml`). Besides the methods of `Model`, the tests cover resuming an interrupted run from its checkpoint, the GPU and logger set-up of the constructor, the helpers in `ModelUtils.py`, and the GPU diagnostic below, run with vLLM replaced. The same tests run on GitHub Actions for every push and pull request, on Python 3.10, 3.11 and 3.14 ([`.github/workflows/tests.yml`](./.github/workflows/tests.yml)).

`src/ModelTest.py` can also be run directly as a GPU diagnostic: it loads a configured model and generates answers to two short prompts, checking that none is skipped or cut off and that `toJSON()` parses a JSON answer. Its checkpoint and run metadata go to a temporary folder, so the checkpoint of a real run is not overwritten. `src/runTest.sh` submits it as a Slurm job, writing its output to `data/slurmlogs/`; pass the index of the model in `model_id` to test another model than the first, e.g. a thinking model:

```bash
cd src && sbatch runTest.sh 1
```

## License

Distributed under the GNU Affero General Public License v3.0 (AGPL-3.0). See [`LICENSE`](./LICENSE) for the full text.
