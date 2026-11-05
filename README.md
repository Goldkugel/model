# Model

A thin wrapper around [vLLM](https://github.com/vllm-project/vllm) for prompting a local LLM in batches: manages many independent conversation histories, renders them with the model's own chat template, runs chunked batched generation that stores its answers on disk (resumable, and reusing earlier answers), and logs prompts to disk.

## Features

- **Pydantic-validated configuration** — model selection, sampling parameters and file locations are loaded from YAML and type-checked via `ModelConfig`.
- **Multi-history chat state** — track many independent conversations at once; add messages by broadcasting to all histories, one per history, or by initializing new histories.
- **Prompt templates** — load prompt files from `prompts/` and fill their `{placeholder}` fields per history.
- **Native chat templates** — histories are sent to vLLM's chat interface (`LLM.chat()`), which formats them with the model's own chat template, so any chat model supported by vLLM works.
- **Chunked generation with an answer store** — histories are generated in chunks of `chunk_size`; after every chunk the prompts and answers are stored in a JSON file (keyed by the hash of each prompt, with the settings they were generated with), so an interrupted run (e.g. a killed Slurm job) resumes where it stopped and answers are never generated twice.
- **Context-length check** — a history whose prompt does not fit into `max_model_len` is skipped (and listed in `skippedHistories`) instead of failing its whole chunk.
- **Progress logging** — after every chunk the log shows answers and tokens generated, tokens per second, the elapsed time and the estimated remaining time.
- **Run metadata** — package versions, the model's Hugging Face revision, GPUs, the full configuration, prompt file hashes and generation statistics are written to a JSON file for reproducibility.
- **Multi-GPU** — all visible GPUs are used via tensor parallelism; Mixture-of-Experts models listed in `expert_parallel_models` additionally get expert parallelism.
- **Thinking models** — the reasoning of models such as Qwen3 or DeepSeek-R1 is split off their answers and stored separately; thinking can be switched on or off for hybrid models, and answers cut off at `max_tokens` are listed in `truncatedHistories`.
- **JSON extraction** — `toJSON()` parses the JSON object or array from a model answer, with or without a Markdown code block around it.
- **Errors raise exceptions** — invalid arguments, missing files and mismatched message counts raise an exception instead of returning an error code.
- **Prompt logging** — write every message of every history to a semicolon-delimited log file.

> **CPU is not supported yet.** `Model` requires at least one CUDA GPU and raises a `RuntimeError` otherwise. The `device` config field accepts `"cpu"`, but this currently has no effect apart from a warning in the log.

## Repository Structure

```
.
├── .github/workflows/
│   └── tests.yml        # CI: runs the tests on every push (without vllm and torch)
├── config/
│   └── config.yaml      # Configuration; Model reads the "llm" section, Logger the "logger" section
├── data/
│   ├── logs/            # Log output (Logger log, prompt log and run metadata)
│   └── tmp/             # Answer stores (the checkpoints)
├── prompts/
│   └── mapping.md       # Prompt template (HPO ↔ SNOMED CT mapping)
├── src/                 # The `model` package
│   ├── __init__.py
│   ├── AnswerStore.py   # JSON store of the prompts and answers (the checkpoint)
│   ├── Model.py         # Model class (chat state, prompting, generation, cleanup)
│   ├── ModelConfig.py   # Pydantic configuration schema
│   └── ModelUtils.py    # Role/device constants, file writers and run metadata helpers
├── tests/
│   ├── conftest.py      # Stand-ins for vllm and torch, temporary log folder
│   └── ModelTest.py     # Pytest suite
├── LICENSE
└── pyproject.toml
```

## Requirements & Installation

- Python 3.10+
- A CUDA-capable GPU
- [`vllm`](https://github.com/vllm-project/vllm) 0.9 or newer, [`torch`](https://pytorch.org/), [`pydantic`](https://docs.pydantic.dev/) 2.x, [`PyYAML`](https://pyyaml.org/) and [`logger`](https://github.com/Goldkugel/logger)

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
pip install "logger @ git+https://github.com/Goldkugel/logger.git@v1.5.0" "pydantic>=2" pyyaml pytest
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
  prompt_tmp_file: "answers.json"
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
| `revision` | `dict[str, str]` | `{}` | The Hugging Face commit (or tag) per model ID, so that a run always loads the same model version; a model without an entry loads the cached or latest version. |
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
| `chunk_size` | `int` | `500` | Number of histories generated before the answers are stored. |
| `prompt_tmp_folder` | `str` | `"./data/tmp/"` | Directory of the answer store (the checkpoint). |
| `prompt_tmp_file` | `str` | `"answers.json"` | File name of the answer store, a JSON file (e.g. `answers.json`); the model name and the postfix (`setPostfix()`) are added, e.g. `answers_medgemma-4b-it.json`. |
| `prompt_log_folder` | `str` | `"./data/logs/"` | Directory `logPrompts()` writes to. |
| `prompt_log_file` | `str` | `"prompts.log"` | File name `logPrompts()` writes to; named like the checkpoint, e.g. `prompts_medgemma-4b-it.log`. |
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
    # (Both steps at once, with one dict of values per history:
    # model.addPromptsFromFile(0, concepts).)

    # Generate an answer for every history that has none yet.
    model.generate()

    # None for an answer without valid JSON, and for histories in
    # model.skippedHistories (prompt longer than max_model_len).
    for i in range(len(concepts)):
        print(model.toJSON(i))

    # Prompts and answers by turn (counted from 0); "" for an unanswered turn.
    print(model.getTurnCount(0), model.getPrompt(0, 0), model.getAnswer(0, 0))

    model.logPrompts()

    # "medgemma-4b-it" for "google/medgemma-4b-it", e.g. for file names;
    # the full ID is model.model.
    print(model.getModelName())
```

`Model.getModelCount(config)` returns the number of models in `model_id` without loading one, e.g. to run every configured model in turn:

```python
for index in range(Model.getModelCount("./config/config.yaml")):
    with Model("./config/config.yaml", index=index) as model:
        ...
```

Messages can also be added directly. A single-item list is broadcast to every history; a list with one item per history is added one-to-one; on empty histories, one new history is created per item. The role must be `"system"`, `"user"` or `"assistant"`, and `message` must be a list, even for a single message:

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

**Cut-off answers.** Every answer that stopped at `max_tokens` is listed in `model.truncatedHistories` and in the run metadata. A thinking model cut off during its reasoning gets an empty answer (`toJSON()` returns `None`); the next `generate()` call replaces it by a new answer; empty answers are not stored. With the same seed and settings the new answer is cut off again, so raise `max_tokens` first. Until then, `addPrompt()` refuses to add another turn to such a history. With DeepSeek-R1-style templates, which open the reasoning in the prompt, an output cut off before `</think>` contains no tag at all; it is recognised from the template and also stored as reasoning with an empty answer.

Cut-off answers are marked with `"truncated": True` on the message (`messageTruncatedElement`), also in the answer store. `toJSON()` then only accepts a JSON value that starts at the answer's first bracket, so a cut-off array such as `[{...}, {"label": "li` yields `None`, not its first element.

Models that use OpenAI's "harmony" format (gpt-oss) put their reasoning into a separate channel instead of `<think>` tags and are not supported yet.

### The answer store: resuming and reusing answers

`generate()` stores the prompts and answers after every chunk in the answer store, a JSON file: `prompt_tmp_file` in `prompt_tmp_folder`, with the model name and the postfix added to its name, e.g. `answers_medgemma-4b-it_fewshot.json`. Before generating, it looks up every history in the store and uses the stored answer if there is one; only the others are generated. To resume an interrupted run, build the same histories again and call `generate()`:

```python
model.setPostfix("fewshot")
model.addPromptsFromFile(0, concepts)
model.generate()   # uses the answers of an earlier run, generates the rest
```

`hasCheckpoint()` tells whether the store exists. Deleting it starts over; `setPostfix(postfix)` gives runs of the same model with other settings their own store and prompt log. The run metadata records the paths used (`files`), and every `generate()` call how many answers it generated, took from the store (`stored`) and copied to identical histories (`shared`).

**Layout.** The store holds the settings the answers depend on and the histories as a tree by turn. Every prompt (system or user message) is a node keyed by the SHA-256 of its text, with its answer and the following prompts under `next`:

```json
{
  "format": 1,
  "settings": {"model": "google/medgemma-4b-it", "revision": "...", "seed": 42, "temperature": 0.01, "...": "..."},
  "histories": {
    "<SHA-256 of the first prompt>": {
      "role": "user",
      "prompt": "Label: Microcephaly",
      "answer": {"text": "...", "reasoning": "...", "truncated": true},
      "next": {"<SHA-256 of the second prompt>": {"role": "user", "prompt": "...", "answer": {"text": "..."}}}
    }
  }
}
```

`reasoning` and `truncated` are only present for answers that have them. A stored answer is used only if the history's earlier answers are the stored ones as well, since an answer depends on them. Histories with the same prompts share their nodes: a turn several histories share (e.g. a first prompt about the same concept) is generated once, and identical histories in one `generate()` call get the same answer. Changing a prompt (also the system prompt) leads to new nodes, so its answers are generated anew while the other answers are kept.

**Settings.** The answers depend on `model_id` and the model's revision, `seed`, `temperature`, `top_p`, `max_tokens`, `max_model_len`, `dtype`, `quantization`, `enable_thinking`, `reasoning_start_tag` and `reasoning_end_tag`. A store written with other values of these is refused with a `ValueError` naming them (an unknown revision, e.g. of a model missing from the local cache, counts as no difference); the other settings (folders, `chunk_size`, GPU memory, batching) may change.

**Restoring histories.** A history is put back together exactly as it was generated: its prompts come from the caller, and every answer, with its reasoning and cut-off mark, from the node of its prompt, but only while the history's earlier answers are the stored ones. The turns stored after a prompt always followed the answer stored on it (or no answer, for two prompts in a row), so answers of different histories are never combined.

**Concurrent runs.** Every write holds a lock on `<store>.lock`. If another run wrote the store in between, its answers are merged in first, so runs sharing a store (e.g. the jobs of a Slurm job array) keep each other's answers. If both runs answered the same prompt differently, the run writing last keeps its answer and the turns after it; the other run's turns after its answer are not stored.

### Errors

All methods raise an exception on failure; none returns an error code:

| Method | Raises |
|---|---|
| `Model()` | `FileNotFoundError` for a missing config file, `KeyError` if it has no `llm` section, `pydantic.ValidationError` for unknown keys or invalid values, `ValueError` for an invalid model index, `RuntimeError` if no CUDA device is visible |
| `addPrompt()` | `TypeError` if `message` is a string instead of a list; `ValueError` for an unknown role, if the number of messages is neither 1, nor the number of histories, nor used to create new histories, if a history ends with an empty answer, or if an answer (`"assistant"`) would not follow a user prompt |
| `addPromptFromFile()` | `IndexError` for an invalid prompt index, `FileNotFoundError` for a missing prompt file, `ValueError` if `prompt_folder` is empty or `amount` is missing for empty histories |
| `formatPromptFromFile()` | `IndexError` for an invalid history index, `KeyError` for a placeholder without a value, `ValueError` for a malformed template (e.g. an unescaped `{`) |
| `addPromptsFromFile()` | As `addPromptFromFile()` and `formatPromptFromFile()`; `ValueError` if `values` is empty or does not have one dict per history |
| `generate()` | `ValueError` if the answer store is not one or was written with other settings, `RuntimeError` if there is something to generate but the model was closed with `close()` |
| `getPrompt()`, `getAnswer()`, `getTurnCount()` | `IndexError` for an invalid history index or turn |
| `setPostfix()` | `ValueError` if the postfix contains a path separator |
| `toJSON()` | `IndexError` for an invalid history index; returns `None` if the history has no answer or the answer contains no JSON |

`generate()` returns the number of answers it generated (not counting stored answers and copies for identical histories).

Failures that do not stop a run are written to the log instead: a cleanup step of `close()` that fails, a model revision or GPU name that cannot be read for the run metadata (stored as `null`), and `device: "cpu"`.

### Run metadata

`Model` writes a record of every run to `metadata_folder/metadata_file` when the model is loaded, and updates it after every `generate()` call. Each update holds a lock on `<metadata_file>.lock`, so runs sharing the file do not lose each other's records; where file locks are not supported, this is logged and the update runs without one. The file is a JSON list with one entry per `Model` instance, so resumed runs are appended rather than overwritten:

| Key | Content |
|---|---|
| `run_id`, `started`, `updated` | Unique id of the `Model` instance, its start time and the time of the last update |
| `host`, `slurm_job_id`, `python` | Host name, `SLURM_JOB_ID` (if any) and Python version |
| `packages` | Installed versions of `model`, `logger`, `vllm`, `torch` and `transformers` |
| `model` | `model_id`, its index in the config and its revision: the configured one, otherwise the commit of the cached snapshot (also when huggingface_hub finds the snapshot incomplete, from the cache's `refs/main`); `null` for a local directory |
| `gpus` | Number and names of the GPUs and the CUDA version |
| `prompt_files` | SHA-256 of every configured prompt file (`null` if missing) |
| `config` | The complete `llm` configuration |
| `files` | Paths of the answer store and the prompt log |
| `generations` | One entry per `generate()` call: start and end time, duration, number of histories, answers generated, taken from the store (`stored`) and copied to identical histories (`shared`), skipped and truncated history indices, prompt and generated tokens, and the approximate number of reasoning tokens |

### Changes in 3.0

Version 3.0 replaces the CSV checkpoint by the answer store (see [The answer store](#the-answer-store-resuming-and-reusing-answers)):

- `generate()` uses stored answers and generates only the others, so resuming needs no extra call; identical histories are generated once. `exportHistoriesToFile()` and `importHistoriesFromFile()` were removed: build the histories again and call `generate()`.
- The store records the settings the answers depend on and refuses a store written with other ones. Runs sharing a store merge their answers instead of being refused.
- Checkpoints of 2.x (CSV) cannot be read; a store file that is not an answer store is refused with a `ValueError`. Use a new `prompt_tmp_file` name (e.g. `answers.json`) or delete the old checkpoints.
- The store and the prompt log are named after the model, e.g. `answers_medgemma-4b-it.json` for `prompt_tmp_file: "answers.json"`, and `setPostfix()` adds a postfix to both.
- New: `getModelName()`, `Model.getModelCount(config)`, `hasCheckpoint()`, `getTurnCount()`, `getPrompt()`, `getAnswer()` and `addPromptsFromFile()`.
- The file lock and the atomic writes come from the logger package (v1.5.0: `lockedFile()`, `writeAtomically()`, `writeJSON()`); `model.ModelUtils.lockedFile` and `writeJSON` still work. `writeCheckpointCSV()` and the checkpoint column constants were removed, and pandas is no longer needed.

### Changes in 2.1

Version 2.1 is compatible with 2.0: existing code and configuration files keep working, except calls that used to produce wrong histories silently and now raise an exception (see below).

- `Model(config=...)` also accepts a dict or a `ModelConfig` instead of a file path.
- Thinking models: the reasoning is stored apart from the answer, and answers cut off at `max_tokens` are listed in `truncatedHistories` (see [Thinking models](#thinking-models)). New, optional config options: `enable_thinking`, `reasoning_start_tag`, `reasoning_end_tag`.
- Importing `model` no longer overwrites `CUDA_VISIBLE_DEVICES`, so the GPUs selected by Slurm or by hand are used.
- `vllm` 0.9 or newer is required.
- Input that used to produce wrong histories now raises: `addPrompt()` with a string instead of a list (`TypeError`), with an unknown role such as `"model"`, or after an empty answer (`ValueError`); `generate()` after `close()` (`RuntimeError`).
- `generate()` also replaces empty answers (thinking models cut off during their reasoning), not only after resuming.
- `importHistoriesFromFile()` keeps empty prompts and rows without text, so every history keeps its index.
- Concurrent runs no longer overwrite each other: a checkpoint is only written by the run that created or imported it, it records the model and is not resumed with another one, and updates of the run metadata are locked. An answer must follow a user prompt.
- Cut-off answers are marked, so `toJSON()` returns no fragments of cut-off JSON, and a DeepSeek-R1-style answer cut off during its reasoning is no longer taken for the answer.
- Failures that were swallowed silently are logged (see [Errors](#errors)), and a model whose run metadata cannot be written is released before the error is raised.
- Repository layout: the package lives in `src/` and the tests in `tests/`; the GPU diagnostic `model.diagnostic` and its Slurm job were removed. Import paths (`from model import Model`, `model.ModelUtils`, ...) are unchanged, but an editable install (`pip install -e .`) must be reinstalled once.

### Migrating from 1.x

- Methods that returned `-1` now raise an exception (see [Errors](#errors)); replace checks such as `if model.importHistoriesFromFile() > 0:` by `try`/`except FileNotFoundError`.
- `addPrompt()` with a mismatched number of messages raises `ValueError` instead of logging and adding nothing.
- `toJSON()` raises `IndexError` for an invalid index instead of returning `None`.
- `generate()` returns the number of generated answers instead of `None`, and skips prompts longer than `max_model_len` (see `skippedHistories`).
- New config options: `enforce_eager`, `gpu_memory_utilization`, `dtype`, `quantization`, `enable_prefix_caching`, `metadata_folder`, `metadata_file`. All are optional: `enforce_eager` defaults to the formerly hard-coded `true`, the others to vLLM's usual defaults.

## Running Unit Tests

The test suite needs neither a GPU nor a downloaded model, nor `vllm` and `torch` installed: tests that touch the engine replace `torch.cuda` and the vLLM classes via `monkeypatch`, and [`tests/conftest.py`](./tests/conftest.py) registers stand-ins for `vllm` and `torch` if they are not installed. It also points the logger at a temporary folder, so the tests write nothing into the repository and can be run from any folder:

```bash
pytest -v
```

`pytest` collects every `*Test.py` file in [`tests/`](./tests/README.md) (configured in `pyproject.toml`). Besides the methods of `Model`, the tests cover the answer store (resuming an interrupted run, reusing stored answers, refusing other settings, runs sharing a store), the GPU and logger set-up of the constructor and the helpers in `ModelUtils.py`. The same tests run on GitHub Actions for every push and pull request, on Python 3.10, 3.11 and 3.14 ([`.github/workflows/tests.yml`](./.github/workflows/tests.yml)).

## License

Distributed under the GNU Affero General Public License v3.0 (AGPL-3.0). See [`LICENSE`](./LICENSE) for the full text.
