# Source

The source code of the `model` package (installed from this folder, see `pyproject.toml`).

| File | Content |
|---|---|
| `__init__.py` | Package exports: `Model`, `ModelConfig`, `messageRoleElement`, `messageTextElement`, `gpu`, `cpu` |
| `Model.py` | `Model` class: loads a model into vLLM, manages the conversation histories, fills prompt templates, generates answers in chunks (skipping prompts longer than the context, logging progress), writes and reads checkpoints, writes the run metadata, parses JSON answers and frees GPU memory |
| `ModelConfig.py` | `ModelConfig`, the pydantic schema of the `llm` config section |
| `ModelUtils.py` | Role, device and dictionary-key constants, `writePrompt()` for the prompt log, atomic writers for checkpoints (`writeCheckpointCSV()`) and JSON (`writeJSON()`), and helpers for the run metadata (package versions, model revision, file hashes) |
| `ModelTest.py` | Pytest suite, plus a GPU diagnostic when run as a script |
| `runTest.sh` | Slurm job running the GPU diagnostic |

## Import side effects

Importing `model` sets the following environment variables, so that vLLM enumerates the GPUs in a fixed order:

- `CUDA_DEVICE_ORDER=PCI_BUS_ID`
- `VLLM_TARGET_DEVICE=cuda`

`CUDA_VISIBLE_DEVICES` is not changed: `Model` uses every GPU it selects (e.g. the GPUs Slurm allocated to the job), or every GPU of the machine if it is not set.

It also sets the `vllm` logger to `ERROR` to suppress vLLM's start-up output.

## Message histories

A history is a list of messages in the chat-template format, e.g.:

```python
[
    {"role": "system",    "content": "You are a concise medical assistant."},
    {"role": "user",      "content": "What is microcephaly?"},
    {"role": "assistant", "content": "..."},
]
```

`Model.messageHistories` holds one such list per independent conversation. `generate()` answers every history whose last message is not from the `assistant`, except those whose prompt does not fit into `max_model_len` (listed in `Model.skippedHistories`).

## Tests

```bash
pytest src/ModelTest.py -v
```

Run the tests from the repository root. They need neither a GPU nor a downloaded model, and `vllm` and `torch` need not be installed (see `conftest.py` in the repository root). To check the GPU setup on the cluster, submit the diagnostic from this folder (it changes to the project root itself):

```bash
sbatch runTest.sh
```

Its output is written to `data/slurmlogs/`.
