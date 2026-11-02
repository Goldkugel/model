# Source

The source code of the `model` package, in `model/` (installed from this folder, see `pyproject.toml`).

| File in `model/` | Content |
|---|---|
| `__init__.py` | Package exports: `Model`, `ModelConfig`, `messageRoleElement`, `messageTextElement`, `messageReasoningElement`, `messageTruncatedElement`, `gpu`, `cpu` |
| `Model.py` | `Model` class: loads a model into vLLM (configured from a YAML file, a dict or a `ModelConfig`), manages the conversation histories, fills prompt templates, generates answers in chunks (skipping prompts longer than the context, splitting off the reasoning of thinking models, logging progress), writes and reads checkpoints, writes the run metadata, parses JSON answers and frees GPU memory |
| `ModelConfig.py` | `ModelConfig`, the pydantic schema of the `llm` config section |
| `ModelUtils.py` | Role, device and dictionary-key constants, `writePrompt()` for the prompt log, atomic writers for checkpoints (`writeCheckpointCSV()`) and JSON (`writeJSON()`), the locked run metadata update (`updateRunRecord()`), and helpers for the run metadata (package versions, model revision, file hashes) |
| `diagnostic.py` | GPU diagnostic, run with `python -m model.diagnostic [index]` (see [`scripts/`](../scripts/README.md)) |

## Import side effects

Importing `model` sets the following environment variables, so that vLLM enumerates the GPUs in a fixed order:

- `CUDA_DEVICE_ORDER=PCI_BUS_ID`
- `VLLM_TARGET_DEVICE=cuda`

`CUDA_VISIBLE_DEVICES` is not changed: `Model` uses every GPU it selects (e.g. the GPUs Slurm allocated to the job), or every GPU of the machine if it is not set.

It also sets the `vllm` logger to `ERROR` to suppress vLLM's start-up output, and sets `sys.dont_write_bytecode = True`, so Python writes no `.pyc` files for the rest of the process.

## Message histories

A history is a list of messages in the chat-template format, e.g.:

```python
[
    {"role": "system",    "content": "You are a concise medical assistant."},
    {"role": "user",      "content": "What is microcephaly?"},
    {"role": "assistant", "content": "..."},
]
```

An answer of a thinking model also has a `"reasoning"` key (`messageReasoningElement`) holding its reasoning, and an answer cut off at `max_tokens` has `"truncated": True` (`messageTruncatedElement`); only `role` and `content` are sent to the model. An `assistant` message always follows a `user` message.

`Model.messageHistories` holds one such list per independent conversation. `generate()` answers every history whose last message is not from the `assistant`, except those whose prompt does not fit into `max_model_len` (listed in `Model.skippedHistories`).

## Tests

The tests are in [`tests/`](../tests/README.md); the Slurm job for the GPU diagnostic is in [`scripts/`](../scripts/README.md).
