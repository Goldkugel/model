# Logs

Log files written while running `Model`. `*.log` and `*.json` files are ignored by git.

| File | Written by | Config |
|---|---|---|
| `output.log` | [`logger`](https://github.com/Goldkugel/logger), throughout the run | `logger.folder`, `logger.file_name` |
| `prompts.log` | `Model.logPrompts()` | `llm.prompt_log_folder`, `llm.prompt_log_file` |
| `run_metadata.json` | `Model`, when the model is loaded and after every `generate()` call | `llm.metadata_folder`, `llm.metadata_file` |

## `prompts.log`

A semicolon-delimited file with one row per message of every conversation history. It is overwritten on every call of `logPrompts()`.

```
history;prompt;role;length;text
0;0;user;5012;You are a senior medical informatician ...
0;1;assistant;54;{"mapping_type": "EXACT_MATCH", "confidence": 9}
```

| Column | Content |
|---|---|
| `history` | Index of the conversation history |
| `prompt` | Index of the message within that history |
| `role` | `system`, `user` or `assistant` |
| `length` | Length of the original message in characters |
| `text` | The message, with line breaks replaced by spaces and `;` by `,` |

Because the text is altered to keep one message per row, this file is meant for reading only. To save and restore histories, use the checkpoint in [`data/tmp/`](../tmp/README.md).

## `run_metadata.json`

A JSON list with one entry per `Model` instance, recording what is needed to reproduce a run: package versions, the model and its Hugging Face commit hash, GPUs, the full `llm` configuration, the SHA-256 of every prompt file and statistics of every `generate()` call. Entries of earlier runs are kept. See [Run metadata](../../README.md#run-metadata) in the main README for every key.

```json
[
  {
    "run_id": "3f2c9b...",
    "started": "2026-10-29T10:15:02+01:00",
    "model": {"id": "google/medgemma-4b-it", "index": 0, "revision": "a1b2c3..."},
    "gpus": {"count": 2, "names": ["NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-80GB"], "cuda": "12.8"},
    "generations": [
      {"histories": 1200, "answers": 1198, "skipped": [17, 803], "prompt_tokens": 5904211, "generated_tokens": 61822, "seconds": 812.4}
    ]
  }
]
```

Some keys are left out in this example.
