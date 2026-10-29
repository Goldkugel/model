# Logs

Log files written while running `Model`. `*.log` files are ignored by git.

| File | Written by | Config |
|---|---|---|
| `output.log` | [`logger`](https://github.com/Goldkugel/logger), throughout the run | `logger.folder`, `logger.file_name` |
| `prompts.log` | `Model.logPrompts()` | `llm.prompt_log_folder`, `llm.prompt_log_file` |

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
