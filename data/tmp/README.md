# Checkpoints

`Model.generate()` writes all conversation histories to a checkpoint file in this folder after every chunk of `chunk_size` histories (file name: `llm.prompt_tmp_file`, folder: `llm.prompt_tmp_folder` in [`config/config.yaml`](../../config/config.yaml)). If a run is interrupted, e.g. by a Slurm time limit, at most one chunk of work is lost. `*.tmp` files are ignored by git.

A checkpoint belongs to one run: `Model` only overwrites a checkpoint it created or imported itself, and raises `RuntimeError` otherwise. Runs at the same time (several models, or the jobs of a Slurm job array) therefore each need their own `prompt_tmp_file`.

## Resuming a run

Load the checkpoint and generate again; histories that already end with an answer are skipped. The column names depend on `prompt_files`, so resume with the same `prompt_files` the checkpoint was written with, and with the same model (a checkpoint of another model is refused):

```python
model.importHistoriesFromFile()
model.generate()
```

## File format

A semicolon-delimited CSV with every value quoted and one row per history:

| Column | Content |
|---|---|
| `_model` | Model that wrote the checkpoint |
| `system` | System prompt, if the history has one |
| `<prompt>` | Prompt of turn *n*, named after the stem of `prompt_files[n]` (e.g. `mapping` for `mapping.md`), or `prompt<n+1>` beyond the configured files |
| `<prompt>_answer` | Model answer to that prompt; empty if not generated yet, or if a thinking model was cut off during its reasoning (generated again on resume) |
| `<prompt>_reasoning` | Reasoning of a thinking model's answer; the column only exists if some answer of that turn has reasoning |
| `<prompt>_truncated` | `true` for an answer cut off at `max_tokens`; the column only exists if some answer of that turn was cut off |

```
"_model";"mapping";"mapping_answer"
"google/medgemma-4b-it";"You are a senior medical informatician ...";"{""mapping_type"": ""NO_MATCH"", ""confidence"": 7}"
```

The checkpoint is written to `<file>.tmp` first and then moved over the previous checkpoint in one step, so an interrupted write never leaves a corrupted file. A leftover `<file>.tmp` is only a failed write and can be deleted.
