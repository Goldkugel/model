# Checkpoints

`Model.generate()` writes all conversation histories to a checkpoint file in this folder after every chunk of `chunk_size` histories (file name: `llm.prompt_tmp_file`, folder: `llm.prompt_tmp_folder` in [`config/config.yaml`](../../config/config.yaml)). If a run is interrupted, e.g. by a Slurm time limit, at most one chunk of work is lost. `*.tmp` files are ignored by git.

## Resuming a run

Load the checkpoint and generate again; histories that already end with an answer are skipped. The column names depend on `prompt_files`, so resume with the same `prompt_files` the checkpoint was written with:

```python
model.importHistoriesFromFile()
model.generate()
```

## File format

A semicolon-delimited CSV with every value quoted and one row per history:

| Column | Content |
|---|---|
| `system` | System prompt, if the history has one |
| `<prompt>` | Prompt of turn *n*, named after the stem of `prompt_files[n]` (e.g. `mapping` for `mapping.md`), or `prompt<n+1>` beyond the configured files |
| `<prompt>_answer` | Model answer to that prompt; empty if not generated yet, or if a thinking model was cut off during its reasoning (generated again on resume) |
| `<prompt>_reasoning` | Reasoning of a thinking model's answer; the column only exists if some answer of that turn has reasoning |

```
"mapping";"mapping_answer"
"You are a senior medical informatician ...";"{""mapping_type"": ""NO_MATCH"", ""confidence"": 7}"
```

The checkpoint is written to `<file>.tmp` first and then moved over the previous checkpoint in one step, so an interrupted write never leaves a corrupted file. A leftover `<file>.tmp` is only a failed write and can be deleted.
