# Answer stores

`Model.generate()` stores the prompts and answers of all conversation histories in an answer store, a JSON file in this folder, after every chunk of `chunk_size` histories (file name: `llm.prompt_tmp_file`, folder: `llm.prompt_tmp_folder` in [`config/config.yaml`](../../config/config.yaml)). The model name and the postfix set with `Model.setPostfix()` are added to the file name, e.g. `answers_medgemma-4b-it.json` or `answers_medgemma-4b-it_fewshot.json` for `prompt_tmp_file: "answers.json"`. If a run is interrupted, e.g. by a Slurm time limit, at most one chunk of work is lost.

The answer stores (`*.json`) and the `*.tmp` and `*.lock` files are ignored by git.

## Resuming a run

Build the same histories again and call `generate()`: it uses the stored answers and only generates the others. A store written with other settings (model, revision, seed, sampling and reasoning settings, see `Model._answerSettings()`) is refused with a `ValueError`; delete it, or use another `prompt_tmp_file` or postfix.

```python
model.addPromptsFromFile(0, concepts)
model.generate()
```

`Model.hasCheckpoint()` tells whether the store of the current model and postfix exists.

## File format

The settings the answers depend on, and the histories as a tree by turn. Every system or user prompt is a node keyed by the SHA-256 of its text; the answer to it is stored on the node, and the following prompts under `next`:

```json
{
  "format": 1,
  "settings": {"model": "google/medgemma-4b-it", "revision": "...", "seed": 2898231092, "...": "..."},
  "histories": {
    "<SHA-256 of the first prompt>": {
      "role": "user",
      "prompt": "You are a senior medical informatician ...",
      "answer": {"text": "{\"mapping_type\": \"NO_MATCH\", \"confidence\": 7}"},
      "next": {}
    }
  }
}
```

An answer has `reasoning` for a thinking model and `"truncated": true` if it was cut off at `max_tokens`. Empty answers (a thinking model cut off during its reasoning) are not stored, so they are generated again. See [The answer store](../../README.md#the-answer-store-resuming-and-reusing-answers) in the main README for details.

The store is written to `<file>.tmp` first and then moved over the previous store in one step, so an interrupted write never leaves a corrupted file; a leftover `<file>.tmp` is only a failed write and can be deleted. Every write holds a lock on `<file>.lock` (empty, and safe to delete when no run is active) and merges the answers other runs wrote to the same store in the meantime, so concurrent runs (several jobs of a Slurm job array) may share one store.
