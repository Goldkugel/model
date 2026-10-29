# Prompts

This directory contains the prompt template files (`.md` or `.txt`) loaded by the `Model` class.

| File | Content |
|---|---|
| `mapping.md` | Rates the semantic relation between an HPO concept and a SNOMED CT concept; answers with a JSON object `{"mapping_type": ..., "confidence": ...}` |

## Configuration

The prompt files and their order are set in the `llm` section of [`config/config.yaml`](../config/config.yaml). The position in `prompt_files` is the `index` used in code, and the file name (without extension) names the file's columns in the generation checkpoint:

```yaml
llm:
  prompt_folder: "./prompts/"
  prompt_files:
    - "mapping.md"
```

## Loading and filling a prompt

```python
from model import Model

model = Model(config="./config/config.yaml")

# Load prompt_files[0] into 'amount' new conversation histories
model.addPromptFromFile(index=0, amount=2)

# Fill the {placeholders} of history 0 and 1; each dict needs a value for
# every placeholder of the template (hpo_label, hpo_synonyms, ..., sct_children)
model.formatPromptFromFile(0, {"hpo_label": "Microcephaly", "sct_term": "Microcephaly"})
model.formatPromptFromFile(1, {"hpo_label": "Renal dysplasia", "sct_term": "Renal dysplasia"})

model.generate()
```

`addPromptFromFile()` always adds the file as a **user** message. To give the model a system prompt, add it directly with `model.addPrompt(role="system", message=[...])` before loading the template.

If histories already exist, `addPromptFromFile()` adds the prompt to every one of them as the next turn, and `amount` is ignored.

## Writing prompt templates

- **Format:** plain text (`.txt`) or Markdown (`.md`), UTF-8 encoded.
- **Placeholders:** `{name}` is replaced by `formatPromptFromFile()`; every placeholder in the file needs a value, or formatting fails.
- **Literal braces:** write `{{` and `}}` for braces that should appear in the prompt, e.g. in a JSON example (see the end of `mapping.md`).
- **No turn markers:** write only the prompt text, without tokens such as `<start_of_turn>`; `Model` formats messages with the model's own chat template.
