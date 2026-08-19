Markdown

# Prompts Directory

This directory contains the prompt template files (`.md` or `.txt`) loaded and processed by the `Model` class during inference.

---

## Directory Structure

Place each prompt template in an individual file within this folder:

```text
.
├── config/
│   └── config.yaml
├── data/
│   └── logs/
├── prompts/
│   ├── README.md
│   ├── system_prompt.md
│   ├── user_prompt_1.md
│   └── user_prompt_2.md
└── src/
```

Usage & Configuration

The order and selection of prompts loaded from this directory are controlled via your config/config.yaml file under the llm section.

## 1. config.yaml Configuration

Specify the directory path and the list of prompt filenames in execution order:

```yaml
llm:
  prompt_folder: "./prompts/"
  prompt_files:
    - "system_prompt.md"
    - "user_prompt_1.md"
```

## 2. Loading Prompts in Code

You can load a prompt file into the active conversation history by index matching the list in prompt_files:

```python

from model import Model

# Initialize the model using your configuration file
model = Model(config="./config/config.yaml")

# Load the first prompt file specified in prompt_files (index 0)
# 'amount' sets the number of parallel conversation histories to create
model.addPromptFromFile(index=0, amount=1)

# Generate response
model.generate()

# Optional: Log conversations to data/logs/
model.logPrompts()
```

Writing Prompt Templates

    - File Format: Files can be plain text (.txt) or Markdown (.md).
    - Encoding: All files must be UTF-8 encoded.
    - Content: Write the prompt text directly in the file without hardcoded turn markers (such as <start_of_turn>). The Model class automatically formats messages using the target model's tokenizer chat template.