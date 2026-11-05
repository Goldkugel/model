# Configuration

`config.yaml` is the configuration file read by `Model` (default path: `./config/config.yaml`, relative to the working directory). `Model` also accepts the same settings as a dict or a `ModelConfig` instead of a file. It has two top-level sections:

| Section | Read by |
|---|---|
| `llm` | `Model` (validated by `ModelConfig`) |
| `logger` | [`logger`](https://github.com/Goldkugel/logger), which `Model` uses for its run log. If the file passed to `Model` has no `logger` section, the logger falls back to `./config/config.yaml`, and fails if that has none either (see [Configuration sources](../README.md#configuration-sources)). |

When `model` is used inside a larger pipeline, the pipeline's config file may contain further sections (e.g. `adapter`); they are ignored here.

## `llm` section

See [`config.yaml`](./config.yaml) for the settings in use.

Every field is optional and falls back to its default. Unknown fields are rejected, so a typo raises a validation error instead of being ignored silently. See the [field reference](../README.md#configuration) in the main README for the meaning, type and default of every field.

All paths are relative to the working directory the script is started from, not to this folder.

## `logger` section

```yaml
logger:
  folder: "./data/logs/"
  file_name: "output.log"
  header_char: "-"
  header_length: 60
  log_runtime: True
  format: "%Y-%m-%d %H:%M:%S"
  encoding: "utf-8"
  level: "info"
  runtime_format: "minutes"
  console: True
  colors:
    info: "white"
    warning: "yellow"
    error: "red"
```

See the [`logger` documentation](https://github.com/Goldkugel/logger) for these fields.
