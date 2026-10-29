import sys

# Prevent Python bytecode caching
sys.dont_write_bytecode = True

import json
import os
import types

import pytest
import torch
import yaml
from pydantic import ValidationError

from model            import Model, ModelConfig, messageRoleElement, messageTextElement
from model.ModelUtils import systemRole, userRole, modelRole, writePrompt

# The Model module itself (not the class re-exported as `model.Model`), for
# monkeypatching names it imported from vllm.
model_module = sys.modules["model.Model"]


# --- Fixtures ---------------------------------------------------------------

@pytest.fixture
def model_config_data() -> dict:
    """Base "llm" config dict matching ModelConfig's schema; override per test."""
    return {
        "model_id": ["dummy/model-a", "dummy/model-b"],
        "seed": 42,
        "temperature": 0.01,
        "top_p": 0.95,
        "max_model_len": 8192,
        "max_num_batched_tokens": 16384,
        "max_tokens": 2048,
        "prompt_log_folder": "./data/logs/",
        "prompt_log_file": "prompts.log",
        "device": "cuda",
        "prompt_folder": "./prompts/",
        "prompt_files": ["prompt_a.md", "prompt_b.md"],
        "prompt_tmp_file": "raw.tmp",
        "prompt_tmp_folder": "./data/tmp/",
        "chunk_size": 500,
    }


@pytest.fixture
def bare_model(model_config_data, tmp_path) -> Model:
    """
    A Model instance with a validated config but without an actual vLLM
    engine loaded.

    Model.__init__ requires a real GPU, so tests that only exercise history
    management (addPrompt, export/import, toJSON, ...) construct the object
    via __new__ (bypassing __init__) and set up just the state those methods
    rely on -- config and messageHistories, with llm left as None.
    """
    model = Model.__new__(Model)
    model.config = ModelConfig.model_validate(model_config_data)
    # Keep files written by generate() out of the repository
    model.config.prompt_tmp_folder = str(tmp_path / "tmp")
    model.config.metadata_folder = str(tmp_path / "logs")
    model.messageHistories = []
    model.llm = None
    model.sampling_params = None
    model.model = model_config_data["model_id"][0]
    model.skippedHistories = []
    model.runId = "test-run"
    model.runStarted = "2026-01-01T00:00:00+00:00"
    model.modelIndex = 0
    model.gpuCount = 0
    model.generationStats = []
    return model


class FakeTokenizer:
    """Chat-template stand-in: one token per character of every message."""

    def apply_chat_template(self, history, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        return [0] * sum(len(message[messageTextElement]) for message in history)


class FakeLLM:
    """
    vLLM stand-in for generate(): answers every history with
    "answer to <last prompt>" and records the size of every chat() batch.
    """

    def __init__(self):
        self.batches = []

    def get_tokenizer(self):
        return FakeTokenizer()

    def chat(self, histories, sampling_params, use_tqdm):
        self.batches.append(len(histories))
        results = []
        for history in histories:
            text = f"answer to {history[-1][messageTextElement]}"
            results.append(types.SimpleNamespace(
                prompt_token_ids=[0] * 10,
                outputs=[types.SimpleNamespace(text=f"  {text}\n", token_ids=[0] * 3)],
            ))
        return results


def _write_yaml_config(tmp_path, model_config_data: dict) -> str:
    """Writes model_config_data under the "llm" section to a temp YAML file."""
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump({"llm": model_config_data}, f)
    return str(config_path)


# --- ModelConfig --------------------------------------------------------------

class TestModelConfig:

    def test_config_initializes_with_expected_defaults(self):
        config = ModelConfig()
        assert config.model_id == []
        assert config.enforce_eager is True
        assert config.gpu_memory_utilization == 0.9
        assert config.dtype == "auto"
        assert config.quantization is None
        assert config.enable_prefix_caching is True
        assert config.metadata_file == "run_metadata.json"
        assert config.device == "cuda"
        assert config.temperature == 0.01
        assert config.top_p == 0.95
        assert config.chunk_size == 500
        assert config.expert_parallel_models == ["Qwen3-30B"]

    def test_config_rejects_extra_undefined_fields(self):
        with pytest.raises(ValidationError):
            ModelConfig(invalid_param=123)  # type: ignore

    def test_config_rejects_temperature_out_of_range(self):
        with pytest.raises(ValidationError):
            ModelConfig(temperature=2.5)

    def test_config_rejects_gpu_memory_utilization_out_of_range(self):
        with pytest.raises(ValidationError):
            ModelConfig(gpu_memory_utilization=1.5)

    def test_repository_config_contains_every_option(self):
        with open("./config/config.yaml", encoding="utf-8") as f:
            llm_section = yaml.safe_load(f)["llm"]
        assert set(llm_section) == set(ModelConfig.model_fields)

    def test_config_rejects_invalid_device(self):
        with pytest.raises(ValidationError):
            ModelConfig(device="mps")  # type: ignore

    def test_config_loads_expert_parallel_models_override(self, model_config_data):
        model_config_data["expert_parallel_models"] = ["Mixtral", "DeepSeek"]
        config = ModelConfig.model_validate(model_config_data)
        assert config.expert_parallel_models == ["Mixtral", "DeepSeek"]


# --- Model constructor guard clauses -------------------------------------------

class TestModelConstructorGuards:
    """
    Exercises the __init__ guard clauses without requiring a real GPU or
    loading vLLM, by controlling torch.cuda via monkeypatch.
    """

    def test_raises_file_not_found_for_missing_config(self, tmp_path):
        missing_path = str(tmp_path / "does_not_exist.yaml")
        with pytest.raises(FileNotFoundError):
            Model(config=missing_path)

    def test_raises_value_error_for_out_of_range_index(self, tmp_path, model_config_data):
        config_path = _write_yaml_config(tmp_path, model_config_data)
        with pytest.raises(ValueError):
            Model(config=config_path, index=len(model_config_data["model_id"]))

    def test_raises_runtime_error_when_cuda_unavailable(
        self, tmp_path, model_config_data, monkeypatch
    ):
        config_path = _write_yaml_config(tmp_path, model_config_data)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(RuntimeError):
            Model(config=config_path, index=0)

    def test_raises_runtime_error_when_no_gpu_visible(
        self, tmp_path, model_config_data, monkeypatch
    ):
        config_path = _write_yaml_config(tmp_path, model_config_data)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
        with pytest.raises(RuntimeError):
            Model(config=config_path, index=0)

    def test_passes_engine_options_to_vllm_and_writes_metadata(
        self, tmp_path, model_config_data, monkeypatch
    ):
        model_config_data.update({
            "metadata_folder": str(tmp_path / "logs"),
            "enforce_eager": False,
            "gpu_memory_utilization": 0.8,
            "dtype": "bfloat16",
            "quantization": "fp8",
            "enable_prefix_caching": False,
        })
        config_path = _write_yaml_config(tmp_path, model_config_data)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

        captured = {}
        monkeypatch.setattr(model_module, "LLM", lambda **kwargs: captured.setdefault("llm", kwargs))
        monkeypatch.setattr(model_module, "SamplingParams", lambda **kwargs: captured.setdefault("sampling", kwargs))

        Model(config=config_path, index=0)

        llm_kwargs = captured["llm"]
        assert llm_kwargs["seed"] == model_config_data["seed"]
        assert llm_kwargs["max_num_batched_tokens"] == model_config_data["max_num_batched_tokens"]
        assert llm_kwargs["enforce_eager"] is False
        assert llm_kwargs["gpu_memory_utilization"] == 0.8
        assert llm_kwargs["dtype"] == "bfloat16"
        assert llm_kwargs["quantization"] == "fp8"
        assert llm_kwargs["enable_prefix_caching"] is False
        assert captured["sampling"]["seed"] == model_config_data["seed"]

        runs = json.loads((tmp_path / "logs" / "run_metadata.json").read_text(encoding="utf-8"))
        assert len(runs) == 1
        assert runs[0]["model"]["id"] == "dummy/model-a"
        assert runs[0]["gpus"]["count"] == 1
        assert runs[0]["config"]["dtype"] == "bfloat16"
        assert runs[0]["generations"] == []


# --- Model.addPrompt -----------------------------------------------------------

class TestAddPrompt:

    def test_initializes_new_histories_from_empty(self, bare_model):
        count = bare_model.addPrompt(userRole, ["prompt a", "prompt b"])

        assert count == 2
        assert bare_model.messageHistories[0] == [
            {messageRoleElement: userRole, messageTextElement: "prompt a"}
        ]
        assert bare_model.messageHistories[1] == [
            {messageRoleElement: userRole, messageTextElement: "prompt b"}
        ]

    def test_broadcasts_single_message_to_all_histories(self, bare_model):
        bare_model.addPrompt(userRole, ["a", "b"])
        bare_model.addPrompt(modelRole, ["shared reply"])

        assert bare_model.messageHistories[0][-1] == {
            messageRoleElement: modelRole, messageTextElement: "shared reply"
        }
        assert bare_model.messageHistories[1][-1] == {
            messageRoleElement: modelRole, messageTextElement: "shared reply"
        }

    def test_aligns_one_message_per_history(self, bare_model):
        bare_model.addPrompt(userRole, ["a", "b"])
        bare_model.addPrompt(modelRole, ["reply a", "reply b"])

        assert bare_model.messageHistories[0][-1][messageTextElement] == "reply a"
        assert bare_model.messageHistories[1][-1][messageTextElement] == "reply b"

    def test_mismatched_message_count_raises_and_adds_nothing(self, bare_model):
        bare_model.addPrompt(userRole, ["a", "b", "c"])
        with pytest.raises(ValueError):
            bare_model.addPrompt(modelRole, ["reply a", "reply b"])

        assert all(len(history) == 1 for history in bare_model.messageHistories)

    def test_none_message_is_a_no_op(self, bare_model):
        count = bare_model.addPrompt(userRole, None)

        assert count == 0
        assert bare_model.messageHistories == []


# --- Model.addPromptFromFile ----------------------------------------------------

class TestAddPromptFromFile:

    def test_loads_prompt_and_initializes_histories(self, tmp_path, bare_model):
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        (prompt_dir / "prompt_a.md").write_text("Hello {name}", encoding="utf-8")

        bare_model.config.prompt_folder = str(prompt_dir)
        bare_model.config.prompt_files = ["prompt_a.md"]

        count = bare_model.addPromptFromFile(index=0, amount=2)

        assert count == 2
        assert len(bare_model.messageHistories) == 2
        assert bare_model.messageHistories[0][0][messageTextElement] == "Hello {name}"

    def test_invalid_index_raises(self, bare_model):
        with pytest.raises(IndexError):
            bare_model.addPromptFromFile(index=5)

    def test_missing_file_raises(self, tmp_path, bare_model):
        bare_model.config.prompt_folder = str(tmp_path)
        bare_model.config.prompt_files = ["missing.md"]

        with pytest.raises(FileNotFoundError):
            bare_model.addPromptFromFile(index=0, amount=1)

    def test_empty_prompt_folder_raises(self, bare_model):
        bare_model.config.prompt_folder = ""
        bare_model.config.prompt_files = ["prompt_a.md"]

        with pytest.raises(ValueError):
            bare_model.addPromptFromFile(index=0, amount=1)

    def test_missing_amount_on_empty_histories_raises(self, tmp_path, bare_model):
        (tmp_path / "prompt_a.md").write_text("Hello", encoding="utf-8")
        bare_model.config.prompt_folder = str(tmp_path)
        bare_model.config.prompt_files = ["prompt_a.md"]

        with pytest.raises(ValueError):
            bare_model.addPromptFromFile(index=0)


# --- Model.formatPromptFromFile -------------------------------------------------

class TestFormatPromptFromFile:

    def test_formats_placeholders_in_last_message(self, bare_model):
        bare_model.addPrompt(userRole, ["Hello {name}, you are {age}"])
        delta = bare_model.formatPromptFromFile(index=0, values={"name": "Ada", "age": 30})

        formatted = bare_model.messageHistories[0][0][messageTextElement]
        assert formatted == "Hello Ada, you are 30"
        assert delta == len(formatted) - len("Hello {name}, you are {age}")

    def test_missing_placeholder_key_raises(self, bare_model):
        bare_model.addPrompt(userRole, ["Hello {name}"])
        with pytest.raises(KeyError):
            bare_model.formatPromptFromFile(index=0, values={})

    def test_unescaped_brace_raises_value_error(self, bare_model):
        bare_model.addPrompt(userRole, ["An open { brace"])
        with pytest.raises(ValueError):
            bare_model.formatPromptFromFile(index=0, values={})

    def test_empty_histories_raises(self, bare_model):
        with pytest.raises(IndexError):
            bare_model.formatPromptFromFile(index=0, values={})

    def test_out_of_bounds_index_raises(self, bare_model):
        bare_model.addPrompt(userRole, ["Hello"])
        with pytest.raises(IndexError):
            bare_model.formatPromptFromFile(index=5, values={})


# --- Model._get_column_names ----------------------------------------------------

class TestGetColumnNames:

    def test_uses_prompt_file_stem_when_available(self, bare_model):
        bare_model.config.prompt_files = ["hpo_mapping.md", "sct_mapping.md"]

        assert bare_model._get_column_names(0) == ("hpo_mapping", "hpo_mapping_answer")
        assert bare_model._get_column_names(1) == ("sct_mapping", "sct_mapping_answer")

    def test_falls_back_to_generic_name_beyond_configured_files(self, bare_model):
        bare_model.config.prompt_files = ["hpo_mapping.md"]

        assert bare_model._get_column_names(1) == ("prompt2", "prompt2_answer")


# --- Model.exportHistoriesToFile / importHistoriesFromFile ----------------------

class TestExportImportHistoriesRoundTrip:

    def test_round_trip_preserves_prompt_and_answer_pairs(self, tmp_path, bare_model):
        bare_model.config.prompt_tmp_folder = str(tmp_path) + os.sep
        bare_model.config.prompt_tmp_file = "raw.tmp"
        bare_model.config.prompt_files = ["prompt_a.md"]

        bare_model.addPrompt(userRole, ["What is the capital of France?"])
        bare_model.addPrompt(modelRole, ["Paris"])

        written = bare_model.exportHistoriesToFile()
        assert written == 1

        bare_model.messageHistories = []
        imported = bare_model.importHistoriesFromFile()

        assert imported == 1
        assert bare_model.messageHistories[0] == [
            {messageRoleElement: userRole, messageTextElement: "What is the capital of France?"},
            {messageRoleElement: modelRole, messageTextElement: "Paris"},
        ]

    def test_round_trip_keeps_answers_that_look_like_missing_or_numbers(self, tmp_path, bare_model):
        bare_model.config.prompt_tmp_folder = str(tmp_path) + os.sep
        bare_model.config.prompt_files = ["prompt_a.md"]

        bare_model.addPrompt(userRole, ["q1", "q2", "q3"])
        bare_model.addPrompt(modelRole, ["NA", "None", "1"])
        expected = [list(history) for history in bare_model.messageHistories]

        bare_model.exportHistoriesToFile()
        bare_model.messageHistories = []
        bare_model.importHistoriesFromFile()

        assert bare_model.messageHistories == expected

    def test_round_trip_preserves_system_role(self, tmp_path, bare_model):
        bare_model.config.prompt_tmp_folder = str(tmp_path) + os.sep
        bare_model.config.prompt_files = ["prompt_a.md"]

        bare_model.addPrompt(systemRole, ["You are concise."])
        bare_model.addPrompt(userRole, ["Question"])
        bare_model.addPrompt(modelRole, ["Answer"])
        expected = [list(history) for history in bare_model.messageHistories]

        bare_model.exportHistoriesToFile()
        bare_model.messageHistories = []
        bare_model.importHistoriesFromFile()

        assert bare_model.messageHistories == expected

    def test_round_trip_with_non_identifier_prompt_file_name(self, tmp_path, bare_model):
        bare_model.config.prompt_tmp_folder = str(tmp_path) + os.sep
        bare_model.config.prompt_files = ["01_hpo-mapping.md"]

        bare_model.addPrompt(userRole, ["Question"])
        bare_model.addPrompt(modelRole, ["Answer"])
        expected = [list(history) for history in bare_model.messageHistories]

        bare_model.exportHistoriesToFile()
        bare_model.messageHistories = []

        assert bare_model.importHistoriesFromFile() == 1
        assert bare_model.messageHistories == expected

    def test_export_leaves_no_temporary_file(self, tmp_path, bare_model):
        bare_model.config.prompt_tmp_folder = str(tmp_path) + os.sep
        bare_model.config.prompt_tmp_file = "raw.tmp"

        bare_model.addPrompt(userRole, ["Question"])
        bare_model.exportHistoriesToFile()

        assert sorted(os.listdir(tmp_path)) == ["raw.tmp"]

    def test_export_with_no_histories_raises(self, bare_model):
        with pytest.raises(ValueError):
            bare_model.exportHistoriesToFile()

    def test_import_with_missing_file_raises(self, tmp_path, bare_model):
        bare_model.config.prompt_tmp_folder = str(tmp_path) + os.sep
        bare_model.config.prompt_tmp_file = "does_not_exist.tmp"

        with pytest.raises(FileNotFoundError):
            bare_model.importHistoriesFromFile()

    def test_import_with_empty_file_raises(self, tmp_path, bare_model):
        bare_model.config.prompt_tmp_folder = str(tmp_path) + os.sep
        bare_model.config.prompt_tmp_file = "empty.tmp"
        (tmp_path / "empty.tmp").write_text("", encoding="utf-8")

        with pytest.raises(ValueError):
            bare_model.importHistoriesFromFile()


# --- Model.toJSON ----------------------------------------------------------------

class TestToJSON:

    def test_extracts_json_from_markdown_code_block(self, bare_model):
        bare_model.addPrompt(userRole, ["classify this"])
        bare_model.addPrompt(modelRole, ['Sure, here you go:\n```json\n{"label": "kidney"}\n```'])

        assert bare_model.toJSON(0) == {"label": "kidney"}

    def test_extracts_json_object_without_code_block(self, bare_model):
        bare_model.addPrompt(userRole, ["classify this"])
        bare_model.addPrompt(modelRole, ['The result is {"label": "kidney"} as requested.'])

        assert bare_model.toJSON(0) == {"label": "kidney"}

    def test_ignores_brackets_in_trailing_prose(self, bare_model):
        bare_model.addPrompt(userRole, ["classify this"])
        bare_model.addPrompt(modelRole, ['{"label": "kidney"}\nNote: see {ref} and [1].'])

        assert bare_model.toJSON(0) == {"label": "kidney"}

    def test_skips_brackets_in_leading_prose(self, bare_model):
        bare_model.addPrompt(userRole, ["classify this"])
        bare_model.addPrompt(modelRole, ['Answer [final]: {"label": "kidney"}'])

        assert bare_model.toJSON(0) == {"label": "kidney"}

    def test_returns_none_for_invalid_json(self, bare_model):
        bare_model.addPrompt(userRole, ["classify this"])
        bare_model.addPrompt(modelRole, ["not json at all"])

        assert bare_model.toJSON(0) is None

    def test_returns_none_when_last_message_is_not_from_model(self, bare_model):
        bare_model.addPrompt(userRole, ["classify this"])

        assert bare_model.toJSON(0) is None

    def test_raises_for_out_of_bounds_index(self, bare_model):
        with pytest.raises(IndexError):
            bare_model.toJSON(0)


# --- Model.reset / getMessageHistories --------------------------------------------

class TestResetAndGetMessageHistories:

    def test_reset_clears_histories(self, bare_model):
        bare_model.addPrompt(userRole, ["a"])
        bare_model.reset()

        assert bare_model.getMessageHistories() == []


# --- Model.generate ------------------------------------------------------------------

class TestGenerate:

    def test_returns_immediately_when_no_prompts(self, bare_model):
        # self.llm is None; if the empty-history guard clause didn't
        # short-circuit, this would raise AttributeError on
        # self.llm.get_tokenizer() instead of returning cleanly.
        assert bare_model.generate() == 0

    def test_answers_unanswered_histories_in_chunks(self, bare_model):
        bare_model.llm = FakeLLM()
        bare_model.config.chunk_size = 2
        bare_model.addPrompt(userRole, ["a", "b", "c", "d", "e"])
        # An already answered history is not sent again
        bare_model.messageHistories[1].append({messageRoleElement: modelRole, messageTextElement: "done"})

        assert bare_model.generate() == 4

        assert bare_model.llm.batches == [2, 2]
        assert bare_model.messageHistories[0][-1] == {messageRoleElement: modelRole, messageTextElement: "answer to a"}
        assert bare_model.messageHistories[1][-1][messageTextElement] == "done"
        assert all(history[-1][messageRoleElement] == modelRole for history in bare_model.messageHistories)
        assert os.path.isfile(os.path.join(bare_model.config.prompt_tmp_folder, bare_model.config.prompt_tmp_file))

    def test_skips_prompts_longer_than_the_context(self, bare_model):
        bare_model.llm = FakeLLM()
        bare_model.config.max_model_len = 10
        bare_model.addPrompt(userRole, ["short", "x" * 10, "y" * 50])

        assert bare_model.generate() == 1

        assert bare_model.skippedHistories == [1, 2]
        assert bare_model.llm.batches == [1]
        assert bare_model.toJSON(1) is None
        assert bare_model.messageHistories[2] == [{messageRoleElement: userRole, messageTextElement: "y" * 50}]

    def test_chunk_without_fitting_prompts_is_not_sent(self, bare_model):
        bare_model.llm = FakeLLM()
        bare_model.config.max_model_len = 3
        bare_model.addPrompt(userRole, ["too long"])

        assert bare_model.generate() == 0
        assert bare_model.llm.batches == []

    def test_records_statistics_in_run_metadata(self, bare_model):
        bare_model.llm = FakeLLM()
        bare_model.config.max_model_len = 10
        bare_model.addPrompt(userRole, ["a", "b", "z" * 20])

        bare_model.generate()

        path = os.path.join(bare_model.config.metadata_folder, bare_model.config.metadata_file)
        runs = json.loads(open(path, encoding="utf-8").read())
        stats = runs[0]["generations"][0]
        assert stats["histories"] == 3
        assert stats["answers"] == 2
        assert stats["skipped"] == [2]
        assert stats["prompt_tokens"] == 20
        assert stats["generated_tokens"] == 6


# --- Model._writeRunMetadata ----------------------------------------------------------

class TestRunMetadata:

    def test_records_prompt_hashes_and_package_versions(self, tmp_path, bare_model):
        (tmp_path / "prompt_a.md").write_text("Hello", encoding="utf-8")
        bare_model.config.prompt_folder = str(tmp_path)
        bare_model.config.prompt_files = ["prompt_a.md", "missing.md"]

        bare_model._writeRunMetadata()

        path = os.path.join(bare_model.config.metadata_folder, bare_model.config.metadata_file)
        run = json.loads(open(path, encoding="utf-8").read())[0]
        assert run["prompt_files"] == {
            # sha256("Hello")
            "prompt_a.md": "185f8db32271fe25f561a6fc938b2e264306ec304eda518007d1764826381969",
            "missing.md": None,
        }
        assert run["packages"]["model"] is not None
        assert run["model"] == {"id": "dummy/model-a", "index": 0, "revision": None}

    def test_keeps_other_runs_and_replaces_its_own(self, bare_model):
        path = os.path.join(bare_model.config.metadata_folder, bare_model.config.metadata_file)
        os.makedirs(bare_model.config.metadata_folder)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([{"run_id": "earlier-run"}], f)

        bare_model._writeRunMetadata()
        bare_model._writeRunMetadata()

        runs = json.loads(open(path, encoding="utf-8").read())
        assert [run["run_id"] for run in runs] == ["earlier-run", "test-run"]

    def test_rejects_a_file_that_is_not_a_list(self, bare_model):
        path = os.path.join(bare_model.config.metadata_folder, bare_model.config.metadata_file)
        os.makedirs(bare_model.config.metadata_folder)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"run_id": "x"}, f)

        with pytest.raises(ValueError):
            bare_model._writeRunMetadata()


# --- Model.logPrompts --------------------------------------------------------------

class TestLogPrompts:

    def test_writes_log_file_at_configured_path(self, tmp_path, bare_model):
        bare_model.config.prompt_log_folder = str(tmp_path) + os.sep
        bare_model.config.prompt_log_file = "prompts.log"
        bare_model.addPrompt(userRole, ["Hello"])

        bare_model.logPrompts()

        log_path = tmp_path / "prompts.log"
        assert log_path.is_file()
        assert "Hello" in log_path.read_text(encoding="utf-8")


# --- Model.close / context manager -------------------------------------------------

class TestClose:

    def test_close_is_safe_to_call_without_a_loaded_llm(self, bare_model):
        bare_model.close()  # must not raise even though self.llm is None
        assert bare_model.llm is None

    def test_context_manager_calls_close_on_exit(self, bare_model):
        with bare_model as m:
            assert m is bare_model
        assert bare_model.llm is None

    def test_garbage_collection_keeps_distributed_environment(self, bare_model, monkeypatch):
        # Another Model instance may still use the process-wide distributed
        # environment, so __del__ must not tear it down.
        calls = []
        monkeypatch.setattr(model_module, "destroy_model_parallel", lambda: calls.append("parallel"))
        monkeypatch.setattr(model_module, "destroy_distributed_environment", lambda: calls.append("env"))

        bare_model.llm = object()
        bare_model.__del__()

        assert bare_model.llm is None
        assert calls == []


# --- ModelUtils.writePrompt ---------------------------------------------------------

class TestWritePrompt:

    def test_writes_header_and_expected_row_count(self, tmp_path):
        histories = [
            [
                {messageRoleElement: userRole, messageTextElement: "Hello"},
                {messageRoleElement: modelRole, messageTextElement: "Hi there"},
            ],
        ]
        path = str(tmp_path / "logs" / "prompts.log")

        total = writePrompt(path, histories)

        assert total == 2
        lines = open(path, encoding="utf-8").read().splitlines()
        assert lines[0] == "history;prompt;role;length;text"
        assert len(lines) == 3  # header + 2 message rows

    def test_sanitizes_embedded_semicolons_and_newlines(self, tmp_path):
        histories = [[
            {messageRoleElement: userRole, messageTextElement: "line one;line two\nline three"}
        ]]
        path = str(tmp_path / "prompts.log")

        writePrompt(path, histories)

        content = open(path, encoding="utf-8").read()
        assert "line one,line two line three" in content
        assert content.count("\n") == 2  # header + one data row


# ==============================================================================
# Manual Slurm/GPU diagnostic (not a pytest test) -- see src/runTest.sh.
# Run directly with `python3 src/ModelTest.py` on a GPU node.
# ==============================================================================

def main():
    print("=" * 50)
    print("SLURM & GPU ENVIRONMENT DIAGNOSTIC")
    print("=" * 50)

    # 1. Check Slurm & CUDA visibility
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set")
    print(f"CUDA_VISIBLE_DEVICES : {cuda_visible}")

    cuda_available = torch.cuda.is_available()
    print(f"PyTorch CUDA Available: {cuda_available}")

    if not cuda_available:
        print("\n❌ ERROR: CUDA is not available to PyTorch.")
        print("Check if the CUDA module is loaded or if PyTorch CPU-only is installed.")
        sys.exit(1)

    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU Device Count     : {device_count}")
    print(f"Active GPU           : {device_name}")

    print("\n" + "=" * 50)
    print("INITIALIZING MODEL")
    print("=" * 50)

    # 2. Load the first configured model into vLLM
    try:
        model = Model()
        print("✅ Model successfully initialized.")
    except Exception as e:
        print(f"\n❌ ERROR initializing Model: {e}")
        sys.exit(1)

    # 3. Free GPU memory and shut down vLLM's distributed environment
    model.close()

    print("\n🎉 ALL CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
