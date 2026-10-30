import sys

# Prevent Python bytecode caching
sys.dont_write_bytecode = True

import importlib
import json
import os
import pathlib
import tempfile
import types

import pytest
import torch
import yaml
from logger import Logger
from pydantic import ValidationError

from model            import (
    Model, ModelConfig, messageRoleElement, messageTextElement, messageReasoningElement,
)
import model as model_package
from model.ModelUtils import (
    systemRole, userRole, modelRole, writePrompt, writeJSON, modelRevision, packageVersions,
)

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
    rely on -- config, messageHistories and the run metadata fields, with
    llm left as None. Tests of generate() assign a FakeLLM.
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
    model.truncatedHistories = []
    model.runId = "test-run"
    model.runStarted = "2026-01-01T00:00:00+00:00"
    model.modelIndex = 0
    model.gpuCount = 0
    model.generationStats = []
    return model


class FakeTokenizer:
    """
    Chat-template stand-in: one token per character of every message.
    Records the messages and extra template arguments of every call.
    """

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, history, tokenize, add_generation_prompt, **kwargs):
        assert tokenize and add_generation_prompt
        self.calls.append((history, kwargs))
        return [0] * sum(len(message[messageTextElement]) for message in history)

    def encode(self, text, add_special_tokens=True):
        return [0] * len(text)


class FakeLLM:
    """
    vLLM stand-in for generate(): answers every history with
    "answer to <last prompt>" and records the size of every chat() batch.
    """

    def __init__(self):
        self.batches = []
        self.tokenizer = FakeTokenizer()
        # Messages and extra keyword arguments of every chat() call
        self.chat_calls = []

    def get_tokenizer(self):
        return self.tokenizer

    def answer(self, history) -> tuple[str, str]:
        """(text, finish_reason) of the answer to one history."""
        return f"  answer to {history[-1][messageTextElement]}\n", "stop"

    def chat(self, histories, sampling_params, use_tqdm, **kwargs):
        self.batches.append(len(histories))
        self.chat_calls.append((histories, kwargs))
        results = []
        for history in histories:
            text, finish_reason = self.answer(history)
            results.append(types.SimpleNamespace(
                prompt_token_ids=[0] * 10,
                outputs=[types.SimpleNamespace(
                    text=text, token_ids=[0] * 3, finish_reason=finish_reason
                )],
            ))
        return results


class FakeThinkingLLM(FakeLLM):
    """
    FakeLLM answering like a thinking model: reasoning in <think> tags
    (with a JSON draft in it) before a JSON answer. A prompt starting with
    "long" is cut off during the reasoning.
    """

    def answer(self, history) -> tuple[str, str]:
        prompt = history[-1][messageTextElement]
        if prompt.startswith("long"):
            return '<think>\nStill reasoning about {"draft": true}', "length"
        return (f'<think>\nThe user asks {prompt}. Draft: {{"draft": true}}\n</think>\n\n'
                f'{{"answer": "{prompt}"}}', "stop")


def _write_yaml_config(tmp_path, model_config_data: dict) -> str:
    """Writes model_config_data under the "llm" section to a temp YAML file."""
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump({"llm": model_config_data}, f)
    return str(config_path)


def _fake_engine(monkeypatch, gpus: int = 1) -> dict:
    """
    Makes Model.__init__ run without a GPU: torch reports `gpus` CUDA
    devices, and LLM and SamplingParams record their keyword arguments in
    the returned dict instead of starting vLLM.
    """
    captured = {}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: gpus)
    monkeypatch.setattr(model_module, "LLM", lambda **kwargs: captured.setdefault("llm", kwargs))
    monkeypatch.setattr(model_module, "SamplingParams", lambda **kwargs: captured.setdefault("sampling", kwargs))
    return captured


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


# --- Configuration sources -----------------------------------------------------

class TestModelConfigSources:
    """
    Model(config=...) accepts a YAML path, a dict (the whole configuration or
    just the llm section) or a ModelConfig. Tested via Model._resolveConfig,
    plus one constructor run with vLLM replaced.
    """

    def test_accepts_a_path_and_a_path_object(self, tmp_path, model_config_data):
        config_path = _write_yaml_config(tmp_path, model_config_data)
        for source in (config_path, pathlib.Path(config_path)):
            config, logger_path, logger_settings = Model._resolveConfig(source)
            assert config == ModelConfig.model_validate(model_config_data)
            # The file has no logger section, so the Logger is left alone.
            assert logger_path is None
            assert logger_settings is None

    def test_path_with_logger_section_is_handed_to_the_logger(self, tmp_path, model_config_data):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({"llm": model_config_data, "logger": {}}))
        _, logger_path, _ = Model._resolveConfig(str(config_path))
        assert logger_path == str(config_path)

    def test_missing_section_raises_key_error(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("other: {}\n")
        with pytest.raises(KeyError, match="llm"):
            Model._resolveConfig(str(config_path))

    def test_accepts_a_dict_of_llm_settings(self, model_config_data):
        config, logger_path, logger_settings = Model._resolveConfig(model_config_data)
        assert config == ModelConfig.model_validate(model_config_data)
        assert logger_path is None
        assert logger_settings is None

    def test_accepts_a_full_configuration_dict(self, model_config_data):
        logger_data = {"folder": "./data/logs/"}
        config, logger_path, logger_settings = Model._resolveConfig(
            {"llm": model_config_data, "logger": logger_data}
        )
        assert config == ModelConfig.model_validate(model_config_data)
        assert logger_path is None
        assert logger_settings == logger_data

    def test_empty_section_in_a_dict_falls_back_to_defaults(self):
        config, _, _ = Model._resolveConfig({"llm": None})
        assert config == ModelConfig()

    def test_rejects_unknown_keys_in_a_dict(self):
        with pytest.raises(ValidationError):
            Model._resolveConfig({"invalid_param": 123})

    def test_accepts_a_model_config_and_copies_it(self, model_config_data):
        original = ModelConfig.model_validate(model_config_data)
        config, logger_path, logger_settings = Model._resolveConfig(original)
        assert config == original
        assert config is not original
        assert logger_path is None
        assert logger_settings is None

    def test_constructor_uses_a_dict(self, tmp_path, model_config_data, monkeypatch):
        model_config_data["metadata_folder"] = str(tmp_path / "logs")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(model_module, "LLM", lambda **kwargs: None)
        monkeypatch.setattr(model_module, "SamplingParams", lambda **kwargs: None)

        model = Model(config={"llm": model_config_data}, index=1)
        assert model.model == "dummy/model-b"
        assert model.config.seed == model_config_data["seed"]


    def test_constructor_configures_the_logger_from_a_full_dict(
        self, tmp_path, model_config_data, monkeypatch
    ):
        # Logger is a process-wide singleton; monkeypatch restores its config
        monkeypatch.setattr(Logger, "config", None)
        _fake_engine(monkeypatch)
        model_config_data["metadata_folder"] = str(tmp_path / "logs")

        Model(config={"llm": model_config_data, "logger": {"folder": str(tmp_path / "run")}})

        assert Logger.config.folder == str(tmp_path / "run")

    def test_constructor_configures_the_logger_from_the_yaml_file(
        self, tmp_path, model_config_data, monkeypatch
    ):
        monkeypatch.setattr(Logger, "config", None)
        _fake_engine(monkeypatch)
        model_config_data["metadata_folder"] = str(tmp_path / "logs")
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "llm": model_config_data, "logger": {"folder": str(tmp_path / "run")}
        }))

        Model(config=str(config_path))

        assert Logger.config.folder == str(tmp_path / "run")

    def test_constructor_keeps_an_already_configured_logger(
        self, tmp_path, model_config_data, monkeypatch
    ):
        Logger()
        configured = Logger.config
        _fake_engine(monkeypatch)
        model_config_data["metadata_folder"] = str(tmp_path / "logs")

        Model(config={"llm": model_config_data, "logger": {"folder": str(tmp_path / "run")}})

        assert Logger.config is configured


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


    def test_shards_the_model_across_every_visible_gpu(
        self, tmp_path, model_config_data, monkeypatch
    ):
        model_config_data["metadata_folder"] = str(tmp_path / "logs")
        captured = _fake_engine(monkeypatch, gpus=4)

        model = Model(config=model_config_data)

        assert captured["llm"]["tensor_parallel_size"] == 4
        assert model.gpuCount == 4

    @pytest.mark.parametrize("index, expected", [(0, True), (1, False)])
    def test_enables_expert_parallelism_for_listed_models(
        self, tmp_path, model_config_data, monkeypatch, index, expected
    ):
        model_config_data["metadata_folder"] = str(tmp_path / "logs")
        model_config_data["expert_parallel_models"] = ["model-a"]
        captured = _fake_engine(monkeypatch)

        Model(config=model_config_data, index=index)

        assert captured["llm"]["enable_expert_parallel"] is expected

    def test_import_keeps_the_gpu_selection(self, monkeypatch):
        # A selection such as Slurm's must survive importing the module;
        # it used to be reset to "0,...,n-1".
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

        importlib.reload(model_module)

        assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"
        assert os.environ["VLLM_TARGET_DEVICE"] == "cuda"


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


    def test_broadcasts_to_existing_histories_and_ignores_amount(self, tmp_path, bare_model):
        (tmp_path / "prompt_a.md").write_text("Follow-up", encoding="utf-8")
        bare_model.config.prompt_folder = str(tmp_path)
        bare_model.addPrompt(userRole, ["a", "b"])

        assert bare_model.addPromptFromFile(index=0, amount=5) == 2

        assert [history[-1][messageTextElement] for history in bare_model.messageHistories] == ["Follow-up"] * 2


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


    def test_round_trip_of_multi_turn_histories_of_different_lengths(self, tmp_path, bare_model):
        bare_model.config.prompt_tmp_folder = str(tmp_path) + os.sep
        bare_model.messageHistories = [
            [
                {messageRoleElement: userRole, messageTextElement: "q1"},
                {messageRoleElement: modelRole, messageTextElement: "a1"},
                {messageRoleElement: userRole, messageTextElement: "q2"},
                {messageRoleElement: modelRole, messageTextElement: "a2"},
            ],
            [
                {messageRoleElement: userRole, messageTextElement: "q1"},
                {messageRoleElement: modelRole, messageTextElement: "a1"},
            ],
        ]
        expected = [list(history) for history in bare_model.messageHistories]

        bare_model.exportHistoriesToFile()
        header = (tmp_path / "raw.tmp").read_text(encoding="utf-8").splitlines()[0]
        assert header == '"prompt_a";"prompt_a_answer";"prompt_b";"prompt_b_answer"'

        bare_model.messageHistories = []
        bare_model.importHistoriesFromFile()

        assert bare_model.messageHistories == expected

    def test_consecutive_user_prompts_get_their_own_columns(self, tmp_path, bare_model):
        bare_model.config.prompt_tmp_folder = str(tmp_path) + os.sep
        bare_model.addPrompt(userRole, ["first"])
        bare_model.addPrompt(userRole, ["second"])
        expected = [list(history) for history in bare_model.messageHistories]

        bare_model.exportHistoriesToFile()
        bare_model.messageHistories = []
        bare_model.importHistoriesFromFile()

        assert bare_model.messageHistories == expected

    def test_import_without_prompt_columns_raises(self, tmp_path, bare_model):
        # E.g. a checkpoint written with other prompt_files
        bare_model.config.prompt_tmp_folder = str(tmp_path) + os.sep
        (tmp_path / "raw.tmp").write_text('"other";"other_answer"\n"q";"a"\n', encoding="utf-8")

        with pytest.raises(ValueError, match="no prompt/answer columns"):
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


    def test_falls_back_when_the_code_block_is_not_json(self, bare_model):
        bare_model.addPrompt(userRole, ["classify this"])
        bare_model.addPrompt(modelRole, ['Draft:\n```\nnot json\n```\nFinal: {"label": "kidney"}'])

        assert bare_model.toJSON(0) == {"label": "kidney"}

    def test_extracts_json_array(self, bare_model):
        bare_model.addPrompt(userRole, ["list them"])
        bare_model.addPrompt(modelRole, ['The labels are ["kidney", "liver"].'])

        assert bare_model.toJSON(0) == ["kidney", "liver"]

    def test_parses_a_bare_scalar(self, bare_model):
        bare_model.addPrompt(userRole, ["rate this"])
        bare_model.addPrompt(modelRole, ["7"])

        assert bare_model.toJSON(0) == 7

    def test_returns_none_for_an_empty_answer(self, bare_model):
        bare_model.addPrompt(userRole, ["classify this"])
        bare_model.addPrompt(modelRole, ["   "])

        assert bare_model.toJSON(0) is None


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


class InterruptedLLM(FakeLLM):
    """FakeLLM whose second chat() call fails, like a job killed mid-run."""

    def chat(self, histories, sampling_params, use_tqdm, **kwargs):
        if self.batches:
            raise RuntimeError("job killed")
        return super().chat(histories, sampling_params, use_tqdm, **kwargs)


class BatchEncodingTokenizer(FakeTokenizer):
    """Tokenizer returning a dict-like BatchEncoding, as some transformers versions do."""

    def apply_chat_template(self, history, tokenize, add_generation_prompt, **kwargs):
        return {"input_ids": super().apply_chat_template(history, tokenize, add_generation_prompt, **kwargs)}


class TestGenerateResumeAndTurns:

    def test_resumes_from_the_checkpoint_after_an_interruption(self, bare_model):
        bare_model.config.chunk_size = 2
        bare_model.addPrompt(userRole, ["a", "b", "c", "d"])
        bare_model.llm = InterruptedLLM()

        with pytest.raises(RuntimeError, match="job killed"):
            bare_model.generate()

        # A new run loads the checkpoint of the first chunk ...
        bare_model.reset()
        bare_model.importHistoriesFromFile()
        assert [len(history) for history in bare_model.messageHistories] == [2, 2, 1, 1]

        # ... and only generates the histories without an answer
        bare_model.llm = FakeLLM()
        assert bare_model.generate() == 2
        assert bare_model.llm.batches == [2]
        assert [history[-1][messageTextElement] for history in bare_model.messageHistories] == [
            "answer to a", "answer to b", "answer to c", "answer to d"
        ]

    def test_second_turn_sends_the_whole_conversation(self, bare_model):
        bare_model.llm = FakeLLM()
        bare_model.addPrompt(userRole, ["first"])
        bare_model.generate()
        bare_model.addPrompt(userRole, ["second"])

        assert bare_model.generate() == 1

        sent, _ = bare_model.llm.chat_calls[-1]
        assert [message[messageTextElement] for message in sent[0]] == ["first", "answer to first", "second"]
        assert bare_model.messageHistories[0][-1][messageTextElement] == "answer to second"

    def test_context_check_accepts_a_batch_encoding(self, bare_model):
        bare_model.llm = FakeLLM()
        bare_model.llm.tokenizer = BatchEncodingTokenizer()
        bare_model.config.max_model_len = 10
        bare_model.addPrompt(userRole, ["short", "x" * 10])

        assert bare_model.generate() == 1
        assert bare_model.skippedHistories == [1]


# --- Thinking models ------------------------------------------------------------

class TestSplitReasoning:

    @pytest.mark.parametrize("text, expected", [
        # Qwen3: reasoning in <think> tags before the answer
        ("<think>\nSome reasoning.\n</think>\n\nThe answer.", ("Some reasoning.", "The answer.")),
        # DeepSeek-R1: the start tag is part of the prompt, not the output
        ("Some reasoning.\n</think>\nThe answer.", ("Some reasoning.", "The answer.")),
        # Cut off during the reasoning: no answer
        ("<think>\nSome reasoning that never", ("Some reasoning that never", "")),
        # Thinking switched off: Qwen3 emits an empty think block
        ("<think>\n\n</think>\n\nThe answer.", ("", "The answer.")),
        # Non-thinking model: the whole output is the answer
        ("  The answer.\n", ("", "The answer.")),
        # Only the first end tag closes the reasoning
        ("R</think>Answer mentioning </think> literally", ("R", "Answer mentioning </think> literally")),
    ])
    def test_splits_reasoning_and_answer(self, bare_model, text, expected):
        assert bare_model._splitReasoning(text) == expected

    def test_splitting_can_be_disabled(self, bare_model):
        bare_model.config.reasoning_end_tag = None
        text = "<think>R</think>A"
        assert bare_model._splitReasoning(text) == ("", text)

    def test_custom_tags(self, bare_model):
        bare_model.config.reasoning_start_tag = "<reasoning>"
        bare_model.config.reasoning_end_tag = "</reasoning>"
        assert bare_model._splitReasoning("<reasoning>R</reasoning> A") == ("R", "A")

    def test_config_rejects_an_empty_end_tag(self):
        with pytest.raises(ValidationError):
            ModelConfig(reasoning_end_tag="")


class TestGenerateWithThinkingModels:

    def test_stores_reasoning_separately_from_the_answer(self, bare_model):
        bare_model.llm = FakeThinkingLLM()
        bare_model.addPrompt(userRole, ["q1"])

        assert bare_model.generate() == 1

        answer = bare_model.messageHistories[0][-1]
        assert answer[messageTextElement] == '{"answer": "q1"}'
        assert answer[messageReasoningElement] == 'The user asks q1. Draft: {"draft": true}'
        # toJSON() reads the answer, not the JSON draft in the reasoning
        assert bare_model.toJSON(0) == {"answer": "q1"}
        assert bare_model.truncatedHistories == []

    def test_non_thinking_answers_have_no_reasoning_key(self, bare_model):
        bare_model.llm = FakeLLM()
        bare_model.addPrompt(userRole, ["q1"])
        bare_model.generate()
        assert messageReasoningElement not in bare_model.messageHistories[0][-1]

    def test_answer_cut_off_during_reasoning_is_empty_and_flagged(self, bare_model):
        bare_model.llm = FakeThinkingLLM()
        bare_model.addPrompt(userRole, ["q1", "long question"])

        bare_model.generate()

        cut_off = bare_model.messageHistories[1][-1]
        assert cut_off[messageTextElement] == ""
        assert cut_off[messageReasoningElement] == 'Still reasoning about {"draft": true}'
        assert bare_model.toJSON(1) is None
        assert bare_model.truncatedHistories == [1]

    def test_reasoning_is_not_sent_back_to_the_model(self, bare_model):
        bare_model.llm = FakeThinkingLLM()
        bare_model.addPrompt(userRole, ["q1"])
        bare_model.generate()
        bare_model.addPrompt(userRole, ["q2"])
        bare_model.generate()

        sent_histories, _ = bare_model.llm.chat_calls[-1]
        assert all(set(message) == {messageRoleElement, messageTextElement}
                   for message in sent_histories[0])
        checked_history, _ = bare_model.llm.tokenizer.calls[-1]
        assert all(messageReasoningElement not in message for message in checked_history)
        # The stored history keeps the reasoning
        assert messageReasoningElement in bare_model.messageHistories[0][1]

    @pytest.mark.parametrize("enable_thinking", [True, False])
    def test_passes_enable_thinking_to_chat_and_context_check(self, bare_model, enable_thinking):
        bare_model.llm = FakeThinkingLLM()
        bare_model.config.enable_thinking = enable_thinking
        bare_model.addPrompt(userRole, ["q1"])

        bare_model.generate()

        assert bare_model.llm.chat_calls[0][1] == {
            "chat_template_kwargs": {"enable_thinking": enable_thinking}
        }
        assert bare_model.llm.tokenizer.calls[0][1] == {"enable_thinking": enable_thinking}

    def test_leaves_the_template_default_without_enable_thinking(self, bare_model):
        bare_model.llm = FakeThinkingLLM()
        bare_model.addPrompt(userRole, ["q1"])

        bare_model.generate()

        assert bare_model.llm.chat_calls[0][1] == {}
        assert bare_model.llm.tokenizer.calls[0][1] == {}

    def test_records_truncation_and_reasoning_tokens_in_run_metadata(self, bare_model):
        bare_model.llm = FakeThinkingLLM()
        bare_model.addPrompt(userRole, ["q1", "long question"])

        bare_model.generate()

        path = os.path.join(bare_model.config.metadata_folder, bare_model.config.metadata_file)
        stats = json.loads(open(path, encoding="utf-8").read())[0]["generations"][0]
        assert stats["truncated"] == [1]
        # FakeTokenizer.encode: one token per character of both reasonings
        assert stats["reasoning_tokens"] == (
            len('The user asks q1. Draft: {"draft": true}')
            + len('Still reasoning about {"draft": true}')
        )

    def test_checkpoint_round_trip_keeps_reasoning(self, tmp_path, bare_model):
        bare_model.config.prompt_tmp_folder = str(tmp_path) + os.sep
        bare_model.config.prompt_files = ["prompt_a.md"]
        bare_model.llm = FakeThinkingLLM()
        bare_model.addPrompt(userRole, ["q1", "q2"])
        bare_model.generate()
        expected = [list(history) for history in bare_model.messageHistories]

        bare_model.exportHistoriesToFile()
        header = open(os.path.join(tmp_path, bare_model.config.prompt_tmp_file), encoding="utf-8").readline()
        assert "prompt_a_reasoning" in header

        bare_model.messageHistories = []
        bare_model.importHistoriesFromFile()
        assert bare_model.messageHistories == expected

    def test_checkpoint_without_reasoning_has_no_reasoning_column(self, tmp_path, bare_model):
        bare_model.config.prompt_tmp_folder = str(tmp_path) + os.sep
        bare_model.config.prompt_files = ["prompt_a.md"]
        bare_model.addPrompt(userRole, ["q1"])
        bare_model.addPrompt(modelRole, ["a1"])

        bare_model.exportHistoriesToFile()

        header = open(os.path.join(tmp_path, bare_model.config.prompt_tmp_file), encoding="utf-8").readline()
        assert "reasoning" not in header


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


    def _record_cleanup(self, monkeypatch, failing: str = "") -> list:
        """Replaces the cleanup steps of close() by recorders; `failing` raises."""
        calls = []

        def step(name):
            def run():
                calls.append(name)
                if name == failing:
                    raise RuntimeError(name)
            return run

        monkeypatch.setattr(model_module, "destroy_model_parallel", step("parallel"))
        monkeypatch.setattr(model_module, "destroy_distributed_environment", step("env"))
        monkeypatch.setattr(torch.distributed, "destroy_process_group", step("group"))
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "empty_cache", step("cache"), raising=False)
        return calls

    def test_close_releases_the_engine_and_gpu_memory(self, bare_model, monkeypatch):
        calls = self._record_cleanup(monkeypatch)
        bare_model.llm = object()

        bare_model.close()

        assert bare_model.llm is None
        assert calls == ["parallel", "env", "group", "cache"]

    def test_close_continues_after_a_failing_cleanup_step(self, bare_model, monkeypatch):
        calls = self._record_cleanup(monkeypatch, failing="parallel")

        bare_model.close()

        assert calls == ["parallel", "env", "group", "cache"]


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


# --- ModelUtils: atomic writes and run metadata helpers ------------------------------

class TestModelUtils:

    def test_failed_write_keeps_the_previous_file(self, tmp_path):
        path = str(tmp_path / "data.json")
        writeJSON({"version": 1}, path)

        # object() is not JSON-serializable, so the write fails halfway
        with pytest.raises(TypeError):
            writeJSON({"version": 2, "broken": object()}, path)

        assert json.loads(open(path, encoding="utf-8").read()) == {"version": 1}
        assert sorted(os.listdir(tmp_path)) == ["data.json"]

    def test_package_versions_of_missing_packages_are_none(self):
        versions = packageVersions(["model", "no-such-package-xyz"])
        assert versions["model"] is not None
        assert versions["no-such-package-xyz"] is None

    def test_model_revision_of_a_local_directory_is_none(self, tmp_path):
        assert modelRevision(str(tmp_path)) is None

    def test_model_revision_is_the_cached_snapshot_hash(self, tmp_path, monkeypatch):
        snapshot = tmp_path / "models--org--name" / "snapshots" / "abc123"
        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = lambda model_id, local_files_only: str(snapshot) + os.sep
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

        assert modelRevision("org/name") == "abc123"

    def test_model_revision_without_a_cached_snapshot_is_none(self, monkeypatch):
        def not_cached(model_id, local_files_only):
            raise FileNotFoundError(model_id)
        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = not_cached
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

        assert modelRevision("org/name") is None


# --- Package exports -------------------------------------------------------------------

class TestPackageExports:

    def test_every_exported_name_is_importable(self):
        assert set(model_package.__all__) == {
            "Model", "ModelConfig", "messageRoleElement", "messageTextElement",
            "messageReasoningElement", "gpu", "cpu",
        }
        for name in model_package.__all__:
            assert getattr(model_package, name) is not None


# --- GPU diagnostic ----------------------------------------------------------------------

class DiagnosticLLM(FakeLLM):
    """FakeLLM answering the diagnostic prompts correctly, or cut off."""

    finish_reason = "stop"

    def answer(self, history) -> tuple[str, str]:
        if "JSON" in history[-1][messageTextElement]:
            return '{"answer": 4}', self.finish_reason
        return "A small head.", self.finish_reason


class TestGpuDiagnostic:
    """
    Runs main() with vLLM and the GPU replaced, so the diagnostic submitted
    by runTest.sh cannot break unnoticed. It reads ./config/config.yaml,
    like the other tests run from the repository root.
    """

    @pytest.fixture
    def diagnostic(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["ModelTest.py"])
        monkeypatch.setattr(tempfile, "mkdtemp", lambda prefix: str(tmp_path))
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Fake GPU", raising=False)
        monkeypatch.setattr(model_module, "SamplingParams", lambda **kwargs: None)
        monkeypatch.setattr(model_module, "LLM", lambda **kwargs: DiagnosticLLM())
        return tmp_path

    def test_passes_with_a_working_model(self, diagnostic, capsys):
        main()

        output = capsys.readouterr().out
        assert "ALL CHECKS PASSED" in output
        assert "WARNING" not in output
        # The test generation does not touch the checkpoint of a real run
        assert (diagnostic / "prompts_raw.tmp").is_file()

    def test_fails_when_answers_are_cut_off(self, diagnostic, monkeypatch, capsys):
        monkeypatch.setattr(DiagnosticLLM, "finish_reason", "length")

        with pytest.raises(SystemExit) as exit_info:
            main()

        assert exit_info.value.code == 1
        assert "cut off at max_tokens" in capsys.readouterr().out

    def test_fails_without_cuda(self, diagnostic, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        with pytest.raises(SystemExit) as exit_info:
            main()

        assert exit_info.value.code == 1


# ==============================================================================
# Manual Slurm/GPU diagnostic (not a pytest test) -- see src/runTest.sh.
# Run directly with `python3 src/ModelTest.py [index]` on a GPU node, from the
# repository root; `index` selects the model in config.model_id (default 0).
# ==============================================================================

# Prompts of the smoke test: the first checks toJSON(), the second a free answer
diagnostic_prompts = [
    'Reply with only this JSON object and nothing else: {"answer": 4}',
    "Name one symptom of microcephaly in one sentence.",
]


def main():
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

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

    # 2. Load the selected model into vLLM. The checkpoint and run metadata
    # of the test generation go to a temporary folder, so the checkpoint of
    # a real run is never overwritten.
    with open("./config/config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    output_dir = tempfile.mkdtemp(prefix="modeltest_")
    config["llm"]["prompt_tmp_folder"] = output_dir
    config["llm"]["metadata_folder"] = output_dir

    try:
        model = Model(config, index=index)
        print(f"✅ Model '{model.model}' successfully initialized.")
    except Exception as e:
        print(f"\n❌ ERROR initializing Model: {e}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("GENERATING")
    print("=" * 50)
    print(f"enable_thinking      : {model.config.enable_thinking}")
    print(f"Output folder        : {output_dir}")

    # 3. Generate an answer for every diagnostic prompt
    errors = []
    try:
        model.addPrompt(userRole, diagnostic_prompts)
        generated = model.generate()

        for idx, history in enumerate(model.getMessageHistories()):
            message = history[-1]
            print(f"\n--- History {idx} ---")
            print(f"Prompt   : {history[0][messageTextElement]}")
            print(f"Reasoning: {len(message.get(messageReasoningElement, ''))} characters")
            print(f"Answer   : {message.get(messageTextElement, '') if message[messageRoleElement] == modelRole else '(none)'}")

        if generated != len(diagnostic_prompts):
            errors.append(f"{generated} of {len(diagnostic_prompts)} answers generated.")
        if model.skippedHistories:
            errors.append(f"Histories {model.skippedHistories} exceed max_model_len.")
        if model.truncatedHistories:
            errors.append(f"Histories {model.truncatedHistories} were cut off at max_tokens "
                          f"({model.config.max_tokens}); increase max_tokens for thinking models.")
        if model.toJSON(0) != {"answer": 4}:
            print("\n⚠️  WARNING: toJSON() did not return {\"answer\": 4} for history 0.")
    except Exception as e:
        errors.append(f"Generation failed: {e!r}")
    finally:
        # 4. Free GPU memory and shut down vLLM's distributed environment
        model.close()

    if errors:
        for error in errors:
            print(f"\n❌ ERROR: {error}")
        sys.exit(1)

    print("\n🎉 ALL CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
