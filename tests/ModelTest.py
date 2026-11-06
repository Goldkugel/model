import sys

import importlib
import json
import multiprocessing
import os
import pathlib
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
    updateRunRecord, messageTruncatedElement, describeError,
)
import model.ModelUtils as model_utils
import logger.FileUtils as logger_file_utils
from conftest import REPO_ROOT

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
    A Model with a validated config but no vLLM engine (llm is None).

    Model.__init__ requires a GPU, so the object is built via __new__ with
    just the state the other methods rely on. Tests of generate() assign a
    FakeLLM.
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
    model.store = None
    model.revision = None
    model.revisionLooked = False
    model.postfix = ""
    return model


@pytest.fixture
def logged(monkeypatch) -> list[str]:
    """Records every message passed to Logger.log() during the test."""
    messages = []
    monkeypatch.setattr(Logger, "log", lambda self, message, *args, **kwargs: messages.append(message))
    return messages


@pytest.fixture
def logged_levels(monkeypatch) -> list[tuple[str, str]]:
    """Records (message, level) of every Logger.log() call during the test."""
    messages = []
    monkeypatch.setattr(
        Logger, "log",
        lambda self, message, cmdline=True, level="info": messages.append((message, level)),
    )
    return messages


def _checkpoint_path(model: Model) -> pathlib.Path:
    """Path of the checkpoint file of `model`."""
    return pathlib.Path(model._checkpointPath())


def _metadata_path(model: Model) -> str:
    """Path of the run metadata file of `model`."""
    return os.path.join(model.config.metadata_folder, model.config.metadata_file)


class FakeTokenizer:
    """
    Chat-template stand-in: one token per character of every message.
    Records the messages and extra template arguments of every call. With
    tokenize=False it returns the prompt as text, ending with
    `generation_prompt` (e.g. "<think>\n" for a DeepSeek-R1-style template).
    """

    generation_prompt = ""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, history, tokenize, add_generation_prompt, **kwargs):
        assert add_generation_prompt
        self.calls.append((history, kwargs))
        if not tokenize:
            return "".join(message[messageTextElement] for message in history) + self.generation_prompt
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
    """Defaults and validation of ModelConfig."""

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
        with open(REPO_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
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

class TestModelConfigRevisions:
    """llm.revision must name models of model_id."""

    def test_revision_for_unknown_model_is_rejected(self):
        with pytest.raises(ValidationError, match="not in model_id"):
            ModelConfig(model_id=["org/model-a"], revision={"org/model-b": "abc123"})

    def test_model_ids_and_revisions_are_stripped(self):
        config = ModelConfig(model_id=[" org/model-a "], revision={" org/model-a ": " abc123 "})
        assert config.model_id == ["org/model-a"]
        assert config.revision == {"org/model-a": "abc123"}


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

    def test_empty_logger_section_in_a_dict_means_the_logger_defaults(self, model_config_data):
        _, _, logger_settings = Model._resolveConfig({"llm": model_config_data, "logger": None})
        assert logger_settings == {}

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

    def test_failing_engine_start_releases_the_gpus(
        self, tmp_path, model_config_data, monkeypatch, logged_levels
    ):
        config_path = _write_yaml_config(tmp_path, model_config_data)
        _fake_engine(monkeypatch)
        calls = []

        def fail(**kwargs):
            raise RuntimeError("CUDA out of memory")

        monkeypatch.setattr(model_module, "LLM", fail)
        monkeypatch.setattr(model_module, "destroy_model_parallel", lambda: calls.append("parallel"))
        monkeypatch.setattr(model_module, "destroy_distributed_environment", lambda: calls.append("env"))

        with pytest.raises(RuntimeError, match="out of memory"):
            Model(config=config_path, index=0)

        assert calls == ["parallel", "env"]
        assert ("Loading model into cuda failed; releasing the GPUs.", "error") in logged_levels

    def test_guard_failures_are_logged_as_errors(
        self, tmp_path, model_config_data, monkeypatch, logged_levels
    ):
        config_path = _write_yaml_config(tmp_path, model_config_data)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        with pytest.raises(RuntimeError):
            Model(config=config_path, index=0)

        assert ("No GPU available.", "error") in logged_levels

    def test_cpu_device_is_a_warning(self, tmp_path, model_config_data, monkeypatch, logged_levels):
        model_config_data.update({"device": "cpu", "metadata_folder": str(tmp_path / "logs")})
        config_path = _write_yaml_config(tmp_path, model_config_data)
        _fake_engine(monkeypatch)

        Model(config=config_path, index=0)

        assert ("Device 'cpu' is not supported yet; the model is loaded on the GPU.",
                "warning") in logged_levels

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

    def test_warns_that_cpu_is_not_supported(self, tmp_path, model_config_data, monkeypatch, logged):
        model_config_data.update({"metadata_folder": str(tmp_path / "logs"), "device": "cpu"})
        _fake_engine(monkeypatch)

        Model(config=model_config_data)

        assert any("'cpu' is not supported" in message for message in logged)

    def test_releases_the_engine_if_the_run_metadata_cannot_be_written(
        self, tmp_path, model_config_data, monkeypatch
    ):
        # The caller gets no Model to close(), so __init__ must do it
        (tmp_path / "run_metadata.json").write_text("{broken", encoding="utf-8")
        model_config_data["metadata_folder"] = str(tmp_path)
        _fake_engine(monkeypatch)
        closed = []
        monkeypatch.setattr(Model, "close", lambda self: closed.append(self.llm))

        with pytest.raises(json.JSONDecodeError):
            Model(config=model_config_data)

        assert len(closed) == 1 and closed[0] is not None

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
    """The three cases of addPrompt(): new histories, broadcast, one message per history."""

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


class TestAddPromptValidation:
    """Inputs addPrompt() rejects without adding anything."""

    def test_a_string_instead_of_a_list_raises(self, bare_model):
        # It used to be split into one history per character
        with pytest.raises(TypeError):
            bare_model.addPrompt(userRole, "hello")
        assert bare_model.messageHistories == []

    def test_an_unknown_role_raises(self, bare_model):
        bare_model.addPrompt(userRole, ["q"])

        with pytest.raises(ValueError, match="unknown role 'model'"):
            bare_model.addPrompt("model", ["a"])
        assert len(bare_model.messageHistories[0]) == 1

    def test_a_turn_after_an_empty_answer_raises(self, bare_model):
        bare_model.addPrompt(userRole, ["q1", "q2"])
        bare_model.addPrompt(modelRole, ["a1", ""])

        with pytest.raises(ValueError, match=r"histories \[1\] end with an empty answer"):
            bare_model.addPrompt(userRole, ["follow-up"])
        assert [len(history) for history in bare_model.messageHistories] == [2, 2]


    @pytest.mark.parametrize("before", [
        [],                                   # a new history
        [(systemRole, "Be concise.")],        # after a system prompt
        [(userRole, "q"), (modelRole, "a")],  # after another answer
    ])
    def test_an_answer_without_a_prompt_before_it_raises(self, bare_model, before):
        # The checkpoint used to bring it back with an empty prompt inserted
        for role, text in before:
            bare_model.addPrompt(role, [text])

        with pytest.raises(ValueError, match="must follow a user prompt"):
            bare_model.addPrompt(modelRole, ["answer"])
        assert sum(len(history) for history in bare_model.messageHistories) == len(before)


# --- Model.addPromptFromFile ----------------------------------------------------

class TestAddPromptFromFile:
    """Loading a prompt template with addPromptFromFile()."""

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
    """Filling placeholders with formatPromptFromFile()."""

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


# --- Model.toJSON ----------------------------------------------------------------

class TestToJSON:
    """Parsing the last answer of a history with toJSON()."""

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
    """reset() and getMessageHistories()."""

    def test_reset_clears_histories(self, bare_model):
        bare_model.addPrompt(userRole, ["a"])
        bare_model.reset()

        assert bare_model.getMessageHistories() == []


# --- Model.getModelName ---------------------------------------------------------------

class TestGetModelName:
    """getModelName(): the model ID without the organisation."""

    @pytest.mark.parametrize("model_id, name", [
        ("google/medgemma-4b-it", "medgemma-4b-it"),
        ("Qwen3-4B", "Qwen3-4B"),
        ("/models/org/Mistral-7B/", "Mistral-7B"),
    ])
    def test_returns_the_id_without_the_organisation(self, bare_model, model_id, name):
        bare_model.model = model_id

        assert bare_model.getModelName() == name

    def test_returns_none_without_a_model(self, bare_model):
        bare_model.model = None

        assert bare_model.getModelName() is None

    def test_names_a_given_model_id(self, bare_model):
        bare_model.model = None

        assert bare_model.getModelName("mistralai/Mistral-7B-Instruct-v0.2") == "Mistral-7B-Instruct-v0.2"


# --- Model.getModelCount --------------------------------------------------------------

class TestGetModelCount:
    """getModelCount(), which reads the config without loading a model."""

    def test_counts_the_models_of_a_config_file(self, tmp_path, model_config_data):
        path = _write_yaml_config(tmp_path, model_config_data)

        assert Model.getModelCount(path) == 2

    def test_accepts_a_dict_and_a_model_config(self, model_config_data):
        model_config_data["model_id"] = ["a/one", "b/two", "c/three"]

        assert Model.getModelCount(model_config_data) == 3
        assert Model.getModelCount({"llm": model_config_data}) == 3
        assert Model.getModelCount(ModelConfig.model_validate(model_config_data)) == 3

    def test_loads_no_model(self, model_config_data, monkeypatch):
        monkeypatch.setattr(Model, "__init__", lambda *args, **kwargs: pytest.fail("Model loaded"))

        assert Model.getModelCount(model_config_data) == 2

    def test_missing_config_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Model.getModelCount(str(tmp_path / "missing.yaml"))


# --- Model.setPostfix / hasCheckpoint ------------------------------------------------

class TestFileNames:
    """File names with model name and postfix (setPostfix(), hasCheckpoint())."""

    def test_file_names_contain_the_model_name(self, bare_model):
        assert os.path.basename(bare_model._checkpointPath()) == "raw_model-a.tmp"
        assert os.path.basename(bare_model._promptLogPath()) == "prompts_model-a.log"

    def test_postfix_follows_the_model_name(self, bare_model):
        bare_model.setPostfix("Test_FewShot")

        assert os.path.basename(bare_model._checkpointPath()) == "raw_model-a_Test_FewShot.tmp"
        assert os.path.basename(bare_model._promptLogPath()) == "prompts_model-a_Test_FewShot.log"

    def test_empty_postfix_removes_it(self, bare_model):
        bare_model.setPostfix("x")
        bare_model.setPostfix("")

        assert os.path.basename(bare_model._checkpointPath()) == "raw_model-a.tmp"

    @pytest.mark.parametrize("postfix", ["a/b", "../x"])
    def test_postfix_with_a_path_separator_raises(self, bare_model, postfix):
        with pytest.raises(ValueError, match="path separator"):
            bare_model.setPostfix(postfix)

    def test_has_checkpoint_follows_the_postfix(self, bare_model):
        assert not bare_model.hasCheckpoint()

        bare_model.llm = FakeLLM()
        bare_model.addPrompt(userRole, ["q"])
        bare_model.generate()

        assert bare_model.hasCheckpoint()
        bare_model.setPostfix("other")
        assert not bare_model.hasCheckpoint()

    def test_runs_with_other_postfixes_do_not_share_a_checkpoint(self, bare_model, tmp_path):
        bare_model.llm = FakeLLM()
        bare_model.setPostfix("first")
        bare_model.addPrompt(userRole, ["q"])
        bare_model.generate()

        bare_model.reset()
        bare_model.setPostfix("second")
        bare_model.addPrompt(userRole, ["q"])

        # The answer of the first postfix is not used
        assert bare_model.generate() == 1

        stores = sorted(name for name in os.listdir(tmp_path / "tmp") if name.endswith(".tmp"))
        assert stores == ["raw_model-a_first.tmp", "raw_model-a_second.tmp"]

    def test_run_metadata_records_the_file_names(self, bare_model):
        bare_model.setPostfix("Test")
        bare_model._writeRunMetadata()

        runs = json.loads(open(_metadata_path(bare_model), encoding="utf-8").read())
        assert runs[-1]["files"] == {
            "checkpoint": bare_model._checkpointPath(),
            "prompt_log": bare_model._promptLogPath(),
        }


# --- Model.setSeed ------------------------------------------------------------------

class TestSetSeed:
    """Switching the seed of a loaded model (setSeed())."""

    @pytest.fixture
    def sampling(self, monkeypatch) -> list[dict]:
        """Records the keyword arguments of every SamplingParams()."""
        calls = []
        monkeypatch.setattr(model_module, "SamplingParams", lambda **kwargs: calls.append(kwargs) or kwargs)
        return calls

    def test_updates_the_config_and_the_sampling_params(self, bare_model, sampling):
        config = bare_model.config

        bare_model.setSeed(7)

        assert bare_model.config.seed == 7
        assert bare_model.sampling_params == {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            "seed": 7,
        }
        # The config is copied, the other settings are kept
        assert config.seed == 42
        assert bare_model.config.model_dump() == {**config.model_dump(), "seed": 7}

    def test_does_not_load_the_model_again(self, bare_model, sampling):
        bare_model.llm = llm = FakeLLM()

        bare_model.setSeed(7)

        assert bare_model.llm is llm

    @pytest.mark.parametrize("seed", ["7", 7.0, True, None])
    def test_a_seed_that_is_no_int_raises(self, bare_model, sampling, seed):
        with pytest.raises(TypeError, match="must be an int"):
            bare_model.setSeed(seed)

        assert bare_model.config.seed == 42 and sampling == []

    def test_a_store_of_another_seed_is_refused(self, bare_model, sampling):
        bare_model.llm = FakeLLM()
        bare_model.addPrompt(userRole, ["q"])
        bare_model.generate()

        bare_model.reset()
        bare_model.setSeed(7)
        bare_model.addPrompt(userRole, ["q"])

        with pytest.raises(ValueError, match=r"seed: 42 -> 7"):
            bare_model.generate()
        # Nothing was generated, and the store keeps the answers of seed 42
        assert bare_model.llm.batches == [1]
        assert _stored(bare_model)["settings"]["seed"] == 42

    def test_the_loaded_store_is_not_reused_after_a_seed_change(self, bare_model, sampling):
        # Also if setSeed() is bypassed, the store of the old seed is not
        # used for the answers of the new one
        bare_model.llm = FakeLLM()
        bare_model.addPrompt(userRole, ["q"])
        bare_model.generate()

        bare_model.reset()
        bare_model.config = bare_model.config.model_copy(update={"seed": 7})
        bare_model.addPrompt(userRole, ["q"])

        with pytest.raises(ValueError, match=r"seed: 42 -> 7"):
            bare_model.generate()

    def test_every_seed_gets_its_own_store_with_a_postfix(self, bare_model, sampling, tmp_path):
        bare_model.llm = FakeLLM()
        for seed in (1, 2, 1):
            bare_model.reset()
            bare_model.setSeed(seed)
            bare_model.setPostfix(f"seed{seed}")
            bare_model.addPrompt(userRole, ["q"])
            bare_model.generate()

        # Seed 1 is generated once; its second run takes the stored answer
        assert bare_model.llm.batches == [1, 1]
        assert [stats["seed"] for stats in bare_model.generationStats] == [1, 2, 1]
        assert [stats["stored"] for stats in bare_model.generationStats] == [0, 0, 1]
        stores = sorted(name for name in os.listdir(tmp_path / "tmp") if name.endswith(".tmp"))
        assert stores == ["raw_model-a_seed1.tmp", "raw_model-a_seed2.tmp"]
        assert _stored(bare_model)["settings"]["seed"] == 1

    def test_generate_passes_the_new_sampling_params(self, bare_model, sampling, monkeypatch):
        bare_model.llm = FakeLLM()
        passed = []
        chat = bare_model.llm.chat
        monkeypatch.setattr(bare_model.llm, "chat", lambda histories, sampling_params, **kwargs:
                            passed.append(sampling_params) or chat(histories, sampling_params, **kwargs))

        bare_model.setSeed(7)
        bare_model.setPostfix("seed7")
        bare_model.addPrompt(userRole, ["q"])
        bare_model.generate()

        assert [params["seed"] for params in passed] == [7]


# --- Model.getTurnCount / getPrompt / getAnswer --------------------------------------

class TestTurns:
    """getTurnCount(), getPrompt() and getAnswer()."""

    def test_turn_count_without_histories_is_zero(self, bare_model):
        assert bare_model.getTurnCount() == 0

    def test_prompts_and_answers_by_turn(self, bare_model):
        bare_model.addPrompt(systemRole, ["system"])
        bare_model.addPrompt(userRole, ["q1"])
        bare_model.addPrompt(modelRole, ["  a1 "])
        bare_model.addPrompt(userRole, ["q2"])

        assert bare_model.getTurnCount(0) == 2
        assert bare_model.getPrompt(0, 0) == "q1"
        assert bare_model.getPrompt(0, 1) == "q2"
        assert bare_model.getAnswer(0, 0) == "a1"
        # Not answered yet
        assert bare_model.getAnswer(0, 1) == ""

    def test_unknown_turn_or_history_raises(self, bare_model):
        bare_model.addPrompt(userRole, ["q1"])

        with pytest.raises(IndexError, match="Turn 1"):
            bare_model.getPrompt(0, 1)
        with pytest.raises(IndexError):
            bare_model.getAnswer(5, 0)
        with pytest.raises(IndexError):
            bare_model.getTurnCount(5)


# --- Model.addPromptsFromFile ---------------------------------------------------------

class TestAddPromptsFromFile:
    """addPromptsFromFile(): one template, one dict of values per history."""

    @pytest.fixture
    def templated_model(self, bare_model, tmp_path):
        (tmp_path / "first.md").write_text("Label: {label}", encoding="utf-8")
        (tmp_path / "second.md").write_text("Synonym: {synonym}", encoding="utf-8")
        bare_model.config.prompt_folder = str(tmp_path)
        bare_model.config.prompt_files = ["first.md", "second.md"]
        return bare_model

    def test_creates_and_fills_one_history_per_dict(self, templated_model):
        values = [{"label": "A", "synonym": "a"}, {"label": "B", "synonym": "b"}]

        assert templated_model.addPromptsFromFile(0, values) == 2
        templated_model.addPrompt(modelRole, ["x", "y"])
        assert templated_model.addPromptsFromFile(1, values) == 2

        assert [templated_model.getPrompt(i, 0) for i in range(2)] == ["Label: A", "Label: B"]
        assert [templated_model.getPrompt(i, 1) for i in range(2)] == ["Synonym: a", "Synonym: b"]

    def test_needs_one_dict_per_history(self, templated_model):
        templated_model.addPromptsFromFile(0, [{"label": "A"}, {"label": "B"}])

        with pytest.raises(ValueError, match="one dict per history"):
            templated_model.addPromptsFromFile(1, [{"synonym": "a"}])

    def test_empty_values_raise(self, templated_model):
        with pytest.raises(ValueError, match="No values"):
            templated_model.addPromptsFromFile(0, [])

    def test_missing_placeholder_raises(self, templated_model):
        with pytest.raises(KeyError):
            templated_model.addPromptsFromFile(0, [{"synonym": "a"}])

    def test_failure_creates_no_histories(self, templated_model):
        # The second dict lacks the placeholder; the first history must not
        # stay behind with its filled template.
        with pytest.raises(KeyError):
            templated_model.addPromptsFromFile(0, [{"label": "A"}, {"synonym": "b"}])

        assert templated_model.getMessageHistories() == []

    def test_failure_leaves_existing_histories_unchanged(self, templated_model):
        templated_model.addPromptsFromFile(0, [{"label": "A"}, {"label": "B"}])
        templated_model.addPrompt(modelRole, ["x", "y"])
        before = json.loads(json.dumps(templated_model.getMessageHistories()))

        with pytest.raises(KeyError):
            templated_model.addPromptsFromFile(1, [{"synonym": "a"}, {"label": "b"}])

        assert templated_model.getMessageHistories() == before

    def test_malformed_template_leaves_the_histories_unchanged(self, templated_model, tmp_path):
        (tmp_path / "broken.md").write_text("Label: {label", encoding="utf-8")
        templated_model.config.prompt_files = ["broken.md"]

        with pytest.raises(ValueError, match="Malformed"):
            templated_model.addPromptsFromFile(0, [{"label": "A"}])

        assert templated_model.getMessageHistories() == []


# --- Model.generate ------------------------------------------------------------------

class TestGenerate:
    """generate(): chunks, skipped prompts and the answer store file."""

    def test_returns_immediately_when_no_prompts(self, bare_model):
        # self.llm is None, so getting past the guard clause would raise
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
        assert os.path.isfile(_checkpoint_path(bare_model))

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

        path = _metadata_path(bare_model)
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
    """Resuming an interrupted generate() and multi-turn conversations."""

    def test_resumes_from_the_checkpoint_after_an_interruption(self, bare_model):
        bare_model.config.chunk_size = 2
        bare_model.addPrompt(userRole, ["a", "b", "c", "d"])
        bare_model.llm = InterruptedLLM()

        with pytest.raises(RuntimeError, match="job killed"):
            bare_model.generate()

        # A new run with the same prompts uses the answers stored after the
        # first chunk ...
        new_run = Model.__new__(Model)
        new_run.__dict__.update(bare_model.__dict__, store=None, messageHistories=[])
        new_run.addPrompt(userRole, ["a", "b", "c", "d"])
        new_run.llm = FakeLLM()

        # ... and only generates the histories without an answer
        assert new_run.generate() == 2
        assert new_run.llm.batches == [2]
        bare_model = new_run
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

    def test_empty_answer_is_generated_again(self, bare_model):
        bare_model.llm = FakeThinkingLLM()
        bare_model.addPrompt(userRole, ["long question", "short"])
        bare_model.generate()
        assert bare_model.messageHistories[0][-1][messageTextElement] == ""

        # E.g. after raising max_tokens: only the empty answer is replaced
        bare_model.llm = FakeLLM()
        assert bare_model.generate() == 1

        assert bare_model.messageHistories[0] == [
            {messageRoleElement: userRole, messageTextElement: "long question"},
            {messageRoleElement: modelRole, messageTextElement: "answer to long question"},
        ]
        assert len(bare_model.messageHistories[1]) == 2

    def test_generate_after_close_raises_a_clear_error(self, bare_model):
        bare_model.addPrompt(userRole, ["q"])
        bare_model.close()

        with pytest.raises(RuntimeError, match="closed"):
            bare_model.generate()
        assert len(bare_model.messageHistories[0]) == 1

    def test_context_check_accepts_a_batch_encoding(self, bare_model):
        bare_model.llm = FakeLLM()
        bare_model.llm.tokenizer = BatchEncodingTokenizer()
        bare_model.config.max_model_len = 10
        bare_model.addPrompt(userRole, ["short", "x" * 10])

        assert bare_model.generate() == 1
        assert bare_model.skippedHistories == [1]


class ScriptedLLM(FakeLLM):
    """FakeLLM answering every prompt with the same (text, finish_reason)."""

    def __init__(self, text: str, finish_reason: str = "stop"):
        super().__init__()
        self.text = text
        self.finish_reason = finish_reason

    def answer(self, history) -> tuple[str, str]:
        return self.text, self.finish_reason


def _record_updates(path: str, worker: int, count: int) -> None:
    """Adds `count` run records for `worker` to the run metadata at `path`."""
    for i in range(count):
        updateRunRecord(path, {"run_id": f"{worker}-{i}"})


class TestConcurrentRuns:
    """
    Runs sharing a checkpoint or run metadata file (two models, or the jobs
    of a Slurm job array started with one config) must not destroy each
    other's work.
    """

    def test_runs_sharing_a_store_keep_each_others_answers(self, bare_model):
        bare_model.llm = FakeLLM()
        other_run = Model.__new__(Model)
        other_run.__dict__.update(bare_model.__dict__, store=None, messageHistories=[])
        other_run.llm = FakeLLM()
        # Both load the (empty) store before either writes it
        bare_model._answerStore()
        other_run._answerStore()

        bare_model.addPrompt(userRole, ["a"])
        bare_model.generate()
        other_run.addPrompt(userRole, ["b"])
        other_run.generate()

        stored = json.loads(_checkpoint_path(bare_model).read_text(encoding="utf-8"))
        assert sorted(node["prompt"] for node in stored["histories"].values()) == ["a", "b"]

    def test_a_store_of_another_model_is_refused(self, bare_model):
        bare_model.llm = FakeLLM()
        bare_model.addPrompt(userRole, ["q"])
        bare_model.generate()
        # Another organisation's model of the same name has the same file name
        other_run = Model.__new__(Model)
        other_run.__dict__.update(bare_model.__dict__, store=None, messageHistories=[], model="other-org/model-a")
        other_run.addPrompt(userRole, ["q"])
        other_run.llm = FakeLLM()

        with pytest.raises(ValueError, match="model: 'dummy/model-a' -> 'other-org/model-a'"):
            other_run.generate()
        # Refused before anything was generated
        assert other_run.llm.batches == []

    @pytest.mark.skipif(
        logger_file_utils.fcntl is None or "fork" not in multiprocessing.get_all_start_methods(),
        reason="needs file locks and fork",
    )
    def test_concurrent_metadata_updates_keep_every_record(self, tmp_path):
        path = str(tmp_path / "run_metadata.json")
        context = multiprocessing.get_context("fork")
        workers = [context.Process(target=_record_updates, args=(path, worker, 25)) for worker in range(4)]
        for process in workers:
            process.start()
        for process in workers:
            process.join()

        runs = json.loads(open(path, encoding="utf-8").read())
        assert len(runs) == 100
        assert all(process.exitcode == 0 for process in workers)


def _stored(model: Model) -> dict:
    """The content of the answer store file of `model`."""
    return json.loads(_checkpoint_path(model).read_text(encoding="utf-8"))


def _new_run(model: Model, **changes) -> Model:
    """A second Model with the settings of `model`, no histories and no loaded store."""
    run = Model.__new__(Model)
    run.__dict__.update(model.__dict__, store=None, messageHistories=[], **changes)
    run.config = model.config.model_copy(deep=True)
    run.llm = FakeLLM()
    return run


class TestAnswerStore:
    """The prompts and answers in the JSON answer store (src/AnswerStore.py)."""

    def test_stores_settings_and_a_tree_by_turn(self, bare_model):
        bare_model.llm = FakeLLM()
        bare_model.addPrompt(systemRole, ["system"])
        bare_model.addPrompt(userRole, ["first"])
        bare_model.generate()
        bare_model.addPrompt(userRole, ["second"])
        bare_model.generate()

        stored = _stored(bare_model)
        assert stored["format"] == 1
        assert stored["settings"]["model"] == "dummy/model-a"
        assert stored["settings"]["seed"] == bare_model.config.seed
        (system,) = stored["histories"].values()
        assert system["role"] == "system" and "answer" not in system
        (first,) = system["next"].values()
        assert first["prompt"] == "first" and first["answer"] == {"text": "answer to first"}
        (second,) = first["next"].values()
        assert second["answer"] == {"text": "answer to second"}
        assert list(system["next"]) == [model_package.AnswerStore.promptKeyOf("first")]

    def test_a_shared_first_turn_is_generated_once(self, bare_model):
        bare_model.llm = FakeLLM()
        bare_model.addPrompt(userRole, ["concept"] * 3)
        bare_model.generate()
        bare_model.addPrompt(userRole, ["synonym a", "synonym b", "synonym c"])
        bare_model.generate()

        assert bare_model.llm.batches == [1, 3]
        assert [bare_model.getAnswer(i, 0) for i in range(3)] == ["answer to concept"] * 3
        (concept,) = _stored(bare_model)["histories"].values()
        assert len(concept["next"]) == 3

    def test_identical_histories_get_the_same_answer(self, bare_model):
        bare_model.llm = FakeLLM()
        bare_model.addPrompt(userRole, ["q", "r", "q"])

        assert bare_model.generate() == 2
        assert [bare_model.getAnswer(i, 0) for i in range(3)] == ["answer to q", "answer to r", "answer to q"]
        assert bare_model.generationStats[-1]["shared"] == 1

    def test_stored_answers_are_used_instead_of_generated(self, bare_model):
        bare_model.llm = FakeLLM()
        bare_model.addPrompt(userRole, ["a", "b"])
        bare_model.generate()

        run = _new_run(bare_model)
        run.addPrompt(userRole, ["b", "c", "a"])

        assert run.generate() == 1
        assert run.llm.batches == [1]
        assert [run.getAnswer(i, 0) for i in range(3)] == ["answer to b", "answer to c", "answer to a"]
        assert run.generationStats[-1]["stored"] == 2

    def test_a_stored_answer_needs_the_same_earlier_answers(self, bare_model):
        bare_model.llm = FakeLLM()
        bare_model.addPrompt(userRole, ["first"])
        bare_model.generate()
        bare_model.addPrompt(userRole, ["second"])
        bare_model.generate()

        # The same prompts, but another answer to the first one
        run = _new_run(bare_model)
        run.addPrompt(userRole, ["first"])
        run.addPrompt(modelRole, ["another answer"])
        run.addPrompt(userRole, ["second"])

        assert run.generate() == 1

    def test_answers_can_be_used_after_close(self, bare_model):
        bare_model.llm = FakeLLM()
        bare_model.addPrompt(userRole, ["q"])
        bare_model.generate()

        run = _new_run(bare_model)
        run.llm = None
        run.addPrompt(userRole, ["q"])

        assert run.generate() == 0
        assert run.getAnswer(0, 0) == "answer to q"

    def test_empty_answers_are_not_stored(self, bare_model):
        bare_model.llm = FakeThinkingLLM()
        bare_model.addPrompt(userRole, ["long question"])
        bare_model.generate()
        assert bare_model.getAnswer(0, 0) == ""

        (node,) = _stored(bare_model)["histories"].values()
        assert "answer" not in node

    def test_skipped_histories_and_their_copies_are_not_stored(self, bare_model):
        bare_model.llm = FakeLLM()
        bare_model.config.max_model_len = 10
        bare_model.addPrompt(userRole, ["x" * 10, "short", "x" * 10])

        assert bare_model.generate() == 1
        assert bare_model.skippedHistories == [0, 2]
        assert len(_stored(bare_model)["histories"]) == 1

    def test_other_settings_are_refused(self, bare_model):
        bare_model.llm = FakeLLM()
        bare_model.addPrompt(userRole, ["q"])
        bare_model.generate()

        run = _new_run(bare_model)
        run.config.temperature = 0.7
        run.config.chunk_size = 1
        run.addPrompt(userRole, ["q"])

        with pytest.raises(ValueError, match="temperature: 0.01 -> 0.7") as error:
            run.generate()
        # Settings that do not change the answers may change
        assert "chunk_size" not in str(error.value)
        assert run.llm.batches == []

    def test_an_unknown_revision_is_no_difference(self, bare_model, monkeypatch):
        bare_model.llm = FakeLLM()
        bare_model.revision = "abc123"
        bare_model.revisionLooked = True
        bare_model.addPrompt(userRole, ["q"])
        bare_model.generate()

        # E.g. the model is not in the local Hugging Face cache
        monkeypatch.setattr(model_module, "modelRevision", lambda model_id, configured=None: None)
        run = _new_run(bare_model, revision=None, revisionLooked=False)
        run.addPrompt(userRole, ["q"])

        assert run.generate() == 0

    def test_histories_are_restored_as_they_were(self, bare_model):
        # Two turns with a system prompt, reasoning, a cut-off (empty) answer
        # and a first turn two histories share
        bare_model.llm = FakeThinkingLLM()
        bare_model.addPrompt(systemRole, ["system"] * 3)
        bare_model.addPrompt(userRole, ["concept", "concept", "other concept"])
        bare_model.generate()
        bare_model.addPrompt(userRole, ["synonym a", "long synonym b", "synonym c"])
        bare_model.generate()
        expected = [[dict(message) for message in history] for history in bare_model.messageHistories]

        run = _new_run(bare_model)
        run.llm = FakeThinkingLLM()
        run.addPrompt(systemRole, ["system"] * 3)
        run.addPrompt(userRole, ["concept", "concept", "other concept"])
        assert run.generate() == 0
        run.addPrompt(userRole, ["synonym a", "long synonym b", "synonym c"])
        # Only the empty answer, which is not stored, is generated again
        assert run.generate() == 1

        assert run.messageHistories == expected
        assert run.messageHistories[0][2][messageReasoningElement].startswith("The user asks concept")
        assert run.truncatedHistories == [1]

    def test_answers_of_another_run_for_a_differing_answer_are_left_out(self, bare_model):
        bare_model.llm = FakeLLM()
        bare_model._answerStore()
        other_run = _new_run(bare_model)
        other_run._answerStore()

        # Both answer "first" differently and go on from there
        bare_model.addPrompt(userRole, ["first"])
        bare_model.addPrompt(modelRole, ["my answer"])
        bare_model.addPrompt(userRole, ["second"])
        bare_model.generate()
        other_run.addPrompt(userRole, ["first"])
        other_run.addPrompt(modelRole, ["other answer"])
        other_run.addPrompt(userRole, ["second"])
        other_run.generate()

        # The run writing last keeps its answer and the turns after it; the
        # turns generated after the other answer are not attached to it
        (first,) = _stored(bare_model)["histories"].values()
        assert first["answer"] == {"text": "other answer"}
        assert len(first["next"]) == 1

        def continue_after(answer: str) -> int:
            run = _new_run(bare_model)
            run.addPrompt(userRole, ["first"])
            run.addPrompt(modelRole, [answer])
            run.addPrompt(userRole, ["second"])
            return run.generate()

        assert continue_after("other answer") == 0
        assert continue_after("my answer") == 1

    def test_two_prompts_in_a_row_do_not_take_answers_after_an_answer(self, bare_model):
        bare_model.llm = FakeLLM()
        bare_model.addPrompt(userRole, ["first"])
        bare_model.generate()
        bare_model.addPrompt(userRole, ["second"])
        bare_model.generate()

        run = _new_run(bare_model)
        run.addPrompt(userRole, ["first"])
        run.addPrompt(userRole, ["second"])

        assert run.generate() == 1
        sent, _ = run.llm.chat_calls[-1]
        assert [message[messageTextElement] for message in sent[0]] == ["first", "second"]

    def test_a_file_that_is_no_answer_store_is_refused(self, bare_model):
        path = _checkpoint_path(bare_model)
        path.parent.mkdir(parents=True)
        path.write_text('{"something": "else"}', encoding="utf-8")
        bare_model.llm = FakeLLM()
        bare_model.addPrompt(userRole, ["q"])

        with pytest.raises(ValueError, match="not an answer store"):
            bare_model.generate()


class TestAnswerStoreMerge:
    """
    _merge() combines the trees of two runs sharing an answer store: turns
    are only merged below equal answers, since they depend on them.
    """

    @staticmethod
    def _node(prompt, answer=None, following=None):
        node = {"role": "user", "prompt": prompt}
        if answer is not None:
            node["answer"] = {"text": answer}
        if following is not None:
            node["next"] = following
        return node

    def test_different_answers_keep_the_target_and_its_turns(self):
        from model.AnswerStore import _merge
        target = {"p": self._node("p", "mine", {"q": self._node("q", "after mine")})}
        source = {"p": self._node("p", "theirs", {"r": self._node("r", "after theirs")})}

        _merge(target, source)

        assert target["p"]["answer"] == {"text": "mine"}
        assert set(target["p"]["next"]) == {"q"}

    def test_a_prompt_without_answer_takes_the_source_answer_and_turns(self):
        from model.AnswerStore import _merge
        target = {"p": self._node("p")}
        source = {"p": self._node("p", "theirs", {"r": self._node("r", "next")})}

        _merge(target, source)

        assert target["p"]["answer"] == {"text": "theirs"}
        assert set(target["p"]["next"]) == {"r"}

    def test_equal_answers_merge_the_following_turns(self):
        from model.AnswerStore import _merge
        target = {"p": self._node("p", "same", {"q": self._node("q", "a")})}
        source = {"p": self._node("p", "same", {"r": self._node("r", "b")})}

        _merge(target, source)

        assert set(target["p"]["next"]) == {"q", "r"}


class TestCutOffAnswers:
    """Answers cut off at max_tokens."""

    def test_cut_off_reasoning_opened_by_the_template_is_no_answer(self, bare_model):
        # DeepSeek-R1-style: the template ends the prompt with <think>, so the
        # cut-off output has no tags and is all reasoning
        bare_model.llm = ScriptedLLM('The user wants {"draft": true} ... still thinking', "length")
        bare_model.llm.tokenizer.generation_prompt = "<think>\n"
        bare_model.addPrompt(userRole, ["q"])

        bare_model.generate()

        answer = bare_model.messageHistories[0][-1]
        assert answer[messageTextElement] == ""
        assert answer[messageReasoningElement].startswith("The user wants")
        assert bare_model.toJSON(0) is None

    def test_finished_output_without_tags_stays_the_answer(self, bare_model):
        bare_model.llm = ScriptedLLM('{"label": "kidney"}')
        bare_model.llm.tokenizer.generation_prompt = "<think>\n"
        bare_model.addPrompt(userRole, ["q"])

        bare_model.generate()

        assert bare_model.toJSON(0) == {"label": "kidney"}

    def test_cut_off_json_does_not_yield_a_fragment(self, bare_model):
        bare_model.llm = ScriptedLLM('[{"label": "kidney"}, {"label": "li', "length")
        bare_model.addPrompt(userRole, ["q"])

        bare_model.generate()

        assert bare_model.messageHistories[0][-1][messageTruncatedElement] is True
        assert bare_model.toJSON(0) is None

    def test_complete_json_before_the_cut_is_parsed(self, bare_model):
        bare_model.llm = ScriptedLLM('{"label": "kidney"} because the kidney is aff', "length")
        bare_model.addPrompt(userRole, ["q"])

        bare_model.generate()

        assert bare_model.toJSON(0) == {"label": "kidney"}

    def test_truncation_mark_survives_the_checkpoint(self, bare_model):
        bare_model.llm = ScriptedLLM('[{"label": "kidney"}, {"label": "li', "length")
        bare_model.addPrompt(userRole, ["q"])
        bare_model.generate()
        expected = [list(history) for history in bare_model.messageHistories]

        bare_model.reset()
        bare_model.store = None
        bare_model.addPrompt(userRole, ["q"])

        assert bare_model.generate() == 0
        assert bare_model.messageHistories == expected
        assert bare_model.truncatedHistories == [0]
        assert bare_model.toJSON(0) is None


# --- Thinking models ------------------------------------------------------------

class TestSplitReasoning:
    """_splitReasoning() for thinking and non-thinking models."""

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
    """generate() with the reasoning of thinking models."""

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

        path = _metadata_path(bare_model)
        stats = json.loads(open(path, encoding="utf-8").read())[0]["generations"][0]
        assert stats["truncated"] == [1]
        # FakeTokenizer.encode: one token per character of both reasonings
        assert stats["reasoning_tokens"] == (
            len('The user asks q1. Draft: {"draft": true}')
            + len('Still reasoning about {"draft": true}')
        )

    def test_checkpoint_round_trip_keeps_reasoning(self, tmp_path, bare_model):
        bare_model.llm = FakeThinkingLLM()
        bare_model.addPrompt(userRole, ["q1", "q2"])
        bare_model.generate()
        expected = [list(history) for history in bare_model.messageHistories]

        bare_model.reset()
        bare_model.store = None
        bare_model.addPrompt(userRole, ["q1", "q2"])
        bare_model.generate()

        assert bare_model.messageHistories == expected

    def test_answer_without_reasoning_stores_none(self, tmp_path, bare_model):
        bare_model.llm = FakeLLM()
        bare_model.addPrompt(userRole, ["q1"])
        bare_model.generate()

        stored = json.loads(_checkpoint_path(bare_model).read_text(encoding="utf-8"))
        (node,) = stored["histories"].values()
        assert node["answer"] == {"text": "answer to q1"}


# --- Model._writeRunMetadata ----------------------------------------------------------

class TestRunMetadata:
    """The run metadata file written by _writeRunMetadata()."""

    def test_records_prompt_hashes_and_package_versions(self, tmp_path, bare_model):
        (tmp_path / "prompt_a.md").write_text("Hello", encoding="utf-8")
        bare_model.config.prompt_folder = str(tmp_path)
        bare_model.config.prompt_files = ["prompt_a.md", "missing.md"]

        bare_model._writeRunMetadata()

        path = _metadata_path(bare_model)
        run = json.loads(open(path, encoding="utf-8").read())[0]
        assert run["prompt_files"] == {
            # sha256("Hello")
            "prompt_a.md": "185f8db32271fe25f561a6fc938b2e264306ec304eda518007d1764826381969",
            "missing.md": None,
        }
        assert run["packages"]["model"] is not None
        assert run["model"] == {"id": "dummy/model-a", "index": 0, "revision": None}

    def test_unreadable_gpu_name_is_null_and_logged(self, bare_model, monkeypatch, logged):
        def broken(index):
            raise RuntimeError("NVML not available")
        monkeypatch.setattr(torch.cuda, "get_device_name", broken, raising=False)
        bare_model.gpuCount = 1

        bare_model._writeRunMetadata()

        path = _metadata_path(bare_model)
        assert json.loads(open(path, encoding="utf-8").read())[0]["gpus"]["names"] == [None]
        assert any("name of GPU 0" in message for message in logged)

    def test_keeps_other_runs_and_replaces_its_own(self, bare_model):
        path = _metadata_path(bare_model)
        os.makedirs(bare_model.config.metadata_folder)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([{"run_id": "earlier-run"}], f)

        bare_model._writeRunMetadata()
        bare_model._writeRunMetadata()

        runs = json.loads(open(path, encoding="utf-8").read())
        assert [run["run_id"] for run in runs] == ["earlier-run", "test-run"]

    def test_rejects_a_file_that_is_not_a_list(self, bare_model):
        path = _metadata_path(bare_model)
        os.makedirs(bare_model.config.metadata_folder)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"run_id": "x"}, f)

        with pytest.raises(ValueError):
            bare_model._writeRunMetadata()


# --- Model.logPrompts --------------------------------------------------------------

class TestLogPrompts:
    """Writing the prompt log with logPrompts()."""

    def test_writes_log_file_at_configured_path(self, tmp_path, bare_model):
        bare_model.config.prompt_log_folder = str(tmp_path) + os.sep
        bare_model.config.prompt_log_file = "prompts.log"
        bare_model.addPrompt(userRole, ["Hello"])

        bare_model.logPrompts()

        log_path = tmp_path / "prompts_model-a.log"
        assert log_path.is_file()
        assert "Hello" in log_path.read_text(encoding="utf-8")


# --- Model.close / context manager -------------------------------------------------

class TestClose:
    """close(), the context manager and __del__."""

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
        monkeypatch.setattr(torch.distributed, "is_available", lambda: True, raising=False)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True, raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "empty_cache", step("cache"), raising=False)
        return calls

    def test_close_releases_the_engine_and_gpu_memory(self, bare_model, monkeypatch):
        calls = self._record_cleanup(monkeypatch)
        bare_model.llm = object()

        bare_model.close()

        assert bare_model.llm is None
        assert calls == ["parallel", "env", "group", "cache"]

    def test_close_continues_after_a_failing_cleanup_step_and_logs_it(
        self, bare_model, monkeypatch, logged
    ):
        calls = self._record_cleanup(monkeypatch, failing="parallel")

        bare_model.close()

        assert calls == ["parallel", "env", "group", "cache"]
        assert any("destroy_model_parallel failed" in message for message in logged)

    def test_close_skips_a_process_group_that_does_not_exist(self, bare_model, monkeypatch, logged):
        # Destroying a missing process group raises; it must not be tried
        calls = self._record_cleanup(monkeypatch)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False, raising=False)

        bare_model.close()

        assert calls == ["parallel", "env", "cache"]
        assert not any("failed" in message for message in logged)


# --- ModelUtils.writePrompt ---------------------------------------------------------

class TestWritePrompt:
    """The semicolon-delimited prompt log of writePrompt()."""

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


# --- ModelUtils: atomic writes, run metadata helpers and errors ----------------------

class TestModelUtils:
    """Helpers of ModelUtils: writeJSON(), packageVersions(), modelRevision(), describeError()."""

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

    def test_model_revision_without_a_cached_snapshot_is_none_and_logged(self, monkeypatch, logged):
        def not_cached(model_id, local_files_only):
            raise FileNotFoundError(model_id)
        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = not_cached
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

        assert modelRevision("org/name") is None
        assert any("revision of model 'org/name'" in message for message in logged)

    def test_model_revision_is_the_configured_one(self):
        assert modelRevision("org/name", "def456") == "def456"

    def test_model_revision_of_an_incomplete_snapshot_is_the_cached_commit(
        self, tmp_path, monkeypatch, logged
    ):
        def incomplete(model_id, local_files_only):
            raise RuntimeError("The cached snapshot for 'org/name' (revision 'main') is incomplete.")
        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = incomplete
        constants = types.ModuleType("huggingface_hub.constants")
        constants.HF_HUB_CACHE = str(tmp_path)
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
        monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
        refs = tmp_path / "models--org--name" / "refs"
        refs.mkdir(parents=True)
        (refs / "main").write_text("abc123\n")

        assert modelRevision("org/name") == "abc123"
        assert not any("revision of model" in message for message in logged)

    def test_a_cut_error_leaves_no_parenthesis_open(self):
        error = RuntimeError("The cached snapshot for 'google/medgemma-4b-it' (revision 'main') "
                             "misses files that the model needs to be loaded by vLLM.")
        described = describeError(error, width=70)
        assert len(described) <= 70
        assert described.endswith(" (...)")
        assert described.count("(") == described.count(")")


# --- Log messages ----------------------------------------------------------------------

def _assert_task_structure(messages: list[str], tmp_path=None) -> None:
    """
    Checks the structure of logged messages: a task starts as "<Task>..."
    and ends as exactly "<Task> completed." (tasks may nest), every other
    message is one sentence ending in "." (not "..."), results are rows of
    their own instead of being folded into the completion row, and no
    message is longer than 100 characters (a temporary path counts as one
    short word).
    """
    open_tasks = []
    for message in messages:
        short = message.replace(str(tmp_path), "<tmp>") if tmp_path else message
        assert len(short) <= 100, message
        if message.endswith("..."):
            open_tasks.append(message[:-len("...")])
        elif open_tasks and message == f"{open_tasks[-1]} completed.":
            open_tasks.pop()
        else:
            assert message.endswith("."), message
            assert " completed" not in message, message
    assert open_tasks == [], f"tasks without a completion row: {open_tasks}"


class TestLogMessages:
    """Every task logged as "<Task>..." ends with "<Task> completed."."""

    def test_loading_the_model(self, tmp_path, model_config_data, monkeypatch, logged):
        model_config_data.update({"metadata_folder": str(tmp_path / "logs"), "device": "cpu"})
        _fake_engine(monkeypatch)

        Model(config=model_config_data)

        assert "Loading model at index 0: 'dummy/model-a' completed." in logged
        _assert_task_structure(logged, tmp_path)

    def test_adding_a_prompt_from_file(self, tmp_path, bare_model, logged):
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        (prompt_dir / "prompt_a.md").write_text("Hello", encoding="utf-8")
        bare_model.config.prompt_folder = str(prompt_dir)
        bare_model.config.prompt_files = ["prompt_a.md"]

        bare_model.addPromptFromFile(index=0, amount=2)

        assert logged[-1] == "Loading prompt from file 'prompt_a.md' completed."
        _assert_task_structure(logged, tmp_path)

    def test_generating_in_chunks_with_skipped_shared_and_stored_answers(self, tmp_path, bare_model, logged):
        bare_model.llm = FakeLLM()
        bare_model.config.chunk_size = 2
        bare_model.config.max_model_len = 10
        bare_model.addPrompt(userRole, ["a", "b"])
        bare_model.generate()

        run = _new_run(bare_model)
        run.addPrompt(userRole, ["a", "c", "d", "c", "z" * 20])
        run.generate()

        assert "Generating answers for 5 histories completed." in logged
        _assert_task_structure(logged, tmp_path)

    def test_generating_only_stored_answers(self, tmp_path, bare_model, logged):
        bare_model.llm = FakeLLM()
        bare_model.addPrompt(userRole, ["a"])
        bare_model.generate()

        run = _new_run(bare_model)
        run.addPrompt(userRole, ["a"])
        run.generate()

        assert logged[-1] == "Generating answers for 1 histories completed."
        _assert_task_structure(logged, tmp_path)


# --- Package exports -------------------------------------------------------------------

class TestPackageExports:
    """What the model package exports."""

    def test_import_does_not_change_the_bytecode_setting(self, monkeypatch):
        """Importing the package must not switch off .pyc files for the whole process."""
        monkeypatch.setattr(sys, "dont_write_bytecode", False)
        importlib.reload(model_package)
        assert sys.dont_write_bytecode is False

    def test_every_exported_name_is_importable(self):
        assert set(model_package.__all__) == {
            "Model", "ModelConfig", "messageRoleElement", "messageTextElement",
            "messageReasoningElement", "messageTruncatedElement", "gpu", "cpu",
        }
        for name in model_package.__all__:
            assert getattr(model_package, name) is not None

