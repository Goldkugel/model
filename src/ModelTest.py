import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

from unittest.mock import MagicMock, patch
import os
import types
import yaml
import pytest

# --- Import-time safety ----------------------------------------------------
# Model.py, at module import time, does:
#   from vllm import LLM, SamplingParams
#   from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
#   gpus = int(torch.cuda.device_count())
#   if gpus > 0: os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([i for i in range(1, gpus + 1)])
#
# The last line is a known bug (str.join() on a list of ints) that crashes
# the import outright on any machine with a visible GPU. To keep this test
# suite runnable regardless of environment - no real vLLM package required,
# no GPU required - vllm is stubbed out and torch.cuda.device_count() is
# forced to 0 before Model is imported, so that branch is never entered.
if "vllm" not in sys.modules:
    vllm_stub = types.ModuleType("vllm")
    vllm_stub.LLM = MagicMock(name="LLM")
    vllm_stub.SamplingParams = MagicMock(name="SamplingParams")
    sys.modules["vllm"] = vllm_stub

    vllm_distributed_stub = types.ModuleType("vllm.distributed")
    vllm_distributed_stub.destroy_distributed_environment = MagicMock(
        name="destroy_distributed_environment"
    )
    vllm_distributed_stub.destroy_model_parallel = MagicMock(
        name="destroy_model_parallel"
    )
    sys.modules["vllm.distributed"] = vllm_distributed_stub

import torch
_original_device_count = torch.cuda.device_count
torch.cuda.device_count = lambda: 0

from .Model import Model as ModelClass
from .ModelUtils import *

torch.cuda.device_count = _original_device_count


@pytest.fixture(autouse=True)
def _reset_shared_class_state():
    """
    messageHistories (and llm/sampling_params/model) are defined as
    class-level attributes on Model rather than being set in __init__,
    so instances can share mutable state across tests unless reset here.
    This fixture exists purely to keep this test suite isolated; it's
    also a symptom of the same bug flagged for the class itself (see
    test_message_histories_are_shared_across_instances_without_reset).
    """
    ModelClass.messageHistories = []
    yield
    ModelClass.messageHistories = []


@pytest.fixture
def config_path(tmp_path):
    """
    Write a minimal, valid 'llm' section config YAML to a temp file and
    return its path, so Model.__init__ can load it without touching the
    real project config.
    """
    config = {
        "llm": {
            "model_id": ["fake/test-model"],
        }
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(config, f)
    return str(path)


class TestModelInit:

    def test_init_loads_config_from_yaml(self, config_path):
        model = ModelClass(config=config_path)
        assert model.config.model_id == ["fake/test-model"]

    def test_init_with_empty_model_id_list_does_not_attempt_to_load(self, tmp_path):
        config = {"llm": {"model_id": []}}
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(config, f)
        model = ModelClass(config=str(path))
        assert model.llm is None


class TestAddPrompt:

    def test_broadcasts_single_message_to_all_existing_histories(self, config_path):
        model = ModelClass(config=config_path)
        model.messageHistories = [[], []]

        count = model.addPrompt(role=userRole, message=["hi"])

        assert count == 2
        assert all(
            h[-1] == {messageRoleElement: userRole, messageTextElement: "hi"}
            for h in model.messageHistories
        )

    def test_appends_one_message_per_existing_history(self, config_path):
        model = ModelClass(config=config_path)
        model.messageHistories = [[], []]

        count = model.addPrompt(role=userRole, message=["first", "second"])

        assert count == 2
        assert model.messageHistories[0][-1][messageTextElement] == "first"
        assert model.messageHistories[1][-1][messageTextElement] == "second"

    def test_initializes_new_histories_when_none_exist(self, config_path):
        model = ModelClass(config=config_path)
        model.messageHistories = []

        count = model.addPrompt(role=userRole, message=["a", "b", "c"])

        assert count == 3
        assert len(model.messageHistories) == 3
        assert model.messageHistories[2][0][messageTextElement] == "c"

    def test_returns_zero_when_message_count_matches_no_case(self, config_path):
        # 2 messages, but 3 existing histories: not a single-message
        # broadcast (len(message) != 1), not one-per-history
        # (len(message) != len(histories)), and histories already exist
        # (so "initialize new histories" doesn't apply either).
        model = ModelClass(config=config_path)
        model.messageHistories = [[], [], []]

        count = model.addPrompt(role=userRole, message=["one", "two"])

        assert count == 0
        assert model.messageHistories == [[], [], []]

class TestResetAndGetters:

    def test_reset_clears_histories(self, config_path):
        model = ModelClass(config=config_path)
        model.messageHistories = [
            [{messageRoleElement: userRole, messageTextElement: "hi"}]
        ]

        model.reset()

        assert model.getMessageHistories() == []

    def test_get_message_histories_returns_current_histories(self, config_path):
        model = ModelClass(config=config_path)
        history = [{messageRoleElement: userRole, messageTextElement: "hi"}]
        model.messageHistories = [history]

        assert model.getMessageHistories() == [history]


class TestGenerate:

    def test_builds_gemma_style_prompt_and_appends_response(self, config_path):
        model = ModelClass(config=config_path)
        model.messageHistories = [
            [{messageRoleElement: userRole, messageTextElement: "What is HPO?"}]
        ]
        sampling_params = MagicMock(name="sampling_params")
        model.sampling_params = sampling_params

        fake_output = MagicMock()
        fake_output.outputs = [
            MagicMock(text="  HPO stands for Human Phenotype Ontology.  ")
        ]
        model.llm = MagicMock()
        model.llm.generate.return_value = [fake_output]

        model.generate()

        expected_prompt = f"{startTurn}{userRole}\nWhat is HPO?{endTurn}\n{startTurn}{modelRole}"
        model.llm.generate.assert_called_once_with(
            [expected_prompt], sampling_params, use_tqdm=False
        )
        assert model.messageHistories[0][-1] == {
            messageRoleElement: modelRole,
            messageTextElement: "HPO stands for Human Phenotype Ontology.",
        }

    def test_folds_a_system_message_into_the_following_user_turn(self, config_path):
        model = ModelClass(config=config_path)
        model.messageHistories = [
            [
                {messageRoleElement: systemRole, messageTextElement: "Be concise."},
                {messageRoleElement: userRole, messageTextElement: "What is HPO?"},
            ]
        ]
        model.sampling_params = MagicMock()
        fake_output = MagicMock()
        fake_output.outputs = [MagicMock(text="short answer")]
        model.llm = MagicMock()
        model.llm.generate.return_value = [fake_output]

        model.generate()

        expected_prompt = (
            f"{startTurn}{userRole}\nBe concise.\n\n"
            f"What is HPO?{endTurn}\n"
            f"{startTurn}{modelRole}"
        )
        called_prompts = model.llm.generate.call_args.args[0]
        assert called_prompts == [expected_prompt]

    def test_builds_one_prompt_per_history_independently(self, config_path):
        model = ModelClass(config=config_path)
        model.messageHistories = [
            [{messageRoleElement: userRole, messageTextElement: "History A"}],
            [{messageRoleElement: userRole, messageTextElement: "History B"}],
        ]
        model.sampling_params = MagicMock()
        model.llm = MagicMock()
        model.llm.generate.return_value = [
            MagicMock(outputs=[MagicMock(text="response A")]),
            MagicMock(outputs=[MagicMock(text="response B")]),
        ]

        model.generate()

        called_prompts = model.llm.generate.call_args.args[0]
        assert len(called_prompts) == 2
        assert "History A" in called_prompts[0]
        assert "History B" in called_prompts[1]
        assert model.messageHistories[0][-1][messageTextElement] == "response A"
        assert model.messageHistories[1][-1][messageTextElement] == "response B"

class TestLogPrompts:

    def test_logs_to_configured_path_with_stored_histories(self, config_path):
        model = ModelClass(config=config_path)
        model.messageHistories = [
            [{messageRoleElement: userRole, messageTextElement: "hi"}]
        ]

        with patch("Model.writePrompt") as mock_write:
            model.logPrompts()

        expected_path = os.path.join(
            model.config.prompt_log_folder, model.config.prompt_log_file
        )
        mock_write.assert_called_once_with(expected_path, model.messageHistories)


class TestDel:

    def test_del_does_not_raise_even_when_every_cleanup_step_fails(self, config_path):
        model = ModelClass(config=config_path)

        with patch("Model.destroy_model_parallel", side_effect=RuntimeError("no group")), \
             patch("Model.destroy_distributed_environment", side_effect=RuntimeError("no group")), \
             patch("torch.distributed.destroy_process_group", side_effect=AssertionError()), \
             patch("torch.cuda.empty_cache", side_effect=RuntimeError("no cuda")):
            model.__del__()  # should not raise, despite every step failing


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))