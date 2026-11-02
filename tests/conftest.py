"""
Pytest configuration, loaded before the test modules are imported.

The tests never load a real model: everything that needs a GPU either
bypasses Model.__init__ or replaces torch.cuda and the vLLM classes via
monkeypatch. vllm and torch are therefore only needed because Model.py
imports them. If they are not installed (e.g. in CI or on a laptop without
a GPU), minimal stand-ins are registered here so the package can still be
imported. When the real packages are installed, they are used unchanged.

The Logger writes to a temporary folder during the tests, so they leave no
log in the repository; tests that read repository files use REPO_ROOT, so
they do not depend on the working directory.
"""

import importlib.util
import pathlib
import sys

import pytest
from logger import Logger, LoggerConfig

# Root of the repository, which holds config/ and prompts/
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
import types


def _is_installed(name: str) -> bool:
    """Returns whether the top-level package `name` can be imported."""
    return importlib.util.find_spec(name) is not None


def _not_installed(package: str):
    """
    Returns a stand-in for a class of `package` that fails loudly when
    instantiated, so a test that accidentally needs the real package does
    not pass silently.
    """
    def fail(*args, **kwargs):
        raise RuntimeError(f"'{package}' is not installed; this test needs the real package.")
    return fail


if not _is_installed("torch"):
    torch = types.ModuleType("torch")
    # No GPU: Model.__init__ raises unless a test monkeypatches these.
    torch.cuda = types.SimpleNamespace(
        device_count=lambda: 0,
        is_available=lambda: False,
        empty_cache=lambda: None,
    )
    torch.distributed = types.SimpleNamespace(
        is_available=lambda: False,
        is_initialized=lambda: False,
        destroy_process_group=lambda: None,
    )
    sys.modules["torch"] = torch

if not _is_installed("vllm"):
    vllm = types.ModuleType("vllm")
    vllm.LLM = _not_installed("vllm")
    vllm.SamplingParams = _not_installed("vllm")

    vllm_distributed = types.ModuleType("vllm.distributed")
    vllm_distributed.destroy_distributed_environment = lambda: None
    vllm_distributed.destroy_model_parallel = lambda: None

    vllm.distributed = vllm_distributed
    sys.modules["vllm"] = vllm
    sys.modules["vllm.distributed"] = vllm_distributed


@pytest.fixture(scope="session", autouse=True)
def _log_to_temporary_folder(tmp_path_factory):
    """Points the process-wide Logger at a temporary folder for all tests."""
    Logger.config = LoggerConfig(folder=str(tmp_path_factory.mktemp("logs")))
