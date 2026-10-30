import sys
# Prevent Python from writing .pyc bytecode cache files to keep container environments clean
sys.dont_write_bytecode = True

from typing             import Optional
from vllm               import LLM, SamplingParams
from vllm.distributed   import destroy_distributed_environment, destroy_model_parallel
from .ModelConfig       import ModelConfig
from .ModelUtils        import (
    systemRole,
    userRole,
    modelRole,
    chatRoles,
    cpu,
    messageRoleElement,
    messageTextElement,
    messageReasoningElement,
    systemColumn,
    reasoningColumnSuffix,
    writePrompt,
    writeCheckpointCSV,
    writeJSON,
    metadataPackages,
    packageVersions,
    modelRevision,
    fileSHA256,
    target_device,
)
from logger             import Logger, LoggerConfig
from pathlib            import Path
from datetime           import datetime
import pandas           as pd
import logging
import os
import platform
import socket
import time
import uuid
import contextlib
import gc
import torch
import yaml
import json
import re
import csv

# Suppress verbose vLLM initialization and internal status logs
logging.getLogger("vllm").setLevel(logging.ERROR)

# Ensure CUDA devices are enumerated by PCI bus ID
# This guarantees consistent GPU ordering across runs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Tell vLLM to use its CUDA backend (CPU execution is not supported yet).
# This is also vLLM's default; setting it overrides a "cpu" inherited from
# the environment, which would select vLLM's CPU backend.
os.environ[target_device] = "cuda"

# CUDA_VISIBLE_DEVICES is deliberately left alone: it selects the GPUs (e.g.
# set by Slurm), and vLLM uses all of them via tensor_parallel_size, which
# __init__ takes from torch.cuda.device_count() within that selection.

# Global configuration constants
# YAML key where LLM config options are stored
configuration_section: str = "llm"       
# YAML key of the logger settings in the same file (see the logger package)
logger_section: str = "logger"
# Default path for the configuration file
standard_directory: str = "./config/config.yaml"  

def _duration(seconds: float) -> str:
    """Formats a number of seconds as H:MM:SS."""
    seconds = int(round(seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _rate(tokens: int, seconds: float) -> str:
    """Formats tokens per second, or "-" if no time has passed."""
    return f"{tokens / seconds:.0f}" if seconds > 0 else "-"


class Model:
    """
    Wrapper class around the vLLM engine to manage multi-turn chat sessions,
    batch prompt formatting, text generation, and GPU resource cleanup.
    """

    def __init__(
        self,
        config: str | os.PathLike | dict | ModelConfig = standard_directory,
        index: int = 0,
    ):
        """
        Load and validate configuration, select the model at `index`, and
        initialize the vLLM engine on GPU.

        :param config: Where the settings come from:
            - a path to a YAML configuration file with an "llm" section
              (default: ./config/config.yaml relative to the working
              directory);
            - a dict, either the content of such a file (with "llm" and
              optionally "logger" keys) or just the llm settings themselves;
            - a ready ModelConfig (a copy is stored).
            If the Logger has not been configured yet, it is configured from
            the same source: the "logger" section of the file, or the
            "logger" key of a full configuration dict.
        :param index: Index into config.model_id selecting which configured
            model to load.
        :raises FileNotFoundError: If the config file does not exist.
        :raises KeyError: If the config file has no "llm" section.
        :raises pydantic.ValidationError: If the llm settings contain unknown
            keys or invalid values.
        :raises ValueError: If `index` is out of range for config.model_id.
        :raises RuntimeError: If no CUDA device is available or visible to
            PyTorch. This also applies when config.device is "cpu", since
            CPU execution is not supported yet.
        """
        self.messageHistories: list[list[dict]] = []
        self.llm: Optional[LLM] = None
        self.sampling_params: Optional[SamplingParams] = None
        self.model: Optional[str] = None
        self.config: Optional[ModelConfig] = None
        # Indices of the histories the last generate() call skipped because
        # their prompt does not fit into config.max_model_len
        self.skippedHistories: list[int] = []
        # Indices of the histories whose answer the last generate() call cut
        # off at config.max_tokens (finish_reason "length")
        self.truncatedHistories: list[int] = []
        # Run metadata (see _writeRunMetadata()): one statistics record per
        # generate() call, under an id unique to this instance
        self.runId: str = uuid.uuid4().hex
        self.runStarted: str = datetime.now().astimezone().isoformat(timespec="seconds")
        self.modelIndex: int = index
        self.gpuCount: int = 0
        self.generationStats: list[dict] = []

        # --- Step 1: Load and Validate Configuration ---
        self.config, logger_path, logger_settings = Model._resolveConfig(config)

        # The Logger is a process-wide singleton. If the entry point has not
        # configured it yet, configure it from the same source as the model:
        # the same YAML file, or the "logger" key of the same dict. Without
        # either, Logger() falls back to ./config/config.yaml.
        if Logger.config is None and logger_path is not None:
            l = Logger(logger_path)
        else:
            if Logger.config is None and logger_settings is not None:
                Logger.config = LoggerConfig.model_validate(logger_settings)
            l = Logger()

        # --- Step 2: Validate Model Selection Index ---
        if not (0 <= index < len(self.config.model_id)):
            l.log(f"No model specified or index {index} out of range.")
            raise ValueError(
                f"Index {index} out of range for config.model_id "
                f"({len(self.config.model_id)} entries)."
            )

        model = self.config.model_id[index]
        self.model = model
        l.log(f"Loading model at index {index}: '{model}'...")

        device = self.config.device
        if device == cpu:
            l.log("WARNING: device 'cpu' is not supported yet; the model is loaded on the GPU.")
        l.log(f"Scanning device '{device}'...")

        # --- Step 3: Check CUDA Availability ---
        # Fail before vLLM is started, which would otherwise abort with a
        # far less readable error.
        cuda_ok = torch.cuda.is_available()
        l.log(f"Cuda available: {cuda_ok}.")

        if not cuda_ok:
            l.log("No GPU available.")
            raise RuntimeError("No GPU available: CUDA is not accessible to PyTorch.")

        gpus = torch.cuda.device_count()
        self.gpuCount = gpus
        l.log(f"GPU Amount: {gpus}.")

        if gpus == 0:
            l.log("No GPU visible.")
            raise RuntimeError("No GPU visible despite CUDA being reported as available.")

        l.log(f"Scanning device '{device}' completed.")
        
        # --- Step 4: Configure vLLM Sampling Parameters ---
        # The per-request seed makes each answer reproducible regardless of
        # how requests are batched.
        self.sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            seed=self.config.seed
        )

        # --- Step 5: Initialize the vLLM Engine ---
        l.log("Loading model...")
        
        self.llm = LLM(
            model=model,
            # Shard the model across all visible GPUs
            tensor_parallel_size=gpus,
            # True skips CUDA graph capture: slower generation, but faster
            # start-up and less GPU memory
            enforce_eager=self.config.enforce_eager,
            max_model_len=self.config.max_model_len,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            seed=self.config.seed,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            dtype=self.config.dtype,
            quantization=self.config.quantization,
            enable_prefix_caching=self.config.enable_prefix_caching,
            # Mixture-of-Experts models (configured via
            # config.expert_parallel_models) need expert parallelism to
            # shard their experts across the available GPUs; dense models
            # ignore this flag.
            enable_expert_parallel=any(
                pattern in model for pattern in self.config.expert_parallel_models
            ),
        )

        l.log(f"Loading model into {device} completed.")

        # --- Step 6: Record versions, model revision and GPUs ---
        # If this fails (e.g. a corrupt metadata file), the caller gets no
        # Model to close(), so release the engine here before re-raising.
        try:
            self._writeRunMetadata()
        except Exception:
            l.log("Writing the run metadata failed; releasing the loaded model.")
            self.close()
            raise

    @staticmethod
    def _resolveConfig(
        config: str | os.PathLike | dict | ModelConfig,
    ) -> tuple[ModelConfig, Optional[str], Optional[dict]]:
        """
        Turn the `config` argument of __init__ into a ModelConfig.

        Returns the ModelConfig, the YAML file it came from if that file has
        a "logger" section (or None), and the "logger" section of a full
        configuration dict (or None), so __init__ can configure the Logger
        from the same source.

        :raises FileNotFoundError: If the config file does not exist.
        :raises KeyError: If the config file has no "llm" section.
        """
        # Copy a given ModelConfig so later changes by the caller do not
        # silently change the model's configuration.
        if isinstance(config, ModelConfig):
            return config.model_copy(deep=True), None, None

        if isinstance(config, dict):
            # A full configuration (the parsed YAML file) has an "llm" key,
            # which is not a valid llm setting itself, so the two kinds of
            # dict cannot be confused.
            if configuration_section in config:
                section = config[configuration_section] or {}
                return ModelConfig.model_validate(section), None, config.get(logger_section)
            return ModelConfig.model_validate(config), None, None

        path = os.fspath(config)
        if not Path(path).is_file():
            Logger().log(f"Config file not found at '{path}'")
            raise FileNotFoundError(f"Config file not found at '{path}'")

        # An empty file parses to None, so fall back to an empty dict to get
        # the clearer "missing section" error below instead of a TypeError.
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if configuration_section not in data:
            raise KeyError(f"Config file '{path}' has no '{configuration_section}' section.")

        # Only hand the path on to the Logger if the file has logger settings.
        logger_path = path if logger_section in data else None
        return ModelConfig.model_validate(data[configuration_section] or {}), logger_path, None

    def addPrompt(self, role: str = userRole, message: Optional[list[str]] = None) -> int:
        """
        Appends messages with the given role to the conversation histories.

        Handles three input cases:
        1. Single message broadcast: Appends 1 message to ALL active histories.
        2. Multi-message alignment: Appends message[i] to active history[i].
        3. Initialization: Creates N new histories if none currently exist.

        :param role: The sender role (e.g., userRole, modelRole, systemRole).
        :param message: A list of message strings to add. None or an empty
            list adds nothing.
        :return: Total number of active message histories.
        :raises TypeError: If `message` is a single string instead of a list.
        :raises ValueError: If `role` is not systemRole, userRole or
            modelRole; if the number of messages neither is 1, nor matches
            the number of histories, nor initializes empty histories; or if
            a history the messages would be added to ends with an empty
            answer (see generate()). Nothing is added in these cases.
        """
        l = Logger()

        if message is None:
            message = []

        # A string is a sequence too: without this check, "hello" would be
        # added as the five messages "h", "e", "l", "l", "o".
        if isinstance(message, str):
            l.log("addPrompt: message is a string, not a list of strings.")
            raise TypeError("addPrompt: 'message' must be a list of strings, not a string.")

        # Other roles (e.g. Gemini's "model") are not understood by the chat
        # templates, by generate() or by the checkpoint, which would store
        # them as user prompts.
        if message and role not in chatRoles:
            error = f"addPrompt: unknown role '{role}'; expected one of {list(chatRoles)}."
            l.log(error)
            raise ValueError(error)

        # An empty answer (a thinking model cut off during its reasoning)
        # must be generated again before the next turn: the checkpoint does
        # not store empty answers, so a resumed run would see two prompts in
        # a row and never answer the first one.
        if message and self.messageHistories:
            empty = [
                idx for idx, history in enumerate(self.messageHistories)
                if Model._endsWithEmptyAnswer(history)
            ]
            if empty:
                error = (
                    f"addPrompt: histories {empty} end with an empty answer; "
                    f"call generate() again (with a larger max_tokens) first."
                )
                l.log(error)
                raise ValueError(error)

        # Case 1: Broadcast a single prompt to every existing conversation
        # thread
        if len(message) == 1 and len(self.messageHistories) > 0:
            for history in self.messageHistories:
                history.append({
                    messageRoleElement: role,
                    messageTextElement: message[0]
                })

        # Case 2: Append one distinct message per existing conversation thread
        elif (len(message) == len(self.messageHistories) and
            len(self.messageHistories) > 0):
            for idx, history in enumerate(self.messageHistories):
                history.append({
                    messageRoleElement: role,
                    messageTextElement: message[idx]
                })

        # Case 3: Initialize new conversation threads when history is
        # currently empty
        elif len(self.messageHistories) == 0 and len(message) > 0:
            for msg in message:
                self.messageHistories.append([{
                    messageRoleElement: role,
                    messageTextElement: msg
                }])

        # Invalid combination: adding the messages anyway would misalign the
        # conversation threads.
        elif len(message) > 0:
            error = (
                f"addPrompt: message count ({len(message)}) does not match "
                f"active history count ({len(self.messageHistories)})."
            )
            l.log(error)
            raise ValueError(error)

        return len(self.messageHistories)

    @staticmethod
    def _endsWithEmptyAnswer(history: list[dict]) -> bool:
        """
        Returns whether `history` ends with an empty model answer, i.e. one
        of a thinking model cut off during its reasoning.
        """
        return (
            bool(history)
            and history[-1].get(messageRoleElement) == modelRole
            and not str(history[-1].get(messageTextElement) or "").strip()
        )

    def addPromptFromFile(self, index: int, amount: int = -1) -> int:
        """
        Reads a prompt template from a file on disk and adds it to the message
        histories. The template is always added with the user role and with
        its {placeholders} unfilled; see formatPromptFromFile().

        :param index: Index of the target filename in config.prompt_files.
        :param amount: Number of duplicate conversation threads to create if
            histories are empty. Ignored otherwise, as the prompt is then
            broadcast to every existing history.
        :return: Number of active message histories.
        :raises IndexError: If `index` is out of range for
            config.prompt_files.
        :raises ValueError: If config.prompt_folder is empty, or if the
            histories are empty and `amount` is not positive.
        :raises FileNotFoundError: If the prompt file does not exist.
        """
        l = Logger()

        # Guard Clause: Validate prompt file index bounds
        if not (0 <= index < len(self.config.prompt_files)):
            l.log(f"Invalid prompt index: {index}")
            raise IndexError(
                f"Prompt index {index} out of range for config.prompt_files "
                f"({len(self.config.prompt_files)} entries)."
            )

        # Guard Clause: Check if prompt folder path is specified
        if not self.config.prompt_folder:
            l.log("No prompt folder specified in config.")
            raise ValueError("No prompt folder specified in config.")

        file_name = self.config.prompt_files[index]
        file_path = Path(self.config.prompt_folder) / file_name

        # Guard Clause: Ensure prompt file exists on disk
        if not file_path.is_file():
            l.log(f"File '{file_name}' not found at {file_path}.")
            raise FileNotFoundError(f"Prompt file '{file_name}' not found at {file_path}.")

        # Guard Clause: Without histories, `amount` says how many to create
        if len(self.messageHistories) == 0 and amount <= 0:
            l.log("Histories empty and valid 'amount' not specified.")
            raise ValueError("Histories are empty: 'amount' must be positive.")

        l.log(f"Loading prompt from file '{file_name}'...")
        prompt = file_path.read_text(encoding="utf-8")
        l.log(f"Loaded prompt with base length {len(prompt)}.")

        # Determine how to append/multiply the loaded prompt
        if len(self.messageHistories) > 0:
            # If histories already exist, broadcast this prompt as a 
            # single item
            prompts = [prompt]
        else:
            # If starting fresh, replicate the prompt 'amount' times to 
            # create N parallel threads
            prompts = [prompt] * amount

        # Add prompt to history and capture return count
        ret = self.addPrompt(userRole, prompts)
        l.log(f"Added the prompt to {ret} histories.")
        return ret

    def formatPromptFromFile(self, index: int, values: dict) -> int:
        """
        Fills the {placeholders} of the last message of one history (usually
        a template added by addPromptFromFile()) with the provided values.
        Literal braces in the template must be escaped as {{ and }}.

        :param index: Index of the history within messageHistories whose last
            message is formatted.
        :param values: Dictionary containing key-value pairs matching {key}
            placeholders.
        :return: Change in the message's length in characters (negative if
            the values are shorter than their placeholders).
        :raises IndexError: If `index` is out of range for messageHistories.
        :raises KeyError: If `values` has no entry for a placeholder.
        :raises ValueError: If the template is malformed, e.g. contains an
            unescaped single brace.
        """
        l = Logger()

        # Guard Clause: Validate history index bounds
        if not (0 <= index < len(self.messageHistories)):
            l.log(f"History index {index} out of bounds.")
            raise IndexError(
                f"History index {index} out of range "
                f"({len(self.messageHistories)} histories)."
            )

        history = self.messageHistories[index]
        message = history[-1]

        raw_template = message[messageTextElement]

        # Step 1: Format template placeholders with dictionary values
        try:
            formatted_prompt = raw_template.format(**values)
        except KeyError as e:
            l.log(f"Missing template placeholder key in values dict: {e}")
            raise
        except (ValueError, IndexError) as e:
            l.log(f"Failed to format prompt template at history {index}: {e}")
            raise ValueError(f"Malformed prompt template at history {index}: {e}") from e

        # Step 2: Update the message content in-place. `message` is the very
        # dict stored in self.messageHistories[index], so no write-back is
        # needed.
        message[messageTextElement] = formatted_prompt

        return len(formatted_prompt) - len(raw_template)

    def generate(self) -> int:
        """
        Generates an answer for every history that does not end with a model
        answer yet. Histories are sent to vLLM's chat interface, which
        formats them with the tokenizer's chat template, in chunks of
        config.chunk_size; after every chunk the answers are appended to
        their histories and all histories are written to the checkpoint file
        (see exportHistoriesToFile()), so an interrupted run can be resumed
        with importHistoriesFromFile().

        For thinking models, the reasoning is split off the answer (see
        _splitReasoning()): the answer becomes the message text, the
        reasoning is stored under messageReasoningElement on the same
        message and is never sent back to the model. Answers cut off at
        config.max_tokens are listed in self.truncatedHistories; a thinking
        model cut off during its reasoning gets an empty answer, which the
        next generate() call replaces by a new one (with the same seed and
        settings, it is cut off again, so raise max_tokens first).

        A history whose prompt does not fit into config.max_model_len is
        skipped instead of failing its whole chunk: it keeps no answer, its
        index is stored in self.skippedHistories, and toJSON() returns None
        for it.

        Progress and throughput are logged after every chunk, and the
        statistics of the call are added to the run metadata file.

        :return: Number of answers generated.
        :raises RuntimeError: If there is something to generate but the
            model was closed with close().
        """
        l = Logger()
        self.skippedHistories = []
        self.truncatedHistories = []

        # 1. Identify indices where the model has not yet generated a response
        # (Histories whose last message role is NOT modelRole), or only an
        # empty one
        unprocessed_indices = [
            idx for idx, history in enumerate(self.messageHistories)
            if history and (
                history[-1].get(messageRoleElement) != modelRole
                or Model._endsWithEmptyAnswer(history)
            )
        ]

        if not unprocessed_indices:
            l.log("No histories without an answer; nothing to generate.")
            return 0

        if self.llm is None:
            l.log("generate() called after close().")
            raise RuntimeError("The model was closed with close(); create a new Model to generate.")

        # Empty answers are replaced: remove them, so the prompt before them
        # is answered again
        for idx in unprocessed_indices:
            if Model._endsWithEmptyAnswer(self.messageHistories[idx]):
                self.messageHistories[idx].pop()

        tokenizer = self.llm.get_tokenizer()
        chunk_size = self.config.chunk_size
        total = len(unprocessed_indices)
        # Number of chunks, i.e. total / chunk_size rounded up
        chunk_count = -(-total // chunk_size)

        l.log(f"Starting generation for {total} unprocessed histories in {chunk_count} chunk(s) of up to {chunk_size}...")

        started = datetime.now().astimezone().isoformat(timespec="seconds")
        run_start = time.perf_counter()
        done = 0
        generated = 0
        prompt_tokens = 0
        output_tokens = 0
        reasoning_tokens = 0

        # 2. Process inference in chunked batches
        for chunk_number, i in enumerate(range(0, total, chunk_size), start=1):
            chunk_indices = unprocessed_indices[i : i + chunk_size]
            chunk_start = time.perf_counter()

            # Leave out prompts that do not fit into the context window: vLLM
            # would reject the whole chunk because of a single one of them
            batch_indices = [idx for idx in chunk_indices if self._fitsContext(tokenizer, idx)]
            chunk_output_tokens = 0

            if batch_indices:
                l.log(f"Running vLLM generation for chunk {chunk_number}/{chunk_count} [Indices {batch_indices[0]} to {batch_indices[-1]}]...")

                # Run batched generation over the chat histories
                results = self.llm.chat(
                    [self._chatMessages(self.messageHistories[idx]) for idx in batch_indices],
                    self.sampling_params,
                    use_tqdm=False,
                    **self._chatTemplateArguments()
                )

                # 3. Append outputs back to their respective histories, with
                # the reasoning of thinking models split off the answer
                for idx, result in zip(batch_indices, results):
                    completion = result.outputs[0]
                    reasoning, answer = self._splitReasoning(str(completion.text))
                    message = {
                        messageRoleElement: modelRole,
                        messageTextElement: answer
                    }
                    if reasoning:
                        message[messageReasoningElement] = reasoning
                        # vLLM only reports the token count of the whole
                        # output, so the reasoning is tokenized again to
                        # estimate its share (approximate: the tags and the
                        # whitespace stripped by _splitReasoning() are lost)
                        reasoning_tokens += len(tokenizer.encode(reasoning, add_special_tokens=False))
                    self.messageHistories[idx].append(message)
                    if getattr(completion, "finish_reason", None) == "length":
                        self.truncatedHistories.append(idx)
                    prompt_tokens += len(result.prompt_token_ids or [])
                    chunk_output_tokens += len(completion.token_ids or [])

                # 4. Write a checkpoint after every chunk, so a killed job
                # (e.g. by a Slurm time limit) loses at most one chunk of work
                self.exportHistoriesToFile()

            output_tokens += chunk_output_tokens
            done += len(chunk_indices)
            generated += len(batch_indices)

            # 5. Log progress, throughput and the estimated remaining time
            chunk_seconds = time.perf_counter() - chunk_start
            elapsed = time.perf_counter() - run_start
            # Extrapolated from the average time per history so far, skipped
            # histories included
            remaining = elapsed / done * (total - done)
            l.log(
                f"Chunk {chunk_number}/{chunk_count}: {len(batch_indices)} answers, "
                f"{chunk_output_tokens} tokens in {chunk_seconds:.1f} s "
                f"({_rate(chunk_output_tokens, chunk_seconds)} tokens/s). "
                f"Progress: {done}/{total} histories, elapsed {_duration(elapsed)}, "
                f"remaining ~{_duration(remaining)}."
            )

        elapsed = time.perf_counter() - run_start
        if self.skippedHistories:
            l.log(f"Skipped {len(self.skippedHistories)} histories whose prompt exceeds max_model_len: {self.skippedHistories}")
        if self.truncatedHistories:
            l.log(f"{len(self.truncatedHistories)} answers were cut off at max_tokens "
                  f"({self.config.max_tokens}): {self.truncatedHistories}")
        if reasoning_tokens:
            l.log(f"Of the generated tokens, about {reasoning_tokens} were reasoning.")
        l.log(
            f"Generation completed: {generated} answers, {prompt_tokens} prompt "
            f"and {output_tokens} generated tokens in {_duration(elapsed)} "
            f"({_rate(output_tokens, elapsed)} generated tokens/s)."
        )

        # 6. Record the statistics of this call in the run metadata
        self.generationStats.append({
            "started": started,
            "finished": datetime.now().astimezone().isoformat(timespec="seconds"),
            "seconds": round(elapsed, 1),
            "histories": total,
            "answers": generated,
            "skipped": list(self.skippedHistories),
            "truncated": list(self.truncatedHistories),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": output_tokens,
            # Re-tokenized from the split-off reasoning, so approximate
            "reasoning_tokens": reasoning_tokens,
        })
        self._writeRunMetadata()

        return generated

    def _fitsContext(self, tokenizer, index: int) -> bool:
        """
        Returns whether the prompt of the history at `index`, formatted with
        the chat template, leaves room for at least one generated token
        within config.max_model_len. Otherwise the index is added to
        self.skippedHistories.
        """
        # Format the prompt exactly as generate() sends it: without stored
        # reasoning and with the same chat template arguments
        token_ids = tokenizer.apply_chat_template(
            self._chatMessages(self.messageHistories[index]),
            tokenize=True,
            add_generation_prompt=True,
            **self._chatTemplateArguments().get("chat_template_kwargs", {})
        )
        # Some transformers versions return a BatchEncoding instead of a list
        if hasattr(token_ids, "keys"):
            token_ids = token_ids["input_ids"]

        if len(token_ids) < self.config.max_model_len:
            return True

        Logger().log(
            f"Skipping history {index}: its prompt has {len(token_ids)} tokens, "
            f"max_model_len is {self.config.max_model_len}."
        )
        self.skippedHistories.append(index)
        return False

    @staticmethod
    def _chatMessages(history: list[dict]) -> list[dict]:
        """
        Returns the messages of `history` as they are sent to the model:
        only role and content, so stored reasoning is never fed back.
        """
        return [
            {
                messageRoleElement: message[messageRoleElement],
                messageTextElement: message[messageTextElement],
            }
            for message in history
        ]

    def _chatTemplateArguments(self) -> dict:
        """
        Extra keyword arguments for LLM.chat(): chat_template_kwargs with
        enable_thinking if config.enable_thinking is set, otherwise none, so
        the chat template's default applies.
        """
        if self.config.enable_thinking is None:
            return {}
        return {"chat_template_kwargs": {"enable_thinking": self.config.enable_thinking}}

    def _splitReasoning(self, text: str) -> tuple[str, str]:
        """
        Splits a model output into (reasoning, answer), both stripped.

        - With config.reasoning_end_tag in the text, the reasoning is the
          text before its first occurrence (without a leading
          config.reasoning_start_tag) and the answer the text after it.
          This also covers chat templates that put the start tag into the
          prompt (e.g. DeepSeek-R1), whose output has no start tag.
        - With a start tag but no end tag, the output was cut off during
          the reasoning: everything is reasoning and the answer is empty.
        - Otherwise (non-thinking models, thinking switched off, or
          splitting disabled with reasoning_end_tag = null) everything is
          the answer.
        """
        end_tag   = self.config.reasoning_end_tag
        start_tag = self.config.reasoning_start_tag
        stripped  = text.strip()

        if not end_tag:
            return "", stripped

        if end_tag in stripped:
            reasoning, _, answer = stripped.partition(end_tag)
            reasoning = reasoning.strip()
            if start_tag and reasoning.startswith(start_tag):
                reasoning = reasoning[len(start_tag):]
            return reasoning.strip(), answer.strip()

        if start_tag and stripped.startswith(start_tag):
            return stripped[len(start_tag):].strip(), ""

        return "", stripped

    def _writeRunMetadata(self) -> None:
        """
        Writes this run's metadata to config.metadata_folder /
        config.metadata_file: package versions, the model and its Hugging
        Face revision, GPUs, host, the full config, the SHA-256 of every
        prompt file and the statistics of every generate() call.

        The file holds a JSON list with one entry per Model instance (keyed
        by self.runId); this instance's entry is replaced, all others are
        kept, so resumed runs append to the same file.

        :raises ValueError: If the existing file is not a JSON list.
        """
        path = os.path.join(self.config.metadata_folder, self.config.metadata_file)

        runs = []
        if os.path.isfile(path):
            with open(path, mode="r", encoding="utf-8") as f:
                runs = json.load(f)
            if not isinstance(runs, list):
                raise ValueError(f"Run metadata file '{path}' does not contain a JSON list.")

        gpu_names = []
        for gpu_index in range(self.gpuCount):
            # The metadata is still written without a name, but the reason
            # is logged
            try:
                gpu_names.append(torch.cuda.get_device_name(gpu_index))
            except Exception as e:
                Logger().log(f"Could not read the name of GPU {gpu_index}: {e!r}")
                gpu_names.append(None)

        prompt_hashes = {
            file_name: fileSHA256(os.path.join(self.config.prompt_folder, file_name))
            for file_name in self.config.prompt_files
        }

        record = {
            "run_id": self.runId,
            "started": self.runStarted,
            "updated": datetime.now().astimezone().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "python": platform.python_version(),
            "packages": packageVersions(metadataPackages),
            "model": {
                "id": self.model,
                "index": self.modelIndex,
                "revision": modelRevision(self.model),
            },
            "gpus": {
                "count": self.gpuCount,
                "names": gpu_names,
                "cuda": getattr(getattr(torch, "version", None), "cuda", None),
            },
            "prompt_files": prompt_hashes,
            "config": self.config.model_dump(),
            "generations": self.generationStats,
        }

        runs = [run for run in runs if run.get("run_id") != self.runId]
        runs.append(record)
        writeJSON(runs, path)

    def _get_column_names(self, turn_idx: int) -> tuple[str, str]:
        """
        Derives the (prompt_column, answer_column) name pair for a given
        turn index, used to keep export/import column naming consistent.

        The prompt column is named after the stem of the corresponding
        config.prompt_files entry (e.g. "hpo_mapping" for "hpo_mapping.md"),
        or "prompt{turn_idx + 1}" if no prompt file exists at that index
        (e.g. for turns added programmatically rather than from a template
        file). The answer column is the prompt column name with "_answer"
        appended; the reasoning column of a thinking model's answer is the
        prompt column name with reasoningColumnSuffix appended.

        :param turn_idx: Zero-based index of the prompt/answer turn.
        :return: Tuple of (prompt_column_name, answer_column_name).
        """
        prompt_files = self.config.prompt_files
        if turn_idx < len(prompt_files):
            base_name = Path(prompt_files[turn_idx]).stem
        else:
            base_name = f"prompt{turn_idx + 1}"
        return base_name, f"{base_name}_answer"

    def exportHistoriesToFile(self) -> int:
        """
        Writes all messageHistories to the checkpoint file
        (config.prompt_tmp_folder / config.prompt_tmp_file), one row per
        history. The system prompt is stored in the "system" column; each
        user prompt and its answer are stored in a prompt/answer column pair
        named by _get_column_names(). The reasoning of a thinking model's
        answer is stored in "<prompt column>_reasoning"; the column only
        exists if some answer of that turn has reasoning.

        :return: Number of rows written.
        :raises ValueError: If there are no histories.
        """
        l = Logger()

        # 1. Resolve output file path
        target_file = os.path.join(
            self.config.prompt_tmp_folder,
            self.config.prompt_tmp_file
        )

        # 2. Nothing to export without histories
        histories = self.getMessageHistories()
        if not histories:
            l.log("No messageHistories available to export.")
            raise ValueError("No message histories to export.")

        # 3. Transform histories into row dictionaries
        rows_data = []
        for history in histories:
            row_dict = {}
            prompt_counter = 0

            for turn in history:
                role = turn.get(messageRoleElement)
                text = turn.get(messageTextElement)
                
                if role == modelRole:
                    # Model response pairs with the current active prompt index
                    prompt_col, ans_col = self._get_column_names(prompt_counter)
                    row_dict[ans_col] = str(text).strip() if text else ""
                    # The reasoning of a thinking model gets its own column
                    reasoning = turn.get(messageReasoningElement)
                    if reasoning:
                        row_dict[prompt_col + reasoningColumnSuffix] = str(reasoning)
                    prompt_counter += 1
                elif role == systemRole:
                    # System prompts get a dedicated column so the import
                    # can restore them with the system role
                    row_dict[systemColumn] = str(text) if text else ""
                else:
                    # User prompts get their own column slot
                    prompt_col, _ = self._get_column_names(prompt_counter)

                    # If this slot is already filled (e.g. two consecutive
                    # user prompts), advance to the next column slot
                    if prompt_col in row_dict:
                        prompt_counter += 1
                        prompt_col, _ = self._get_column_names(prompt_counter)

                    row_dict[prompt_col] = str(text) if text else ""

            rows_data.append(row_dict)

        # 4. Build the DataFrame. Its columns are the union of all rows'
        # keys, so histories with fewer turns get empty cells, which the
        # import skips.
        df = pd.DataFrame(rows_data)

        # 5. Write atomically, so an interrupted write never corrupts the
        # previous checkpoint
        return writeCheckpointCSV(df, target_file)

    def importHistoriesFromFile(self) -> int:
        """
        Replaces self.messageHistories with the histories stored in the
        checkpoint file (config.prompt_tmp_folder / config.prompt_tmp_file),
        as written by exportHistoriesToFile().

        Reconstructs multi-turn conversation histories by mapping the
        "system" column and the prompt/answer column pairs back into
        role-message dictionaries, and restores the reasoning of answers
        from the reasoning columns. Empty answer cells are skipped, so a
        history whose last answer is missing (e.g. an answer cut off during
        the reasoning) is generated again by generate(). Empty prompts are
        kept up to the last turn with text, and a row without any text
        becomes an empty history, so every history keeps its index.

        The column names are derived from config.prompt_files (see
        _get_column_names()), so the checkpoint must be read with the same
        prompt_files it was written with.

        :return: Total number of message histories populated.
        :raises FileNotFoundError: If the checkpoint file does not exist,
            e.g. because no earlier run was interrupted.
        :raises ValueError: If the checkpoint is empty or has no
            prompt/answer columns.
        """
        l = Logger()

        # 1. Load the checkpoint file
        target_file = os.path.join(
            self.config.prompt_tmp_folder,
            self.config.prompt_tmp_file
        )

        if not os.path.exists(target_file):
            l.log(f"File not found at '{target_file}'. Cannot populate messageHistories.")
            raise FileNotFoundError(f"Checkpoint file not found at '{target_file}'.")

        # Read every cell as a literal string: without dtype=str and
        # keep_default_na=False, answers such as "NA" or "None" would be
        # parsed as missing (and regenerated), and "1" would become "1.0".
        try:
            df = pd.read_csv(
                target_file,
                sep=";",
                quoting=csv.QUOTE_ALL,
                dtype=str,
                keep_default_na=False
            )
        except pd.errors.EmptyDataError:
            df = None

        if df is None or df.empty:
            l.log("DataFrame is empty or None. Cannot populate histories.")
            raise ValueError(f"Checkpoint file '{target_file}' contains no histories.")

        # 2. Identify all prompt/answer column pairs present in the DataFrame,
        # turn by turn until the first turn with neither column
        column_pairs = []
        prompt_counter = 0

        while True:
            prompt_col, ans_col = self._get_column_names(prompt_counter)
            
            # Stop when neither column exists in the DataFrame
            if prompt_col not in df.columns and ans_col not in df.columns:
                break

            column_pairs.append((prompt_col, ans_col))
            prompt_counter += 1

        if not column_pairs:
            l.log("No valid prompt/answer columns found in DataFrame.")
            raise ValueError(f"Checkpoint file '{target_file}' has no prompt/answer columns.")

        # 3. Reconstruct messageHistories row by row
        new_message_histories = []

        # Rows are read as dicts keyed by the exact column names, since
        # itertuples() renames columns that are not valid Python identifiers
        # (e.g. "hpo-mapping") and they could then not be looked up.
        for row in df.to_dict(orient="records"):
            history = []

            # Extract system prompt, which always opens the history
            system_val = row.get(systemColumn)
            if Model._hasText(system_val):
                history.append({
                    messageRoleElement: systemRole,
                    messageTextElement: str(system_val)
                })

            # Empty cells stand both for missing answers and for the turns
            # a shorter history does not have. Prompts may be empty
            # themselves, though (e.g. a template filled with ""), so every
            # turn up to the last one with any text is restored with its
            # prompt; skipping an empty prompt would leave its answer
            # without a question.
            last_turn = max(
                (
                    turn for turn, (prompt_col, ans_col) in enumerate(column_pairs)
                    if Model._hasText(row.get(prompt_col)) or Model._hasText(row.get(ans_col))
                ),
                default=-1
            )

            for prompt_col, ans_col in column_pairs[:last_turn + 1]:
                # Extract prompt text (User)
                prompt_val = row.get(prompt_col)
                history.append({
                    messageRoleElement: userRole,
                    messageTextElement: str(prompt_val) if pd.notna(prompt_val) else ""
                })

                # Extract answer text (Model), with its reasoning if any
                ans_val = row.get(ans_col)
                if Model._hasText(ans_val):
                    message = {
                        messageRoleElement: modelRole,
                        messageTextElement: str(ans_val).strip()
                    }
                    reasoning = row.get(prompt_col + reasoningColumnSuffix)
                    if Model._hasText(reasoning):
                        message[messageReasoningElement] = str(reasoning)
                    history.append(message)

            # Rows without any text are kept as empty histories, so the
            # indices of all later histories stay the same
            new_message_histories.append(history)

        # 4. Assign reconstructed histories back to model state
        self.messageHistories = new_message_histories
        l.log(f"Successfully imported {len(self.messageHistories)} message histories.")

        return len(self.messageHistories)

    @staticmethod
    def _hasText(value) -> bool:
        """Returns whether a checkpoint cell holds more than whitespace."""
        return pd.notna(value) and str(value).strip() != ""

    def toJSON(self, index: int) -> Optional[dict | list]:
        """
        Parses the last model answer of the history at the given index into
        a native Python value. Tried in this order: the content of the first
        Markdown code block, the first complete JSON object or array
        anywhere in the answer, and the whole answer as a bare JSON scalar.
        Only the answer is searched, never the reasoning of a thinking model.

        :param index: Index of the target message history.
        :return: Parsed JSON value, or None if the history has no answer
            (e.g. it was skipped by generate()) or the answer contains no
            valid JSON.
        :raises IndexError: If `index` is out of range for messageHistories.
        """
        l = Logger()

        # Guard Clause: Ensure the index is within bounds
        if not (0 <= index < len(self.messageHistories)):
            l.log(f"History index {index} out of bounds.")
            raise IndexError(
                f"History index {index} out of range "
                f"({len(self.messageHistories)} histories)."
            )

        history = self.messageHistories[index]
        
        # Ensure history has messages and the last message came from the model
        if not history or history[-1].get(messageRoleElement) != modelRole:
            l.log(f"No model response found at history index {index}.")
            return None

        text = history[-1].get(messageTextElement, "").strip()
        if not text:
            l.log(f"Empty text at history index {index}.")
            return None

        # Strategy 1: Extract JSON inside markdown code blocks ```json ... ``` or ``` ... ```
        json_code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if json_code_block:
            substring = json_code_block.group(1).strip()
            try:
                return json.loads(substring)
            except json.JSONDecodeError:
                pass  # Fall back to regex scan if code block string isn't valid JSON

        # Strategy 2: Decode the first complete JSON object {...} or array
        # [...] starting at any opening bracket. raw_decode stops at the end
        # of the JSON value, so trailing prose (even prose containing
        # brackets) does not break parsing.
        decoder = json.JSONDecoder()
        for bracket in re.finditer(r"[\{\[]", text):
            try:
                parsed, _ = decoder.raw_decode(text, bracket.start())
                return parsed
            except json.JSONDecodeError:
                continue

        # Strategy 3: Parse the whole text, which only succeeds for a bare
        # JSON scalar such as a number or a string
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            l.log(f"Could not locate a valid JSON object at index {index}.")
            return None

    def getMessageHistories(self) -> list[list[dict]]:
        """
        Returns all current conversation history records.
        """
        return self.messageHistories

    def reset(self) -> None:
        """
        Clears all active conversation histories and resets state.
        """
        self.messageHistories = []
        self.skippedHistories = []
        self.truncatedHistories = []

    def logPrompts(self) -> None:
        """
        Writes every message of every history to the prompt log
        (config.prompt_log_folder / config.prompt_log_file), overwriting it;
        see writePrompt(). The reasoning of thinking models is not logged.
        """
        path = os.path.join(
            self.config.prompt_log_folder, 
            self.config.prompt_log_file
        )
        writePrompt(path, self.messageHistories)

    def close(self) -> None:
        """
        Explicitly cleans up vLLM engine instances, PyTorch CUDA memory, and 
        distributed process groups.
        Helps prevent VRAM leaks when dynamically instantiating or destroying 
        models.
        """
        if hasattr(self, 'llm') and self.llm is not None:
            del self.llm
            self.llm = None

        steps = [
            ("destroy_model_parallel", destroy_model_parallel),
            ("destroy_distributed_environment", destroy_distributed_environment),
        ]
        # The process group is usually gone after vLLM's own teardown, and
        # destroying a missing one raises, so only try if one is left.
        distributed = torch.distributed
        if distributed.is_available() and distributed.is_initialized():
            steps.append(("destroy_process_group", distributed.destroy_process_group))

        # A failing step must not block the following ones, but is logged:
        # it may leave GPU memory or processes behind.
        for name, step in steps:
            try:
                step()
            except Exception as e:
                Logger().log(f"close(): {name} failed: {e!r}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self):
        """Allows usage in context manager blocks (`with Model() as model:`)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensures GPU resource cleanup when exiting a context manager block."""
        self.close()

    def __del__(self) -> None:
        """
        Fallback on object garbage collection: only releases this instance's
        vLLM engine reference. The distributed environment is process-wide,
        so tearing it down here would break any other Model instance still
        alive; that is left to an explicit close() or a `with` block.
        """
        with contextlib.suppress(Exception):
            self.llm = None