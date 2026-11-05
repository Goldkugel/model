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
    messageTruncatedElement,
    writePrompt,
    updateRunRecord,
    metadataPackages,
    packageVersions,
    modelRevision,
    describeError,
    fileSHA256,
    target_device,
)
from .AnswerStore       import AnswerStore
from logger             import Logger, LoggerConfig
from pathlib            import Path
from datetime           import datetime
import json
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
import re

logging.getLogger("vllm").setLevel(logging.ERROR)

# Consistent GPU ordering across runs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# vLLM's default, but overrides a "cpu" inherited from the environment (CPU
# execution is not supported yet).
os.environ[target_device] = "cuda"

# CUDA_VISIBLE_DEVICES is deliberately left alone: it selects the GPUs (e.g.
# set by Slurm), and vLLM uses all of them via tensor_parallel_size.

# Sections of the YAML config file, and its default path
configuration_section: str = "llm"       
logger_section: str = "logger"
standard_directory: str = "./config/config.yaml"  

def _duration(seconds: float) -> str:
    """Formats a number of seconds as H:MM:SS."""
    seconds = int(round(seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _rate(tokens: int, seconds: float) -> str:
    """Formats tokens per second, or "-" if no time has passed."""
    return f"{tokens / seconds:.0f}" if seconds > 0 else "-"


def _firstIndexes(indexes: list, count: int = 10) -> str:
    """Formats the first count indexes, e.g. "[1, 4, 7, ...]", for a short log row."""
    return str(indexes[:count])[:-1] + (", ...]" if len(indexes) > count else "]")


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
        # Histories the last generate() skipped (prompt exceeds max_model_len)
        self.skippedHistories: list[int] = []
        # Histories whose last answer was cut off at max_tokens
        self.truncatedHistories: list[int] = []
        # Run metadata (see _writeRunMetadata())
        self.runId: str = uuid.uuid4().hex
        self.runStarted: str = datetime.now().astimezone().isoformat(timespec="seconds")
        self.modelIndex: int = index
        self.gpuCount: int = 0
        self.generationStats: list[dict] = []
        # The answer store (the checkpoint), loaded by the first generate();
        # see _answerStore()
        self.store: Optional[AnswerStore] = None
        # Looked up once by _modelRevision()
        self.revision: Optional[str] = None
        self.revisionLooked: bool = False
        # See setPostfix()
        self.postfix: str = ""

        self.config, logger_path, logger_settings = Model._resolveConfig(config)

        # The Logger is a process-wide singleton. If it is not configured yet,
        # configure it from the same file or dict; otherwise Logger() falls
        # back to ./config/config.yaml.
        if Logger.config is None and logger_path is not None:
            l = Logger(logger_path)
        else:
            if Logger.config is None and logger_settings is not None:
                Logger(LoggerConfig.model_validate(logger_settings))
            l = Logger()

        if not (0 <= index < len(self.config.model_id)):
            l.log(f"No model specified or index {index} out of range.", level="error")
            raise ValueError(
                f"Index {index} out of range for config.model_id "
                f"({len(self.config.model_id)} entries)."
            )

        model = self.config.model_id[index]
        self.model = model
        l.log(f"Loading model at index {index}: '{model}'...")

        device = self.config.device
        if device == cpu:
            l.log("Device 'cpu' is not supported yet; the model is loaded on the GPU.", level="warning")
        l.log(f"Scanning device '{device}'...")

        # Fail before vLLM starts, whose error would be far less readable.
        cuda_ok = torch.cuda.is_available()
        l.log(f"Cuda available: {cuda_ok}.")

        if not cuda_ok:
            l.log("No GPU available.", level="error")
            raise RuntimeError("No GPU available: CUDA is not accessible to PyTorch.")

        gpus = torch.cuda.device_count()
        self.gpuCount = gpus
        l.log(f"GPU Amount: {gpus}.")

        if gpus == 0:
            l.log("No GPU visible.", level="error")
            raise RuntimeError("No GPU visible despite CUDA being reported as available.")

        l.log(f"Scanning device '{device}' completed.")
        
        # The per-request seed makes each answer independent of the batching.
        self.sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            seed=self.config.seed
        )

        l.log(f"Loading model into {device}...")

        try:
            self.llm = LLM(
                model=model,
                # The pinned commit of llm.revision for model and tokenizer; None
                # loads the latest (or, offline, the cached) version.
                revision=self.config.revision.get(model),
                tokenizer_revision=self.config.revision.get(model),
                tensor_parallel_size=gpus,
                enforce_eager=self.config.enforce_eager,
                max_model_len=self.config.max_model_len,
                max_num_batched_tokens=self.config.max_num_batched_tokens,
                seed=self.config.seed,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                dtype=self.config.dtype,
                quantization=self.config.quantization,
                enable_prefix_caching=self.config.enable_prefix_caching,
                # Mixture-of-Experts models shard their experts across the GPUs
                enable_expert_parallel=any(
                    pattern in model for pattern in self.config.expert_parallel_models
                ),
            )
        except Exception:
            # vLLM may have started worker processes and taken GPU memory
            # before it failed (e.g. out of memory); the caller gets no Model
            # to close(), and the next Model would find the GPUs taken.
            l.log(f"Loading model into {device} failed; releasing the GPUs.", level="error")
            self.close()
            raise

        l.log(f"Loading model into {device} completed.")

        # If this fails (e.g. a corrupt metadata file), the caller gets no
        # Model to close(), so release the engine here.
        try:
            self._writeRunMetadata()
        except Exception:
            l.log("Writing the run metadata failed; releasing the loaded model.", level="error")
            self.close()
            raise

        l.log(f"Loading model at index {index}: '{model}' completed.")

    @staticmethod
    def _resolveConfig(
        config: str | os.PathLike | dict | ModelConfig,
    ) -> tuple[ModelConfig, Optional[str], Optional[dict]]:
        """
        Turn the `config` argument of __init__ into a ModelConfig.

        Returns the ModelConfig, the YAML file it came from if that file has
        a "logger" section (or None), and the "logger" section of a full
        configuration dict ({} for an empty one, None if there is none), so
        __init__ can configure the Logger from the same source.

        :raises FileNotFoundError: If the config file does not exist.
        :raises KeyError: If the config file has no "llm" section.
        """
        # A copy, so later changes by the caller do not affect the model.
        if isinstance(config, ModelConfig):
            return config.model_copy(deep=True), None, None

        if isinstance(config, dict):
            # A full configuration has an "llm" key, which is not a valid
            # setting itself, so the two kinds cannot be confused.
            if configuration_section in config:
                section = config[configuration_section] or {}
                # An empty logger section (None) means the Logger's defaults,
                # as in a YAML file.
                logger_settings = (config[logger_section] or {}) if logger_section in config else None
                return ModelConfig.model_validate(section), None, logger_settings
            return ModelConfig.model_validate(config), None, None

        path = os.fspath(config)
        if not Path(path).is_file():
            Logger().log(f"Config file not found at '{path}'.", level="error")
            raise FileNotFoundError(f"Config file not found at '{path}'")

        # An empty file parses to None; {} gives the clearer error below.
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if configuration_section not in data:
            raise KeyError(f"Config file '{path}' has no '{configuration_section}' section.")

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
            the number of histories, nor initializes empty histories; if a
            history the messages would be added to ends with an empty answer
            (see generate()); or if answers (modelRole) would not follow a
            user prompt. Nothing is added in these cases.
        """
        l = Logger()

        if message is None:
            message = []

        # Otherwise "hello" would be added as five messages "h", "e", ...
        if isinstance(message, str):
            l.log("addPrompt: message is a string, not a list of strings.", level="error")
            raise TypeError("addPrompt: 'message' must be a list of strings, not a string.")

        # Other roles (e.g. Gemini's "model") are not understood by the chat
        # templates or generate(), and the answer store would store them as
        # prompts.
        if message and role not in chatRoles:
            error = f"addPrompt: unknown role '{role}'; expected one of {list(chatRoles)}."
            l.log(error, level="error")
            raise ValueError(error)

        # An empty answer (a thinking model cut off during its reasoning) is
        # not stored, so it must be generated again before the next turn.
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
                l.log(error, level="error")
                raise ValueError(error)

        # The answer store pairs every answer with the prompt before it, so an
        # answer anywhere else would be stored as a prompt or as the answer to
        # a system prompt.
        if message and role == modelRole:
            without_prompt = [
                idx for idx, history in enumerate(self.messageHistories)
                if not history or history[-1].get(messageRoleElement) != userRole
            ]
            if not self.messageHistories or without_prompt:
                where = f"histories {without_prompt}" if without_prompt else "a new history"
                error = f"addPrompt: an answer must follow a user prompt, which {where} lack."
                l.log(error, level="error")
                raise ValueError(error)

        # Case 1: broadcast
        if len(message) == 1 and len(self.messageHistories) > 0:
            for history in self.messageHistories:
                history.append({
                    messageRoleElement: role,
                    messageTextElement: message[0]
                })

        # Case 2: one message per history
        elif (len(message) == len(self.messageHistories) and
            len(self.messageHistories) > 0):
            for idx, history in enumerate(self.messageHistories):
                history.append({
                    messageRoleElement: role,
                    messageTextElement: message[idx]
                })

        # Case 3: new histories
        elif len(self.messageHistories) == 0 and len(message) > 0:
            for msg in message:
                self.messageHistories.append([{
                    messageRoleElement: role,
                    messageTextElement: msg
                }])

        elif len(message) > 0:
            error = (
                f"addPrompt: message count ({len(message)}) does not match "
                f"active history count ({len(self.messageHistories)})."
            )
            l.log(error, level="error")
            raise ValueError(error)

        return len(self.messageHistories)

    def _history(self, index: int) -> list[dict]:
        """
        Returns the history at `index`.

        :raises IndexError: If `index` is out of range for messageHistories
            (negative indices included, unlike plain list indexing).
        """
        if not (0 <= index < len(self.messageHistories)):
            Logger().log(f"History index {index} out of bounds.", level="error")
            raise IndexError(
                f"History index {index} out of range "
                f"({len(self.messageHistories)} histories)."
            )
        return self.messageHistories[index]

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

        if not (0 <= index < len(self.config.prompt_files)):
            l.log(f"Invalid prompt index: {index}.", level="error")
            raise IndexError(
                f"Prompt index {index} out of range for config.prompt_files "
                f"({len(self.config.prompt_files)} entries)."
            )

        if not self.config.prompt_folder:
            l.log("No prompt folder specified in config.", level="error")
            raise ValueError("No prompt folder specified in config.")

        file_name = self.config.prompt_files[index]
        file_path = Path(self.config.prompt_folder) / file_name

        if not file_path.is_file():
            l.log(f"File '{file_name}' not found at {file_path}.", level="error")
            raise FileNotFoundError(f"Prompt file '{file_name}' not found at {file_path}.")

        if len(self.messageHistories) == 0 and amount <= 0:
            l.log("Histories empty and valid 'amount' not specified.", level="error")
            raise ValueError("Histories are empty: 'amount' must be positive.")

        l.log(f"Loading prompt from file '{file_name}'...")
        prompt = file_path.read_text(encoding="utf-8")
        l.log(f"Loaded prompt with base length {len(prompt)}.")

        # Broadcast to existing histories, or create `amount` new ones
        if len(self.messageHistories) > 0:
            prompts = [prompt]
        else:
            prompts = [prompt] * amount

        ret = self.addPrompt(userRole, prompts)
        l.log(f"Added the prompt to {ret} histories.")
        l.log(f"Loading prompt from file '{file_name}' completed.")
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

        history = self._history(index)
        message = history[-1]

        raw_template = message[messageTextElement]

        try:
            formatted_prompt = raw_template.format(**values)
        except KeyError as e:
            l.log(f"Missing template placeholder key in values dict: {e}.", level="error")
            raise
        except (ValueError, IndexError) as e:
            l.log(f"Failed to format prompt template at history {index}.", level="error")
            l.log(f"Reason: {describeError(e)}.", level="error")
            raise ValueError(f"Malformed prompt template at history {index}: {e}") from e

        # `message` is the dict stored in the history, so no write-back needed
        message[messageTextElement] = formatted_prompt

        return len(formatted_prompt) - len(raw_template)

    def generate(self) -> int:
        """
        Generates an answer for every history that does not end with a model
        answer yet.

        Answers in the answer store (the checkpoint, see _answerStore()) are
        reused, and identical histories are generated once. The rest is sent
        to vLLM's chat interface in chunks of config.chunk_size; the answers
        are stored after every chunk, so calling generate() again with the
        same prompts resumes an interrupted run.

        The reasoning of thinking models is split off the answer (see
        _splitReasoning()) and stored under messageReasoningElement; it is
        never sent back to the model. Answers cut off at max_tokens are listed
        in self.truncatedHistories; one cut off during the reasoning is empty,
        not stored, and replaced by the next generate() call (raise
        max_tokens first, or it is cut off again).

        A history whose prompt does not fit into config.max_model_len is
        skipped and listed in self.skippedHistories, instead of failing its
        whole chunk.

        :return: Number of answers generated (not counting stored answers and
            the copies for identical histories).
        :raises ValueError: If the answer store was written with other
            settings (see AnswerStore); checked before anything is generated.
        :raises RuntimeError: If there is something to generate but the
            model was closed with close().
        """
        l = Logger()
        self.skippedHistories = []
        self.truncatedHistories = []

        # Histories without an answer to their last prompt, or an empty one
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

        # Fails before anything is generated if the settings differ
        store = self._answerStore()

        # Remove empty answers, so their prompt is answered again
        for idx in unprocessed_indices:
            if Model._endsWithEmptyAnswer(self.messageHistories[idx]):
                self.messageHistories[idx].pop()

        started = datetime.now().astimezone().isoformat(timespec="seconds")
        run_start = time.perf_counter()
        l.log(f"Generating answers for {len(unprocessed_indices)} histories...")

        missing_indices = []
        reused = 0
        for idx in unprocessed_indices:
            answer = store.answerFor(self.messageHistories[idx])
            if answer is None:
                missing_indices.append(idx)
                continue
            self.messageHistories[idx].append(answer)
            reused += 1
            if answer.get(messageTruncatedElement):
                self.truncatedHistories.append(idx)
        if reused:
            l.log(f"Using {reused} stored answers of {len(unprocessed_indices)} histories without an answer.")

        # Identical histories: the first is generated, the others get copies
        copies: dict[int, list[int]] = {}
        first_of: dict[str, int] = {}
        generate_indices = []
        for idx in missing_indices:
            key = json.dumps(self._chatMessages(self.messageHistories[idx]), ensure_ascii=False)
            if key in first_of:
                copies[first_of[key]].append(idx)
            else:
                first_of[key] = idx
                copies[idx] = []
                generate_indices.append(idx)
        shared = len(missing_indices) - len(generate_indices)

        generated = 0
        prompt_tokens = 0
        output_tokens = 0
        reasoning_tokens = 0

        if generate_indices and self.llm is None:
            l.log("generate() called after close().", level="error")
            raise RuntimeError("The model was closed with close(); create a new Model to generate.")

        if generate_indices:
            tokenizer = self.llm.get_tokenizer()
            # The same for every history, as they share template and settings
            prompt_opens_reasoning = self._promptOpensReasoning(
                tokenizer, self.messageHistories[generate_indices[0]]
            )
            chunk_size = self.config.chunk_size
            total = len(generate_indices)
            # Rounded up
            chunk_count = -(-total // chunk_size)

            l.log(f"{total} histories to generate in {chunk_count} chunk(s) of up to {chunk_size}.")
            if shared:
                l.log(f"{shared} identical histories get the same answers.")

            done = 0

            for chunk_number, i in enumerate(range(0, total, chunk_size), start=1):
                chunk_indices = generate_indices[i : i + chunk_size]
                chunk_start = time.perf_counter()

                # vLLM would reject the whole chunk for a single prompt that
                # does not fit into the context window
                batch_indices = [idx for idx in chunk_indices if self._fitsContext(tokenizer, idx)]
                for idx in chunk_indices:
                    if idx not in batch_indices:
                        self.skippedHistories.extend(copies[idx])
                chunk_output_tokens = 0

                if batch_indices:
                    chunk_task = (
                        f"Running vLLM generation for chunk {chunk_number}/{chunk_count} "
                        f"[Indices {batch_indices[0]} to {batch_indices[-1]}]"
                    )
                    l.log(f"{chunk_task}...")

                    results = self.llm.chat(
                        [self._chatMessages(self.messageHistories[idx]) for idx in batch_indices],
                        self.sampling_params,
                        use_tqdm=False,
                        **self._chatTemplateArguments()
                    )

                    for idx, result in zip(batch_indices, results):
                        completion = result.outputs[0]
                        truncated = getattr(completion, "finish_reason", None) == "length"
                        reasoning, answer = self._splitReasoning(
                            str(completion.text), truncated, prompt_opens_reasoning
                        )
                        message = {
                            messageRoleElement: modelRole,
                            messageTextElement: answer
                        }
                        if truncated:
                            message[messageTruncatedElement] = True
                        if reasoning:
                            message[messageReasoningElement] = reasoning
                            # vLLM only counts the whole output, so the reasoning
                            # is re-tokenized (approximate: tags and stripped
                            # whitespace are lost)
                            reasoning_tokens += len(tokenizer.encode(reasoning, add_special_tokens=False))
                        for target in [idx] + copies[idx]:
                            self.messageHistories[target].append(dict(message))
                            if truncated:
                                self.truncatedHistories.append(target)
                        prompt_tokens += len(result.prompt_token_ids or [])
                        chunk_output_tokens += len(completion.token_ids or [])

                        # Identical histories share their nodes in the store
                        store.add(self.messageHistories[idx])

                    l.log(f"{chunk_task} completed.")

                    # A killed job (e.g. by a Slurm time limit) loses at most one chunk
                    store.save()

                output_tokens += chunk_output_tokens
                done += len(chunk_indices)
                generated += len(batch_indices)

                chunk_seconds = time.perf_counter() - chunk_start
                elapsed = time.perf_counter() - run_start
                # From the average time per history so far, skipped ones included
                remaining = elapsed / done * (total - done)
                l.log(
                    f"Chunk {chunk_number}/{chunk_count}: {len(batch_indices)} answers, "
                    f"{chunk_output_tokens} tokens in {chunk_seconds:.1f} s "
                    f"({_rate(chunk_output_tokens, chunk_seconds)} tokens/s)."
                )
                l.log(
                    f"Progress: {done}/{total} histories, elapsed {_duration(elapsed)}, "
                    f"remaining ~{_duration(remaining)}."
                )

        self.skippedHistories.sort()
        self.truncatedHistories.sort()

        elapsed = time.perf_counter() - run_start
        # At most 10 indexes per row; all are in skippedHistories and
        # truncatedHistories.
        if self.skippedHistories:
            l.log(f"Skipped {len(self.skippedHistories)} histories whose prompt exceeds "
                  f"max_model_len: {_firstIndexes(self.skippedHistories)}.")
        if self.truncatedHistories:
            l.log(f"{len(self.truncatedHistories)} answers were cut off at max_tokens "
                  f"({self.config.max_tokens}): {_firstIndexes(self.truncatedHistories)}.")
        if reasoning_tokens:
            l.log(f"Of the generated tokens, about {reasoning_tokens} were reasoning.")
        l.log(
            f"{generated} answers generated, {reused} stored and "
            f"{shared} shared with identical histories."
        )
        if generated > 0:
            l.log(
                f"{prompt_tokens} prompt and {output_tokens} generated tokens in "
                f"{_duration(elapsed)} ({_rate(output_tokens, elapsed)} generated tokens/s)."
            )
        l.log(f"Generating answers for {len(unprocessed_indices)} histories completed.")

        self.generationStats.append({
            "started": started,
            "finished": datetime.now().astimezone().isoformat(timespec="seconds"),
            "seconds": round(elapsed, 1),
            "histories": len(unprocessed_indices),
            "answers": generated,
            "stored": reused,
            "shared": shared,
            "skipped": list(self.skippedHistories),
            "truncated": list(self.truncatedHistories),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": output_tokens,
            # Approximate, see above
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
        # Exactly as generate() sends it
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

    def _promptOpensReasoning(self, tokenizer, history: list[dict]) -> bool:
        """
        Returns whether the chat template ends the prompt of `history` with
        config.reasoning_start_tag, as DeepSeek-R1's does. The model's output
        then has no start tag, and a cut-off output without an end tag is
        reasoning, not an answer (see _splitReasoning()).
        """
        start_tag = self.config.reasoning_start_tag
        if not start_tag or not self.config.reasoning_end_tag:
            return False
        prompt = tokenizer.apply_chat_template(
            self._chatMessages(history),
            tokenize=False,
            add_generation_prompt=True,
            **self._chatTemplateArguments().get("chat_template_kwargs", {})
        )
        return str(prompt).rstrip().endswith(start_tag)

    def _splitReasoning(
        self, text: str, truncated: bool = False, prompt_opens_reasoning: bool = False
    ) -> tuple[str, str]:
        """
        Splits a model output into (reasoning, answer), both stripped.

        - With the end tag in the text: the reasoning is the text before its
          first occurrence (without a leading start tag, which DeepSeek-R1's
          output lacks), the answer the text after it.
        - With a start tag but no end tag, or a `truncated` output when the
          prompt opened the reasoning (`prompt_opens_reasoning`): cut off
          during the reasoning, so the answer is empty (a JSON draft from the
          reasoning must not be taken for the answer).
        - Otherwise (non-thinking model, thinking off, or reasoning_end_tag
          null): everything is the answer.
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

        if truncated and prompt_opens_reasoning:
            return stripped, ""

        return "", stripped

    def _writeRunMetadata(self) -> None:
        """
        Writes this run's metadata to config.metadata_folder /
        config.metadata_file: package versions, the model and its Hugging
        Face revision, GPUs, host, the full config, the SHA-256 of every
        prompt file and the statistics of every generate() call.

        The file holds a JSON list with one entry per Model instance (keyed
        by self.runId); only this instance's entry is replaced, under a lock
        (see updateRunRecord()).

        :raises ValueError: If the existing file is not a JSON list.
        """
        path = os.path.join(self.config.metadata_folder, self.config.metadata_file)

        gpu_names = []
        for gpu_index in range(self.gpuCount):
            try:
                gpu_names.append(torch.cuda.get_device_name(gpu_index))
            except Exception as e:
                Logger().log(f"Could not read the name of GPU {gpu_index}.", level="warning")
                Logger().log(f"Reason: {describeError(e)}.", level="warning")
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
                "revision": self._modelRevision(),
            },
            "gpus": {
                "count": self.gpuCount,
                "names": gpu_names,
                "cuda": getattr(getattr(torch, "version", None), "cuda", None),
            },
            "prompt_files": prompt_hashes,
            "files": {
                "checkpoint": self._checkpointPath(),
                "prompt_log": self._promptLogPath(),
            },
            "config": self.config.model_dump(),
            "generations": self.generationStats,
        }

        updateRunRecord(path, record)

    def _fileName(self, file_name: str) -> str:
        """
        The name of a file of this model: `file_name` with the model name
        (getModelName()) and the postfix (setPostfix()) added before its
        extension, e.g. "answers_medgemma-4b-it_test.json" for "answers.json".
        """
        stem, extension = os.path.splitext(file_name)
        parts = [stem, self.getModelName(), self.postfix]
        return "_".join(part for part in parts if part) + extension

    def _checkpointPath(self) -> str:
        """
        Returns the path of the checkpoint file: config.prompt_tmp_file in
        config.prompt_tmp_folder, named after the model and the postfix.
        """
        return os.path.join(self.config.prompt_tmp_folder, self._fileName(self.config.prompt_tmp_file))

    def _promptLogPath(self) -> str:
        """
        Returns the path of the prompt log: config.prompt_log_file in
        config.prompt_log_folder, named after the model and the postfix.
        """
        return os.path.join(self.config.prompt_log_folder, self._fileName(self.config.prompt_log_file))

    def setPostfix(self, postfix: str = "") -> None:
        """
        Sets the postfix added to the names of the checkpoint and the prompt
        log after the model name, e.g. "answers_medgemma-4b-it_fewshot.json",
        so runs with other settings get their own files; "" removes it. The
        next generate() call uses the answer store of the new name.

        :param postfix: Text added to the file names.
        :raises ValueError: If `postfix` contains a path separator.
        """
        if "/" in postfix or os.sep in postfix:
            raise ValueError(f"The postfix '{postfix}' must not contain a path separator.")

        self.postfix = postfix
        self.store = None
        Logger().log(f"Checkpoint: '{self._checkpointPath()}'.")
        Logger().log(f"Prompt log: '{self._promptLogPath()}'.")

    def hasCheckpoint(self) -> bool:
        """
        Whether the checkpoint (the answer store) of this model exists (see
        _checkpointPath()), i.e. whether generate() finds answers of an
        earlier run.
        """
        return os.path.isfile(self._checkpointPath())

    def _answerSettings(self) -> dict:
        """
        The settings the answers depend on, stored with them in the answer
        store: answers generated with other settings are not reused (see
        AnswerStore). Settings that only change where files are written or
        how fast the model runs (folders, chunk_size, GPU memory, batching)
        are not part of them.
        """
        return {
            "model": self.model,
            "revision": self._modelRevision(),
            "seed": self.config.seed,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "max_model_len": self.config.max_model_len,
            "dtype": self.config.dtype,
            "quantization": self.config.quantization,
            "enable_thinking": self.config.enable_thinking,
            "reasoning_start_tag": self.config.reasoning_start_tag,
            "reasoning_end_tag": self.config.reasoning_end_tag,
        }

    def _modelRevision(self) -> Optional[str]:
        """
        The Hugging Face revision of the model (see modelRevision()), looked
        up once, so that a model missing from the local cache is logged once.
        """
        if not self.revisionLooked and self.model is not None:
            self.revision = modelRevision(self.model, self.config.revision.get(self.model))
            self.revisionLooked = True
        return self.revision

    def _answerStore(self) -> AnswerStore:
        """
        The answer store (the checkpoint) of this model: its prompts and
        answers in a JSON file (see AnswerStore and _checkpointPath()),
        loaded on first use.

        :raises ValueError: If the file was written with other settings (see
            _answerSettings()).
        """
        if self.store is None or self.store.path != self._checkpointPath():
            store = AnswerStore(self._checkpointPath(), self._answerSettings())
            store.load()
            self.store = store
        return self.store

    def toJSON(self, index: int) -> Optional[dict | list]:
        """
        Parses the last model answer of a history (never the reasoning) into
        a Python value. Tried in order: the first Markdown code block, the
        first complete JSON object or array, the whole answer as a JSON
        scalar. For an answer cut off at max_tokens, the value must start at
        its first bracket, so '[{"a": 1}, {"b":' does not yield {"a": 1}.

        :param index: Index of the target message history.
        :return: Parsed JSON value, or None if the history has no answer
            (e.g. it was skipped by generate()) or the answer contains no
            valid JSON.
        :raises IndexError: If `index` is out of range for messageHistories.
        """
        l = Logger()

        history = self._history(index)
        
        if not history or history[-1].get(messageRoleElement) != modelRole:
            l.log(f"No model response found at history index {index}.")
            return None

        text = history[-1].get(messageTextElement, "").strip()
        if not text:
            l.log(f"Empty text at history index {index}.")
            return None

        json_code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if json_code_block:
            substring = json_code_block.group(1).strip()
            try:
                return json.loads(substring)
            except json.JSONDecodeError:
                pass

        # raw_decode stops at the end of the value, so trailing prose (even
        # with brackets) does not break parsing.
        decoder = json.JSONDecoder()
        truncated = bool(history[-1].get(messageTruncatedElement))
        for bracket in re.finditer(r"[\{\[]", text):
            try:
                parsed, _ = decoder.raw_decode(text, bracket.start())
                return parsed
            except json.JSONDecodeError:
                if truncated:
                    l.log(f"The answer at index {index} was cut off inside its JSON value.")
                    return None
                continue

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            l.log(f"Could not locate a valid JSON object at index {index}.")
            return None

    def getModelName(self, model_id: Optional[str] = None) -> Optional[str]:
        """
        Returns the name of a model as the logs and file names use it: its ID
        without the organisation, e.g. "medgemma-4b-it" for
        "google/medgemma-4b-it" (the last part of a local path), like
        Embedder.getModelName(). Without `model_id`, the name of the loaded
        model (self.model).

        :param model_id: A model ID; None for the loaded model.
        :return: The model name, or None if no model is given or loaded.
        """
        model_id = self.model if model_id is None else model_id
        if model_id is None:
            return None
        return str(model_id).rstrip("/").rsplit("/", 1)[-1]

    @staticmethod
    def getModelCount(
        config: str | os.PathLike | dict | ModelConfig = standard_directory,
    ) -> int:
        """
        Returns the number of models in config.model_id without loading one,
        e.g. to load every model in turn with Model(config, index).

        :param config: The settings, as for Model(): a YAML configuration
            file with an "llm" section, a dict or a ModelConfig.
        :return: The number of configured models.
        :raises FileNotFoundError: If the config file does not exist.
        :raises KeyError: If the config file has no "llm" section.
        :raises pydantic.ValidationError: If the llm settings are invalid.
        """
        return len(Model._resolveConfig(config)[0].model_id)

    def getTurnCount(self, index: int = 0) -> int:
        """
        Returns the number of user prompts (turns) of a history; histories
        built with addPrompt() or addPromptFromFile() all have the same.

        :param index: Index of the history within messageHistories.
        :return: Number of turns, 0 if there are no histories.
        :raises IndexError: If there are histories but `index` is out of range.
        """
        if len(self.messageHistories) == 0:
            return 0
        return sum(1 for message in self._history(index) if message[messageRoleElement] == userRole)

    def _turnPosition(self, index: int, turn: int) -> int:
        """
        Position of the user prompt of `turn` (counted from 0) within the
        history at `index`.

        :raises IndexError: If the history or the turn does not exist.
        """
        history = self._history(index)
        positions = [
            position for position, message in enumerate(history)
            if message[messageRoleElement] == userRole
        ]
        if not (0 <= turn < len(positions)):
            raise IndexError(f"Turn {turn} out of range for history {index} ({len(positions)} turns).")
        return positions[turn]

    def getPrompt(self, index: int, turn: int) -> str:
        """
        Returns the user prompt of a turn of a history.

        :param index: Index of the history within messageHistories.
        :param turn: Turn within the history, counted from 0.
        :return: The prompt text.
        :raises IndexError: If the history or the turn does not exist.
        """
        return str(self._history(index)[self._turnPosition(index, turn)][messageTextElement])

    def getAnswer(self, index: int, turn: int) -> str:
        """
        Returns the answer of the model to a turn of a history, without the
        reasoning of thinking models.

        :param index: Index of the history within messageHistories.
        :param turn: Turn within the history, counted from 0.
        :return: The answer text, "" if the turn has no answer (yet), e.g.
            because its prompt is longer than config.max_model_len.
        :raises IndexError: If the history or the turn does not exist.
        """
        history = self._history(index)
        position = self._turnPosition(index, turn) + 1
        if position >= len(history) or history[position][messageRoleElement] != modelRole:
            return ""
        return str(history[position][messageTextElement]).strip()

    def addPromptsFromFile(self, index: int, values: list[dict]) -> int:
        """
        Adds the prompt template at `index` of config.prompt_files to the
        histories and fills in its {placeholders} with one dict of values per
        history (see addPromptFromFile() and formatPromptFromFile()). Without
        histories, one history is created per dict.

        :param index: Index of the template in config.prompt_files.
        :param values: One dict of placeholder values per history.
        :return: Number of active message histories.
        :raises ValueError: If `values` is empty, or if there are histories
            and `values` does not have one dict per history.
        :raises IndexError: If `index` is out of range for config.prompt_files.
        :raises KeyError: If a dict has no value for a placeholder.
        :raises ValueError: Also if the template is malformed. In this and the
            KeyError case the histories are left as they were before the
            call, so no history keeps an unfilled template.
        """
        if len(values) == 0:
            raise ValueError("No values given: one dict per history is needed.")
        if len(self.messageHistories) > 0 and len(values) != len(self.messageHistories):
            raise ValueError(
                f"{len(values)} dicts of values for {len(self.messageHistories)} histories; "
                f"one dict per history is needed."
            )

        had_histories = len(self.messageHistories) > 0
        ret = self.addPromptFromFile(index, len(values))
        try:
            for history_index, history_values in enumerate(values):
                self.formatPromptFromFile(history_index, history_values)
        except (KeyError, ValueError):
            # Remove the template again: generate() would otherwise send the
            # histories before the failing one filled and the rest unfilled.
            if had_histories:
                for history in self.messageHistories:
                    history.pop()
            else:
                self.messageHistories = []
            raise
        return ret

    def getMessageHistories(self) -> list[list[dict]]:
        """
        Returns all current conversation history records.
        """
        return self.messageHistories

    def reset(self) -> None:
        """
        Clears all histories and the lists of skipped and truncated ones; the
        answer store and the run statistics are kept.
        """
        self.messageHistories = []
        self.skippedHistories = []
        self.truncatedHistories = []

    def logPrompts(self) -> None:
        """
        Writes every message of every history to the prompt log (see
        _promptLogPath() and writePrompt()), overwriting it. The reasoning of
        thinking models is not logged.
        """
        writePrompt(self._promptLogPath(), self.messageHistories)

    def close(self) -> None:
        """
        Releases the vLLM engine, CUDA memory and the distributed process
        groups, so loading another model does not leak GPU memory.
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

        # A failing step must not block the others, but is logged, as it may
        # leave GPU memory or processes behind.
        for name, step in steps:
            try:
                step()
            except Exception as e:
                Logger().log(f"close(): {name} failed.", level="warning")
                Logger().log(f"Reason: {describeError(e)}.", level="warning")

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