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
    messageRoleElement,
    messageTextElement,
    systemColumn,
    writePrompt,
    writeCheckpointCSV,
    writeJSON,
    metadataPackages,
    packageVersions,
    modelRevision,
    fileSHA256,
    target_device,
)
from logger             import Logger
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
# Tell vLLM to use its CUDA backend (CPU execution is not supported yet)
os.environ[target_device] = "cuda"

# Expose every CUDA device PyTorch can see to vLLM. This is not config-driven
# (it runs at import time, before any config file is loaded) -- vLLM is later
# given tensor_parallel_size=gpus using this same device count, so the two
# must stay in sync.
detected_gpu_count = int(torch.cuda.device_count())
if detected_gpu_count > 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(detected_gpu_count))

# Global configuration constants
# YAML key where LLM config options are stored
configuration_section: str = "llm"       
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

    def __init__(self, config: str = standard_directory, index: int = 0):
        """
        Load and validate configuration, select the model at `index`, and
        initialize the vLLM engine on GPU.

        :param config: Path to the YAML configuration file containing the
            "llm" section (validated against ModelConfig).
        :param index: Index into config.model_id selecting which configured
            model to load.
        :raises FileNotFoundError: If the config file does not exist.
        :raises ValueError: If `index` is out of range for config.model_id.
        :raises RuntimeError: If no CUDA device is available or visible to
            PyTorch. This also applies when config.device is "cpu", since
            CPU execution is not supported yet.
        """
        l = Logger()

        self.messageHistories: list[list[dict]] = []
        self.llm: Optional[LLM] = None
        self.sampling_params: Optional[SamplingParams] = None
        self.model: Optional[str] = None
        self.config: Optional[ModelConfig] = None
        # Indices of the histories the last generate() call skipped because
        # their prompt does not fit into config.max_model_len
        self.skippedHistories: list[int] = []
        # Run metadata (see _writeRunMetadata()): one statistics record per
        # generate() call, under an id unique to this instance
        self.runId: str = uuid.uuid4().hex
        self.runStarted: str = datetime.now().astimezone().isoformat(timespec="seconds")
        self.modelIndex: int = index
        self.gpuCount: int = 0
        self.generationStats: list[dict] = []

        # --- Step 1: Load and Validate Configuration ---
        config_path = Path(config)
        if not config_path.is_file():
            l.log(f"Config file not found at '{config}'")
            raise FileNotFoundError(f"Config file not found at '{config}'")

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.config = ModelConfig.model_validate(data[configuration_section])

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
        self._writeRunMetadata()

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
        :raises ValueError: If the number of messages neither is 1, nor
            matches the number of histories, nor initializes empty histories.
            Nothing is added in that case.
        """
        if message is None:
            message = []

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
            Logger().log(error)
            raise ValueError(error)

        return len(self.messageHistories)

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

        A history whose prompt does not fit into config.max_model_len is
        skipped instead of failing its whole chunk: it keeps no answer, its
        index is stored in self.skippedHistories, and toJSON() returns None
        for it.

        Progress and throughput are logged after every chunk, and the
        statistics of the call are added to the run metadata file.

        :return: Number of answers generated.
        """
        l = Logger()
        self.skippedHistories = []

        # 1. Identify indices where the model has not yet generated a response
        # (Histories whose last message role is NOT modelRole)
        unprocessed_indices = [
            idx for idx, history in enumerate(self.messageHistories)
            if history and history[-1].get(messageRoleElement) != modelRole
        ]

        if not unprocessed_indices:
            l.log("No histories without an answer; nothing to generate.")
            return 0

        tokenizer = self.llm.get_tokenizer()
        chunk_size = self.config.chunk_size
        total = len(unprocessed_indices)
        chunk_count = -(-total // chunk_size)

        l.log(f"Starting generation for {total} unprocessed histories in {chunk_count} chunk(s) of up to {chunk_size}...")

        started = datetime.now().astimezone().isoformat(timespec="seconds")
        run_start = time.perf_counter()
        done = 0
        generated = 0
        prompt_tokens = 0
        output_tokens = 0

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
                    [self.messageHistories[idx] for idx in batch_indices],
                    self.sampling_params,
                    use_tqdm=False
                )

                # 3. Append outputs back to their respective histories
                for idx, result in zip(batch_indices, results):
                    completion = result.outputs[0]
                    self.messageHistories[idx].append({
                        messageRoleElement: modelRole,
                        messageTextElement: str(completion.text).strip()
                    })
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
            "prompt_tokens": prompt_tokens,
            "generated_tokens": output_tokens,
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
        token_ids = tokenizer.apply_chat_template(
            self.messageHistories[index],
            tokenize=True,
            add_generation_prompt=True
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
            try:
                gpu_names.append(torch.cuda.get_device_name(gpu_index))
            except Exception:
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
        appended.

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
        named by _get_column_names().

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
                    _, ans_col = self._get_column_names(prompt_counter)
                    row_dict[ans_col] = str(text).strip() if text else ""
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

        # 4. Merge converted rows into DataFrame
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
        role-message dictionaries. Empty cells are skipped, so a history
        whose last answer is missing is generated again by generate().

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

        # 2. Identify all prompt/answer column pairs present in the DataFrame
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
            if pd.notna(system_val) and str(system_val).strip() != "":
                history.append({
                    messageRoleElement: systemRole,
                    messageTextElement: str(system_val)
                })

            for prompt_col, ans_col in column_pairs:
                # Extract prompt text (User)
                prompt_val = row.get(prompt_col)
                if pd.notna(prompt_val) and str(prompt_val).strip() != "":
                    history.append({
                        messageRoleElement: userRole,
                        messageTextElement: str(prompt_val)
                    })

                # Extract answer text (Model)
                ans_val = row.get(ans_col)
                if pd.notna(ans_val) and str(ans_val).strip() != "":
                    history.append({
                        messageRoleElement: modelRole,
                        messageTextElement: str(ans_val).strip()
                    })

            if history:
                new_message_histories.append(history)

        # 4. Assign reconstructed histories back to model state
        self.messageHistories = new_message_histories
        l.log(f"Successfully imported {len(self.messageHistories)} message histories.")

        return len(self.messageHistories)

    def toJSON(self, index: int) -> Optional[dict | list]:
        """
        Parses the last generated model response from the history thread at the
        given index, extracts the first valid JSON object or array, and
        converts it into a native Python dict or list.

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

    def logPrompts(self) -> None:
        """
        Writes all recorded conversation histories to a log file defined in 
        the config.
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
            
        # Wrap each destruction step to ensure one failure doesn't block 
        # subsequent cleanup
        with contextlib.suppress(Exception):
            destroy_model_parallel()
        with contextlib.suppress(Exception):
            destroy_distributed_environment()
        with contextlib.suppress(Exception):
            torch.distributed.destroy_process_group()
            
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