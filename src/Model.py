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
    target_device,
)
from logger             import Logger
from pathlib            import Path
import pandas           as pd
import logging
import os
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
            # Skip CUDA graph capture: slower generation, but faster start-up
            # and less GPU memory
            enforce_eager=True,
            max_model_len=self.config.max_model_len,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            seed=self.config.seed,
            # Mixture-of-Experts models (configured via
            # config.expert_parallel_models) need expert parallelism to
            # shard their experts across the available GPUs; dense models
            # ignore this flag.
            enable_expert_parallel=any(
                pattern in model for pattern in self.config.expert_parallel_models
            ),
        )

        l.log(f"Loading model into {device} completed.")

    def addPrompt(self, role: str = userRole, message: list = None) -> int:
        """
        Appends messages with the given role to the conversation histories.

        Handles three input cases:
        1. Single message broadcast: Appends 1 message to ALL active histories.
        2. Multi-message alignment: Appends message[i] to active history[i].
        3. Initialization: Creates N new histories if none currently exist.

        :param role: The sender role (e.g., userRole, modelRole, systemRole).
        :param message: A list of message strings to add.
        :return: Total number of active message histories.
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

        # Invalid combination: message count doesn't broadcast (1), align
        # 1:1 with existing histories, or initialize from empty histories.
        # Log and drop rather than silently misaligning conversation threads.
        elif len(message) > 0:
            Logger().log(
                f"addPrompt: message count ({len(message)}) does not match "
                f"active history count ({len(self.messageHistories)}); no "
                f"prompt was added."
            )

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
        :return: Number of active message histories, or -1 on failure.
        """
        l = Logger()

        # Guard Clause: Validate prompt file index bounds
        if not (0 <= index < len(self.config.prompt_files)):
            l.log(f"Invalid prompt index: {index}")
            return -1

        # Guard Clause: Check if prompt folder path is specified
        if not self.config.prompt_folder:
            l.log("No prompt folder specified in config.")
            return -1

        file_name = self.config.prompt_files[index]
        file_path = Path(self.config.prompt_folder) / file_name

        # Guard Clause: Ensure prompt file exists on disk
        if not file_path.is_file():
            l.log(f"File '{file_name}' not found at {file_path}.")
            return -1

        l.log(f"Loading prompt from file '{file_name}'...")
        prompt = file_path.read_text(encoding="utf-8")
        l.log(f"Loaded prompt with base length {len(prompt)}.")

        # Determine how to append/multiply the loaded prompt
        if len(self.messageHistories) > 0:
            # If histories already exist, broadcast this prompt as a 
            # single item
            prompts = [prompt]
        elif amount > 0:
            # If starting fresh, replicate the prompt 'amount' times to 
            # create N parallel threads
            prompts = [prompt] * amount
        else:
            l.log("Histories empty and valid 'amount' not specified.")
            return -1

        # Add prompt to history and capture return count
        ret = self.addPrompt(userRole, prompts)
        l.log(f"Added {ret} prompt(s) to the model.")
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
            the values are shorter than their placeholders), or -1 on
            failure.
        """
        l = Logger()

        # Guard Clause: Ensure active histories exist
        if not self.messageHistories:
            l.log("Cannot format prompt: messageHistories is empty.")
            return -1

        # Guard Clause: Validate history index bounds
        if not (0 <= index < len(self.messageHistories)):
            l.log(f"History index {index} out of bounds.")
            return -1

        history = self.messageHistories[index]
        message = history[-1]

        raw_template = message[messageTextElement]

        # Step 1: Format template placeholders with dictionary values
        try:
            formatted_prompt = raw_template.format(**values)
        except KeyError as e:
            l.log(f"Missing template placeholder key in values dict: {e}")
            return -1
        except Exception as e:
            l.log(f"Failed to format prompt template at history {index}: {e}")
            return -1

        # Step 2: Update the message content in-place. `message` is the very
        # dict stored in self.messageHistories[index], so no write-back is
        # needed.
        message[messageTextElement] = formatted_prompt

        return len(formatted_prompt) - len(raw_template)

    def generate(self) -> None:
        """
        Generates an answer for every history that does not end with a model
        answer yet. Histories are formatted with the tokenizer's chat
        template and generated in chunks of config.chunk_size; after every
        chunk the answers are appended to their histories and all histories
        are written to the checkpoint file (see exportHistoriesToFile()), so
        an interrupted run can be resumed with importHistoriesFromFile().
        """
        l = Logger()

        # 1. Nothing to do without prompts
        if len(self.messageHistories) == 0 or len(self.messageHistories[0]) == 0:
            l.log("No prompts available for generation.")
            return

        tokenizer = self.llm.get_tokenizer()
        chunk_size = self.config.chunk_size

        # 2. Identify indices where the model has not yet generated a response
        # (Histories whose last message role is NOT modelRole)
        unprocessed_indices = [
            idx for idx, history in enumerate(self.messageHistories)
            if history and history[-1].get(messageRoleElement) != modelRole
        ]

        if not unprocessed_indices:
            l.log("All prompts already have model generations. Skipping.")
            return

        l.log(f"Starting generation for {len(unprocessed_indices)} unprocessed histories in chunks of {chunk_size}...")

        # 3. Process inference in chunked batches
        for i in range(0, len(unprocessed_indices), chunk_size):
            batch_indices = unprocessed_indices[i : i + chunk_size]
            
            # Format chat templates for only the current unprocessed batch
            formatted_prompts = [
                tokenizer.apply_chat_template(
                    self.messageHistories[idx],
                    tokenize=False,
                    add_generation_prompt=True
                )
                for idx in batch_indices
            ]

            l.log(f"Running vLLM generation batch [Indices {batch_indices[0]} to {batch_indices[-1]}]...")

            # Run batched generation
            generatedText = self.llm.generate(
                formatted_prompts,
                self.sampling_params,
                use_tqdm=False
            )

            # Extract generated response texts
            outputs = [str(text.outputs[0].text).strip() for text in generatedText]

            # 4. Append outputs back to their respective histories
            for idx, output in zip(batch_indices, outputs):
                self.messageHistories[idx].append({
                    messageRoleElement: modelRole,
                    messageTextElement: output
                })

            # 5. Write a checkpoint after every chunk, so a killed job (e.g.
            # by a Slurm time limit) loses at most one chunk of work
            self.exportHistoriesToFile()
            l.log(f"Batch checkpoint saved.")

        l.log("Generation completed successfully for all prompts.")

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

        :return: Number of rows written, or -1 if there are no histories.
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
            return -1

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

        :return: Total number of message histories populated, or -1 on failure.
        """
        l = Logger()

        # 1. Load the checkpoint file
        target_file = os.path.join(
            self.config.prompt_tmp_folder,
            self.config.prompt_tmp_file
        )

        if not os.path.exists(target_file):
            l.log(f"File not found at '{target_file}'. Cannot populate messageHistories.")
            return -1

        # Read every cell as a literal string: without dtype=str and
        # keep_default_na=False, answers such as "NA" or "None" would be
        # parsed as missing (and regenerated), and "1" would become "1.0".
        df = pd.read_csv(
            target_file,
            sep=";",
            quoting=csv.QUOTE_ALL,
            dtype=str,
            keep_default_na=False
        )

        if df is None or df.empty:
            l.log("DataFrame is empty or None. Cannot populate histories.")
            return -1

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
            return -1

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
        :return: Parsed JSON object (dict) or array (list), or None if
            parsing fails.
        """
        l = Logger()

        # Guard Clause: Ensure messageHistories exists and index is within bounds
        if not self.messageHistories or not (0 <= index < len(self.messageHistories)):
            l.log(f"History index {index} out of bounds.")
            return None

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