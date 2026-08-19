import sys
# Prevent Python from writing .pyc bytecode cache files to keep container environments clean
sys.dont_write_bytecode = True

from vllm               import LLM, SamplingParams
from vllm.distributed   import destroy_distributed_environment, destroy_model_parallel
from .ModelConfig       import ModelConfig
from .ModelUtils        import *
from logger             import Logger
from pathlib            import Path
import logging
import os
import contextlib
import gc
import torch
import yaml

# Suppress verbose vLLM initialization and internal status logs
logging.getLogger("vllm").setLevel(logging.ERROR)

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
        Loads configuration from a YAML file, validates hardware availability, 
        and initializes the vLLM engine with the selected model.

        :param config: Path to the YAML configuration file.
        :param index: Index of the model ID to load from the config's model_id 
            list.
        """
        l = Logger()
        
        # Instance attribute initialization (prevents class-level state sharing)

        # Stores active conversation threads
        self.messageHistories: list[list[dict]] = []  
        # Holds the active vLLM instance
        self.llm: LLM = None                          
        # Generation parameters (temp, top_p, max_tokens)
        self.sampling_params: SamplingParams = None   
        # Identifier/path of the loaded model
        self.model: str = None                        
        # Validated Pydantic/dataclass configuration
        self.config: ModelConfig = None               

        # --- Step 1: Load and Validate Configuration ---
        config_path = Path(config)
        if not config_path.is_file():
            l.log(f"Config file not found at '{config}'")
            raise FileNotFoundError(f"Config file not found at '{config}'")

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            
        # Parse the 'llm' dictionary section using the ModelConfig validator
        self.config = ModelConfig.model_validate(data[configuration_section])

        # --- Step 2: Validate Model Selection Index ---
        if not (0 <= index < len(self.config.model_id)):
            l.log(f"No model specified or index {index} out of range.")
            return

        model = self.config.model_id[index]
        self.model = model
        l.log(f"Loading model at index {index}: '{model}'...")

        device = self.config.device
        l.log(f"Scanning device {device}...")

        # --- Step 3: Configure Hardware Target (GPU vs. CPU) ---
        gpus = 0
        if device == gpu:
            cuda_ok = torch.cuda.is_available()
            l.log(f"Cuda available: {cuda_ok}.")
            
            if not cuda_ok:
                l.log("No GPU available.")
                return

            gpus = int(torch.cuda.device_count())
            l.log(f"GPU Amount: {gpus}.")

            if gpus == 0:
                l.log("No GPU visible.")
                return

            # Force CUDA to enumerate GPUs by physical PCI bus ID for 
            # consistent ordering.
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ[target_device] = gpu
        else:
            # Set target device environment variable to CPU.
            os.environ[target_device] = cpu

        l.log(f"Scanning device {device} completed.")
        l.log(f"Maximum number of batched tokens: " \
            f"{self.config.max_num_batched_tokens}.")
        l.log(f"Maximum number of new tokens: {self.config.max_tokens}.")
        l.log(f"Temperature: {self.config.temperature}.")
        l.log(f"Max Model Length: {self.config.max_model_len}.")
        
        # --- Step 4: Configure vLLM Sampling Parameters ---
        self.sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens
        )

        # --- Step 5: Initialize the vLLM Engine ---
        l.log("Loading model...")
        
        if device == gpu:
            # Multi-GPU CUDA initialization with Tensor Parallelism.
            self.llm = LLM(
                model=model,
                # Distributes model across all available GPUs.
                tensor_parallel_size=gpus,  
                max_model_len=self.config.max_model_len,
                # Enable expert parallelism only for Mixture-of-Experts (MoE) 
                # architectures.
                enable_expert_parallel=("Qwen3-30B" in model)
            )
        else:
            # CPU single-node execution initialization
            self.llm = LLM(
                model=model,
                device=cpu,
                max_model_len=self.config.max_model_len
            )

        l.log(f"Loading model into {device} completed.")
        l.log(f"Loading model at index {index} completed.")

    def addPrompt(self, role: str = userRole, message: list = None) -> int:
        """
        Appends user or model messages to the conversation histories.

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

        return len(self.messageHistories)

    def addPromptFromFile(self, index: int, amount: int = -1) -> int:
        """
        Reads a prompt template from a file on disk and adds it to the message 
        histories.

        :param index: Index of the target filename in config.prompt_files.
        :param amount: Number of duplicate conversation threads to create if 
            histories are empty.
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
        Formats an existing prompt template already loaded in the message 
        histories at the specified index using the provided key-value 
        dictionary.

        :param index: Index of the message within messageHistories to format.
            The last message added will be formatted.
        :param values: Dictionary containing key-value pairs matching {key} 
            placeholders.
        :return: Total number of characters added to the prompt, or -1 on 
            failure.
        """
        l = Logger()

        # Guard Clause: Ensure active histories exist
        if not self.messageHistories:
            l.log("Cannot format prompt: messageHistories is empty.")
            return -1

        # Guard Clause: Validate turn index bounds for this history thread
        if not (0 <= index < len(self.messageHistories)):
            l.log(f"Turn index {index} out of bounds.")
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
            l.log(f"Failed to format prompt template at turn {index}: {e}")
            return -1

        # Step 2: Update the message content in-place
        message[messageTextElement] = formatted_prompt
        history[len(history) - 1] = message 
        self.messageHistories[index] = history

        return len(formatted_prompt) - len(raw_template)

    def generate(self) -> None:
        """
        Formats all message histories using the model's native tokenizer chat 
        template, runs batch inference through vLLM, and appends outputs back 
        to histories.
        """
        tokenizer = self.llm.get_tokenizer()

        # Pass history dictionaries directly to the template tokenizer
        formatted_prompts = [
            tokenizer.apply_chat_template(
                history,
                tokenize                = False,
                add_generation_prompt   = True
            )
            for history in self.messageHistories
        ]

        # Run batched generation
        generatedText = self.llm.generate(
            formatted_prompts, 
            self.sampling_params, 
            use_tqdm                = False
        )

        # Extract responses and append back to messageHistories
        outputs = [str(text.outputs[0].text).strip() for text in generatedText]
        self.addPrompt(role = modelRole, message = outputs)

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
        """Fallback destructor call on object garbage collection."""
        self.close()