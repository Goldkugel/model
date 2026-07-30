import sys
# Prevent Python from generating .pyc (bytecode cache) files
# Useful for cleaner environments and containerized deployments
sys.dont_write_bytecode = True

# vLLM imports for model loading and inference
from vllm               import LLM, SamplingParams
from vllm.distributed   import destroy_distributed_environment, destroy_model_parallel
from ModelConfig        import ModelConfig
from ModelUtils         import *
from Logger             import Logger
import os
import contextlib
import gc
import torch
import yaml

# Key under which model settings are expected to live in the YAML config file.
configuration_section: str  = "llm"

# Default path to the config file, used if no path is explicitly passed in.
standard_directory: str     = "../config/config.yaml"

# Ensure CUDA devices are enumerated by PCI bus ID
# This guarantees consistent GPU ordering across runs
os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
os.environ["VLLM_TARGET_DEVICE"]    = "cuda"

# Total number of GPUs visible to this process before any restriction below.
gpus = int(torch.cuda.device_count())

# Restrict CUDA_VISIBLE_DEVICES to every detected GPU (intended to produce
# e.g. "0,1,2" for 3 GPUs, so downstream code sees a consistent, explicit
# device list rather than relying on defaults).
if gpus > 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(0, gpus)])


class Model:
    """
    Wrapper class around vLLM to manage:
    - Message histories (chat state)
    - Prompt formatting for different model families
    - Text generation
    - Cleanup of distributed GPU resources
    """

    config: ModelConfig = None
    # Stores multiple independent conversation histories
    messageHistories = []
    # vLLM-related objects
    llm             = None
    sampling_params = None
    model           = None

    def __init__(self, config: str = standard_directory, index: int = 0):
        """
        Load and validate model configuration from a YAML file, then
        load the model at position `index` within the configured
        `model_id` list into vLLM.
        """
        data = None

        self.messageHistories   = []
        self.llm                = None
        self.sampling_params    = None
        self.model              = None

        with open(config, "r") as f:
            data = yaml.safe_load(f)
        self.config = ModelConfig.model_validate(data[configuration_section])

        l = Logger()
        l.log(f"Loading model at index {index}...")
        if len(self.config.model_id) > 0 and index < len(self.config.model_id):
            model = self.config.model_id[index]
            l.log(f"Model ID: '{model}'.")
            l.log(f"Cuda available: {torch.cuda.is_available()}.")
            l.log(f"Device count: {torch.cuda.device_count()}.")
            l.log(f"Maximum number fof batched tokens: {self.config.max_num_batched_tokens}.")
            l.log(f"Maximum number of new tokens: {self.config.max_tokens}.")
            l.log(f"Temperature: {self.config.temperature}.")
            l.log(f"Max Model Length: {self.config.max_model_len}.")
            l.log(f"GPUs: {gpus}.")
            l.log("Loading model into gpu...")
            # Initialize vLLM engine
            self.llm = LLM(
                model                   = model,
                # Number of GPUs used for tensor parallelism
                tensor_parallel_size    = gpus,
                max_model_len           = self.config.max_model_len,
                # Enable expert parallelism only for specific models
                enable_expert_parallel  = (
                    model == "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
                )
            )
            # Sampling configuration for text generation
            self.sampling_params = SamplingParams(
                temperature             = self.config.temperature,
                top_p                   = self.config.top_p,
                max_tokens              = self.config.max_tokens
            )
            l.log("Loading model into gpu completed.")
            l.log(f"Loading model at index {index} completed.")
        else:
            l.log("No model specified or index out of range.")

    def addPrompt(self, role : str = userRole, message : list = None) -> int:
        """
        Add a message (or messages) to the conversation histories.

        Supports:
        - Broadcasting a single message to all histories
        - Appending one message per history
        - Creating new histories if none exist

        Returns:
            Number of active message histories
        """
        ret = 0

        if message is None:
            message = []

        # Case 1: Single message broadcast to all histories
        if len(message) == 1 and len(self.messageHistories) > 0:
            for index in range(0, len(self.messageHistories)):
                self.messageHistories[index].append({
                    messageRoleElement : role,
                    messageTextElement : message[0]
                })
            ret = len(self.messageHistories)
        else:
            # Case 2: One message per history
            if len(message) == len(self.messageHistories):
                for index in range(0, len(self.messageHistories)):
                    self.messageHistories[index].append({
                        messageRoleElement : role,
                        messageTextElement : message[index]
                    })
                ret = len(self.messageHistories)
            else:
                # Case 3: Initialize histories when none exist
                if len(self.messageHistories) == 0 and len(message) > 0:
                    for index in range(0, len(message)):
                        history = []
                        history.append({
                            messageRoleElement : role,
                            messageTextElement : message[index]
                        })
                        self.messageHistories.append(history)
                    ret = len(self.messageHistories)
        return ret

    def generate(self) -> None:
        """
        Generate the next assistant response for every stored conversation
        history, using Gemma-style prompt formatting (<start_of_turn>/
        <end_of_turn>), and append each response to its history.

        System messages have no dedicated Gemma turn type, so they're
        folded into the following user turn instead (`s` tracks whether
        the previous message was a system message still awaiting its
        user turn to be appended/closed).
        """
        inputs = []
        l = Logger()
        # Build a prompt for each conversation history
        for messageHistory in self.messageHistories:
            prompt = ""
            s = False # Tracks whether a system message was just processed
            for message in messageHistory:
                if message[messageRoleElement] == systemRole:
                    # System messages are converted into user turns
                    prompt += f"{startTurn}{userRole}\n" \
                            f"{message[messageTextElement]}\n\n"
                    s = True
                else:
                    if s:
                        # Close system-injected user message
                        prompt +=  f"{message[messageTextElement]}{endTurn}\n"
                        s = False
                    else:
                        # Standard user/assistant turn
                        prompt += f"{startTurn}{message[messageRoleElement]}\n" \
                            f"{message[messageTextElement]}{endTurn}\n"
            # Prepare model to generate the next assistant response
            prompt += f"{startTurn}{modelRole}"
            inputs.append(prompt)

        # Run inference
        generatedText = self.llm.generate(inputs, self.sampling_params, use_tqdm=False)
        outputs = []

        # Extract and clean generated text
        for text in generatedText:
            outputs.append(text.outputs[0].text)

        # Append model responses to histories
        self.addPrompt(role = modelRole, message = outputs)

    def getMessageHistories(self) -> list[list[object]]:
        """
        Return all stored conversation histories.
        """
        return self.messageHistories

    def reset(self) -> None:
        """
        Clear all conversation histories.
        """
        self.messageHistories = []

    def logPrompts(self) -> None:
        """
        Log every prompt from every history to a file, at the path
        configured by `prompt_log_folder`/`prompt_log_file`.
        """
        path = os.path.join(self.config.prompt_log_folder, self.config.prompt_log_file)
        writePrompt(path, self.messageHistories)

    def __del__(self) -> None:
        """
        Destructor to aggressively clean up GPU and distributed resources.
        Prevents memory leaks and CUDA context issues. Each cleanup step
        is wrapped individually so that one failing step (e.g. no
        distributed process group was ever initialized) doesn't stop the
        remaining steps from running.
        """
        try:
            del self.llm
        except:
            pass
        try:
            destroy_model_parallel()
        except:
            pass
        try:
            destroy_distributed_environment()
        except:
            pass
        try:
            with contextlib.suppress(AssertionError):
                torch.distributed.destroy_process_group()
        except:
            pass
        try:
            gc.collect()
        except:
            pass
        try:
            torch.cuda.empty_cache()
        except:
           pass