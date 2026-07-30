import sys
# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

from pydantic import BaseModel, ConfigDict

class ModelConfig(BaseModel):
    """
    Pydantic model defining and validating the configuration schema for
    prompting an LLM (model selection, sampling parameters, and
    prompt logging).
    """

    model_config                = ConfigDict(extra = "forbid")

    # One or more model identifiers to load/prompt (e.g. a HuggingFace
    # model ID such as "google/medgemma-27b-text-it").
    model_id: list              = []

    # Random seed used for reproducible generation.
    seed: int                   = 2898231092

    # Sampling temperature: controls randomness of generation (0 =
    # deterministic/greedy, higher values = more random).
    temperature: float          = 0.01

    # Nucleus sampling probability mass: only tokens within the top
    # `top_p` cumulative probability are considered at each step.
    top_p: float                = 0.95

    # Maximum context length (in tokens) the model is configured to
    # support, covering prompt + generated tokens combined.
    max_model_len: int          = 8192

    # Maximum number of tokens batched together across concurrent
    # requests (a vLLM-style throughput/memory tuning parameter).
    max_num_batched_tokens: int = 16384

    # Maximum number of tokens to generate per output sequence.
    max_tokens: int             = 2048

    # Directory where the prompt log file will be written (relative or
    # absolute path).
    prompt_log_folder: str      = "../data/logs/"

    # Name of the prompt log file to write to within `prompt_log_folder`.
    prompt_log_file: str        = "prompts.log"