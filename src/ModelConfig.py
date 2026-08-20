import sys

# Prevent Python from writing .pyc bytecode cache files to keep container environments clean
sys.dont_write_bytecode = True

from typing     import Literal
from pydantic   import BaseModel, ConfigDict, Field

class ModelConfig(BaseModel):
    """
    Pydantic model defining and validating the configuration schema for
    prompting an LLM (model selection, sampling parameters, and
    prompt logging).
    """

    # Forbid extra fields in the YAML file to catch typos early
    model_config = ConfigDict(extra="forbid")

    # One or more model identifiers to load/prompt (e.g. "google/medgemma-27b-text-it").
    model_id: list[str] = Field(
        default_factory=list,
        description="List of HuggingFace or local model repository identifiers."
    )

    # Random seed used for reproducible generation across runs.
    seed: int = Field(
        default=2898231092,
        description="Random seed for generation determinism."
    )

    # Sampling temperature: controls randomness (0.0 = greedy/deterministic).
    temperature: float = Field(
        default=0.01,
        ge=0.0,
        le=2.0,
        description="Sampling temperature controlling output randomness."
    )

    # Nucleus sampling probability mass boundary.
    top_p: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
        description="Top-p (nucleus) sampling threshold."
    )

    # Maximum context length (in tokens) covering combined prompt and output length.
    max_model_len: int = Field(
        default=8192,
        gt=0,
        description="Maximum model context window size in tokens."
    )

    # Maximum number of tokens batched together across concurrent requests (vLLM memory tuning).
    max_num_batched_tokens: int = Field(
        default=16384,
        gt=0,
        description="vLLM batch engine token throughput limit."
    )

    # Maximum number of new tokens generated per output sequence.
    max_tokens: int = Field(
        default=2048,
        gt=0,
        description="Maximum generation length per request."
    )

    # Directory where prompt log files are saved.
    prompt_log_folder: str = Field(
        default="./data/logs/",
        description="Directory target for execution logs."
    )

    # Log filename within `prompt_log_folder`.
    prompt_log_file: str = Field(
        default="prompts.log",
        description="Output filename for CSV prompt histories."
    )

    # Target hardware execution device. Restricted strictly to "gpu" or "cpu".
    device: Literal["gpu", "cpu"] = Field(
        default="gpu",
        description="Target execution device platform ('gpu' or 'cpu')."
    )

    # Directory containing prompt template files (.md or .txt).
    prompt_folder: str = Field(
        default="./prompts/",
        description="Directory containing input prompt template files."
    )

    # List of prompt files to process in sequential prompting order.
    prompt_files: list[str] = Field(
        default_factory=list,
        description="Ordered sequence of prompt template filenames."
    )

    # The file where the temporary data for the prompts are stored.
    prompt_tmp_file: str = Field(
        default="raw.tmp",
        description="Filename for the temporary files created to store the generated text."
    )

    # The folder where the temporary data is stored.
    prompt_tmp_folder: str = Field(
        default="./data/tmp/",
        description="Directory containing the temporary files."
    )

    # How many prompts are being processed at one time.
    chunk_size: int = Field(
        default=500,
        gt=0,
        description="Chunks of prompts processed at the same time. Afterwards the current status is stored in the temporary file."
    )