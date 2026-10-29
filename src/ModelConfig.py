import sys

# Prevent Python from writing .pyc bytecode cache files to keep container environments clean
sys.dont_write_bytecode = True

from typing     import Literal
from pydantic   import BaseModel, ConfigDict, Field

class ModelConfig(BaseModel):
    """
    Pydantic model defining and validating the "llm" section of the
    configuration file: model selection, vLLM engine and sampling
    parameters, prompt templates, checkpointing and prompt logging.
    """

    # Forbid extra fields in the YAML file to catch typos early
    model_config = ConfigDict(extra="forbid")

    # One or more model identifiers to load/prompt (e.g. "google/medgemma-27b-text-it").
    model_id: list[str] = Field(
        default_factory=list,
        description="List of HuggingFace or local model repository identifiers."
    )

    # Random seed passed to vLLM and to every sampling request, for
    # reproducible generation across runs.
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

    # Maximum number of tokens vLLM processes in one engine step across
    # concurrent requests (throughput vs. GPU memory trade-off).
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

    # Directory where the prompt log written by Model.logPrompts() is saved.
    prompt_log_folder: str = Field(
        default="./data/logs/",
        description="Directory of the prompt log file."
    )

    # Log filename within `prompt_log_folder`.
    prompt_log_file: str = Field(
        default="prompts.log",
        description="Filename of the semicolon-delimited prompt log."
    )

    # Target hardware execution device. Only "cuda" is supported yet: "cpu"
    # passes validation but has no effect, and Model still requires a GPU.
    device: Literal["cpu", "cuda"] = Field(
        default="cuda",
        description="Target execution device; only 'cuda' is supported yet."
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

    # Checkpoint file holding all histories, written after every generation
    # chunk and read back by Model.importHistoriesFromFile() to resume a run.
    prompt_tmp_file: str = Field(
        default="raw.tmp",
        description="Filename of the generation checkpoint."
    )

    # Directory where the checkpoint file is stored.
    prompt_tmp_folder: str = Field(
        default="./data/tmp/",
        description="Directory of the generation checkpoint."
    )

    # Number of histories generated before the next checkpoint is written.
    # Smaller chunks lose less work when a job is killed, larger chunks
    # batch more efficiently.
    chunk_size: int = Field(
        default=500,
        gt=0,
        description="Number of histories generated per chunk; a checkpoint is written after every chunk."
    )

    # Substrings of model_id that require vLLM's expert-parallel mode, i.e.
    # Mixture-of-Experts architectures whose experts should be sharded across
    # the available GPUs. A model is loaded with enable_expert_parallel=True
    # if any of these substrings appear in its model_id.
    expert_parallel_models: list[str] = Field(
        default_factory=lambda: ["Qwen3-30B"],
        description="Substrings of model_id that require enable_expert_parallel=True (e.g. MoE architectures)."
    )