import sys

# Prevent Python from writing .pyc bytecode cache files to keep container environments clean
sys.dont_write_bytecode = True

from typing     import Literal, Optional
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

    # Skip CUDA graph capture. True starts faster and needs less GPU memory,
    # False generates faster once the model is loaded.
    enforce_eager: bool = Field(
        default=True,
        description="Run vLLM in eager mode without CUDA graphs."
    )

    # Fraction of every GPU's memory vLLM may use for weights and KV cache.
    gpu_memory_utilization: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
        description="Fraction of GPU memory reserved by vLLM."
    )

    # Data type of weights and activations, e.g. "auto" (as in the model's
    # config), "bfloat16", "float16".
    dtype: str = Field(
        default="auto",
        description="Data type of the model weights and activations."
    )

    # Quantization method, e.g. "awq", "gptq", "fp8"; null loads the weights
    # as stored (vLLM detects pre-quantized checkpoints itself).
    quantization: Optional[str] = Field(
        default=None,
        description="Quantization method passed to vLLM, or null."
    )

    # Reuse the KV cache of shared prompt prefixes across requests, which
    # speeds up histories that start with the same prompt template.
    enable_prefix_caching: bool = Field(
        default=True,
        description="Reuse the KV cache of common prompt prefixes."
    )

    # Thinking (reasoning) models such as Qwen3 or DeepSeek-R1 write their
    # reasoning before the answer, between reasoning_start_tag and
    # reasoning_end_tag. Model.generate() stores it separately from the
    # answer (see Model._splitReasoning()). Reasoning takes up output
    # tokens, so these models need a much larger max_tokens (e.g. 8192 to
    # 32768) and a matching max_model_len; most model cards also recommend
    # a temperature of about 0.6 instead of near-greedy decoding, which
    # tends to make the reasoning repeat itself.

    # Switches thinking on (True) or off (False) for hybrid models such as
    # Qwen3, by passing enable_thinking to the chat template. null leaves
    # the chat template's default (on for Qwen3); models whose template has
    # no such switch ignore it.
    enable_thinking: Optional[bool] = Field(
        default=None,
        description="enable_thinking passed to the chat template, or null for the template's default."
    )

    # Marker that opens the reasoning. Some chat templates (e.g.
    # DeepSeek-R1) put it into the prompt, so the output only contains the
    # closing marker; both cases are handled.
    reasoning_start_tag: str = Field(
        default="<think>",
        description="Marker that opens the reasoning of a thinking model."
    )

    # Marker that closes the reasoning; the answer is the text after its
    # first occurrence. null disables the splitting, so the whole output is
    # stored as the answer.
    reasoning_end_tag: Optional[str] = Field(
        default="</think>",
        min_length=1,
        description="Marker that closes the reasoning of a thinking model, or null to disable splitting."
    )

    # Maximum number of new tokens generated per output sequence. For
    # thinking models this includes the reasoning.
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

    # Directory of the run metadata file (versions, model revision, GPUs,
    # config, prompt file hashes and generation statistics).
    metadata_folder: str = Field(
        default="./data/logs/",
        description="Directory of the run metadata file."
    )

    # Run metadata filename within `metadata_folder`. Holds a JSON list with
    # one entry per Model instance.
    metadata_file: str = Field(
        default="run_metadata.json",
        description="Filename of the run metadata JSON file."
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