from typing     import Literal, Optional
from pydantic   import BaseModel, ConfigDict, Field

class ModelConfig(BaseModel):
    """
    Pydantic model defining and validating the "llm" section of the
    configuration file: model selection, vLLM engine and sampling
    parameters, prompt templates, checkpointing and prompt logging.
    """

    # Unknown keys (typos) are errors
    model_config = ConfigDict(extra="forbid")

    # e.g. "google/medgemma-27b-text-it"
    model_id: list[str] = Field(
        default_factory=list,
        description="List of HuggingFace or local model repository identifiers."
    )

    # Passed to vLLM and to every sampling request
    seed: int = Field(
        default=2898231092,
        description="Random seed for generation determinism."
    )

    # 0.0 = greedy
    temperature: float = Field(
        default=0.01,
        ge=0.0,
        le=2.0,
        description="Sampling temperature controlling output randomness."
    )

    top_p: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
        description="Top-p (nucleus) sampling threshold."
    )

    # Prompt plus output; generate() skips histories whose prompt alone
    # reaches it.
    max_model_len: int = Field(
        default=8192,
        gt=0,
        description="Maximum model context window size in tokens."
    )

    # Tokens per vLLM engine step across requests (throughput vs. GPU memory)
    max_num_batched_tokens: int = Field(
        default=16384,
        gt=0,
        description="vLLM batch engine token throughput limit."
    )

    # True starts faster and needs less GPU memory, False generates faster
    enforce_eager: bool = Field(
        default=True,
        description="Run vLLM in eager mode without CUDA graphs."
    )

    gpu_memory_utilization: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
        description="Fraction of GPU memory reserved by vLLM."
    )

    # e.g. "auto" (as in the model's config), "bfloat16", "float16"
    dtype: str = Field(
        default="auto",
        description="Data type of the model weights and activations."
    )

    # e.g. "awq", "gptq", "fp8"; null loads the weights as stored (vLLM
    # detects pre-quantized checkpoints itself)
    quantization: Optional[str] = Field(
        default=None,
        description="Quantization method passed to vLLM, or null."
    )

    enable_prefix_caching: bool = Field(
        default=True,
        description="Reuse the KV cache of common prompt prefixes."
    )

    # Thinking models (Qwen3, DeepSeek-R1) write their reasoning between
    # reasoning_start_tag and reasoning_end_tag; it is stored apart from the
    # answer (see Model._splitReasoning()). They need a much larger max_tokens
    # (e.g. 8192 to 32768) and most model cards recommend a temperature of
    # about 0.6, as near-greedy decoding makes the reasoning repeat itself.

    # For hybrid models such as Qwen3; null keeps the template's default (on
    # for Qwen3), templates without the switch ignore it.
    enable_thinking: Optional[bool] = Field(
        default=None,
        description="enable_thinking passed to the chat template, or null for the template's default."
    )

    # Some templates (e.g. DeepSeek-R1) put it into the prompt, so the output
    # only has the end tag; both cases are handled.
    reasoning_start_tag: str = Field(
        default="<think>",
        description="Marker that opens the reasoning of a thinking model."
    )

    # The answer is the text after its first occurrence
    reasoning_end_tag: Optional[str] = Field(
        default="</think>",
        min_length=1,
        description="Marker that closes the reasoning of a thinking model, or null to disable splitting."
    )

    # Includes the reasoning of thinking models
    max_tokens: int = Field(
        default=2048,
        gt=0,
        description="Maximum generation length per request."
    )

    # Written by Model.logPrompts()
    prompt_log_folder: str = Field(
        default="./data/logs/",
        description="Directory of the prompt log file."
    )

    prompt_log_file: str = Field(
        default="prompts.log",
        description="Filename of the semicolon-delimited prompt log."
    )

    # "cpu" passes validation but has no effect; Model still requires a GPU.
    device: Literal["cpu", "cuda"] = Field(
        default="cuda",
        description="Target execution device; only 'cuda' is supported yet."
    )

    # Run metadata: versions, model revision, GPUs, config, prompt file
    # hashes and generation statistics
    metadata_folder: str = Field(
        default="./data/logs/",
        description="Directory of the run metadata file."
    )

    # A JSON list with one entry per Model instance
    metadata_file: str = Field(
        default="run_metadata.json",
        description="Filename of the run metadata JSON file."
    )

    # Prompt templates (.md or .txt)
    prompt_folder: str = Field(
        default="./prompts/",
        description="Directory containing input prompt template files."
    )

    prompt_files: list[str] = Field(
        default_factory=list,
        description="Ordered sequence of prompt template filenames."
    )

    # The answer store (the checkpoint, see AnswerStore), written after every
    # chunk; the model name and the postfix are added to its name.
    prompt_tmp_file: str = Field(
        default="answers.json",
        description="Filename of the answer store (a JSON file)."
    )

    prompt_tmp_folder: str = Field(
        default="./data/tmp/",
        description="Directory of the generation checkpoint."
    )

    # Smaller chunks lose less work when a job is killed, larger ones batch
    # more efficiently.
    chunk_size: int = Field(
        default=500,
        gt=0,
        description="Number of histories generated per chunk; a checkpoint is written after every chunk."
    )

    # Mixture-of-Experts models, whose experts are sharded across the GPUs
    expert_parallel_models: list[str] = Field(
        default_factory=lambda: ["Qwen3-30B"],
        description="Substrings of model_id that require enable_expert_parallel=True (e.g. MoE architectures)."
    )