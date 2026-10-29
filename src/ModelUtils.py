import sys
# Prevent Python from writing .pyc bytecode cache files to keep container 
# environments clean
sys.dont_write_bytecode = True

from importlib  import metadata
from typing     import Optional
from logger     import Logger
import pandas   as pd
import hashlib
import json
import os
import csv

# ==============================================================================
# Role Definitions
# Standard chat message roles matching conventions used by chat-tuned LLM 
# interfaces.
# ==============================================================================
systemRole      = "system"
userRole        = "user"
modelRole       = "assistant"

# ==============================================================================
# Execution Device Identifiers
# ==============================================================================
# Values of ModelConfig.device; only gpu is supported yet
gpu             = "cuda"
cpu             = "cpu"
# Name of the environment variable selecting vLLM's backend
target_device   = "VLLM_TARGET_DEVICE"

# ==============================================================================
# History Key Constants
# Keys of the message dictionaries that make up a conversation history, in
# the format expected by the tokenizer's chat template.
# ==============================================================================
messageRoleElement = "role"
messageTextElement = "content"

# ==============================================================================
# Checkpoint Column Constants
# Column of the checkpoint CSV that holds a history's system prompt, so that
# importHistoriesFromFile() can restore it with the system role.
# ==============================================================================
systemColumn = "system"


# ==============================================================================
# Logging Utilities
# ==============================================================================

def writePrompt(path: str, histories: list[list[dict]]) -> int:
    """
    Writes prompt and response histories to a semicolon-delimited log file,
    one message per row, for human inspection. The text is sanitized, so
    unlike the checkpoint the log cannot be read back into histories.

    :param path: Destination file path for the prompt log.
    :param histories: List of conversation threads, where each thread is a list
        of dictionaries containing role and message content keys.
    :return: Total number of individual messages logged across all histories.
    """
    total_written = 0

    l = Logger()
    file_name = os.path.basename(path)
    l.log(f"Logging prompts in '{file_name}'...")

    # Ensure output parent directories exist before writing
    parent_dir = os.path.dirname(path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    with open(path, mode="w", encoding="utf-8") as prompt_log_file:
        # Write CSV header row
        header = "history;prompt;role;length;text\n"
        prompt_log_file.write(header)

        # Iterate over each conversation history thread
        for history_idx, history in enumerate(histories):
            # Iterate over each message turn within the history thread
            for message_idx, prompt in enumerate(history):
                role = prompt[messageRoleElement]
                text = str(prompt[messageTextElement])
                
                # Replace newlines with spaces to maintain one log entry per 
                # line, and sanitize embedded semicolons to prevent column 
                # misalignment.
                sanitized_text = text.replace('\r\n', ' ').replace('\n', ' ').replace(';', ',')

                # Format row: history_index; message_index; role; 
                # raw_character_length; sanitized_text
                log_line = f"{history_idx};{message_idx};{role};{len(text)};{sanitized_text}\n"
                
                prompt_log_file.write(log_line)
                total_written += 1

    l.log(f"Logging prompts in '{file_name}' completed.")
    
    return total_written

# ==============================================================================
# File Utilities
# ==============================================================================

def _writeAtomically(path: str, write) -> None:
    """
    Calls write(tmp_path) to write the file to "<path>.tmp", which then
    replaces `path` in a single operation. An interrupted write (e.g. a Slurm
    job being killed) therefore never leaves a partially written file.

    :param path: Destination file path.
    :param write: Function writing the complete file to the path it is given.
    """
    # Ensure output parent directories exist before writing
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    tmp_path = path + ".tmp"
    try:
        write(tmp_path)
        # os.replace is atomic on both POSIX and Windows
        os.replace(tmp_path, path)
    finally:
        # Only left behind if writing or replacing failed
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def writeCheckpointCSV(data: pd.DataFrame, path: str) -> int:
    """
    Writes a DataFrame atomically (see _writeAtomically()) to a
    semicolon-delimited, fully quoted CSV file.

    :param data: DataFrame to write.
    :param path: Destination file path of the checkpoint.
    :return: Number of rows written.
    """
    l = Logger()
    file_name = os.path.basename(path)
    l.log(f"Writing checkpoint '{file_name}'...")

    _writeAtomically(path, lambda tmp_path: data.to_csv(
        tmp_path,
        sep=";",
        encoding="utf-8",
        index=False,
        quoting=csv.QUOTE_ALL
    ))

    l.log(f"Writing checkpoint '{file_name}' completed.")

    return len(data)


def writeJSON(data, path: str) -> None:
    """
    Writes a JSON-serializable object atomically (see _writeAtomically()),
    indented for reading.

    :param data: Object to write.
    :param path: Destination file path.
    """
    def write(tmp_path: str) -> None:
        with open(tmp_path, mode="w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

    _writeAtomically(path, write)


# ==============================================================================
# Run Metadata Utilities
# ==============================================================================

# Packages whose installed versions are recorded in the run metadata
metadataPackages = ["model", "logger", "vllm", "torch", "transformers"]


def packageVersions(packages: list[str]) -> dict[str, Optional[str]]:
    """
    Returns the installed version of every package in `packages`, or None
    for a package that is not installed.
    """
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def modelRevision(model_id: str) -> Optional[str]:
    """
    Returns the Hugging Face commit hash of the locally cached snapshot of
    `model_id`, i.e. the exact model version vLLM loads. Returns None for a
    local model directory, or if the snapshot cannot be found.
    """
    if os.path.isdir(model_id):
        return None
    try:
        # huggingface_hub is installed with vllm; imported here so the
        # package still imports without it (e.g. in CI)
        from huggingface_hub import snapshot_download
        snapshot = snapshot_download(model_id, local_files_only=True)
        # Cached snapshots live in .../snapshots/<commit hash>
        return os.path.basename(os.path.normpath(snapshot))
    except Exception:
        return None


def fileSHA256(path: str) -> Optional[str]:
    """
    Returns the SHA-256 hex digest of the file at `path`, or None if it
    does not exist.
    """
    if not os.path.isfile(path):
        return None
    with open(path, mode="rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
