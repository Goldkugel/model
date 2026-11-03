from importlib  import metadata
from typing     import Optional
from logger     import Logger, lockedFile, makeParentFolder, writeJSON
import hashlib
import json
import os

# ==============================================================================
# Role Definitions
# Standard chat message roles matching conventions used by chat-tuned LLM 
# interfaces.
# ==============================================================================
systemRole      = "system"
userRole        = "user"
modelRole       = "assistant"
# Every role a message may have
chatRoles       = (systemRole, userRole, modelRole)

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
# Reasoning of a thinking model, stored on its answer message next to the
# answer text. It is not sent back to the model: Model passes only the role
# and content keys to vLLM and the chat template.
messageReasoningElement = "reasoning"
# Set to True on an answer that was cut off at max_tokens. toJSON() then only
# accepts JSON that starts at the answer's first bracket, so it does not
# return a fragment of a cut-off object or array.
messageTruncatedElement = "truncated"

# ==============================================================================
# Logging Utilities
# ==============================================================================

def writePrompt(path: str, histories: list[list[dict]]) -> int:
    """
    Writes prompt and response histories to a semicolon-delimited log file,
    one message per row, for human inspection. The text is sanitized, so
    the log cannot be read back into histories (the answer store is used
    to restore them). Only
    the message text is written, not the reasoning of thinking models.

    :param path: Destination file path for the prompt log.
    :param histories: List of conversation threads, where each thread is a list
        of dictionaries containing role and message content keys.
    :return: Total number of individual messages logged across all histories.
    """
    total_written = 0

    l = Logger()
    file_name = os.path.basename(path)
    l.log(f"Logging prompts in '{file_name}'...")

    makeParentFolder(path)

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

# The answer store (see AnswerStore), the run metadata and the file lock use
# the file helpers of the logger package: writeAtomically(), writeJSON() and
# lockedFile().

def fileStamp(path: str) -> Optional[tuple[int, int, int]]:
    """
    Returns (inode, size, modification time) of the file at `path`, or None
    if it does not exist. Every write of another process changes the stamp:
    writeAtomically() replaces the file, which gives it a new inode.
    """
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        return None
    return (stat.st_ino, stat.st_size, stat.st_mtime_ns)


def updateRunRecord(path: str, record: dict, key: str = "run_id") -> None:
    """
    Replaces the record with the same `key` in the JSON list at `path`, or
    appends it, keeping all other records. The read and the write happen
    under one lock (see the logger's lockedFile()), so records of runs updating the file
    at the same time are not lost.

    :param path: Path of the JSON file; created if it does not exist.
    :param record: Record to store; must contain `key`.
    :param key: Key identifying a record.
    :raises ValueError: If the existing file is not a JSON list.
    """
    with lockedFile(path):
        records = []
        if os.path.isfile(path):
            with open(path, mode="r", encoding="utf-8") as f:
                records = json.load(f)
            if not isinstance(records, list):
                raise ValueError(f"Run metadata file '{path}' does not contain a JSON list.")

        records = [entry for entry in records if entry.get(key) != record[key]]
        records.append(record)
        writeJSON(records, path)


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
    local model directory, or if the snapshot cannot be found; the latter is
    logged, since the run metadata then lacks the model version.
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
    except Exception as e:
        Logger().log(f"Could not determine the revision of model '{model_id}': {e!r}")
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
