import sys
# Prevent Python from writing .pyc bytecode cache files to keep container 
# environments clean
sys.dont_write_bytecode = True

from importlib  import metadata
from typing     import Optional
from logger     import Logger
import pandas   as pd
import contextlib
import hashlib
import json
import os
import csv

# File locks (see lockedFile()) are only available on POSIX systems
try:
    import fcntl
except ImportError:
    fcntl = None

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
# Checkpoint Column Constants
# Column of the checkpoint CSV that holds a history's system prompt, so that
# importHistoriesFromFile() can restore it with the system role, and the
# suffix of the reasoning columns.
# ==============================================================================
systemColumn = "system"
# Suffix of the column holding the reasoning of a turn's answer, e.g.
# "mapping_reasoning" next to "mapping" and "mapping_answer". Only written
# for histories with reasoning.
reasoningColumnSuffix = "_reasoning"
# Suffix of the column marking cut-off answers with "true", e.g.
# "mapping_truncated". Only written if some answer of that turn was cut off.
truncatedColumnSuffix = "_truncated"
# Column holding the model that wrote the checkpoint, so that a checkpoint
# is never resumed with another model. The leading underscore keeps it apart
# from the prompt columns, which are named after prompt files.
modelColumn = "_model"


# ==============================================================================
# Logging Utilities
# ==============================================================================

def writePrompt(path: str, histories: list[list[dict]]) -> int:
    """
    Writes prompt and response histories to a semicolon-delimited log file,
    one message per row, for human inspection. The text is sanitized, so
    unlike the checkpoint the log cannot be read back into histories. Only
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

    # Next to the target, so os.replace() stays within one file system
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


def fileStamp(path: str) -> Optional[tuple[int, int, int]]:
    """
    Returns (inode, size, modification time) of the file at `path`, or None
    if it does not exist. Every write of another process changes the stamp:
    _writeAtomically() replaces the file, which gives it a new inode.
    """
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        return None
    return (stat.st_ino, stat.st_size, stat.st_mtime_ns)


@contextlib.contextmanager
def lockedFile(path: str):
    """
    Holds an exclusive lock on "<path>.lock" while the block runs, so that
    processes updating the same file (e.g. the jobs of a Slurm job array)
    take turns. Where file locks are not supported (Windows, or a cluster
    file system mounted without lock support), this is logged and the block
    runs without a lock.

    :param path: Path of the file the lock protects.
    """
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    lock_path = path + ".lock"
    with open(lock_path, mode="a", encoding="utf-8") as handle:
        locked = False
        if fcntl is None:
            Logger().log(f"File locks are not supported here; updating '{path}' without a lock.")
        else:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX)
                locked = True
            except OSError as e:
                Logger().log(f"Could not lock '{lock_path}' ({e!r}); updating '{path}' without a lock.")
        try:
            yield
        finally:
            if locked:
                fcntl.flock(handle, fcntl.LOCK_UN)


def updateRunRecord(path: str, record: dict, key: str = "run_id") -> None:
    """
    Replaces the record with the same `key` in the JSON list at `path`, or
    appends it, keeping all other records. The read and the write happen
    under one lock (see lockedFile()), so records of runs updating the file
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
