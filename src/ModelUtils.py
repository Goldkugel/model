from importlib  import metadata
from typing     import Optional
from logger     import Logger, lockedFile, makeParentFolder, writeJSON
import hashlib
import json
import os
import textwrap

# Chat message roles, as the chat templates expect them
systemRole      = "system"
userRole        = "user"
modelRole       = "assistant"
chatRoles       = (systemRole, userRole, modelRole)

# Values of ModelConfig.device; only gpu is supported yet
gpu             = "cuda"
cpu             = "cpu"
# Name of the environment variable selecting vLLM's backend
target_device   = "VLLM_TARGET_DEVICE"

# Keys of the messages of a history, as the chat template expects them
messageRoleElement = "role"
messageTextElement = "content"
# Reasoning of a thinking model, stored on its answer message; never sent
# back to the model, which only gets role and content.
messageReasoningElement = "reasoning"
# True on an answer cut off at max_tokens (see Model.toJSON())
messageTruncatedElement = "truncated"

def writePrompt(path: str, histories: list[list[dict]]) -> int:
    """
    Writes the histories to a semicolon-delimited log file, one message per
    row, for human inspection. The text is sanitized, so the log cannot be
    read back (the answer store restores histories). The reasoning of
    thinking models is not written.

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
        header = "history;prompt;role;length;text\n"
        prompt_log_file.write(header)

        for history_idx, history in enumerate(histories):
            for message_idx, prompt in enumerate(history):
                role = prompt[messageRoleElement]
                text = str(prompt[messageTextElement])
                
                # One row per message, and no extra columns
                sanitized_text = text.replace('\r\n', ' ').replace('\n', ' ').replace(';', ',')

                # The length is that of the original text
                log_line = f"{history_idx};{message_idx};{role};{len(text)};{sanitized_text}\n"
                
                prompt_log_file.write(log_line)
                total_written += 1

    l.log(f"Logging prompts in '{file_name}' completed.")
    
    return total_written

# Atomic writes and file locks come from the logger package (writeJSON(),
# lockedFile()).

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
    appends it, keeping all other records. Read and write happen under one
    lock (the logger's lockedFile()), so concurrent runs lose no records.

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


def describeError(e: BaseException, width: int = 90) -> str:
    """
    An exception as "<type>: <message>" of at most about `width` characters
    for one log row, the message cut at a word (library messages can be very
    long).
    """
    name = type(e).__name__
    message = " ".join(str(e).split())
    if len(name) + 2 + len(message) > width:
        message = textwrap.shorten(message.rstrip("."), width=max(width - len(name) - 8, 10),
                                   placeholder="")
        # A parenthesis the cut leaves open is dropped with the rest.
        if message.count("(") > message.count(")"):
            message = message[:message.rindex("(")].rstrip()
        message = f"{message.rstrip(' .,;:')} (...)"
    return f"{name}: {message}" if message else name


def modelRevision(model_id: str, configured: Optional[str] = None) -> Optional[str]:
    """
    Returns the Hugging Face revision of `model_id` that vLLM loads: the
    configured one (config.revision), otherwise the commit hash of the
    locally cached snapshot. Returns None for a local model directory, or if
    no cached version is found; the latter is logged, since the run metadata
    then lacks the model version.
    """
    if configured:
        return configured
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
        # vLLM downloads only the files it needs, so huggingface_hub may find
        # the snapshot incomplete; the cache still records its commit.
        revision = cachedRevision(model_id)
        if revision is not None:
            return revision
        Logger().log(f"Could not determine the revision of model '{model_id}'.")
        Logger().log(f"Reason: {describeError(e)}.")
        return None


def cachedRevision(model_id: str) -> Optional[str]:
    """
    Returns the commit the Hugging Face cache records for the main branch of
    `model_id` (models--<org>--<name>/refs/main), or None if there is none.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:
        return None
    path = os.path.join(HF_HUB_CACHE, "models--" + model_id.replace("/", "--"), "refs", "main")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read().strip() or None


def fileSHA256(path: str) -> Optional[str]:
    """
    Returns the SHA-256 hex digest of the file at `path`, or None if it
    does not exist.
    """
    if not os.path.isfile(path):
        return None
    with open(path, mode="rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
