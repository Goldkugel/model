import sys
# Prevent Python from writing .pyc bytecode cache files to keep container 
# environments clean
sys.dont_write_bytecode = True

from logger import Logger
import os

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
gpu             = "cuda"
cpu             = "cpu"
target_device   = "VLLM_TARGET_DEVICE"

# ==============================================================================
# Raw Special Token Identifiers (Without Enclosing Delimiters)
# Used to construct exact turn and boundary tokens across distinct model 
# families.
# ==============================================================================

# --- Gemma Family Identifiers ---
startTurnID     = "start_of_turn"
endTurnID       = "end_of_turn"

# --- Llama Family Identifiers ---
startHeaderID   = "start_header_id"
endHeaderID     = "end_header_id"
endOfTextID     = "eot_id"
beginOfTextID   = "begin_of_text"
endOfTextID2    = "end_of_text"

# ==============================================================================
# Structural Delimiters
# Characters used to encapsulate raw token IDs into valid special-token tags.
# ==============================================================================
startTag        = "<"
endTag          = ">"
bar             = "|"
unusedTokens    = "<unused95>"

# ==============================================================================
# Fully Formed Special Tokens
# Assembled token tags corresponding to respective tokenizer template formats.
# ==============================================================================

# --- Gemma Tag Definitions ---
startTurn       = f"{startTag}{startTurnID}{endTag}"        # e.g., "<start_of_turn>"
endTurn         = f"{startTag}{endTurnID}{endTag}"          # e.g., "<end_of_turn>"

# --- Llama Tag Definitions ---
startHeader     = f"{startTag}{bar}{startHeaderID}{bar}{endTag}"  # e.g., "<|start_header_id|>"
endHeader       = f"{startTag}{bar}{endHeaderID}{bar}{endTag}"    # e.g., "<|end_header_id|>"
endOfText       = f"{startTag}{bar}{endOfTextID}{bar}{endTag}"    # e.g., "<|eot_id|>"
beginOfText     = f"{startTag}{bar}{beginOfTextID}{bar}{endTag}"  # e.g., "<|begin_of_text|>"
endOfText2      = f"{startTag}{bar}{endOfTextID2}{bar}{endTag}"   # e.g., "<|end_of_text|>"

# ==============================================================================
# History Key Constants
# Property dictionary keys used inside message objects passed to writePrompt().
# ==============================================================================
messageRoleElement = "role"
messageTextElement = "content"


# ==============================================================================
# Logging Utilities
# ==============================================================================

def writePrompt(path: str, histories: list[list[dict]]) -> int:
    """
    Writes prompt and response histories to a semicolon-delimited CSV log file.

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