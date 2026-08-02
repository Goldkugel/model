import sys
# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

from logger import Logger
import os

# Chat message roles, matching the conventions used by most chat-tuned
# LLM prompt formats.
systemRole          = "system"
userRole            = "user"
modelRole           = "assistant"

# Special token identifiers (without the surrounding <...>/<|...|>
# delimiters) used by different model families to mark turn/message
# boundaries in their raw prompt format.

# For Gemma
startTurnID         = "start_of_turn"
endTurnID           = "end_of_turn"

# For Llama
startHeaderID       = "start_header_id"
endHeaderID         = "end_header_id"
endOfTextID         = "eot_id"
beginOfTextID       = "begin_of_text"
endOfTextID2        = "end_of_text"

# Delimiter characters used to build the full special-token strings below.
startTag            = "<"
endTag              = ">"
bar                 = "|"
unusedTokens        = "<unused95>"

# Fully-formed special tokens, one per model family, built from the
# identifiers and delimiters above.

# For Gemma
startTurn           = f"{startTag}{startTurnID}{endTag}"
endTurn             = f"{startTag}{endTurnID}{endTag}"

# For Llama
startHeader         = f"{startTag}{bar}{startHeaderID}{bar}{endTag}"
endHeader           = f"{startTag}{bar}{endHeaderID}{bar}{endTag}"
endOfText           = f"{startTag}{bar}{endOfTextID}{bar}{endTag}"
beginOfText         = f"{startTag}{bar}{beginOfTextID}{bar}{endTag}"
endOfText2          = f"{startTag}{bar}{endOfTextID2}{bar}{endTag}"

# Keys used within each message dict passed into writePrompt(): the
# speaker role and the message text itself.
messageRoleElement  = "role"
messageTextElement  = "message"


def writePrompt(path: str, histories: list) -> int:
    """
    Write one or more prompt/response histories to a semicolon-delimited
    log file for later inspection.

    `histories` is a list of histories, where each history is itself a
    list of messages (dicts with a `messageRoleElement` and a
    `messageTextElement` key) representing one full conversation - e.g.
    system/user/assistant turns exchanged with the model.

    One row is written per message, across all histories, in the form:
    history index ; message index within that history ; role ;
    message length (characters) ; message text (newlines stripped).

    Returns the total number of rows (messages) written.
    """
    ret = 0

    l = Logger()
    file_name = os.path.basename(path)
    l.log(f"Logging prompts in '{file_name}'...")

    with open(path, mode="w") as prompt_log_file:
        header = "history;prompt;role;length;text\n"
        prompt_log_file.write(header)
        for index in range(0, len(histories)):
            history = histories[index]
            for index2 in range(0, len(histories[index])):
                prompt      = history[index2]
                role        = prompt[messageRoleElement]
                text        = str(prompt[messageTextElement])
                text2       = text.replace('\n', '')
                message     = f"{index};{index2};{role};{len(text)};{text2}" + "\n" 
                prompt_log_file.write(message)
                ret = ret + 1

    l.log(f"Logging prompts in '{file_name}' completed.")
    
    return ret