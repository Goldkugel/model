"""
GPU diagnostic: checks that CUDA is visible, that a configured model loads
into vLLM, and that it generates answers. Run on a GPU node from the
repository root (it reads ./config/config.yaml), e.g. via
scripts/runTest.sh:

    python3 -m model.diagnostic [index]

`index` selects the model in config.model_id (default 0). Exits with code 1
if a check fails.
"""

import os
import sys
import tempfile

import torch
import yaml

from .Model      import Model, standard_directory
from .ModelUtils import (
    userRole,
    modelRole,
    messageRoleElement,
    messageTextElement,
    messageReasoningElement,
)

# Prompts of the smoke test: the first checks toJSON(), the second a free answer
diagnostic_prompts = [
    'Reply with only this JSON object and nothing else: {"answer": 4}',
    "Name one symptom of microcephaly in one sentence.",
]


def main():
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    print("=" * 50)
    print("SLURM & GPU ENVIRONMENT DIAGNOSTIC")
    print("=" * 50)

    # 1. Check Slurm & CUDA visibility
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set")
    print(f"CUDA_VISIBLE_DEVICES : {cuda_visible}")

    cuda_available = torch.cuda.is_available()
    print(f"PyTorch CUDA Available: {cuda_available}")

    if not cuda_available:
        print("\n❌ ERROR: CUDA is not available to PyTorch.")
        print("Check if the CUDA module is loaded or if PyTorch CPU-only is installed.")
        sys.exit(1)

    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU Device Count     : {device_count}")
    print(f"Active GPU           : {device_name}")

    print("\n" + "=" * 50)
    print("INITIALIZING MODEL")
    print("=" * 50)

    # 2. Load the selected model into vLLM. The checkpoint and run metadata
    # of the test generation go to a temporary folder, so the checkpoint of
    # a real run is never overwritten.
    with open(standard_directory, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    output_dir = tempfile.mkdtemp(prefix="modeltest_")
    config["llm"]["prompt_tmp_folder"] = output_dir
    config["llm"]["metadata_folder"] = output_dir

    try:
        model = Model(config, index=index)
        print(f"✅ Model '{model.model}' successfully initialized.")
    except Exception as e:
        print(f"\n❌ ERROR initializing Model: {e}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("GENERATING")
    print("=" * 50)
    print(f"enable_thinking      : {model.config.enable_thinking}")
    print(f"Output folder        : {output_dir}")

    # 3. Generate an answer for every diagnostic prompt
    errors = []
    try:
        model.addPrompt(userRole, diagnostic_prompts)
        generated = model.generate()

        for idx, history in enumerate(model.getMessageHistories()):
            message = history[-1]
            print(f"\n--- History {idx} ---")
            print(f"Prompt   : {history[0][messageTextElement]}")
            print(f"Reasoning: {len(message.get(messageReasoningElement, ''))} characters")
            print(f"Answer   : {message.get(messageTextElement, '') if message[messageRoleElement] == modelRole else '(none)'}")

        if generated != len(diagnostic_prompts):
            errors.append(f"{generated} of {len(diagnostic_prompts)} answers generated.")
        if model.skippedHistories:
            errors.append(f"Histories {model.skippedHistories} exceed max_model_len.")
        if model.truncatedHistories:
            errors.append(f"Histories {model.truncatedHistories} were cut off at max_tokens "
                          f"({model.config.max_tokens}); increase max_tokens for thinking models.")
        if model.toJSON(0) != {"answer": 4}:
            print("\n⚠️  WARNING: toJSON() did not return {\"answer\": 4} for history 0.")
    except Exception as e:
        errors.append(f"Generation failed: {e!r}")
    finally:
        # 4. Free GPU memory and shut down vLLM's distributed environment
        model.close()

    if errors:
        for error in errors:
            print(f"\n❌ ERROR: {error}")
        sys.exit(1)

    print("\n🎉 ALL CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
