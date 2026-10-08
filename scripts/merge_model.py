import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ================================================================
# Configuration
# ================================================================

BASE_MODEL = "HuggingFaceTB/SmolLM2-1.7B"

ADAPTER_PATH = "outputs/smollm2-devops-lora"

OUTPUT_PATH = "outputs/smollm2-devops-merged"


# ================================================================
# Start
# ================================================================

print("=" * 70)
print("SmolLM2 + LoRA MODEL MERGE")
print("=" * 70)

print(f"Base model : {BASE_MODEL}")
print(f"LoRA       : {ADAPTER_PATH}")
print(f"Output     : {OUTPUT_PATH}")
print("=" * 70)


# ================================================================
# Validate adapter
# ================================================================

if not os.path.exists(ADAPTER_PATH):
    raise FileNotFoundError(
        f"LoRA adapter not found: {ADAPTER_PATH}"
    )

adapter_file = os.path.join(
    ADAPTER_PATH,
    "adapter_model.safetensors"
)

if not os.path.exists(adapter_file):
    raise FileNotFoundError(
        f"LoRA weights not found: {adapter_file}"
    )


# ================================================================
# Create output directory
# ================================================================

os.makedirs(
    OUTPUT_PATH,
    exist_ok=True
)


# ================================================================
# Load tokenizer
# ================================================================

print("\n[1/5] Loading tokenizer...")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_PATH
    )
    print("Tokenizer loaded from adapter.")
except Exception:
    print("Adapter tokenizer not found.")
    print("Loading tokenizer from base model...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ================================================================
# Load base model
# ================================================================

print("\n[2/5] Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
)

print("Base model loaded.")


# ================================================================
# Load LoRA adapter
# ================================================================

print("\n[3/5] Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

print("LoRA adapter loaded.")


# ================================================================
# Merge LoRA into base model
# ================================================================

print("\n[4/5] Merging LoRA adapter into base model...")

merged_model = model.merge_and_unload()

print("LoRA successfully merged.")


# ================================================================
# Save merged model
# ================================================================

print("\n[5/5] Saving standalone model...")

merged_model.save_pretrained(
    OUTPUT_PATH,
    safe_serialization=True,
)

tokenizer.save_pretrained(
    OUTPUT_PATH
)

print("Model saved.")


# ================================================================
# Finished
# ================================================================

print("\n")
print("=" * 70)
print("MERGE COMPLETED")
print("=" * 70)

print(f"\nStandalone model:")
print(OUTPUT_PATH)

print("\nFiles:")

for filename in sorted(os.listdir(OUTPUT_PATH)):
    filepath = os.path.join(
        OUTPUT_PATH,
        filename
    )

    if os.path.isfile(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)

        print(
            f"  {filename:<45} "
            f"{size_mb:>10.2f} MB"
        )

print("\n")
print("You can now load this model WITHOUT PEFT/LoRA.")
print("=" * 70)
