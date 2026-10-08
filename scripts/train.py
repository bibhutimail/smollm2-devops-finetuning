import os

# -------------------------------------------------------------------
# MPS fallback
# -------------------------------------------------------------------
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


# ===================================================================
# Configuration
# ===================================================================

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B"

TRAIN_FILE = "data/train.jsonl"
VALIDATION_FILE = "data/validation.jsonl"

OUTPUT_DIR = "outputs/smollm2-devops-lora"


# ===================================================================
# Helper
# ===================================================================

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ===================================================================
# Device
# ===================================================================

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


# ===================================================================
# Startup information
# ===================================================================

print_section("SmolLM2 LoRA Fine-Tuning")

print(f"PyTorch version : {torch.__version__}")
print(f"Device          : {DEVICE}")
print(f"MPS available   : {torch.backends.mps.is_available()}")

if DEVICE == "mps":
    print("Apple Silicon MPS acceleration enabled.")

print("=" * 70)


# ===================================================================
# Check dataset files
# ===================================================================

print("\nLoading dataset...")

if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(
        f"Training dataset not found: {TRAIN_FILE}"
    )

if not os.path.exists(VALIDATION_FILE):
    raise FileNotFoundError(
        f"Validation dataset not found: {VALIDATION_FILE}"
    )


# ===================================================================
# Load dataset
# ===================================================================

dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VALIDATION_FILE,
    },
)

print(dataset)

print(f"\nTraining examples   : {len(dataset['train'])}")
print(f"Validation examples : {len(dataset['validation'])}")


# ===================================================================
# Validate dataset columns
# ===================================================================

required_columns = [
    "instruction",
    "input",
    "output",
]

for split_name in ["train", "validation"]:

    columns = dataset[split_name].column_names

    for column in required_columns:
        if column not in columns:
            raise ValueError(
                f"Missing required column '{column}' "
                f"in {split_name} dataset."
            )


# ===================================================================
# Format dataset
# ===================================================================

print("\nFormatting training dataset...")


def format_example(example):
    instruction = str(example["instruction"]).strip()
    input_text = str(example["input"]).strip()
    output = str(example["output"]).strip()

    if input_text:
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
            f"{output}"
        )

    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
        f"{output}"
    )


train_dataset = dataset["train"].map(
    lambda example: {
        "text": format_example(example)
    }
)

validation_dataset = dataset["validation"].map(
    lambda example: {
        "text": format_example(example)
    }
)

print("Dataset formatting completed.")


# ===================================================================
# Display sample
# ===================================================================

print_section("SAMPLE TRAINING EXAMPLE")

print(train_dataset[0]["text"])

print("=" * 70)


# ===================================================================
# Load tokenizer
# ===================================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded.")


# ===================================================================
# Load model
# ===================================================================

print("\nLoading SmolLM2-1.7B...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
)

print("Base model loaded.")


# -------------------------------------------------------------------
# Gradient checkpointing compatibility
# -------------------------------------------------------------------

model.config.use_cache = False


# ===================================================================
# Move model to device
# ===================================================================

print(f"\nMoving model to {DEVICE}...")

model = model.to(DEVICE)

print("Model moved successfully.")


# ===================================================================
# LoRA configuration
# ===================================================================

print("\nConfiguring LoRA...")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM",

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)

print("LoRA configuration created.")


# ===================================================================
# Training configuration
# ===================================================================

print("\nCreating training configuration...")

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    # Start with a small experiment
    num_train_epochs=1,
    max_steps=3,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,

    learning_rate=1e-4,

    max_length=128,
    dataset_text_field="text",

    # Disable evaluation and checkpoints during this test
    eval_strategy="no",
    save_strategy="no",

    logging_steps=1,
    logging_first_step=True,
    report_to="none",

    fp16=False,
    bf16=False,

    # Test without gradient checkpointing first
    gradient_checkpointing=False,

    remove_unused_columns=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,

    seed=42,
)


# ===================================================================
# Create Trainer
# ===================================================================

print("\nCreating SFTTrainer...")

trainer = SFTTrainer(
    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    processing_class=tokenizer,

    peft_config=lora_config,
)

print("SFTTrainer created successfully.")


# ===================================================================
# Trainable parameters
# ===================================================================

print_section("TRAINABLE PARAMETERS")

trainer.model.print_trainable_parameters()


# ===================================================================
# Start training
# ===================================================================

print_section("STARTING TRAINING")

print("Training started...")
print("This may take some time on Apple M2.")

train_result = trainer.train()


# ===================================================================
# Training results
# ===================================================================

print_section("TRAINING COMPLETED")

print("Training metrics:")

metrics = train_result.metrics

for key, value in metrics.items():
    print(f"{key}: {value}")


# ===================================================================
# Save model
# ===================================================================

print("\nSaving LoRA adapter...")

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

trainer.save_model(OUTPUT_DIR)

print("LoRA adapter saved.")


# ===================================================================
# Save tokenizer
# ===================================================================

print("\nSaving tokenizer...")

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print("Tokenizer saved.")


# ===================================================================
# Final information
# ===================================================================

print_section("DONE")

print(
    f"""
Fine-tuning completed successfully.

Base model:
{MODEL_NAME}

Training examples:
{len(train_dataset)}

Validation examples:
{len(validation_dataset)}

Device:
{DEVICE}

LoRA output:
{OUTPUT_DIR}

You can now use the LoRA adapter for inference.
"""
)

print("=" * 70)
