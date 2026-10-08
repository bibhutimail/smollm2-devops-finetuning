import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "outputs/smollm2-devops-merged"
MAX_NEW_TOKENS = 80

# ============================================================
# Device
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("=" * 70)
print("SmolLM2 DEVOPS - INTERACTIVE CHAT")
print("=" * 70)
print(f"PyTorch : {torch.__version__}")
print(f"Device  : {device}")
print(f"Model   : {MODEL_PATH}")
print()

# ============================================================
# Load Tokenizer
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded.")

# ============================================================
# Load Model - ONLY ONCE
# ============================================================

print("\nLoading merged model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16
)

model = model.to(device)
model.eval()

print("Model loaded successfully.")

# ============================================================
# Warm-up
# ============================================================

print("\nWarming up model...")

warmup_prompt = "What is Kubernetes?"

warmup_inputs = tokenizer(
    warmup_prompt,
    return_tensors="pt"
)

warmup_input_ids = warmup_inputs["input_ids"].to(device)
warmup_attention_mask = warmup_inputs["attention_mask"].to(device)

warmup_start = time.time()

with torch.no_grad():
    _ = model.generate(
        input_ids=warmup_input_ids,
        attention_mask=warmup_attention_mask,
        max_new_tokens=2,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

if device.type == "mps":
    torch.mps.synchronize()

warmup_time = time.time() - warmup_start

print(f"Warm-up completed in {warmup_time:.2f} seconds")

# ============================================================
# Interactive Question Loop
# ============================================================

print("\n" + "=" * 70)
print("MODEL READY")
print("=" * 70)
print("Ask your DevOps questions.")
print("Type 'exit', 'quit', or 'q' to stop.")
print("=" * 70)

while True:

    print()
    question = input("You: ").strip()

    # --------------------------------------------------------
    # Exit
    # --------------------------------------------------------

    if question.lower() in ["exit", "quit", "q"]:
        print("\nGoodbye!")
        break

    if not question:
        continue

    # --------------------------------------------------------
    # Prepare prompt
    # --------------------------------------------------------

    prompt = f"""### Instruction:
{question}

### Response:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    if device.type == "mps":
        torch.mps.synchronize()

    start_time = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    if device.type == "mps":
        torch.mps.synchronize()

    generation_time = time.time() - start_time

    # --------------------------------------------------------
    # Decode ONLY newly generated tokens
    # --------------------------------------------------------

    generated_ids = output_ids[0][input_ids.shape[1]:]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    # --------------------------------------------------------
    # Remove accidental training-example continuation
    # --------------------------------------------------------

    if "### Instruction:" in response:
        response = response.split("### Instruction:")[0].strip()

    if "### Response:" in response:
        response = response.split("### Response:")[0].strip()

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    generated_tokens = len(generated_ids)

    tokens_per_second = (
        generated_tokens / generation_time
        if generation_time > 0
        else 0
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\nModel:")

    print(response)

    print(
        f"\n[{generated_tokens} tokens | "
        f"{generation_time:.2f}s | "
        f"{tokens_per_second:.2f} tokens/sec]"
    )
