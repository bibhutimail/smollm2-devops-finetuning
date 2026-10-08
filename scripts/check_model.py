import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B"

# Select device
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Device: {device}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)

model = model.to(device)
model.eval()

print("Model loaded successfully.")

prompt = "Explain Kubernetes in simple terms."

print(f"\nPrompt: {prompt}")
print("Tokenizing...")

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

inputs = {key: value.to(device) for key, value in inputs.items()}

print("Starting generation...")

start_time = time.time()

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

# Synchronize MPS before measuring time
if device == "mps":
    torch.mps.synchronize()

elapsed = time.time() - start_time

response = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)

print("\nResponse:")
print(response)

print(f"\nGeneration time: {elapsed:.2f} seconds")
print(f"Output tokens: {output.shape[-1]}")
