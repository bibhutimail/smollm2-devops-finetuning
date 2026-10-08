# SmolLM2 LoRA Fine-Tuning

Fine-tune [`HuggingFaceTB/SmolLM2-1.7B`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) with LoRA for DevOps and Kubernetes question answering. The project supports Apple Silicon (MPS), NVIDIA CUDA, and CPU execution.

## Project structure

```text
smillm2-lora/
├── data/
│   ├── train.jsonl
│   └── validation.jsonl
├── scripts/
│   ├── check_model.py
│   ├── merge_model.py
│   ├── test_merged_model.py
│   └── train.py
└── outputs/
    ├── smollm2-devops-lora/
    └── smollm2-devops-merged/
```

The workflow is:

1. Verify the base model with `check_model.py`.
2. Train and save a LoRA adapter with `train.py`.
3. Merge the adapter into the base model with `merge_model.py`.
4. Test the standalone merged model with `test_merged_model.py`.

## Requirements

- Python 3.10 or newer
- Enough memory to load the 1.7B-parameter base model
- Internet access for the first model download
- Optional: Apple Silicon or an NVIDIA GPU for acceleration

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install torch transformers datasets peft trl accelerate safetensors
```

> On Windows, activate the environment with `venv\Scripts\activate`.

## Dataset format

Both dataset files use JSON Lines format. Each line must contain `instruction`, `input`, and `output` fields:

```json
{"instruction":"What is Kubernetes?","input":"","output":"Kubernetes is an open-source container orchestration platform..."}
```

When `input` is present, the training prompt is formatted as:

```text
### Instruction:
<instruction>

### Input:
<input>

### Response:
<output>
```

## Verify the base model

Run this optional smoke test to download the base model and verify text generation:

```bash
python scripts/check_model.py
```

## Fine-tune

Run commands from the project root because the scripts use relative paths:

```bash
python scripts/train.py
```

The current configuration is a small smoke test:

- LoRA rank: `16` — uses rank-16 adapter matrices, balancing adaptation capacity with trainable parameter count and memory use.
- LoRA alpha: `32` — scales the LoRA updates; with rank 16, the effective scaling factor is `32 / 16 = 2`.
- Target modules: `q_proj`, `k_proj`, `v_proj`, and `o_proj` — applies LoRA to the attention query, key, value, and output projection layers.
- Learning rate: `1e-4` — controls how much the trainable LoRA parameters change during each optimizer step.
- Maximum sequence length: `128` — limits each formatted training example to 128 tokens, truncating content that exceeds the limit.
- Batch size: `1` — processes one training example per device before each gradient update.
- Maximum training steps: `3` — stops after only three optimizer steps, making this a pipeline verification rather than meaningful fine-tuning.
- Evaluation and intermediate checkpoints: disabled — skips validation during training and saves only the final adapter after training completes.

Edit `training_args` in `scripts/train.py` for a full training run—for example, increase `max_steps` or remove it and use `num_train_epochs`, then enable an evaluation and save strategy if required.

The trained adapter and tokenizer are saved to:

```text
outputs/smollm2-devops-lora/
```

The directory contains PEFT adapter configuration and weights plus tokenizer
files. It is not a complete standalone model and still needs the
`HuggingFaceTB/SmolLM2-1.7B` base model when loaded directly.

## Merge the model

After training finishes, merge the LoRA adapter into the base model:

```bash
python scripts/merge_model.py
```

The script:

- checks for `outputs/smollm2-devops-lora/adapter_model.safetensors`;
- loads the tokenizer from the adapter directory, falling back to the base
  model tokenizer;
- loads the base model in `float32`;
- applies the LoRA adapter and calls `merge_and_unload()`; and
- saves a standalone safe-serialization model to:

```text
outputs/smollm2-devops-merged/
```

The merged output includes model configuration, generation configuration,
tokenizer files, a SafeTensors index, and the generated model weight shards.
It can be loaded with Transformers without attaching a PEFT adapter.

## Test the merged model

Start the interactive test after merging:

```bash
python scripts/test_merged_model.py
```

The script loads the merged model once in `float16`, performs a short warm-up,
and opens an interactive DevOps question loop. Responses use deterministic
generation with up to 80 new tokens. Each response reports generated tokens,
elapsed time, and tokens per second.

Enter `exit`, `quit`, or `q` to stop. Change `MODEL_PATH` or
`MAX_NEW_TOKENS` near the top of `scripts/test_merged_model.py` when needed.

## Device selection

`train.py` automatically chooses a device in this order:

1. Apple Metal Performance Shaders (`mps`)
2. NVIDIA CUDA (`cuda`)
3. CPU

`test_merged_model.py` and `check_model.py` currently choose MPS when available
and otherwise use CPU. The merge script loads and merges the model in CPU
`float32` memory.

`PYTORCH_ENABLE_MPS_FALLBACK=1` is enabled by `train.py` so unsupported MPS
operations can run on the CPU.

## Notes

- `outputs/smollm2-devops-lora/` is the small LoRA adapter output.
- `outputs/smollm2-devops-merged/` is the larger standalone model output and
  contains the base weights with the LoRA changes merged into them.
- Re-running the merge writes to the existing merged output directory. Remove
  obsolete files manually if the model's shard layout changes.
- The first run can take time because model files are downloaded from Hugging Face.
- Training, merging, and generation require substantial memory; CPU execution
  may be slow.
- If macOS reports a LibreSSL warning from `urllib3`, use a current Python installation linked against OpenSSL and recreate the virtual environment.
