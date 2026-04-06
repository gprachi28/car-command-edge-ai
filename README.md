# Car Command Edge AI

Fine-tuning and quantizing small language models for English car voice command understanding, benchmarked on Apple Silicon (M4 Pro).

## What this is

An edge AI pipeline that trains three efficient LLMs on a synthetic car command dataset (14 intents, 1,571 utterances) and compares their performance across quantization levels. The goal is to demonstrate the latency/memory/accuracy trade-offs of running compressed models on-device.

**Models:** Llama 3.2 3B · Qwen 2.5 3B · SmolLM2 1.7B
**Quantization:** 4-bit and 8-bit MLX format
**Hardware:** Apple M4 Pro

## Dataset

- **Type:** Synthetic — generated locally via Ollama (`llama3.1:8b`)
- **Intents:** 14 car command intents with structured slot output
- **Size:** 1,571 utterances (1,252 train / 319 test), stratified 80/20 split
- **Format:** `Command: <utterance>\nAction: {"intent": "...", "slots": {...}}`
- **Generation:** Batched (20 examples/call), validated, resume-safe incremental writes

## Architecture

```
generate_dataset.py  (Ollama llama3.1:8b → 14 intents, 1,571 utterances)  ✅
        Stratified split 80/20 → train.jsonl / test.jsonl
        │
        └─► finetune_mlx.py  (MLX-LM LoRA — native Apple Silicon)
                3 models: Llama-3.2-3B · Qwen-2.5-3B · SmolLM2-1.7B
                │
                └─► quantize.py  (MLX-LM)
                        4-bit + 8-bit per model → 6 quantized variants
                        │
                        └─► benchmark.py
                                9 variants (3 BF16 + 6 quantized): TTFT · TPS · RAM · accuracy
                                │
                                └─► comparison.py
                                        RESULTS.md + MODEL_CARD.md
                                        │
                                        └─► demo_cli.py
                                                text input → quantized LLM
                                                    → {"intent": "set_climate", "slots": {...}}
```

## Status

| Phase | Status |
|-------|--------|
| Synthetic dataset generation (Ollama llama3.1:8b) | ✅ Complete |
| MLX-LM LoRA fine-tuning (3 models) | ✅ Complete |
| MLX quantization (4-bit + 8-bit) | ✅ Complete |
| Benchmarking (9 variants) | ✅ Complete |
| Results + model card | Pending |

## Quantization Results

| Model | Fine-tuned (BF16) | 4-bit | 8-bit | 4-bit reduction | 8-bit reduction |
|-------|:-----------------:|:-----:|:-----:|:---------------:|:---------------:|
| SmolLM2 1.7B | 3,268 MB | 922 MB | 1,738 MB | 71.8% | 46.8% |
| Qwen 2.5 3B | 5,897 MB | 1,667 MB | 3,138 MB | 71.7% | 46.8% |
| Llama 3.2 3B | 6,144 MB | 1,740 MB | 3,272 MB | 71.7% | 46.7% |

## Benchmark Results

Evaluated on 50 held-out test examples (stratified 20% split). Intent classification accuracy (14 classes).

| Variant | Size (MB) | TTFT (ms) | TPS | RAM (MB) | Accuracy |
|---------|----------:|----------:|----:|---------:|---------:|
| smollm2-finetuned | 3,268 | 84.0 | 69.3 | 3,602 | 94.0% |
| smollm2-4bit | 922 | 55.6 | 198.7 | 3,610 | 90.0% |
| smollm2-8bit | 1,738 | 66.4 | 121.2 | 3,610 | 94.0% |
| qwen-finetuned | 5,897 | 181.9 | 39.7 | 6,377 | 94.0% |
| qwen-4bit | 1,667 | 130.7 | 122.6 | 6,384 | 90.0% |
| qwen-8bit | 3,138 | 149.4 | 72.1 | 6,384 | 92.0% |
| llama-finetuned | 6,144 | 165.2 | 38.7 | 6,656 | 92.0% |
| llama-4bit | 1,740 | 120.9 | 124.7 | 1,928 | 92.0% |
| llama-8bit | 3,272 | 134.1 | 71.3 | 3,564 | 92.0% |

**Note on Llama training:** Llama 3.2 3B did not converge with the same hyperparameters used for SmolLM2 and Qwen (lr=2e-4, LoRA rank=8). The model produced degenerate repetitive output and near-zero accuracy. Dropping to lr=2e-5 with LoRA rank=32 resolved the instability completely — all three Llama variants reached 92% accuracy.

_Hardware: Apple M4 Pro. Intent classification on 14 car command intents._

## Quick Start

```bash
# Requirements: Python 3.11+, Apple Silicon Mac, Ollama
pip install -r requirements.txt

# Generate synthetic dataset (requires Ollama running with llama3.1:8b)
ollama serve
python -m src.generate_dataset

# Fine-tune all models with MLX-LM LoRA (requires HF_TOKEN in .env for Llama)
python -m src.finetune_mlx

# Run interactive demo (after training + quantization)
python src/demo_cli.py --model smollm2-1.7b-4bit
```

## Project Structure

```
src/
├── generate_dataset.py  # Ollama dataset generation (14 intents, 1,571 utterances)
├── dataset.py           # Shared split/save/metadata utilities
├── finetune_mlx.py      # MLX-LM LoRA fine-tuning (active)
├── finetune.py          # HF TRL + LoRA fine-tuning (reference / learning)
├── quantize.py          # MLX 4-bit and 8-bit conversion
├── benchmark.py         # Latency, TPS, memory, accuracy measurement
├── comparison.py        # Generate results table and model card
├── demo_cli.py          # Interactive car command demo
└── utils.py             # Shared config and helpers
```

## Requirements

See `requirements.txt`. Key dependencies: `mlx-lm`, `transformers`, `trl`, `ollama`, `torch`.

Set `HF_TOKEN` in `.env` (required for Llama 3.2 3B — gated model). See `.env.example`.
