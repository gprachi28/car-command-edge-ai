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
                                9 variants: TTFT · TPS · RAM · accuracy
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
| MLX-LM LoRA fine-tuning (3 models) | In progress |
| MLX quantization (4-bit + 8-bit) | Pending |
| Benchmarking (9 variants) | Pending |
| Results + model card | Pending |

## Results

_To be populated after benchmarking._

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
