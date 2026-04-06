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
TTFT target: < 200 ms (real-time voice assistant threshold for in-car use).

| Variant | Size (MB) | TTFT (ms) | TPS | RAM (MB) | Accuracy | Tokens/resp | Power (W) | mWh/token |
|---------|----------:|----------:|----:|---------:|---------:|------------:|----------:|----------:|
| smollm2-finetuned | 3,268 | 74.4 | 72.5 | 3,602 | 94.0% | 26.5 | 11.5 | 0.054 |
| smollm2-4bit | 922 | 55.1 | 202.2 | 3,610 | 90.0% | 28.9 | 14.3 | 0.028 |
| smollm2-8bit | 1,738 | 67.7 | 122.6 | 3,610 | 94.0% | 26.7 | 12.8 | 0.040 |
| qwen-finetuned | 5,897 | 178.4 | 40.4 | 6,377 | 94.0% | 23.3 | 11.7 | 0.107 |
| qwen-4bit | 1,667 | 135.2 | 124.3 | 6,384 | 90.0% | 23.4 | 13.4 | 0.053 |
| qwen-8bit | 3,138 | 163.2 | 71.1 | 6,384 | 92.0% | 26.1 | 13.3 | 0.079 |
| llama-finetuned | 6,144 | 163.5 | 39.0 | 6,656 | 92.0% | 22.3 | 11.1 | 0.104 |
| llama-4bit | 1,740 | 119.3 | 126.0 | 1,928 | 92.0% | 21.9 | 12.8 | 0.050 |
| llama-8bit | 3,272 | 133.7 | 71.6 | 3,564 | 92.0% | 22.2 | 12.2 | 0.070 |

**Key insights:**
- **SmolLM2 4-bit is the strongest edge candidate**: 922 MB, 55 ms TTFT, 90% accuracy, 0.028 mWh/token — smallest model, lowest latency, most energy-efficient.
- **All variants pass the 200 ms TTFT target** (< 200 ms real-time voice assistant threshold). SmolLM2 is fastest (55–74 ms); Llama is slowest at 163 ms.
- **8-bit quantization is lossless for accuracy** on this task (SmolLM2: 94%, Llama: 92%) while halving disk size.
- **Output tokens avg 22–29**: confirms car commands are short — TPS matters less than TTFT for this use case.
- **4-bit draws slightly more power but uses far less energy per token** due to higher throughput (smollm2-4bit: 14.3 W → 0.028 mWh/token vs. smollm2-finetuned: 11.5 W → 0.054 mWh/token).
- **Llama power is comparable to Qwen** at similar quantization levels (llama-4bit: 12.8 W / 0.050 mWh/token vs. qwen-4bit: 13.4 W / 0.053 mWh/token).

**Note on Llama training:** With the same hyperparameters as SmolLM2/Qwen (lr=2e-4, LoRA rank=8), Llama produced degenerate output and near-zero accuracy. Dropping to lr=2e-5 with LoRA rank=32 resolved this — all three variants reach 92% accuracy.

_Hardware: Apple M4 Pro (~273 TOPS Neural Engine). Target cockpit SoC: 30–50 TOPS, ≤16 GB RAM._

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
