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

**Examples from the dataset:**

| Command | Intent | Slots |
|---------|--------|-------|
| `Turn off the fan for rear zone.` | `set_climate` | `{"fan_speed": null, "zone": "rear"}` |
| `Turn the heat up on all seats to high` | `seat_control` | `{"heat": "high", "seat": "all"}` |
| `Open the sunroof about halfway through!` | `window_control` | `{"window": "sunroof", "action": "open", "percentage": 50}` |
| `Where is the nearest gas station?` | `navigate` | `{"destination_type": "gas_station"}` |
| `How's the lane assist doing?` | `safety_assist` | `{"feature": "lane_assist", "action": "status"}` |
| `Switch to sport, please` | `drive_mode` | `{"mode": "sport"}` |

The model is trained to produce the `Action` JSON given the `Command` — intent classification plus structured slot extraction in one pass.

## Architecture

```
generate_dataset.py  (Ollama llama3.1:8b → 14 intents, 1,571 utterances)
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

## Fine-tuning Results

3 epochs, 939 iters, batch 4, grad accumulation 2 (effective batch 8), seed 42. Loss curves saved to `data/results/loss_curves/`.

| Model | Val loss (start → end) | Train loss (final) | Peak RAM | lr | LoRA rank |
|-------|------------------------|-------------------|----------|----|-----------|
| SmolLM2 1.7B | 3.018 → 0.647 | 0.392 | 4.2 GB | 2e-4 | 8 |
| Qwen 2.5 3B | 3.105 → 0.727 | 0.515 | 7.1 GB | 2e-4 | 8 |
| Llama 3.2 3B | — → 0.690 | 0.428 | 7.6 GB | 2e-5 | 32 |

**Note on Llama training:** With the same hyperparameters as SmolLM2/Qwen (lr=2e-4, LoRA rank=8), Llama produced degenerate output and near-zero accuracy. Dropping to lr=2e-5 with LoRA rank=32 resolved this.

## Quantization Results

| Model | Fine-tuned (BF16) | 4-bit | 8-bit | 4-bit reduction | 8-bit reduction |
|-------|:-----------------:|:-----:|:-----:|:---------------:|:---------------:|
| SmolLM2 1.7B | 3,268 MB | 922 MB | 1,738 MB | 71.8% | 46.8% |
| Qwen 2.5 3B | 5,897 MB | 1,667 MB | 3,138 MB | 71.7% | 46.8% |
| Llama 3.2 3B | 6,144 MB | 1,740 MB | 3,272 MB | 71.7% | 46.7% |

## Benchmark Results

Evaluated on 319 held-out test examples (317 eval + 2 warmup discarded), stratified 20% split. Intent classification accuracy (14 classes).
TTFT target: < 200 ms (real-time voice assistant threshold for in-car use). Each variant benchmarked in its own process for accurate RAM measurement.


| Variant | Size (MB) ↓ | TTFT (ms) ↓ | TPS ↑ | RAM (MB) ↓ | Intent acc ↑ | Slot acc ↑ | Output tokens ↓ | Power (W) ↓ | Energy/token (mWh) ↓ |
|---------|----------:|----------:|----:|---------:|---------:|----------:|------------:|----------:|----------:|
| smollm2-finetuned | 3,268 | 78.6 | 71.4 | 3,612 | 95.9% | **59.6%** | 26.4 | 12.5 | 0.060 |
| smollm2-4bit | **922** | **54.8** | **199.3** | **1,108** | 95.0% | 53.3% | 26.4 | 16.7 | **0.034** |
| smollm2-8bit | 1,738 | 64.5 | 120.9 | 1,969 | **96.2%** | 59.0% | 26.5 | 14.8 | 0.046 |
| qwen-finetuned | 5,897 | 180.4 | 40.0 | 6,385 | 93.1% | 48.3% | 24.4 | 12.2 | 0.112 |
| qwen-4bit | 1,667 | 136.9 | 123.9 | 1,833 | 92.4% | 48.6% | 23.7 | 14.3 | 0.057 |
| qwen-8bit | 3,138 | 152.4 | 73.1 | 3,412 | 92.7% | 47.3% | 24.9 | 13.3 | 0.076 |
| llama-finetuned | 6,144 | 165.5 | 38.8 | 6,662 | 95.9% | 52.7% | 22.0 | **11.5** | 0.108 |
| llama-4bit | 1,740 | 119.7 | 125.2 | 1,935 | 95.3% | 48.9% | **21.5** | 13.6 | 0.054 |
| llama-8bit | 3,272 | 133.6 | 70.9 | 3,568 | 95.6% | 51.7% | 21.9 | 13.3 | 0.077 |

**Key insights:**
- **SmolLM2-8bit is the highest accuracy variant** at 96.2% — while being only 1,738 MB and 64.5 ms TTFT. 8-bit quantization is lossless (or marginally better within noise) for all three models.
- **SmolLM2-4bit is the strongest edge candidate**: 922 MB, 54.8 ms TTFT, 95.0% accuracy, 0.034 mWh/token — smallest, fastest, most energy-efficient.
- **All 9 variants pass the 200 ms TTFT target.** SmolLM2 is 54–78 ms; Qwen BF16 is the slowest at 180 ms, still within threshold.
- **Qwen accuracy is consistently lower** than SmolLM2 and Llama across all variants (92–93% vs 95–96%). SmolLM2 achieves better accuracy at every quantization level despite being the smallest model.
- **4-bit quantization costs ≤1% accuracy** across all models (SmolLM2: −0.9%, Qwen: −0.7%, Llama: −0.6%) while cutting size by ~72%.
- **4-bit draws more power (W) but less energy per token** due to higher throughput — smollm2-4bit: 16.7 W → 0.034 mWh/token vs. smollm2-finetuned: 12.5 W → 0.060 mWh/token.
- **Slot accuracy is 47–60%, well below intent accuracy.** The gap is primarily caused by models generating extra plausible slots not in the ground truth (e.g. adding `"brightness": 100` when the label omits it). Exact-match scoring penalises any extra key, so these numbers understate true extraction quality. SmolLM2-finetuned leads at 59.6%; Qwen is lowest at 47–48%.
- **Output tokens avg 21–27**: car commands are short — TPS matters less than TTFT for this use case.



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

# Plot training loss curves
python -m src.plot_losses

# Benchmark all 9 variants sequentially (runs each in its own process for accurate RAM)
bash scripts/run_benchmark.sh

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
├── plot_losses.py       # Plot train/val loss curves from loss logs
├── comparison.py        # Generate results table and model card
├── demo_cli.py          # Interactive car command demo
└── utils.py             # Shared config and helpers
```

## Requirements

See `requirements.txt`. Key dependencies: `mlx-lm`, `transformers`, `trl`, `ollama`, `torch`.

Set `HF_TOKEN` in `.env` (required for Llama 3.2 3B — gated model). See `.env.example`.
