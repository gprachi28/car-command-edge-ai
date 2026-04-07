# Model Card — Car Command Edge AI

**Task:** Natural language → structured intent + slot extraction for in-car voice commands
**Domain:** Automotive / voice AI
**Language:** English
**Hardware:** Apple M4 Pro (benchmarked); target deployment: cockpit SoC (30–50 TOPS, ≤16 GB RAM)

---

## Overview

Three compact language models fine-tuned with LoRA on a synthetic English car command dataset, then quantized to 4-bit and 8-bit MLX format and benchmarked across 9 variants on Apple Silicon. The goal is to demonstrate model compression trade-offs (latency, memory, accuracy, energy) for on-device automotive voice AI.

Given a spoken car command, the model outputs a structured JSON response:

```
Command: Turn off the fan for rear zone.
Action: {"intent": "set_climate", "slots": {"fan_speed": null, "zone": "rear"}}

Command: Where is the nearest gas station?
Action: {"intent": "navigate", "slots": {"destination_type": "gas_station"}}

Command: Open the sunroof about halfway through!
Action: {"intent": "window_control", "slots": {"window": "sunroof", "action": "open", "percentage": 50}}
```

---

## Models

| Model | Parameters | Base | Fine-tuned with |
|-------|:----------:|------|----------------|
| SmolLM2 1.7B | 1.7B | HuggingFaceTB/SmolLM2-1.7B | MLX-LM LoRA |
| Qwen 2.5 3B | 3B | Qwen/Qwen2.5-3B | MLX-LM LoRA |
| Llama 3.2 3B | 3B | meta-llama/Llama-3.2-3B (gated) | MLX-LM LoRA |

---

## Training

### Dataset

- **Type:** Synthetic — generated locally via Ollama (`llama3.1:8b`)
- **Size:** 1,571 utterances across 14 intents (1,252 train / 319 test), stratified 80/20 split
- **Intents:** `set_climate`, `navigate`, `play_media`, `adjust_volume`, `call_contact`, `read_message`, `seat_control`, `set_lighting`, `window_control`, `cruise_control`, `safety_assist`, `vehicle_info`, `drive_mode`, `connectivity`
- **Format:** `{"text": "Command: <utterance>\nAction: <json>"}`
- **Why synthetic:** Avoids domain-transfer confounds from NLU datasets (SNIPS, SLURP) with mismatched intent schemas; keeps the benchmark clean.

### Hyperparameters

| | SmolLM2 1.7B | Qwen 2.5 3B | Llama 3.2 3B |
|-|:------------:|:-----------:|:------------:|
| Framework | MLX-LM LoRA | MLX-LM LoRA | MLX-LM LoRA |
| Epochs | 3 | 3 | 3 |
| Iterations | 939 | 939 | 939 |
| Batch size | 4 | 4 | 4 |
| Gradient accumulation | 2 (effective batch 8) | 2 (effective batch 8) | 2 (effective batch 8) |
| Learning rate | 2e-4 | 2e-4 | 2e-5 |
| LoRA rank | 8 | 8 | 32 |
| LoRA layers | 8 | 8 | 8 |
| Trainable params | 0.18% (3.0M) | 0.11% (3.3M) | ~0.4% |
| Peak training RAM | 4.2 GB | 7.1 GB | 7.6 GB |

Gradient accumulation of 2 means weights are updated every 2 batches, so each update sees 4 × 2 = 8 examples. This gives the same training dynamics as a physical batch of 8 at half the peak RAM cost per step.

**Note on Llama:** lr=2e-4 / rank=8 caused training divergence — the model memorised a degenerate repetitive output pattern with near-zero accuracy. lr=2e-5 and rank=32 resolved this completely. SmolLM2 and Qwen used identical hyperparameters to allow a controlled comparison.

### Training Results (val loss start → end)

| Model | Val loss | Train loss (final) | Speed |
|-------|:--------:|:-----------------:|-------|
| SmolLM2 1.7B | 3.018 → **0.647** | 0.392 | ~3.5 it/sec, ~550 tok/sec |
| Qwen 2.5 3B | 3.105 → 0.727 | 0.515 | ~2.3 it/sec, ~320 tok/sec |
| Llama 3.2 3B | — → 0.690 | 0.428 | ~2.5 it/sec, ~288 tok/sec |

Loss curves: `data/results/loss_curves/`

---

## Quantization

All models quantized from fine-tuned BF16 using `mlx_lm convert -q`. Effective precision: ~4.5-bit (4-bit) and ~8.5-bit (8-bit).

| Model | BF16 | 4-bit | 8-bit | 4-bit reduction | 8-bit reduction |
|-------|-----:|------:|------:|:---------------:|:---------------:|
| SmolLM2 1.7B | 3,268 MB | 922 MB | 1,738 MB | 71.8% | 46.8% |
| Qwen 2.5 3B | 5,897 MB | 1,667 MB | 3,138 MB | 71.7% | 46.8% |
| Llama 3.2 3B | 6,144 MB | 1,740 MB | 3,272 MB | 71.7% | 46.7% |

Compression ratios are near-identical across all three models, as expected for uniform linear quantization. All 9 quantized variants verified loadable before benchmarking.

---

## Benchmark Results

Evaluated on 317 held-out test examples. Each variant run in a separate process for accurate peak RAM measurement. Power measured via macOS `powermetrics` (GPU + CPU combined).

| Variant | Size (MB) ↓ | TTFT (ms) ↓ | TPS ↑ | RAM (MB) ↓ | Intent acc ↑ | Slot acc ↑ | Power (W) | Energy/token (mWh) ↓ |
|---------|----------:|----------:|----:|---------:|---------:|----------:|----------:|-------------------:|
| smollm2-finetuned | 3,268 | 78.6 | 71.4 | 3,612 | 95.9% | **59.6%** | 12.5 | 0.060 |
| smollm2-4bit | **922** | **54.8** | **199.3** | **1,108** | 95.0% | 53.3% | 16.7 | **0.034** |
| smollm2-8bit | 1,738 | 64.5 | 120.9 | 1,969 | **96.2%** | 59.0% | 14.8 | 0.046 |
| qwen-finetuned | 5,897 | 180.4 | 40.0 | 6,385 | 93.1% | 48.3% | 12.2 | 0.112 |
| qwen-4bit | 1,667 | 136.9 | 123.9 | 1,833 | 92.4% | 48.6% | 14.3 | 0.057 |
| qwen-8bit | 3,138 | 152.4 | 73.1 | 3,412 | 92.7% | 47.3% | 13.3 | 0.076 |
| llama-finetuned | 6,144 | 165.5 | 38.8 | 6,662 | 95.9% | 52.7% | **11.5** | 0.108 |
| llama-4bit | 1,740 | 119.7 | 125.2 | 1,935 | 95.3% | 48.9% | 13.6 | 0.054 |
| llama-8bit | 3,272 | 133.6 | 70.9 | 3,568 | 95.6% | 51.7% | 13.3 | 0.077 |

> **Note on latency:** All 9 variants meet the 200 ms TTFT target when measured in isolation. In a full voice pipeline (STT → LLM → TTS), SmolLM2 variants (55–79 ms) leave substantial headroom; Qwen BF16 (180 ms) and Llama BF16 (166 ms) leave very little margin and would be at risk on constrained automotive hardware once STT and TTS latency is added.

See `RESULTS.md` for the full per-intent slot accuracy breakdown and extended analysis.

---

## Intended Use

- **Intended use:** Research and portfolio demonstration of LLM fine-tuning and quantization techniques for automotive voice AI.
- **Target users:** ML engineers and researchers evaluating edge AI model compression trade-offs.
- **Suitable for:** Intent classification and structured slot extraction from short English car commands.

---

## Limitations

**Dataset:**
- Synthetic data generated by a single LLM (`llama3.1:8b`) — utterance diversity and naturalness is limited compared to real user speech.
- English only. No multilingual capability was evaluated.
- No noise robustness evaluation — all inputs are clean text; real ASR output includes transcription errors, disfluencies, and partial phrases.
- Test set is held out from the same synthetic distribution — does not measure generalisation to real-world commands.

**Slot accuracy:**
- Exact-match slot scoring (47–60%) understates practical extraction quality. Models frequently generate extra plausible slots not in the ground truth label, which counts as a full failure under exact match.
- `navigate` (4–23%) and `set_climate` (23–46%) remain weak. The model tends to fill in inferred context beyond what the utterance explicitly states.
- Qwen produces 3–6 malformed JSON outputs per 317 examples under quantization — a reliability concern for production use.

**Hardware context:**
- Benchmarked on M4 Pro (~273 TOPS). Automotive cockpit SoCs typically provide 30–50 TOPS with ≤16 GB shared RAM. On target hardware, TPS will be ~4–6× lower; TTFT may increase but car commands are short enough (21–27 output tokens) that all variants are expected to remain within the 200 ms threshold.
- Power measurements are approximations via macOS `powermetrics` — not equivalent to automotive SoC power envelopes.

**Scope:**
- No voice input pipeline (ASR) — text-only evaluation.
- Models were not evaluated on safety-critical commands (emergency braking, collision warnings) — intent extraction accuracy for those intents was not separately validated.

---

## Recommendations

| Use case | Recommended variant | Rationale |
|----------|--------------------|-----------| 
| Tightest memory / lowest latency | `smollm2-4bit` | 922 MB, 54.8 ms TTFT, 95% accuracy, 0.034 mWh/token |
| Highest accuracy | `smollm2-8bit` | 96.2% intent accuracy, 64.5 ms TTFT, 1,969 MB RAM |
| BF16 reference baseline | `smollm2-finetuned` | Best slot accuracy (59.6%), no quantization loss |
| Avoid | `qwen-*` | Consistently lower accuracy + higher parse failure rate |

---

## Reproducibility

```bash
# Dataset generation (requires Ollama + llama3.1:8b)
python -m src.generate_dataset

# Fine-tuning (MLX-LM LoRA, Apple Silicon)
python -m src.finetune_mlx

# Quantization
python -m src.quantize

# Benchmarking (per-process isolation for accurate RAM)
bash scripts/run_benchmark.sh
```

See `docs/SETUP.md` for full environment setup. Seed: 42. All runs deterministic given the same MLX version.

---

*Benchmarked: 2026-04-06. Hardware: Apple M4 Pro, macOS 15. MLX-LM 0.31+.*
