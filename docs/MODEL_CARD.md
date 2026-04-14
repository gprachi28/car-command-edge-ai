---
language:
  - en
license: mit
tags:
  - edge-ai
  - automotive
  - intent-classification
  - slot-filling
  - mlx
  - lora
  - quantization
  - voice-ai
base_model:
  - HuggingFaceTB/SmolLM2-1.7B
  - Qwen/Qwen2.5-3B
  - meta-llama/Llama-3.2-3B
pipeline_tag: text-generation
library_name: mlx-lm
datasets:
  - synthetic (generated via Ollama llama3.1:8b)
---

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
Command: Cool the front down to 20.
Action: {"intent": "set_climate", "slots": {"zone": "front", "temperature": 20, "mode": "cool"}}

Command: Navigate to the nearest gas station
Action: {"intent": "navigate", "slots": {"destination_type": "gas_station"}}

Command: Open the sunroof about halfway.
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
- **Size:** ~1,200 utterances across 14 intents (~960 train / ~240 test), stratified 80/20 split
- **Generation:** Density-tier approach — each intent generated across `full` (max slots), `partial` (mid-range), and `minimal` (single-slot) tiers with tier-specific gold examples embedded in the prompt. Inline validation at generation time rejects None/null slot values, out-of-schema keys, and question/status utterances.
- **Intents:** `set_climate`, `navigate`, `play_media`, `adjust_volume`, `call_contact`, `read_message`, `seat_control`, `set_lighting`, `window_control`, `cruise_control`, `safety_assist`, `vehicle_info`, `drive_mode`, `connectivity`
- **Format:** `{"text": "Command: <utterance>\nAction: <json>"}`
- **Why synthetic:** No existing public dataset covers this intent+slot schema; synthetic generation ensures clean, domain-matched training and evaluation data with controlled slot density distribution.

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

**Note on Llama:** Llama 3.2 3B required lr=2e-5 and LoRA rank=32 for stable convergence; SmolLM2 and Qwen used identical hyperparameters (lr=2e-4, rank=8).

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

Compression ratios are near-identical across all three models.

---

## Benchmark Results

Evaluated on 229 examples (231-example stratified 20% hold-out, 14 intent classes; 2 warmup examples discarded). Hardware: Apple M4 Pro.

| Variant | Size (MB) ↓ | TTFT (ms) ↓ | TPS ↑ | RAM (MB) ↓ | Intent acc ↑ | Slot acc ↑ | Power (W) | Energy/token (mWh) ↓ |
|---------|----------:|----------:|----:|---------:|---------:|----------:|----------:|-------------------:|
| smollm2-finetuned | 3,268 | 79.3 | 71.7 | 3,608 | 98.3% | 66.4% | 12.2 | 0.058 |
| smollm2-4bit | **922** | **54.3** | 189.0 | **1,103** | 92.6% | 59.8% | 16.2 | **0.033** |
| smollm2-8bit | 1,738 | 66.4 | 120.5 | 1,971 | 97.8% | 66.8% | 14.4 | 0.044 |
| qwen-finetuned | 5,897 | 178.8 | 39.4 | 6,385 | **98.3%** | **68.1%** | 12.6 | 0.115 |
| qwen-4bit | 1,667 | 132.6 | 123.3 | 1,833 | 97.8% | **68.1%** | 14.1 | 0.054 |
| qwen-8bit | 3,138 | 152.7 | 72.2 | 3,412 | **98.3%** | 67.7% | 13.4 | 0.077 |
| llama-finetuned | 6,144 | 164.5 | 38.2 | 6,661 | 96.1% | 62.4% | **11.7** | 0.108 |
| llama-4bit | 1,740 | 120.9 | 124.2 | 1,930 | 93.9% | 55.9% | 13.8 | 0.053 |
| llama-8bit | 3,272 | 195.3 ⚠️ | 65.5 | 3,568 | 96.1% | 62.0% | 5.8 | 0.039 |

> **Note on latency:** 8 of 9 variants meet the 200 ms TTFT target when measured in isolation. **Llama-8bit (195.3 ms) is marginal** and should not be used in latency-sensitive pipelines. In a full voice pipeline (STT → LLM → TTS), SmolLM2 variants (54–79 ms) leave substantial headroom; Qwen BF16 (179 ms) and Llama BF16 (165 ms) leave very little margin.

---

## Intended Use

- **Intended use:** Research and portfolio demonstration of LLM fine-tuning and quantization techniques for automotive voice AI.
- **Target users:** ML engineers and researchers evaluating edge AI model compression trade-offs.
- **Suitable for:** Intent classification and structured slot extraction from short English car commands.

---

## Limitations

**Dataset:**
- Synthetic data generated by a single LLM (`llama3.1:8b`) — utterance diversity and naturalness is limited compared to real user speech.
- Density-tier generation and inline validation eliminate the worst data quality failures (null slots, question utterances, out-of-schema keys), but do not capture the full range of real-world phrasing variation or ASR transcription noise.
- English only. No multilingual capability was evaluated.
- No noise robustness evaluation — all inputs are clean text; real ASR output includes transcription errors, disfluencies, and partial phrases.
- Test set is held out from the same synthetic distribution — does not measure generalisation to real-world commands.

**Slot accuracy:**
- Exact-match slot scoring (56–68%) understates practical extraction quality. Models frequently generate extra plausible slots not in the ground truth label, which counts as a full failure under exact match. Slot F1 (precision/recall at key-value level) and schema-filtered slot F1 are captured per-example and give a more accurate picture of extraction quality.
- `navigate` (22–50%) and `set_climate` (39–56%) remain the weakest intents. Models tend to fill in inferred context beyond what the utterance explicitly states.
- `smollm2-4bit` produces 12 malformed JSON outputs per 229 examples (5.2%) — the highest failure rate across all variants and a reliability concern for production use. All other variants produce 0–4 failures. Add a JSON parse fallback when using `smollm2-4bit`.

**Hardware context:**
- Benchmarked on M4 Pro (~273 TOPS). Automotive cockpit SoCs typically provide 30–50 TOPS with ≤16 GB shared RAM. On target hardware, TPS will be ~4–6× lower; TTFT may increase but car commands are short enough (23–28 output tokens) that most variants are expected to remain within the 200 ms threshold. Llama-8bit (195.3 ms on M4 Pro) may exceed it on constrained hardware.
- Power measurements are approximations via macOS `powermetrics` — not equivalent to automotive SoC power envelopes.

**Scope:**
- No voice input pipeline (ASR) — text-only evaluation.
- Models were not evaluated on safety-critical commands (emergency braking, collision warnings) — intent extraction accuracy for those intents was not separately validated.

---

## Recommendations

| Use case | Recommended variant | Rationale |
|----------|--------------------|-----------|
| Tightest memory / lowest latency / lowest energy | `smollm2-4bit` | 922 MB, 54.3 ms TTFT, 92.6% intent acc, 0.033 mWh/token — add JSON parse fallback (5.2% failure rate) |
| Best accuracy at 4-bit | `qwen-4bit` | 97.8% intent accuracy, 132.6 ms TTFT, 1,833 MB RAM, 0.054 mWh/token |
| Highest overall accuracy | `qwen-8bit` | 98.3% intent accuracy, 152.7 ms TTFT, 3,412 MB RAM |
| BF16 reference baseline | `qwen-finetuned` or `smollm2-finetuned` | Both at 98.3% intent accuracy; Qwen leads on slot accuracy (68.1% vs 66.4%) |
| Avoid in latency-sensitive pipelines | `llama-8bit` | 195.3 ms TTFT — within 5 ms of the 200 ms automotive target |

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

## How to Use

```python
from mlx_lm import load, generate

model, tokenizer = load("models/quantized/smollm2-4bit")

prompt = "Command: Cool the front down to 20.\nAction:"
response = generate(model, tokenizer, prompt=prompt, max_tokens=80)
# {"intent": "set_climate", "slots": {"zone": "front", "temperature": 20, "mode": "cool"}}
```

Or use the interactive CLI:

```bash
python -m src.demo_cli --model smollm2-4bit
```

---

## Citation

```bibtex
@misc{car-command-edge-ai-2026,
  author       = {Prachi Govalkar},
  title        = {Car Command Edge AI: Fine-tuning and Quantizing Small LLMs for On-Device Automotive Voice Commands},
  year         = {2026},
  howpublished = {\url{https://github.com/gprachi28/car-command-edge-ai}},
  note         = {Benchmarked on Apple M4 Pro with MLX-LM 0.31+}
}
```

---

*Initial benchmark: 2026-04-06 (v1 dataset). v2 dataset rewrite + full re-benchmark (greedy decoding, wall-clock TPS, brace-depth stop, slot F1): 2026-04-14. Hardware: Apple M4 Pro, macOS 15. MLX-LM 0.31+.*
