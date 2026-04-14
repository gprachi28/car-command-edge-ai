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
| smollm2-finetuned | 3,268 | 74.8 | 72.1 | 3,608 | 98.3% | 66.4% | 11.7 | 0.054 |
| smollm2-4bit | **922** | **54.1** | 200.1 | **1,103** | 96.5% | 61.6% | 15.0 | **0.029** |
| smollm2-8bit | 1,738 | 66.2 | 121.5 | 1,972 | 98.3% | 66.8% | 13.6 | 0.041 |
| qwen-finetuned | 5,897 | 178.6 | 39.8 | 6,385 | **99.6%** | **69.0%** | 12.9 | 0.117 |
| qwen-4bit | 1,667 | 131.4 | **122.8** | 1,833 | 98.3% | **68.6%** | 15.5 | 0.059 |
| qwen-8bit | 3,138 | 152.0 | 72.3 | 3,412 | **99.6%** | **68.6%** | 14.1 | 0.080 |
| llama-finetuned | 6,144 | 165.0 | 38.4 | 6,661 | 97.4% | 62.9% | **12.3** | 0.114 |
| llama-4bit | 1,740 | 120.4 | **123.8** | 1,930 | 94.3% | 55.9% | 15.1 | 0.056 |
| llama-8bit | 3,272 | 133.4 | 70.2 | 3,568 | 96.9% | 62.0% | 14.0 | 0.079 |

> **Note on latency:** All 9 variants meet the 200 ms TTFT target. SmolLM2 variants (54–75 ms) leave substantial headroom for a full voice pipeline (STT → LLM → TTS). Qwen BF16 (179 ms) and Llama BF16 (165 ms) leave little margin and are better used as BF16 reference baselines.

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
- Exact-match slot scoring (56–69% overall) understates practical extraction quality. Models frequently generate extra plausible slots not in the ground truth label, which counts as a full failure under exact match. Slot F1 (precision/recall at key-value level) and schema-filtered slot F1 are captured per-example and give a more accurate picture of extraction quality.
- `navigate` (22–50%) and `set_climate` (39–56%) remain the weakest intents. Models tend to fill in inferred context beyond what the utterance explicitly states.
- `smollm2-4bit` produces 3 malformed JSON outputs per 229 examples (1.3%) — the highest failure rate across all variants, but well below the prior run which saw 12 failures caused by early-stop inference truncation. Add a JSON parse fallback when using `smollm2-4bit`.

**Hardware context:**
- Benchmarked on M4 Pro (~273 TOPS). Automotive cockpit SoCs typically provide 30–50 TOPS with ≤16 GB shared RAM. On target hardware, TPS will be ~4–6× lower; TTFT may increase but car commands are short enough (25–29 output tokens) that most variants are expected to remain within the 200 ms threshold.
- **Total response time (TTFT + generation) is the relevant pipeline metric**, not TTFT alone. The TTS synthesizer can't begin speaking until the full JSON is parsed. Calculated as TTFT + (output_tokens / TPS): smollm2-4bit ~202 ms, qwen-4bit ~342 ms, llama-4bit ~322 ms. smollm2-4bit is the only 4-bit variant where the complete response is ready within 200 ms.
- **4-bit models draw higher peak wattage than BF16.** smollm2-4bit: 15.0 W; qwen-4bit: 15.5 W vs BF16 baselines at 11.7–12.9 W. Automotive SoCs use passive cooling — sustained peak power above the SoC's thermal design power (TDP) causes clock throttling, which collapses TPS and can push total response time beyond the 200 ms target. On thermally constrained hardware, lower-wattage BF16 variants may be preferable depending on the vehicle's thermal envelope.
- **Only `smollm2-4bit` (1,103 MB RAM) can realistically stay always-resident on an 8 GB cockpit SoC.** A system running the OS, navigation maps, media player, and display compositor has roughly 2–3 GB left for the voice AI model. qwen-4bit (1,833 MB) is borderline; all 8-bit and BF16 variants (3,400–6,700 MB) require unloading other systems. If the model isn't always-resident, every voice interaction pays a cold-start penalty — loading a 6 GB BF16 model from flash takes 5–30 seconds.
- Power measurements are approximations via macOS `powermetrics` — not equivalent to automotive SoC power envelopes.

**Safety-critical intent accuracy:**
- Accuracy is reported as a single figure across all 14 intents. Safety-relevant intents are not separated. A wrong slot on `cruise_control` (e.g. incorrect target speed or enable/disable inversion) or `safety_assist` (wrong feature or action) is a different category of failure from a wrong slot on `play_media`. Current slot accuracy: `cruise_control` 71–88%, `safety_assist` 75–83%. A production deployment would apply a higher accuracy threshold or require a driver confirmation prompt for commands in these intents before execution.

**Scope:**
- No voice input pipeline (ASR) — text-only evaluation.
- **Determinism under repeated identical commands is not benchmarked.** Greedy decoding (temp=0.0) should produce the same output for the same input, but this was not verified across multiple runs. A production system requires guaranteed consistent responses — the same command must always produce the same structured output. smollm2-4bit's 1.3% parse failure rate means 3 out of 229 examples produced no valid JSON; it is unknown whether those failures are deterministic or stochastic.

---

## Recommendations

| Use case | Recommended variant | Rationale |
|----------|--------------------|-----------|
| Tightest memory / lowest latency / lowest energy / always-resident | `smollm2-4bit` | 922 MB disk, 1,103 MB RAM, 54.1 ms TTFT, ~202 ms total response, 96.5% intent acc, 0.029 mWh/token — the only variant that fits in 8 GB cockpit RAM alongside OS and navigation. Add JSON parse fallback (1.3% failure rate). |
| Best accuracy at 4-bit | `qwen-4bit` | 98.3% intent accuracy, 131.4 ms TTFT, ~342 ms total response, 1,833 MB RAM, 0.059 mWh/token |
| Highest overall accuracy | `qwen-8bit` | 99.6% intent accuracy, 152.0 ms TTFT, 3,412 MB RAM |
| BF16 reference baseline | `qwen-finetuned` or `smollm2-finetuned` | Qwen leads on both intent accuracy (99.6% vs 98.3%) and slot accuracy (69.0% vs 66.4%) |

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

*Initial benchmark: 2026-04-06 (v1 dataset). v2 dataset rewrite + full re-benchmark (greedy decoding, wall-clock TPS, slot F1): 2026-04-14. EOS-based inference (brace-depth early stop removed), re-benchmarked 2026-04-14. Hardware: Apple M4 Pro, macOS 15. MLX-LM 0.31+.*
