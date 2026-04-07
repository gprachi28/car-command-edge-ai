# Car Command Edge AI

> Fine-tuning, quantizing, and benchmarking small language models for on-device car voice command understanding — Apple Silicon (M4 Pro).

---

## What It Does

An end-to-end edge AI pipeline that takes a natural language car command and produces structured intent + slot JSON:

```
> "Turn off the fan for the rear zone."
  {"intent": "set_climate", "slots": {"fan_speed": null, "zone": "rear"}}

> "Open the sunroof about halfway."
  {"intent": "window_control", "slots": {"window": "sunroof", "action": "open", "percentage": 50}}

> "Navigate to the nearest gas station."
  {"intent": "navigate", "slots": {"destination_type": "gas_station"}}
```

Three compact LLMs are fine-tuned with LoRA, quantized to 4-bit and 8-bit, and benchmarked across 9 variants for latency, memory, accuracy, and energy.

---

## Models & Stack

| | |
|---|---|
| **Models** | Llama 3.2 3B · Qwen 2.5 3B · SmolLM2 1.7B |
| **Fine-tuning** | MLX-LM LoRA — native Apple Silicon, Metal backend |
| **Quantization** | 4-bit and 8-bit MLX format |
| **Dataset** | Synthetic — 14 intents, 1,571 utterances (Ollama `llama3.1:8b`) |
| **Hardware** | Apple M4 Pro (~273 TOPS) |

---

## Pipeline

```
generate_dataset.py   Ollama llama3.1:8b → 14 intents, 1,571 utterances
                      Stratified 80/20 split → train.jsonl / test.jsonl
        │
        └─► finetune_mlx.py    MLX-LM LoRA — 3 models, native Apple Silicon
                │
                └─► quantize.py    4-bit + 8-bit per model → 6 quantized variants
                        │
                        └─► benchmark.py    9 variants: TTFT · TPS · RAM · accuracy · energy
                                │
                                └─► demo_cli.py    text input → structured JSON output
```

---

## Key Results

| Variant | Size (MB) | TTFT (ms) | RAM (MB) | Intent acc | Slot acc | Energy/token |
|---------|----------:|----------:|---------:|-----------:|---------:|-------------:|
| **smollm2-4bit** | **922** | **54.8** | **1,108** | 95.0% | 53.3% | **0.034 mWh** |
| smollm2-8bit | 1,738 | 64.5 | 1,969 | **96.2%** | 59.0% | 0.046 mWh |
| smollm2-finetuned | 3,268 | 78.6 | 3,612 | 95.9% | **59.6%** | 0.060 mWh |
| llama-4bit | 1,740 | 119.7 | 1,935 | 95.3% | 48.9% | 0.054 mWh |
| qwen-4bit | 1,667 | 136.9 | 1,833 | 92.4% | 48.6% | 0.057 mWh |

- **SmolLM2-4bit** is the top edge candidate: smallest (922 MB), fastest (54.8 ms TTFT), most energy-efficient (0.034 mWh/token) at 95% accuracy.
- **4-bit quantization costs ≤1% accuracy** across all models while cutting size by ~72%.
- **All 9 variants meet the 200 ms TTFT target** in isolation. SmolLM2 variants (55–79 ms) leave substantial headroom for a full STT → LLM → TTS pipeline; Qwen BF16 (180 ms) and Llama BF16 (166 ms) leave little margin.
- **SmolLM2 beats both larger models** on intent accuracy at every quantization level, despite being the smallest.

---

## 📄 Docs

| | |
|---|---|
| **[Full Results & Analysis](docs/RESULTS.md)** | Fine-tuning, quantization, benchmark table, per-intent slot accuracy breakdown |
| **[Model Card](docs/MODEL_CARD.md)** | Architecture, training details, limitations, recommendations |
| Setup guide | `docs/SETUP.md` |

---

## Dataset

Synthetic car commands generated via Ollama (`llama3.1:8b`), covering 14 intents at three slot-depth tiers.

| Command | Intent | Slots |
|---------|--------|-------|
| `Turn off the fan for rear zone.` | `set_climate` | `{"fan_speed": null, "zone": "rear"}` |
| `Turn the heat up on all seats to high` | `seat_control` | `{"heat": "high", "seat": "all"}` |
| `Open the sunroof about halfway through!` | `window_control` | `{"window": "sunroof", "action": "open", "percentage": 50}` |
| `Where is the nearest gas station?` | `navigate` | `{"destination_type": "gas_station"}` |
| `How's the lane assist doing?` | `safety_assist` | `{"feature": "lane_assist", "action": "status"}` |
| `Switch to sport, please` | `drive_mode` | `{"mode": "sport"}` |

**1,571 utterances · 14 intents · 1,252 train / 319 test · stratified 80/20 split**

---

## Quick Start

```bash
# Install dependencies (Python 3.11+, Apple Silicon Mac)
pip install -r requirements.txt

# Generate dataset (requires Ollama + llama3.1:8b)
ollama serve
python -m src.generate_dataset

# Fine-tune all three models (MLX-LM LoRA)
python -m src.finetune_mlx

# Quantize to 4-bit and 8-bit
python -m src.quantize

# Benchmark all 9 variants (per-process for accurate RAM)
bash scripts/run_benchmark.sh

# Run the interactive demo
python src/demo_cli.py --model smollm2-1.7b-4bit
```

> Requires `HF_TOKEN` in `.env` for Llama 3.2 3B (gated model). See `.env.example`.

---

## Project Structure

```
src/
├── generate_dataset.py  # Synthetic dataset generation via Ollama
├── finetune_mlx.py      # MLX-LM LoRA fine-tuning (active pipeline)
├── finetune.py          # HF TRL + LoRA (reference / learning)
├── quantize.py          # MLX 4-bit and 8-bit quantization
├── benchmark.py         # Latency, throughput, memory, accuracy, energy
├── demo_cli.py          # Interactive car command demo
└── utils.py             # Shared config and helpers
docs/
├── RESULTS.md           # Full benchmark results and analysis
├── MODEL_CARD.md        # Model card with training details and limitations
└── SETUP.md             # Environment setup and reproduction guide
```

---

_Hardware: Apple M4 Pro (~273 TOPS Neural Engine). Target cockpit SoC: 30–50 TOPS, ≤16 GB RAM._
