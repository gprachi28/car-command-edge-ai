# Car Command Edge AI

> Fine-tuning, quantizing, and benchmarking small language models for on-device car voice command understanding — Apple Silicon (M4 Pro).

---

## What It Does

An end-to-end edge AI pipeline that takes a natural language car command and produces structured intent + slot JSON:

<img width="1114" height="329" alt="demo" src="https://github.com/user-attachments/assets/2817e2de-f007-4b7d-8688-17f50b00dacb" />

Three compact LLMs are fine-tuned with LoRA, quantized to 4-bit and 8-bit, and benchmarked across 9 variants for latency, memory, accuracy, and energy.


---

## Models & Stack

| | |
|---|---|
| **Models** | Llama 3.2 3B · Qwen 2.5 3B · SmolLM2 1.7B |
| **Fine-tuning** | MLX-LM LoRA — native Apple Silicon, Metal backend |
| **Quantization** | 4-bit and 8-bit MLX format |
| **Dataset** | Synthetic — 14 intents, ~1,200 utterances (Ollama `llama3.1:8b`) |
| **Hardware** | Apple M4 Pro (~273 TOPS) |

---

## Pipeline

```
generate_dataset.py   Ollama llama3.1:8b → 14 intents, ~1,200 utterances
                      Density tiers (full/partial/minimal) + inline validation
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
| **smollm2-4bit** | **922** | **54.3** | **1,103** | 92.6% | 59.8% | **0.033 mWh** |
| smollm2-8bit | 1,738 | 66.4 | 1,971 | 97.8% | 66.8% | 0.044 mWh |
| qwen-4bit | 1,667 | 132.6 | 1,833 | **97.8%** | **68.1%** | 0.054 mWh |
| qwen-8bit | 3,138 | 152.7 | 3,412 | **98.3%** | 67.7% | 0.077 mWh |
| llama-4bit | 1,740 | 120.9 | 1,930 | 93.9% | 55.9% | 0.053 mWh |

- **smollm2-4bit** is the best edge candidate: smallest (922 MB), fastest (54.3 ms TTFT), most energy-efficient (0.033 mWh/token). Intent accuracy 92.6%; add a JSON parse fallback (5.2% parse failure rate).
- **4-bit quantization accuracy cost is model-dependent:** Qwen −0.5%, Llama −2.2%, SmolLM2 −5.7%. **8-bit is lossless** for all three (≤0.5% change) while cutting size ~47%.
- **Qwen achieves highest intent accuracy** (97.8–98.3%) at every quantization level — a reversal from v1. On the cleaner v2 dataset, Qwen's structured output learning leads.
- **8 of 9 variants meet the 200 ms TTFT target.** SmolLM2 variants (54–79 ms) leave substantial headroom. Llama-8bit (195.3 ms) is marginal.
- **Slot acc (exact-match) understates real extraction quality** — models generate additional plausible slots not in ground truth, which exact-match penalises. The benchmark also reports slot F1 (precision/recall) and schema-filtered slot F1 (slots filtered to the per-intent allowed key set) for a more meaningful comparison. The demo CLI applies the schema filter automatically.

> **Note on benchmark vs interactive TTFT:** The benchmark numbers above are measured back-to-back with no idle time between queries, which keeps Metal compute units fully active. In interactive use (the demo CLI), macOS throttles the GPU clock and spins down compute units during the pause while you type. The next query has to wait for them to ramp back up before the first token can be computed — adding ~50 ms. Interactive TTFT is typically 100–150 ms for smollm2-4bit, still well within the 200 ms automotive target.

---

## 📄 Docs

| | |
|---|---|
| **[Full Results & Analysis](docs/RESULTS.md)** | Fine-tuning, quantization, benchmark table, per-intent slot accuracy breakdown |
| **[Model Card](docs/MODEL_CARD.md)** | Architecture, training details, limitations, recommendations |
| **[Setup Guide](docs/SETUP.md)** | Environment setup, dependencies, and reproduction steps |

---

## Dataset

Synthetic car commands generated via Ollama (`llama3.1:8b`), covering 14 intents across three slot-density tiers. The generator was rewritten from a flat-batch approach after the v1 dataset produced 13.7% empty-slot examples and ~18% status/query utterances — neither of which are valid car commands.



Each intent is generated in **full** (maximum slots), **partial** (mid-range), and **minimal** (single-slot) tiers with tier-specific gold examples embedded in the prompt. Inline validation at generation time rejects None-valued slots, out-of-schema keys, and question/status utterances — no post-hoc cleaning pass needed.

| Command | Intent | Slots |
|---------|--------|-------|
| `Cool the front down to 20.` | `set_climate` | `{"zone": "front", "temperature": 20, "mode": "cool"}` |
| `Turn the heat up on all seats to high` | `seat_control` | `{"heat": "high", "seat": "all"}` |
| `Open the sunroof about halfway.` | `window_control` | `{"window": "sunroof", "action": "open", "percentage": 50}` |
| `Navigate to the nearest gas station` | `navigate` | `{"destination_type": "gas_station"}` |
| `Enable lane assist` | `safety_assist` | `{"feature": "lane_assist", "action": "enable"}` |
| `Switch to sport, please` | `drive_mode` | `{"mode": "sport"}` |

**~1,200 utterances · 14 intents · ~960 train / ~240 test · stratified 80/20 split**

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
python -m src.demo_cli --model smollm2-4bit
```

> Requires `HF_TOKEN` in `.env` for Llama 3.2 3B (gated model). See `.env.example`.




---

## Project Structure

```
src/
├── generate_dataset.py  # Synthetic dataset generation via Ollama (density tiers)
├── finetune_mlx.py      # MLX-LM LoRA fine-tuning (active pipeline)
├── quantize.py          # MLX 4-bit and 8-bit quantization
├── benchmark.py         # Latency, throughput, memory, accuracy, energy
├── demo_cli.py          # Interactive car command demo
└── utils.py             # Shared config, INTENT_SCHEMA, and helpers
docs/
├── RESULTS.md           # Full benchmark results and analysis
├── MODEL_CARD.md        # Model card with training details and limitations
└── SETUP.md             # Environment setup and reproduction guide
```

---

_Hardware: Apple M4 Pro (~273 TOPS Neural Engine). Target cockpit SoC: 30–50 TOPS, ≤16 GB RAM._

---

Licensed under the [MIT License](LICENSE).
