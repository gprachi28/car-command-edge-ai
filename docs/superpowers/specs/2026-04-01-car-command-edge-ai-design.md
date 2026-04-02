# Car Command Edge AI — Design Specification

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a portfolio-ready edge AI pipeline that fine-tunes, quantizes, and benchmarks three efficient language models (Llama 3.2 3B, Qwen 2.5 3B, SmolLM2 1.7B) for English car command understanding, demonstrating model compression trade-offs on Apple Silicon.

**Architecture:** Independent Python modules for dataset generation, fine-tuning, quantization, benchmarking, and a CLI demo. A synthetic dataset of 14 car intents (~1,571 utterances) is generated via Ollama (llama3.1:8b locally). Models are fine-tuned with MLX-LM LoRA (native Apple Silicon), quantized to 4-bit and 8-bit MLX format, then benchmarked for latency, throughput, memory, and accuracy on M4 Pro. Results are presented in a comparison table and model card.

**Tech Stack:** MLX-LM (fine-tuning + quantization), Ollama / llama3.1:8b (dataset generation), PyTorch/Transformers (reference), Python 3.11+

**Fine-tuning note:** `finetune.py` (HF TRL + LoRA) is kept for reference and learning. `finetune_mlx.py` is the active path — it runs entirely on Metal with no PyTorch/MPS CPU fallbacks and produces adapters already in MLX format.

---

## Architecture Overview

**Three-phase pipeline:**

1. **Preparation:** ✅ Generate synthetic car command dataset via Ollama (llama3.1:8b), validate, split into train/test
2. **Training:** Fine-tune Llama 3.2 3B, Qwen 2.5 3B, and SmolLM2 1.7B using MLX-LM LoRA (`finetune_mlx.py`)
3. **Evaluation:** Quantize each model to 4-bit and 8-bit, benchmark all 9 variants, generate comparison table and model card

**Data flow:**
```
generate_dataset.py  (Ollama llama3.1:8b → 14 intents, 1,571 utterances)  ✅ DONE
        Output: {"command": "cool the front to 20", "action": {"intent": "set_climate", "slots": {...}}}
        Stratified split 80/20 → train.jsonl / test.jsonl
        │
        └─► finetune_mlx.py  (MLX-LM LoRA, native Apple Silicon)
                3 models: Llama-3.2-3B · Qwen-2.5-3B · SmolLM2-1.7B
                [finetune.py kept for reference — HF TRL + LoRA]
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

---

## Project Structure

```
car-command-edge-ai/
├── README.md                          # Landing page with key results
├── RESULTS.md                         # Human-readable comparison table
├── docs/
│   ├── SETUP.md                       # Installation & environment setup
│   ├── ARCHITECTURE.md                # Technical deep-dive on pipeline
│   └── MODEL_CARD.md                  # Generated model card (outputs)
├── data/
│   ├── raw/
│   │   └── synthetic/                 # Raw per-intent JSONL from Gemini generation
│   ├── processed/
│   │   ├── train.jsonl                # 80% training examples
│   │   ├── test.jsonl                 # 20% test examples
│   │   └── metadata.json              # Dataset statistics
│   └── results/
│       ├── comparison_table.csv       # Benchmark results
│       └── quantization_analysis.json # Accuracy per variant
├── src/
│   ├── __init__.py
│   ├── generate_dataset.py            # Gemini Flash generation + validation + split
│   ├── dataset.py                     # Shared split/save/metadata utilities
│   ├── finetune.py                    # HF TRL + LoRA fine-tuning
│   ├── quantize.py                    # MLX quantization (4-bit, 8-bit)
│   ├── benchmark.py                   # Latency, TPS, memory, accuracy
│   ├── comparison.py                  # Generate comparison table & model card
│   ├── demo_cli.py                    # Interactive CLI for car commands
│   └── utils.py                       # Shared utilities & config
├── models/
│   ├── base/
│   │   ├── llama-3.2-3b/              # Base models (HF format)
│   │   ├── qwen-2.5-3b/
│   │   └── smollm2-1.7b/
│   ├── finetuned/
│   │   ├── llama-3.2-3b-finetuned/    # LoRA checkpoints
│   │   ├── qwen-2.5-3b-finetuned/
│   │   └── smollm2-1.7b-finetuned/
│   └── quantized/
│       ├── llama-3.2-3b-4bit/         # MLX quantized variants
│       ├── llama-3.2-3b-8bit/
│       ├── qwen-2.5-3b-4bit/
│       ├── qwen-2.5-3b-8bit/
│       ├── smollm2-1.7b-4bit/
│       └── smollm2-1.7b-8bit/
├── tests/
│   ├── test_dataset.py                # Data loading & splitting
│   ├── test_finetune.py               # Mock fine-tuning
│   └── test_demo_cli.py               # CLI functionality
├── requirements.txt                    # Dependencies
├── .env.example                        # HuggingFace + Gemini API key template
├── .gitignore
└── scripts/
    └── run_pipeline.sh                # End-to-end execution script
```

---

## Module Responsibilities

### `generate_dataset.py`
- Call Ollama (`llama3.1:8b`, local) per intent in batches of 20 examples per call
- Validate responses: known slot keys, non-empty utterance
- Write per-intent JSONL incrementally to `data/raw/synthetic/` (resume-safe, atomic writes)
- Deduplicate on normalised utterance text, stratified 80/20 split, save to `data/processed/`
- Log metadata: intent distribution, avg utterance length; save `metadata.json`
- Supports `--backend gemini` as an alternative (requires `GEMINI_API_KEY`)

### `dataset.py`
- Shared utilities: `split_dataset`, `save_dataset`, `log_metadata`
- Used by `generate_dataset.py`; not an entry point

### `finetune_mlx.py` *(active)*
- Fine-tune all three models using MLX-LM LoRA (native Apple Silicon, Metal backend)
- Per model: train LoRA adapter → fuse into base model → save to `models/finetuned/<key>-mlx/`
- Training config: batch 4, grad_accum 2 (effective batch 8), lr 2e-4, 3 epochs, 8 LoRA layers
- `--mask-prompt`: loss computed on Action output only, not Command input
- Output is already in MLX format — no conversion needed before quantization

### `finetune.py` *(reference only)*
- HF TRL + LoRA fine-tuning — kept for learning and comparison with `finetune_mlx.py`
- Not used in the active pipeline
- Config: LoRA rank 8, alpha 16, lr 2e-4, batch 8, 3 epochs, MPS device map

### `quantize.py`
- Load fused MLX models from `models/finetuned/<key>-mlx/` (already in MLX format)
- Apply 4-bit and 8-bit quantization via `mlx-lm convert`
- Save to `models/quantized/`
- Log model size before/after quantization

### `benchmark.py`
- Load all 9 variants (3 base + 6 quantized)
- For each variant:
  - Measure time-to-first-token (TTFT) on 50 test examples
  - Calculate tokens-per-second (TPS) for 100-token generation
  - Measure peak RAM usage during inference
  - Evaluate accuracy: does model output correct intent + slots?
- Save results to `data/results/comparison_table.csv`

### `comparison.py`
- Read benchmark results
- Generate formatted comparison table (RESULTS.md):
  - Model name, size, TTFT, TPS, RAM, accuracy
  - Highlight trade-offs (4-bit vs 8-bit, model size vs accuracy)
- Generate MODEL_CARD.md:
  - Model architecture, training approach, dataset
  - Performance metrics
  - Comparison of model variants on synthetic test set
  - Intended use, limitations, biases

### `demo_cli.py`
- Load a user-selected quantized model
- Interactive loop:
  - Prompt: `> {user input}`
  - Parse input through model
  - Output structured command: `{"intent": "...", "slots": {...}}`
  - Display readable output: `✓ Intent: set_climate | Slots: {"mode": "cool", "temp": 21}`
- Support multiple commands in one session

### `utils.py`
- Logging configuration
- Config loading from `.env`: `HF_TOKEN` (required), `GEMINI_API_KEY` (optional, only for `--backend gemini`)
- File path helpers
- Shared constants (model IDs, LoRA config, training config)

---

## Dataset Decision

**Datasets explored (2026-04-02):** Kaggle Car-Command, Fluent Speech Commands, SLURP (full and filtered to IoT/transport/weather), SNIPS NLU, MASSIVE, MAC-SLU.

**Why synthetic:** The Kaggle dataset has a circular mapping problem — folder name = utterance = label — so fine-tuning reduces to near-identity text copying, not intent understanding. Other public datasets (SNIPS, SLURP) are wrong domain; filtering SLURP to relevant domains still leaves mismatched intent schemas and slot types, making the benchmark measure domain transfer rather than quantization impact. Synthetic car commands are the right call because: (1) the task — natural language → structured intent + slots — is genuinely hard enough for LLM fine-tuning to add value, (2) the benchmark stays clean with no domain confounds, and (3) the data generation process is transparent and reproducible. Car commands are short and simple enough that a model pre-trained on language already understands them without fine-tuning — the fine-tuning teaches structured extraction, not language understanding.

---

## Data Pipeline

**Dataset:** ✅ Synthetic car commands — 14 intents, **1,571 utterances** generated via Ollama (`llama3.1:8b`, local)
- **Generator:** `src/generate_dataset.py` — batched calls (20 examples/call), validated, incrementally written
- **Intents:** set_climate, navigate, play_media, adjust_volume, call_contact, read_message, seat_control, set_lighting, window_control, cruise_control, safety_assist, vehicle_info, drive_mode, connectivity
- **Slot depth:** Deep (set_climate, navigate, seat_control, set_lighting), Medium (play_media, adjust_volume, window_control, cruise_control, connectivity), Shallow (call_contact, read_message, safety_assist, vehicle_info, drive_mode)
- **Utterances per intent:** 130 (deep), 120 (medium), 100 (shallow)
- **Train/test split:** Stratified 80/20 by intent → **1,252 train / 319 test** (49 duplicates removed)
- **Raw outputs:** Saved per-intent to `data/raw/synthetic/` for traceability (14 files, gitignored)

**Processing:**
1. Call Gemini Flash per intent with slot schema + style variation prompt
2. Validate: known intent, known slot keys, non-empty utterance
3. Deduplicate exact-match utterances across all intents
4. Stratified 80/20 split → `data/processed/train.jsonl` / `test.jsonl`
5. Save metadata: intent distribution, slot coverage stats

**Fine-tuning format:**
```json
{
  "text": "Command: cool the front down to 20\nAction: {\"intent\": \"set_climate\", \"slots\": {\"zone\": \"front\", \"temperature\": 20, \"mode\": \"cool\"}}"
}
```

---

## Fine-Tuning Strategy

**Models:**
- **Llama 3.2 3B** — Meta, efficient, strong reasoning
- **Qwen 2.5 3B** — Alibaba, multilingual, automotive-friendly
- **SmolLM2 1.7B** — Tiny Labs, minimal footprint, fastest inference

**Framework:** MLX-LM LoRA (`finetune_mlx.py`) — native Apple Silicon, Metal backend
- No PyTorch/MPS CPU fallbacks; all ops run on GPU
- Output directly in MLX format; no conversion before quantization
- `finetune.py` (HF TRL) kept alongside for reference and TRL/PEFT learning
- TRL extensible to DPO/preference learning in v2 if needed

**Training parameters:**
- Epochs: 3
- Batch size: 8
- Learning rate: 2e-4
- LoRA rank: 8, alpha: 16
- Max sequence length: 256

**Design choice — identical hyperparameters across all three models:**
Same params are intentional. The goal is a controlled comparison: if Llama outperforms SmolLM2, that should reflect model architecture, not training setup. Tuning each model separately would confound the benchmark results.

**Known risk — batch size 8 on 3B models:**
Batch size 8 is tight on M4 Pro for the 3B models even with bfloat16. SmolLM2 1.7B handles it comfortably. No `gradient_accumulation_steps` fallback is implemented. If a 3B model OOMs mid-epoch, either:
- Drop batch size to 4 with `gradient_accumulation_steps=2` (preserves effective batch of 8), or
- Run SmolLM2 first to validate the pipeline, then attempt Llama/Qwen

**Output:** Three fine-tuned models (LoRA checkpoints)
- Merge with base model before quantization
- Save merged models to `models/finetuned/`

**Apple Silicon implementation notes (discovered 2026-04-02):**
- `device_map={"": "mps"}` used instead of `"auto"` — Accelerate does not route to MPS with `"auto"`
- `torch_dtype=torch.bfloat16` set at load time — halves model memory (~3 GB vs ~6 GB per 3B model)
- `del model, merged` + `torch.mps.empty_cache()` called after each model save — prevents OOM when loading the next model sequentially (CPU and GPU share the same physical RAM on Apple Silicon)

**Runtime estimate:** 20-45 min per model on M4 Pro (total ~2 hours)

---

## Quantization Strategy

**Format:** MLX (Apple Silicon optimized)

**Variants per model:**
- 4-bit quantization — Aggressive compression, fastest
- 8-bit quantization — Moderate compression, better accuracy retention

**9 total variants:**
1. Llama 3.2 3B base (unquantized, reference)
2. Llama 3.2 3B fine-tuned (unquantized, reference)
3. Llama 3.2 3B 4-bit
4. Llama 3.2 3B 8-bit
5. Qwen 2.5 3B base
6. Qwen 2.5 3B fine-tuned
7. Qwen 2.5 3B 4-bit
8. Qwen 2.5 3B 8-bit
9. SmolLM2 1.7B base, fine-tuned, 4-bit, 8-bit (4 variants)

**Metrics captured:**
- Model size (MB) before/after
- Memory footprint (RAM) at inference
- Load time
- Quantization quality (test accuracy)

---

## Benchmarking & Evaluation

**Metrics measured on all 9 variants:**

| Metric | What it measures | How |
|--------|------------------|-----|
| **Size (MB)** | Disk footprint | File size |
| **TTFT (ms)** | Time to first token | Measure latency for 1 token generation |
| **TPS** | Throughput | Generate 100 tokens, calculate tokens/sec |
| **RAM (MB)** | Peak memory usage | Monitor during inference |
| **Accuracy (%)** | Intent + slots understanding | Evaluate on held-out test set |

**Evaluation process:**
1. Load each variant
2. Run on 50 test examples
3. Measure latency, TPS, memory
4. Compare model output intent + slots against ground truth
5. Aggregate results → CSV

**Output format (RESULTS.md):**
```
| Model | Size (MB) | TTFT (ms) | TPS | RAM (MB) | Accuracy (%) |
|-------|-----------|-----------|-----|----------|--------------|
| Llama-3.2-3B base | 6200 | 45 | 22 | 8500 | 87.3% |
| Llama-3.2-3B fine-tuned | 6200 | 46 | 21 | 8500 | 91.2% |
| Llama-3.2-3B 4-bit | 1550 | 38 | 26 | 3200 | 88.9% |
...
```

**Trade-off narrative:** Document what you learn:
- "4-bit quantization reduces memory by 75% with <3% accuracy loss"
- "SmolLM2 achieves 85% accuracy with 2x faster inference than Llama"
- "Qwen fine-tuned on synthetic car commands shows strongest slot accuracy + quantization benefit"

---

## Voice Demo CLI

**Purpose:** Interactive proof-of-concept showing the pipeline end-to-end.

**Features:**
- Load any quantized model (user chooses variant)
- Text input loop: `> "turn on AC to 21 degrees"`
- Model inference → structured output
- Display parsed command: `✓ Intent: set_climate | Slots: {"mode": "cool", "temp": 21}`
- Multi-command session support

**Optional (stretch goal, v2):** mlx-whisper STT integration for voice input

**Example:**
```
$ python src/demo_cli.py --model smollm2-1.7b-4bit

🚗 Car Command Assistant (Quantized)
Model: smollm2-1.7b-4bit | Memory: 850 MB

> turn on the air conditioning
✓ Intent: set_climate | Slots: {"mode": "cool"}

> navigate to nearest gas station
✓ Intent: navigation | Slots: {"destination_type": "gas_station"}
```

---

## Deliverables for v1

**Portfolio outputs:**

1. **RESULTS.md** — Human-readable comparison table with all 9 variants, key insights, trade-offs
2. **MODEL_CARD.md** — Academic-style model card with architecture, training, performance, limitations, and cross-variant comparison
3. **README.md** — Landing page with problem statement, key results, quick start
4. **docs/ARCHITECTURE.md** — Technical explanation of pipeline and design choices
5. **docs/SETUP.md** — Installation and reproduction guide
6. **Working code** — All modules fully functional, tests passing

**Testing:**
- `test_dataset.py` — Validate split/save/dedup utilities
- `test_finetune.py` — Mock fine-tuning tests
- `test_demo_cli.py` — CLI functionality

**Nice-to-have (v2):**
- Voice input via mlx-whisper STT
- DPO fine-tuning for preference learning
- Expanded synthetic dataset with more utterance variation per intent
- Web-based demo dashboard
- **Parallel dataset experiment (v2 research angle):** Train one model variant on a real NLU dataset (e.g. SNIPS adapted) and one on the synthetic dataset, compare accuracy, slot coverage, and generalisation. Demonstrates understanding of dataset quality trade-offs. Strong portfolio differentiator if results tell a clear story.

---

## Success Criteria

**v1 success (1-2 weeks):**
✅ All 3 models fine-tuned and quantized
✅ Comparison table with 9 variants and 5+ metrics
✅ CLI demo runs without errors
✅ Model card + documentation published
✅ Results validated on synthetic test set (stratified 20% hold-out)
✅ Code is reproducible (SETUP.md + requirements.txt)

**What counts as "meaningful insights" for v2 decision:**
- Clear trade-off patterns (e.g., 4-bit is X% faster, Y% less memory, Z% accuracy loss)
- Qwen outperforms on automotive use case
- SmolLM2 shows edge viability
- Interest in expanding to real voice input

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Gemini API quota during dataset generation | Generation script writes incrementally per intent; resume-safe on re-run |
| Fine-tuning takes too long | Skip epochs if needed (train 1 epoch instead of 3), results still valid |
| Quantization errors | Test each variant loads correctly before benchmarking |
| Accuracy too low post-quantization | 8-bit quantization should retain >85% baseline |
| M4 Pro memory pressure | Start with SmolLM2 1.7B (smallest), add larger models incrementally |

---

## Timeline (v1: 1-2 weeks)

- **Week 1:**
  - Day 1-2: Setup, dataset download, module scaffolding
  - Day 3-4: Fine-tune Llama 3.2 3B and Qwen 2.5 3B
  - Day 5: Quantize both models, start SmolLM2
- **Week 2:**
  - Day 1: Finish SmolLM2, run all benchmarks
  - Day 2-3: Generate comparison table, write model card
  - Day 4: CLI demo, documentation polish
  - Day 5: Final testing, commit, cleanup

---

