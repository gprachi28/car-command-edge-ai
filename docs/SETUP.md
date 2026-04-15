# Setup Guide

## Requirements

- **Python** 3.11+
- **Apple Silicon Mac** (M1/M2/M3/M4) — MLX-LM runs natively on Metal; Intel Macs are not supported
- **Ollama** — for dataset generation
- **HuggingFace account** — required for Llama 3.2 3B (gated model)

---

## 1. Clone the repository

```bash
git clone https://github.com/gprachi28/car-command-edge-ai.git
cd car-command-edge-ai
```

## 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- `mlx-lm` — fine-tuning, quantization, and inference on Apple Silicon
- `ollama` — dataset generation

## 4. Configure secrets

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```
HF_TOKEN=your_huggingface_token_here       # Required for Llama 3.2 3B
```

**Llama 3.2 3B is a gated model.** Before the download will work you must:
1. Accept Meta's license at [huggingface.co/meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)
2. Enable "Access to public gated repositories" on your HuggingFace fine-grained token settings

## 5. Install Ollama and pull the generation model

```bash
# Install Ollama (macOS)
brew install ollama

# Pull the model used for dataset generation
ollama pull llama3.1:8b

# Start the Ollama server
ollama serve
```

---

## Running the Pipeline

### Dataset generation

```bash
python -m src.generate_dataset
```

Generates ~1,200 utterances across 14 intents using density tiers (full/partial/minimal) with inline validation. Writes per-intent JSONL files to `data/raw/synthetic/` and the processed split to `data/processed/train.jsonl` / `test.jsonl`. Resume-safe — re-running skips already-completed intents.

### Fine-tuning (MLX-LM LoRA)

```bash
python -m src.finetune_mlx
```

Fine-tunes all three models sequentially. Fused models are saved to `models/finetuned/`. Expects `data/processed/train.jsonl` and `test.jsonl` to exist.

Runtime: ~20–45 min per model on M4 Pro (~2 hours total).

### Plot training loss curves

```bash
python -m src.plot_losses
```

Reads loss logs from `data/results/loss_curves/` and saves PNG plots to the same directory.

### Quantization

```bash
python -m src.quantize
```

Produces 4-bit and 8-bit variants for all three models under `models/quantized/`.

### Benchmarking

```bash
bash scripts/run_benchmark.sh
```

Runs each of the 9 variants in its own subprocess for accurate peak RAM measurement. Results written to `data/results/comparison_table.csv` and per-example predictions to `data/results/predictions/`.

### Interactive demo

```bash
python -m src.demo_cli --model smollm2-4bit
```

Loads the selected quantized model and starts an interactive loop. Type any car command and the model outputs structured JSON.

---

## Pre-commit hooks

```bash
pre-commit install
pre-commit run --all-files
```

Runs `black`, `ruff`, and whitespace checks. Required before committing.

---

## Troubleshooting

**`mlx_lm` not found after install**
Ensure you are on Apple Silicon and the venv is active. MLX does not install on Intel Macs.

**Ollama connection refused during dataset generation**
Run `ollama serve` in a separate terminal before running `generate_dataset.py`.

**Llama download fails with 401**
Check that `HF_TOKEN` is set in `.env` and that you have accepted the gated model licence on HuggingFace.

**OOM during fine-tuning**
Start with SmolLM2 1.7B (smallest). Close other GPU-heavy applications. On Apple Silicon, CPU and GPU share physical RAM — memory pressure from other apps directly affects training headroom.
