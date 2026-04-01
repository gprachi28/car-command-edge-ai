# Car Command Edge AI

Fine-tuning and quantizing small language models for English car voice command understanding, benchmarked on Apple Silicon (M4 Pro).

## What this is

An edge AI pipeline that trains three efficient LLMs on the [Kaggle Car-Command dataset](https://www.kaggle.com/datasets/oortdatahub/car-command) and compares their performance across quantization levels. The goal is to demonstrate the latency/memory/accuracy trade-offs of running compressed models on-device.

**Models:** Llama 3.2 3B · Qwen 2.5 3B · SmolLM2 1.7B  
**Quantization:** 4-bit and 8-bit MLX format  
**Hardware:** Apple M4 Pro

## Dataset

- **Source:** [Kaggle Car-Command](https://www.kaggle.com/datasets/oortdatahub/car-command)
- **Size:** ~2K–5K English car command utterances
- **Categories:** Climate control, navigation, media, windows/doors, lights
- **Split:** 80% train / 20% test

## Status

> Project is in initial setup phase. Results will be added here as experiments complete.

## Results

_To be populated after benchmarking._

## Quick Start

```bash
# Requirements: Python 3.11+, Kaggle API token, Apple Silicon Mac
pip install -r requirements.txt

# Download and prepare dataset
python src/dataset.py

# Run interactive demo (after training)
python src/demo_cli.py --model smollm2-1.7b-4bit
```

## Project Structure

```
src/
├── dataset.py      # Download and split Kaggle Car-Command data
├── finetune.py     # HF TRL + LoRA fine-tuning
├── quantize.py     # MLX 4-bit and 8-bit conversion
├── benchmark.py    # Latency, TPS, memory, accuracy measurement
├── comparison.py   # Generate results table and model card
├── demo_cli.py     # Interactive car command demo
└── utils.py        # Shared config and helpers
```

## Requirements

See `requirements.txt`. Key dependencies: `transformers`, `trl`, `mlx-lm`, `torch`.

Set `HUGGING_FACE_HUB_TOKEN` and `KAGGLE_USERNAME` / `KAGGLE_KEY` in `.env` (see `.env.example`).
