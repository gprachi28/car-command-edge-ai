# Car Command Edge AI

Fine-tuning and quantizing small language models for English car voice command understanding, benchmarked on Apple Silicon (M4 Pro).

## What this is

An edge AI pipeline that trains three efficient LLMs on the [Kaggle Car-Command dataset](https://www.kaggle.com/datasets/oortdatahub/car-command) and compares their performance across quantization levels. The goal is to demonstrate the latency/memory/accuracy trade-offs of running compressed models on-device.

**Models:** Llama 3.2 3B · Qwen 2.5 3B · SmolLM2 1.7B  
**Quantization:** 4-bit and 8-bit MLX format  
**Hardware:** Apple M4 Pro

## Dataset

- **Source:** [Kaggle Car-Command](https://www.kaggle.com/datasets/oortdatahub/car-command)
- **Format:** 8,463 audio recordings (MP3/WAV) across 40 intent classes
- **Structure:** `<intent name>/<timestamp>/<uuid>.mp3` — folder name is the label
- **Preprocessing:** mlx-whisper (`whisper-large-v3-mlx`) transcribes audio → text; up to 75 files sampled per intent (~3,000 examples total); `whisper-small` was replaced due to hallucination loops and wrong-language outputs
- **Split:** 80% train / 20% test

## Architecture

```
Kaggle Car-Command (8,463 audio files, 40 intents)
│   Structure: <intent_name>/<timestamp>/<uuid>.mp3
│
└─► dataset.py
        Sample ≤75 files/intent → mlx-whisper transcription (cached)
        Output: {"command": "turn on the AC", "action": "Turn AC ON"}
        Split 80/20 → train.jsonl / test.jsonl
        │
        └─► finetune.py  (HF TRL + LoRA)
                3 models: Llama-3.2-3B · Qwen-2.5-3B · SmolLM2-1.7B
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
                                                mic → mlx-whisper STT
                                                    → quantized LLM
                                                    → {"intent": "Turn AC ON"}
```

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
