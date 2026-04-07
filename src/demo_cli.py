# TODO: Implement interactive CLI demo for car command inference
#
# Planned behaviour (see docs/superpowers/specs/ for full spec):
#
# Usage:
#   python src/demo_cli.py --model smollm2-1.7b-4bit
#
# On startup:
#   - Load the specified quantized MLX model from models/quantized/<model>/
#   - Print model name and peak RAM at load time
#
# Interactive loop:
#   - Prompt user for a car command string
#   - Run inference through the fine-tuned model
#   - Parse the JSON output: {"intent": "...", "slots": {...}}
#   - Display in a readable format:
#       ✓ Intent: set_climate | Slots: {"mode": "cool", "zone": "front"}
#   - Handle malformed JSON output gracefully (model failed to produce valid JSON)
#   - Exit cleanly on Ctrl+C or empty input
#
# Supported model keys (matching models/quantized/ directory names):
#   smollm2-1.7b-finetuned, smollm2-1.7b-4bit, smollm2-1.7b-8bit
#   qwen-2.5-3b-finetuned,  qwen-2.5-3b-4bit,  qwen-2.5-3b-8bit
#   llama-3.2-3b-finetuned, llama-3.2-3b-4bit, llama-3.2-3b-8bit
#
# Implementation notes:
#   - Use mlx_lm.load() and mlx_lm.generate() for inference
#   - Prompt format must match training: "Command: <utterance>\nAction:"
#   - Cap max_tokens at ~80 (car command outputs are 21-27 tokens on average)
#   - Measure and display TTFT for each inference
