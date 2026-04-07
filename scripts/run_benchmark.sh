#!/usr/bin/env bash
set -euo pipefail # means it stops immediately if any variant fails rather than silently continuing

# Run benchmarks for all 9 model variants sequentially with power measurement.
# Must be run from the project root: bash scripts/run_benchmark.sh

VARIANTS=(
    smollm2-finetuned
    smollm2-4bit
    smollm2-8bit
    qwen-finetuned
    qwen-4bit
    qwen-8bit
    llama-finetuned
    llama-4bit
    llama-8bit
)

echo "Starting benchmark run for ${#VARIANTS[@]} variants (measure-power enabled)"
echo "Results will be upserted into data/results/comparison_table.csv"
echo ""

for variant in "${VARIANTS[@]}"; do
    echo "=========================================="
    echo "Benchmarking: $variant"
    echo "=========================================="
    python -m src.benchmark --variant "$variant" --measure-power
    echo ""
done

echo "All variants complete. Results saved to data/results/comparison_table.csv"
