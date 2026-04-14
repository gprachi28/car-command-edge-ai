#!/usr/bin/env bash
set -euo pipefail # means it stops immediately if any variant fails rather than silently continuing

# Run benchmarks for all 9 model variants sequentially with power measurement.
# Must be run from the project root: bash scripts/run_benchmark.sh
#
# Reproducibility protocol (see design doc for full details):
#   1. Check thermal state is normal before starting: pmset -g therm
#   2. Close all background apps (browser, mail, Slack, Time Machine, iCloud)
#   3. Disable Spotlight indexing: sudo mdutil -a -i off  (re-enable after: sudo mdutil -a -i on)
#   4. Machine must be plugged in (not on battery)
#   5. Run variants smallest-to-largest BF16 size (order below)
#   6. COOLDOWN_S=5 sleep between variants drains the SoC thermal buffer

COOLDOWN_S=5

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
echo "Cooldown between variants: ${COOLDOWN_S}s"
echo ""

for i in "${!VARIANTS[@]}"; do
    variant="${VARIANTS[$i]}"
    echo "=========================================="
    echo "Benchmarking: $variant"
    echo "=========================================="
    python -m src.benchmark --variant "$variant" --measure-power
    echo ""

    # Sleep between variants (skip after last one)
    if [[ $i -lt $(( ${#VARIANTS[@]} - 1 )) ]]; then
        echo "Cooling down (${COOLDOWN_S}s)..."
        sleep "$COOLDOWN_S"
        echo ""
    fi
done

echo "All variants complete. Results saved to data/results/comparison_table.csv"
