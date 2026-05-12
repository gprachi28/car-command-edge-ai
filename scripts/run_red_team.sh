#!/usr/bin/env bash
set -euo pipefail

# Run red team adversarial evaluation against all 9 model variants.
# Must be run from the project root: bash scripts/run_red_team.sh
#
# Runs 23 adversarial inputs across 4 categories:
#   asr_noise, ambiguous, ood_intent, adversarial
#
# Results saved to data/results/red_team/<variant>.jsonl
# Console table printed per variant.
#
# No thermal prep needed — red team does not measure RAM or power.

echo "Starting red team evaluation (23 cases × 9 variants)"
echo ""

python -m src.red_team

echo ""
echo "Red team complete. Results saved to data/results/red_team/"
