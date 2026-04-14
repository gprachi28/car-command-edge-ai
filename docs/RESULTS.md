# Results — Car Command Edge AI

Three fine-tuned LLMs (Llama 3.2 3B, Qwen 2.5 3B, SmolLM2 1.7B) evaluated across 9 variants (BF16 + 4-bit + 8-bit MLX quantization) on a synthetic car command test set.

**Hardware:** Apple M4 Pro (~273 TOPS Neural Engine)
**Dataset:** v2 synthetic dataset — ~1,200 utterances, 229 evaluated examples (231-example stratified 20% hold-out, 14 intent classes; 2 warmup examples discarded)
**Inference:** greedy decoding (`temp=0.0`), runs to EOS or `max_tokens=150`, wall-clock TPS
**TTFT target:** < 200 ms (real-time voice assistant threshold for in-car use)

---

## Fine-Tuning Results

3 epochs, 939 iterations, batch size 4 with gradient accumulation 2 (effective batch 8), seed 42.
Loss curves: `data/results/loss_curves/`

| Model | Trainable params | Val loss (start → end) | Train loss (final) | Peak RAM | lr | LoRA rank |
|-------|:----------------:|:----------------------:|:-----------------:|:--------:|:--:|:---------:|
| SmolLM2 1.7B | 0.18% (3.0M / 1,711M) | 3.018 → **0.647** | 0.392 | 4.2 GB | 2e-4 | 8 |
| Qwen 2.5 3B | 0.11% (3.3M / 3,086M) | 3.105 → 0.727 | 0.515 | 7.1 GB | 2e-4 | 8 |
| Llama 3.2 3B | ~0.4% (rank 32) | — → 0.690 | 0.428 | 7.6 GB | 2e-5 | 32 |

**Key observations:**
- SmolLM2 converged best — lowest final val loss (0.647) at the smallest parameter count.
- Qwen converged well (0.727) with comparable training speed.
- Llama required different hyperparameters: lr=2e-4 / rank=8 (same as SmolLM2/Qwen) caused the model to memorise a degenerate repetitive output with near-zero accuracy. Dropping to lr=2e-5 and raising rank to 32 resolved this.

---

## Quantization Results

All models quantized from fine-tuned BF16 using `mlx_lm convert -q`. Effective precision: ~4.5-bit (4-bit) and ~8.5-bit (8-bit).

| Model | Fine-tuned (BF16) | 4-bit | 8-bit | 4-bit reduction | 8-bit reduction |
|-------|:-----------------:|:-----:|:-----:|:---------------:|:---------------:|
| SmolLM2 1.7B | 3,268 MB | 922 MB | 1,738 MB | 71.8% | 46.8% |
| Qwen 2.5 3B | 5,897 MB | 1,667 MB | 3,138 MB | 71.7% | 46.8% |
| Llama 3.2 3B | 6,144 MB | 1,740 MB | 3,272 MB | 71.7% | 46.7% |

**Key observations:**
- Compression ratios are near-identical across all three models — expected for uniform linear quantization.
- 4-bit Qwen (1,667 MB) is marginally smaller than 4-bit Llama (1,740 MB) despite similar parameter count — likely due to architecture differences in embedding table size.
- All 6 quantized variants verified loadable before benchmarking.

---

## Benchmark Table

| Variant | Size (MB) ↓ | TTFT (ms) ↓ | TPS ↑ | RAM (MB) ↓ | Intent acc ↑ | Slot acc ↑ | Output tokens | Power (W) | Energy/token (mWh) ↓ |
|---------|----------:|----------:|----:|---------:|---------:|----------:|-------------:|----------:|-------------------:|
| smollm2-finetuned | 3,268 | 74.8 | 72.1 | 3,608 | 98.3% | 66.4% | 29.1 | 11.7 | 0.054 |
| smollm2-4bit | **922** | **54.1** | 200.1 | **1,103** | 96.5% | 61.6% | 29.5 | 15.0 | **0.029** |
| smollm2-8bit | 1,738 | 66.2 | 121.5 | 1,972 | 98.3% | 66.8% | 29.3 | 13.6 | 0.041 |
| qwen-finetuned | 5,897 | 178.6 | 39.8 | 6,385 | **99.6%** | **69.0%** | 25.8 | 12.9 | 0.117 |
| qwen-4bit | 1,667 | 131.4 | **122.8** | 1,833 | 98.3% | **68.6%** | 25.8 | 15.5 | 0.059 |
| qwen-8bit | 3,138 | 152.0 | 72.3 | 3,412 | **99.6%** | **68.6%** | 25.7 | 14.1 | 0.080 |
| llama-finetuned | 6,144 | 165.0 | 38.4 | 6,661 | 97.4% | 62.9% | 24.9 | 12.3 | 0.114 |
| llama-4bit | 1,740 | 120.4 | **123.8** | 1,930 | 94.3% | 55.9% | 24.9 | 15.1 | 0.056 |
| llama-8bit | 3,272 | 133.4 | 70.2 | 3,568 | 96.9% | 62.0% | 24.8 | 14.0 | 0.079 |

> **Note on latency:** All 9 variants meet the 200 ms TTFT target. SmolLM2 variants (54–75 ms) leave substantial headroom for a full STT → LLM → TTS pipeline. Qwen BF16 (179 ms) and Llama BF16 (165 ms) leave little margin and are better suited as BF16 reference baselines than production candidates.
>
> **Note on benchmark vs interactive TTFT:** These numbers are measured back-to-back with no idle time between queries, which keeps Metal compute units fully active. In interactive use, macOS throttles the GPU clock and spins down compute units during the pause while a user types or speaks. The next query has to wait for them to ramp back up before the first token can be computed — adding ~50 ms. Interactive TTFT measured at 97–103 ms for smollm2-4bit across a range of car commands — well within the 200 ms automotive target and more representative of real driver use than the benchmark figure.

---

## Key Insights

**Quantization:**
- **4-bit accuracy cost is model-dependent.** Qwen −1.3% (99.6% → 98.3%), SmolLM2 −1.8% (98.3% → 96.5%), Llama −3.1% (97.4% → 94.3%). Qwen is the most quantization-robust; Llama shows the largest 4-bit accuracy cost.
- **8-bit is effectively lossless** for Qwen (99.6% → 99.6%) and SmolLM2 (98.3% → 98.3%), and near-lossless for Llama (97.4% → 96.9%, −0.5%) while cutting model size by ~47% and RAM by ~45%.
- **4-bit draws more power (W) but less energy per token** — higher throughput more than compensates. smollm2-4bit: 15.0 W → 0.029 mWh/token vs. smollm2-finetuned: 11.7 W → 0.054 mWh/token.
- **4-bit compression's primary automotive benefit is memory bandwidth reduction, not disk size.** Autoregressive LLM inference is bandwidth-bound — each generated token requires loading the full weight set from RAM. A cockpit SoC at 30–50 TOPS typically has 50–100 GB/s memory bandwidth vs M4 Pro's ~273 GB/s. 4-bit quantization cuts bytes-per-token transferred by ~72%, directly reducing the bandwidth bottleneck on target hardware. This is why the TOPS-based TPS scaling estimate (4–6× slower) is conservative — the bandwidth benefit compounds on constrained SoCs.
- **4-bit models draw higher peak wattage than BF16, which matters for thermally constrained hardware.** smollm2-4bit peaks at 15.0 W vs smollm2-finetuned at 11.7 W; qwen-4bit at 15.5 W vs qwen-finetuned at 12.9 W. Automotive SoCs rely on coolant loops or passive heatsinks — no fan. Sustained 15 W on a hot day can trigger thermal throttling, collapsing TPS and breaking the latency budget. On thermally constrained hardware, the lower peak power of BF16 variants (11.7–12.9 W) may be preferable despite their larger footprint, depending on the SoC's thermal design power (TDP).

**Model comparison:**
- **Qwen achieves the highest intent accuracy** (98.3–99.6% across all quantization levels) and the best slot accuracy (68.6–69.0%), reversing the v1 ranking. On the v2 dataset, Qwen's structured output learning is consistent at every compression level.
- **SmolLM2-4bit has the best latency, footprint, and energy** (54.1 ms TTFT, 922 MB, 0.029 mWh/token) with 96.5% intent accuracy. Parse failure rate is 3/229 (1.3%); a JSON parse fallback is still recommended for production.
- **Total response time to complete JSON — not just TTFT — determines pipeline latency.** The TTS synthesizer can't begin speaking until the full JSON is parsed, which only happens after all tokens are generated. Calculated as TTFT + (output_tokens / TPS): smollm2-4bit completes in ~202 ms, qwen-4bit in ~342 ms, llama-4bit in ~322 ms. smollm2-4bit is the only 4-bit variant where the complete response is ready within the 200 ms automotive target. Its 200.1 TPS is what makes this possible.
- **All 9 variants pass the 200 ms TTFT target** comfortably. The highest TTFT is qwen-finetuned at 178.6 ms, leaving 21 ms margin.
- **Llama required different hyperparameters:** lr=2e-4/rank=8 produced degenerate repetitive output and near-zero accuracy. Dropping to lr=2e-5/rank=32 resolved this completely (97.4% intent accuracy fine-tuned).

**Recommended variants:**
- **Best edge (size/latency/energy):** `smollm2-4bit` — 922 MB, 54.1 ms TTFT, 96.5% intent accuracy, 0.029 mWh/token. Parse failure rate 1.3% (3/229); add a JSON parse fallback in production.
- **Best accuracy at 4-bit:** `qwen-4bit` — 98.3% intent accuracy, 131.4 ms TTFT, 1,833 MB RAM, 0.059 mWh/token.
- **Best overall accuracy:** `qwen-finetuned` or `qwen-8bit` — both at 99.6% intent accuracy; qwen-8bit (152.0 ms, 3,412 MB) is the better deployment choice over qwen-finetuned BF16 (178.6 ms, 6,385 MB).
- **Only always-resident candidate on 8 GB hardware:** `smollm2-4bit` at 1,103 MB RAM. A cockpit SoC running the OS, navigation maps, media player, and display compositor has roughly 2–3 GB left for the voice AI model. smollm2-4bit fits; qwen-4bit (1,833 MB) is borderline; all 8-bit and BF16 variants (3,400–6,700 MB) require unloading other systems. If the model isn't always-resident, every voice interaction pays a cold-start penalty — loading a 6 GB BF16 model from flash takes 5–30 seconds.

---

## Slot Accuracy — Per-Intent Breakdown

**Two metrics are now reported per variant** (captured in `data/results/predictions/<variant>.jsonl`):
- **Slot acc (exact-match)** — every key-value pair must match the ground truth exactly; any extra or missing key is a full failure. Strict but understates practical quality.
- **Slot F1** — precision/recall/F1 at the key-value level, excluding None-valued ground truth slots. Two variants: raw model output vs ground truth (`slot_f1_pct`), and schema-filtered output vs ground truth (`slot_f1_filtered_pct`). Schema filtering removes slot keys the model generated that are not in the per-intent allowed key set.

The table below reports the original exact-match figures. Slot F1 and schema-filtered F1 are available per-example in the prediction JSONL files; aggregated figures will be added when the comparison table is next regenerated.

Slot accuracy uses exact-match scoring: every key-value pair in the predicted JSON must match the ground truth exactly. Any extra or missing key counts as a failure. This is strict by design but understates practical extraction quality — a model adding an extra plausible slot (e.g. `"brightness": 100` where the label has none) scores 0 despite being correct.

| Intent | Slot depth | sm2-ft | sm2-4bit | sm2-8bit | qw-ft | qw-4bit | qw-8bit | ll-ft | ll-4bit | ll-8bit |
|--------|:----------:|-------:|--------:|--------:|------:|--------:|--------:|------:|--------:|--------:|
| call_contact | shallow | 94% | 81% | 100% | 94% | 94% | 94% | 88% | 88% | 94% |
| read_message | shallow | 94% | 94% | 94% | 88% | 88% | 88% | 56% | 56% | 56% |
| safety_assist | shallow | 75% | 67% | 75% | 83% | 83% | 83% | 75% | 75% | 75% |
| drive_mode | shallow | 88% | 88% | 88% | 88% | 88% | 88% | 88% | 88% | 88% |
| vehicle_info | shallow | 71% | 71% | 71% | 93% | 64% | 93% | 86% | 86% | 86% |
| cruise_control | medium | 76% | 82% | 76% | 82% | 82% | 82% | 88% | 71% | 82% |
| adjust_volume | medium | 72% | 61% | 83% | 61% | 78% | 61% | 56% | 33% | 56% |
| window_control | medium | 53% | 53% | 53% | 60% | 60% | 60% | 53% | 47% | 53% |
| connectivity | medium | 50% | 44% | 44% | 62% | 69% | 62% | 44% | 44% | 44% |
| seat_control | deep | 67% | 56% | 61% | 78% | 67% | 72% | 67% | 50% | 61% |
| set_lighting | deep | 50% | 61% | 50% | 50% | 39% | 50% | 56% | 39% | 56% |
| set_climate | deep | 50% | 56% | 50% | 39% | 56% | 39% | 44% | 39% | 44% |
| play_media | medium | 47% | 35% | 47% | 59% | 53% | 59% | 41% | 35% | 35% |
| navigate | deep | 50% | 22% | 50% | 44% | 50% | 44% | 50% | 50% | 50% |
| **OVERALL** | | **66%** | **62%** | **67%** | **69%** | **69%** | **69%** | **63%** | **56%** | **62%** |

### Slot Accuracy Insights

**Intent depth still drives slot accuracy more than model size:**
- Shallow intents (`call_contact`, `drive_mode`, `read_message`) consistently score 67–100% across all variants. These have 1–2 fixed slots — little room to add spurious keys.
- Deep intents (`navigate`, `set_climate`, `set_lighting`) score 22–61%. Models generate additional plausible but unlabelled slots, and exact-match scoring fails every such example.

**`navigate` (22–50%) and `set_climate` (39–56%) remain the hardest intents:**
Models tend to infer context beyond what the utterance explicitly states — e.g. adding a travel_mode or units slot when none was labelled. smollm2-4bit is the weakest on navigate at 22%; all 8-bit variants reach 44–50%.

**`call_contact` smollm2-4bit recovered to 81% (was 56%):**
The previous low score was caused by parse truncation in the early-stop inference loop. With EOS-based stopping, smollm2-4bit now produces complete JSON for most shallow intents.

**`vehicle_info` shows a Qwen-4bit dip (64% vs 93% for qwen-finetuned/qwen-8bit):**
Qwen-4bit produces extra slots for vehicle_info queries at 4-bit compression — a schema-following regression not present at 8-bit.

**`connectivity` Qwen-4bit leads at 69%:**
Consistent with v2 observations; Qwen-4bit is the strongest connectivity variant across the board.

**`set_climate` and `play_media` remain challenging:**
Models fill in inferred context beyond what the utterance explicitly states. This is a known failure mode of instruction-tuned models on structured extraction.

**Safety-relevant intents are not separated from convenience intents in the accuracy figures:**
A wrong slot on `play_media` plays the wrong song. A wrong slot on `cruise_control` (incorrect target speed, or enable/disable inversion) or `safety_assist` (wrong feature or action) is a different category of failure. Current slot accuracy for these intents: `cruise_control` 71–88%, `safety_assist` 75–83%. A production system would apply a higher accuracy threshold or require a driver confirmation prompt for commands in these intents before execution.

**Parse reliability (out of 229 examples):**
- smollm2-finetuned: 0 failures
- smollm2-8bit: 0 failures
- qwen-finetuned: 0 failures
- qwen-8bit: 0 failures
- qwen-4bit: 1 failure; llama-finetuned: 1 failure; llama-4bit: 1 failure; llama-8bit: 1 failure
- **smollm2-4bit: 3 failures (1.3%)** — the highest rate; add a JSON parse fallback in production

**Practical implication:**
Slot accuracy numbers (56–69% overall) understate real extraction quality. For production use, switching from exact-match to partial-match scoring (credit for each correct key-value pair, penalty for each extra key) would give a more accurate picture. Intent classification accuracy (94–100%) is the more reliable signal for deployment readiness.
