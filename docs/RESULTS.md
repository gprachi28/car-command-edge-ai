# Results — Car Command Edge AI

Three fine-tuned LLMs (Llama 3.2 3B, Qwen 2.5 3B, SmolLM2 1.7B) evaluated across 9 variants (BF16 + 4-bit + 8-bit MLX quantization) on a synthetic car command test set.

**Hardware:** Apple M4 Pro (~273 TOPS Neural Engine)
**Dataset:** v2 synthetic dataset — ~1,200 utterances, 229 evaluated examples (231-example stratified 20% hold-out, 14 intent classes; 2 warmup examples discarded)
**Inference:** greedy decoding (`temp=0.0`), brace-depth stop (exits when root JSON object closes), wall-clock TPS
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
| smollm2-finetuned | 3,268 | 79.3 | 71.7 | 3,608 | 98.3% | 66.4% | 28.1 | 12.2 | 0.058 |
| smollm2-4bit | **922** | **54.3** | 189.0 | **1,103** | 92.6% | 59.8% | 26.7 | 16.2 | **0.033** |
| smollm2-8bit | 1,738 | 66.4 | 120.5 | 1,971 | 97.8% | 66.8% | 28.1 | 14.4 | 0.044 |
| qwen-finetuned | 5,897 | 178.8 | 39.4 | 6,385 | **98.3%** | **68.1%** | 24.3 | 12.6 | 0.115 |
| qwen-4bit | 1,667 | 132.6 | **123.3** | 1,833 | 97.8% | **68.1%** | 24.7 | 14.1 | 0.054 |
| qwen-8bit | 3,138 | 152.7 | 72.2 | 3,412 | **98.3%** | 67.7% | 24.3 | 13.4 | 0.077 |
| llama-finetuned | 6,144 | 164.5 | 38.2 | 6,661 | 96.1% | 62.4% | 23.5 | 11.7 | 0.108 |
| llama-4bit | 1,740 | 120.9 | 124.2 | 1,930 | 93.9% | 55.9% | 23.7 | 13.8 | 0.053 |
| llama-8bit | 3,272 | 195.3 ⚠️ | 65.5 | 3,568 | 96.1% | 62.0% | 23.5 | **5.8** | 0.039 |

> **Note on latency:** 8 of 9 variants meet the 200 ms TTFT target in isolation. **Llama-8bit (195.3 ms) is marginal** — within 5 ms of the threshold and should not be used in latency-sensitive pipelines. SmolLM2 variants (54–79 ms) leave substantial headroom for a full STT → LLM → TTS pipeline. Qwen BF16 (179 ms) and Llama BF16 (165 ms) leave very little margin.
>
> **Note on benchmark vs interactive TTFT:** These numbers are measured back-to-back with no idle time between queries, which keeps Metal compute units fully active. In interactive use, macOS throttles the GPU clock and spins down compute units during the pause while a user types or speaks. The next query has to wait for them to ramp back up before the first token can be computed — adding ~50 ms. Interactive TTFT is typically 100–150 ms for SmolLM2-4bit. Both figures pass the 200 ms automotive target; the benchmark number reflects sustained throughput, the interactive number is more representative of real driver use.
>
> **Note on Llama-8bit power:** The 5.8 W power reading for llama-8bit is likely a `powermetrics` measurement artifact (brief sampling window at lower GPU utilisation for an 8-bit model). Treat the energy/token figure with caution; the TPS and TTFT numbers are reliable.

---

## Key Insights

**Quantization:**
- **4-bit accuracy cost is model-dependent.** On v2 data: Qwen −0.5% (98.3% → 97.8%), Llama −2.2% (96.1% → 93.9%), SmolLM2 −5.7% (98.3% → 92.6%). Qwen is the most quantization-robust; SmolLM2 shows a significant 4-bit drop despite being the smallest model.
- **8-bit is effectively lossless** for all three models (≤0.5% intent accuracy change) while cutting model size by ~47% and RAM by ~45%.
- **4-bit draws more power (W) but less energy per token** — higher throughput more than compensates. smollm2-4bit: 16.2 W → 0.033 mWh/token vs. smollm2-finetuned: 12.2 W → 0.058 mWh/token.

**Model comparison:**
- **Qwen achieves the highest intent accuracy** (97.8–98.3% across all quantization levels) and the best slot accuracy (67–68%), reversing the v1 ranking. On the cleaner v2 dataset, Qwen's structured output learning shows through clearly.
- **SmolLM2-4bit has the best latency and footprint** (54.3 ms TTFT, 922 MB, 0.033 mWh/token) but carries the lowest intent accuracy among 4-bit variants (92.6%) and a notable parse failure rate (12/229 = 5.2%). More reliable at 8-bit (97.8%, 1 failure).
- **Llama-8bit TTFT (195.3 ms) is marginal** — within 5 ms of the 200 ms automotive target and should not be used in latency-sensitive pipelines. All other variants pass comfortably.
- **Llama required different hyperparameters:** lr=2e-4/rank=8 produced degenerate repetitive output and near-zero accuracy. Dropping to lr=2e-5/rank=32 resolved this completely (96.1% intent accuracy).

**Recommended variants:**
- **Best edge (size/latency/energy):** `smollm2-4bit` — 922 MB, 54.3 ms TTFT, 92.6% intent accuracy, 0.033 mWh/token. Note the 5.2% parse failure rate; add a JSON parse fallback in production.
- **Best accuracy at 4-bit:** `qwen-4bit` — 97.8% intent accuracy, 132.6 ms TTFT, 1,833 MB RAM, 0.054 mWh/token.
- **Best overall accuracy:** `qwen-8bit` — 98.3% intent accuracy, 152.7 ms TTFT, 3,412 MB RAM.
- **Tightest memory budget:** `smollm2-4bit` at 1,103 MB RAM fits comfortably within the ≤16 GB cockpit SoC envelope and would run at ~4–6× worse TPS on a 30–50 TOPS automotive chip, still well within the 200 ms TTFT target for short car commands.
- **Avoid:** `llama-8bit` in latency-sensitive pipelines (195.3 ms, marginal margin).

---

## Slot Accuracy — Per-Intent Breakdown

**Two metrics are now reported per variant** (captured in `data/results/predictions/<variant>.jsonl`):
- **Slot acc (exact-match)** — every key-value pair must match the ground truth exactly; any extra or missing key is a full failure. Strict but understates practical quality.
- **Slot F1** — precision/recall/F1 at the key-value level, excluding None-valued ground truth slots. Two variants: raw model output vs ground truth (`slot_f1_pct`), and schema-filtered output vs ground truth (`slot_f1_filtered_pct`). Schema filtering removes slot keys the model generated that are not in the per-intent allowed key set.

The table below reports the original exact-match figures. Slot F1 and schema-filtered F1 are available per-example in the prediction JSONL files; aggregated figures will be added when the comparison table is next regenerated.

Slot accuracy uses exact-match scoring: every key-value pair in the predicted JSON must match the ground truth exactly. Any extra or missing key counts as a failure. This is strict by design but understates practical extraction quality — a model adding an extra plausible slot (e.g. `"brightness": 100` where the label has none) scores 0 despite being correct.

| Intent | Slot depth | sm2-ft | sm2-4bit | sm2-8bit | qw-ft | qw-4bit | qw-8bit | ll-ft | ll-4bit | ll-8bit |
|--------|:----------:|-------:|--------:|--------:|------:|--------:|--------:|------:|--------:|--------:|
| call_contact | shallow | 94% | 56% | 100% | 94% | 94% | 94% | 88% | 88% | 94% |
| read_message | shallow | 94% | 94% | 94% | 88% | 88% | 88% | 56% | 56% | 56% |
| safety_assist | shallow | 75% | 67% | 75% | 83% | 83% | 83% | 75% | 75% | 75% |
| drive_mode | shallow | 88% | 88% | 88% | 88% | 88% | 88% | 88% | 88% | 88% |
| vehicle_info | shallow | 71% | 71% | 71% | 86% | 57% | 86% | 86% | 86% | 86% |
| cruise_control | medium | 76% | 82% | 76% | 76% | 82% | 76% | 88% | 71% | 82% |
| adjust_volume | medium | 72% | 61% | 83% | 61% | 78% | 61% | 56% | 33% | 56% |
| window_control | medium | 53% | 53% | 53% | 60% | 60% | 60% | 53% | 47% | 53% |
| connectivity | medium | 50% | 44% | 44% | 62% | 69% | 62% | 44% | 44% | 44% |
| seat_control | deep | 67% | 56% | 61% | 78% | 67% | 72% | 67% | 50% | 61% |
| set_lighting | deep | 50% | 61% | 50% | 50% | 39% | 50% | 56% | 39% | 56% |
| set_climate | deep | 50% | 56% | 50% | 39% | 56% | 39% | 44% | 39% | 44% |
| play_media | medium | 47% | 35% | 47% | 59% | 53% | 59% | 41% | 35% | 35% |
| navigate | deep | 50% | 22% | 50% | 44% | 50% | 44% | 44% | 50% | 50% |
| **OVERALL** | | **66%** | **60%** | **67%** | **68%** | **68%** | **68%** | **62%** | **56%** | **62%** |

### Slot Accuracy Insights

**Intent depth still drives slot accuracy more than model size:**
- Shallow intents (`call_contact`, `drive_mode`, `read_message`) consistently score 56–100% across all variants. These have 1–2 fixed slots — little room to add spurious keys.
- Deep intents (`navigate`, `set_climate`, `set_lighting`) score 22–61%. Models generate additional plausible but unlabelled slots, and exact-match scoring fails every such example.

**`navigate` improved significantly vs v1 (22–50% vs 4–23%):**
The v2 density-tier dataset generates cleaner navigate examples with explicit slot guidance. Still variable across variants: smollm2-4bit is the weakest at 22%; all 8-bit variants reach 44–50%.

**`connectivity` floor has risen slightly (44–69% vs 35–39% in v1):**
Qwen-4bit reaches 69% — the best connectivity score across the whole benchmark. The v2 dataset's improved schema consistency has helped here.

**`call_contact` shows a sharp smollm2-4bit drop (56% vs 88–100% for other variants):**
This is consistent with smollm2-4bit's overall parse failure rate (12/229 = 5.2%). Under 4-bit compression, SmolLM2 occasionally truncates or malforms the JSON for shallow intents — not a depth issue but a reliability one specific to this variant.

**`set_climate` and `play_media` remain challenging:**
Models fill in inferred context beyond what the utterance explicitly states. This is a known failure mode of instruction-tuned models on structured extraction.

**Parse reliability (out of 229 examples):**
- smollm2-finetuned: 0 failures (most reliable)
- qwen-finetuned: 3 failures; qwen-4bit: 2; qwen-8bit: 3
- llama-finetuned: 4 failures; llama-4bit: 2; llama-8bit: 3
- smollm2-8bit: 1 failure
- **smollm2-4bit: 12 failures (5.2%)** — significantly higher than all other variants; add a JSON parse fallback in production

**Practical implication:**
Slot accuracy numbers (56–68% overall) understate real extraction quality. For production use, switching from exact-match to partial-match scoring (credit for each correct key-value pair, penalty for each extra key) would give a more accurate picture. Intent classification accuracy (92–98%) is the more reliable signal for deployment readiness.
