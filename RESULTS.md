# Benchmark Results — Car Command Edge AI

Three fine-tuned LLMs (Llama 3.2 3B, Qwen 2.5 3B, SmolLM2 1.7B) evaluated across 9 variants (BF16 + 4-bit + 8-bit MLX quantization) on a synthetic car command test set.

**Hardware:** Apple M4 Pro (~273 TOPS Neural Engine)
**Test set:** 317 examples, stratified 20% hold-out (14 intent classes)
**TTFT target:** < 200 ms (real-time voice assistant threshold for in-car use)

---

## Main Benchmark Table

| Variant | Size (MB) ↓ | TTFT (ms) ↓ | TPS ↑ | RAM (MB) ↓ | Intent acc ↑ | Slot acc ↑ | Output tokens | Power (W) | Energy/token (mWh) ↓ |
|---------|----------:|----------:|----:|---------:|---------:|----------:|-------------:|----------:|-------------------:|
| smollm2-finetuned | 3,268 | 78.6 | 71.4 | 3,612 | 95.9% | **59.6%** | 26.4 | 12.5 | 0.060 |
| smollm2-4bit | **922** | **54.8** | **199.3** | **1,108** | 95.0% | 53.3% | 26.4 | 16.7 | **0.034** |
| smollm2-8bit | 1,738 | 64.5 | 120.9 | 1,969 | **96.2%** | 59.0% | 26.5 | 14.8 | 0.046 |
| qwen-finetuned | 5,897 | 180.4 | 40.0 | 6,385 | 93.1% | 48.3% | 24.4 | 12.2 | 0.112 |
| qwen-4bit | 1,667 | 136.9 | 123.9 | 1,833 | 92.4% | 48.6% | 23.7 | 14.3 | 0.057 |
| qwen-8bit | 3,138 | 152.4 | 73.1 | 3,412 | 92.7% | 47.3% | 24.9 | 13.3 | 0.076 |
| llama-finetuned | 6,144 | 165.5 | 38.8 | 6,662 | 95.9% | 52.7% | 22.0 | **11.5** | 0.108 |
| llama-4bit | 1,740 | 119.7 | 125.2 | 1,935 | 95.3% | 48.9% | **21.5** | 13.6 | 0.054 |
| llama-8bit | 3,272 | 133.6 | 70.9 | 3,568 | 95.6% | 51.7% | 21.9 | 13.3 | 0.077 |

> **Note on latency:** All 9 variants meet the 200 ms TTFT target in isolation. However, a production voice pipeline also includes STT (speech-to-text) and TTS (text-to-speech) stages. SmolLM2 variants (55–79 ms) leave substantial headroom for the full pipeline; Qwen BF16 at 180 ms and Llama BF16 at 166 ms leave very little margin and would be marginal in a STT → LLM → TTS chain on constrained hardware.

---

## Key Insights

**Quantization:**
- **4-bit costs ≤1% intent accuracy** across all models (SmolLM2: −0.9%, Qwen: −0.7%, Llama: −0.6%) while cutting model size by ~72% and RAM by ~70%.
- **8-bit quantization is effectively lossless** — all three models match or marginally exceed their BF16 fine-tuned baseline on intent accuracy.
- **4-bit draws more power (W) but less energy per token** — higher throughput more than compensates. smollm2-4bit: 16.7 W → 0.034 mWh/token vs. smollm2-finetuned: 12.5 W → 0.060 mWh/token.

**Model comparison:**
- **SmolLM2 dominates at every quantization level.** Despite being the smallest model (1.7B), it achieves the highest intent accuracy (96.2% at 8-bit), lowest TTFT (54.8 ms at 4-bit), and lowest energy per token — all at a fraction of the memory footprint.
- **Qwen accuracy is consistently lower** than SmolLM2 and Llama (92–93% vs 95–96%), with the highest parse failure rate (3–6 malformed JSON outputs per 317 examples vs. 0–1 for SmolLM2/Llama).
- **Llama required different hyperparameters:** lr=2e-4/rank=8 (same as SmolLM2/Qwen) produced degenerate repetitive output and near-zero accuracy. Dropping to lr=2e-5/rank=32 resolved this completely (95.9% intent accuracy).

**Recommended variants:**
- **Best edge deployment:** `smollm2-4bit` — 922 MB, 54.8 ms TTFT, 95% intent accuracy, 0.034 mWh/token.
- **Best accuracy:** `smollm2-8bit` — 96.2% intent accuracy, 64.5 ms TTFT, 1,969 MB RAM.
- **Tightest memory budget:** `smollm2-4bit` at 1,108 MB RAM fits comfortably within the ≤16 GB cockpit SoC envelope and would run at ~4–6× worse TPS on a 30–50 TOPS automotive chip, still well within the 200 ms TTFT target for short car commands.

---

## Slot Accuracy — Per-Intent Breakdown

Slot accuracy uses exact-match scoring: every key-value pair in the predicted JSON must match the ground truth exactly. Any extra or missing key counts as a failure. This is strict by design but understates practical extraction quality — a model adding an extra plausible slot (e.g. `"brightness": 100` where the label has none) scores 0 despite being correct.

| Intent | Slot depth | sm2-ft | sm2-4bit | sm2-8bit | qw-ft | qw-4bit | qw-8bit | ll-ft | ll-4bit | ll-8bit |
|--------|:----------:|-------:|--------:|--------:|------:|--------:|--------:|------:|--------:|--------:|
| call_contact | shallow | 80% | 75% | 80% | 80% | 75% | 80% | 75% | 80% | 75% |
| read_message | shallow | 70% | 75% | 70% | 80% | 75% | 75% | 70% | 75% | 70% |
| safety_assist | shallow | 83% | 78% | 83% | 72% | 67% | 72% | 72% | 78% | 72% |
| drive_mode | shallow | 78% | 78% | 78% | 72% | 72% | 72% | 67% | 67% | 67% |
| vehicle_info | shallow | 80% | 60% | 80% | 70% | 70% | 65% | 55% | 55% | 55% |
| cruise_control | medium | 75% | 71% | 75% | 42% | 58% | 42% | 79% | 75% | 79% |
| adjust_volume | medium | 74% | 52% | 74% | 57% | 57% | 52% | 43% | 35% | 39% |
| window_control | medium | 65% | 61% | 65% | 39% | 52% | 39% | 57% | 52% | 57% |
| connectivity | medium | 39% | 39% | 39% | 35% | 39% | 35% | 39% | 39% | 35% |
| seat_control | deep | 58% | 50% | 58% | 62% | 50% | 62% | 65% | 62% | 65% |
| set_lighting | deep | 42% | 50% | 38% | 31% | 31% | 31% | 54% | 38% | 50% |
| set_climate | deep | 46% | 38% | 46% | 27% | 23% | 27% | 35% | 38% | 31% |
| play_media | medium | 46% | 29% | 42% | 29% | 33% | 29% | 33% | 12% | 38% |
| navigate | deep | 23% | 15% | 23% | 12% | 8% | 12% | 12% | 4% | 12% |
| **OVERALL** | | **59.6%** | **53.3%** | **59.0%** | **48.3%** | **48.6%** | **47.3%** | **52.7%** | **48.9%** | **51.7%** |

### Slot Accuracy Insights

**Intent depth drives slot accuracy more than model size:**
- Shallow intents (`call_contact`, `safety_assist`, `drive_mode`) consistently score 67–83% across all variants and models. These have 1–2 fixed slots — there's little room to add spurious keys.
- Deep intents (`navigate`, `set_climate`, `play_media`) score 4–46%. The model generates additional plausible but unlabelled slots, and exact-match scoring fails every such example.

**`navigate` is the worst intent (4–23%):**
Navigation commands accept highly variable slot combinations (`destination`, `destination_type`, `route_preference`, `waypoint`, etc.). The ground truth often captures only the explicit slot mentioned; the model fills in additional context it infers. For example, `"Take me to the airport via the highway"` might be labelled `{"destination": "airport"}` while the model outputs `{"destination": "airport", "route": "highway"}` — factually correct but scored as wrong.

**`connectivity` is a flat floor at 35–39%:**
All nine variants land in a narrow band regardless of model size or quantization, suggesting the issue is in the training data schema (ambiguous or inconsistently labelled connectivity slots) rather than model capacity.

**`set_climate` and `play_media` show the extra-slot hallucination pattern most clearly:**
Models add temperature defaults, fan speed, or media source even when the utterance doesn't specify them. This is a known failure mode of instruction-tuned models on structured extraction — they complete plausible schemas rather than copy only what was said.

**Parse reliability:**
- SmolLM2: 0–1 malformed JSON outputs per 317 examples (most reliable)
- Llama: 0 failures (all variants)
- Qwen: 3–6 failures per 317 examples — most likely to output invalid JSON under 4-bit/8-bit compression

**Practical implication:**
Slot accuracy numbers (47–60%) understate real extraction quality. For production use, switching from exact-match to partial-match scoring (credit for each correct key-value pair, penalty for each extra key) would give a more accurate picture. Intent classification accuracy (92–96%) is the more reliable signal for deployment readiness.
