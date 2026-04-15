"""Benchmark all 9 fine-tuned model variants: TTFT, TPS, RAM, accuracy.

Variants benchmarked:
    - 3 fine-tuned BF16 (finetuned/{smollm2,qwen,llama}-mlx/)
    - 6 quantized (quantized/{smollm2,qwen,llama}-{4,8}bit/)

Metrics per variant:
    - TTFT (ms)              : wall-clock ms from prompt submission to first token
    - TTFT pass              : True if TTFT < 200 ms (real-time threshold)
    - TPS                    : generation tokens/sec (from mlx_lm GenerationResponse)
    - RAM (MB)               : peak memory during inference (mlx_lm GenerationResponse)
    - Accuracy (%)           : intent classification accuracy on 319 test examples
    - Output tokens (avg)    : avg generated tokens per example (car commands <25)
    - Power (W)              : avg GPU+CPU power draw during benchmark (opt-in, sudo)
    - Energy/token (mWh/tok) : energy cost per output token — the metric automotive SoC
                               datasheets use (opt-in, needs sudo)

Edge deployment context (cockpit SoC target: 30-50 TOPS, <=16 GB RAM, TTFT < 200 ms):
    - TTFT is the only latency metric that matters for voice assistants in a car.
      A 1-second delay feels broken to a driver.
    - Energy/token matters for thermal throttling on passive-cooled automotive SoCs.
    - TPS matters less — car commands are short JSON outputs (~15-25 tokens).

Power measurement requires sudo and uses macOS powermetrics.
Pass --measure-power to enable.

Results saved to data/results/comparison_table.csv.

Public API:
    - run_benchmark(variant_key, n_samples, measure_power) -> dict
    - benchmark_all(n_samples, measure_power) -> list[dict]
"""

import csv
import gc
import json
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from src.utils import (
    build_variants,
    dir_size_mb,
    filter_slots,
    get_data_dir,
    get_logger,
    get_models_dir,
    get_processed_dir,
    parse_action,
)

logger = get_logger(__name__)

# Edge deployment context (from edge-ai-automotive-context.md):
# TTFT < 200 ms is the real-time target for in-car voice assistants.
# This is noted in results but not enforced as a hard filter here.


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------
# build_variants() is the single source of truth — imported from utils.


# ---------------------------------------------------------------------------
# Test data loading
# ---------------------------------------------------------------------------


def _load_test_examples(processed_dir: Path, n_samples: int | None) -> list[dict]:
    """Load test examples from test.jsonl, plus WARMUP_RUNS extra for warm-up.

    Returns WARMUP_RUNS extra examples so the warmup pool and the evaluation
    pool are non-overlapping. Callers slice accordingly.

    Each example returned as:
        {"prompt": "Command: ...\\nAction: ", "ground_truth": {...}}

    Args:
        processed_dir: Directory containing test.jsonl.
        n_samples: Number of evaluation examples (warmup examples are
            additional). Pass None to load all available examples.

    Returns:
        List of example dicts with length
        min(n_samples + WARMUP_RUNS, available) when n_samples is given,
        or all valid examples + WARMUP_RUNS when n_samples is None.
    """
    test_path = processed_dir / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"test.jsonl not found at {test_path}")

    examples = []
    with test_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec["text"]
            # Split "Command: X\nAction: Y" into prompt + ground truth
            marker = "\nAction: "
            idx = text.index(marker)
            prompt = text[: idx + len(marker)]
            gt_str = text[idx + len(marker) :]
            try:
                ground_truth = json.loads(gt_str)
            except json.JSONDecodeError:
                continue
            examples.append({"prompt": prompt, "ground_truth": ground_truth})
            if n_samples is not None and len(examples) >= n_samples + WARMUP_RUNS:
                break

    logger.info("Loaded %d test examples", len(examples))
    return examples


# ---------------------------------------------------------------------------
# Inference + measurement
# ---------------------------------------------------------------------------

MAX_TOKENS = 150  # enough for any car command JSON action (~80 chars → ~25 tokens max)
WARMUP_RUNS = 2  # discard first N runs to warm up Metal shader cache


def _infer(model, tokenizer, prompt: str) -> tuple[float, float, float, str, int]:
    """Run one inference pass and return metrics.

    Args:
        model: Loaded MLX model.
        tokenizer: Loaded tokenizer.
        prompt: Full prompt string.

    Returns:
        Tuple of (TTFT in ms, generation TPS, peak RAM in MB, generated text,
        output token count).
    """
    ttft_ms = None
    t_first_token: float | None = None
    t_last_token: float | None = None
    peak_ram_mb = 0.0
    output_tokens: list[str] = []
    last_generation_tokens = 0

    t_start = time.perf_counter()
    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=MAX_TOKENS,
        sampler=make_sampler(
            temp=0.0
        ),  # greedy decoding — deterministic, required for JSON
    ):
        t_now = time.perf_counter()
        if ttft_ms is None:
            ttft_ms = (t_now - t_start) * 1000
            t_first_token = t_now
        t_last_token = t_now
        output_tokens.append(response.text)
        last_generation_tokens = response.generation_tokens
        peak_ram_mb = (
            response.peak_memory * 1024
        )  # mlx_lm reports in GB → convert to MB
        if response.finish_reason is not None:
            break

    # Wall-clock TPS: tokens generated / elapsed generation time (excludes prefill).
    # response.generation_tps is unreliable — MLX lazy evaluation means the timer fires
    # before GPU kernels complete, inflating TPS 10-15x.
    if (
        t_first_token is not None
        and t_last_token is not None
        and t_last_token > t_first_token
    ):
        tps = last_generation_tokens / (t_last_token - t_first_token)
    else:
        tps = 0.0

    if ttft_ms is None:
        # Model generated zero tokens — inference failure, not sub-ms TTFT
        logger.warning("_infer: model produced no tokens for prompt: %r", prompt[:80])
        ttft_ms = float("nan")

    return (
        ttft_ms,
        tps,
        peak_ram_mb,
        "".join(output_tokens),
        last_generation_tokens,
    )


def _slot_f1(pred_slots: dict, gt_slots: dict) -> tuple[float, float, float]:
    """Compute slot-level precision, recall, and F1 using key-value exact match.

    None-valued slots in ground truth are excluded — they represent absent optional
    slots and exact-match None comparison is unreliable across models.

    Args:
        pred_slots: Predicted slot dict from model output.
        gt_slots: Ground truth slot dict.

    Returns:
        Tuple of (precision, recall, f1) in [0, 1].
    """
    gt_items = {k: v for k, v in gt_slots.items() if v is not None}
    pred_items = {k: v for k, v in pred_slots.items() if v is not None}

    if not gt_items and not pred_items:
        return 1.0, 1.0, 1.0
    if not pred_items:
        return 0.0, 0.0, 0.0
    if not gt_items:
        # model predicted slots when none were expected
        return 0.0, 1.0, 0.0

    matched = sum(1 for k, v in pred_items.items() if gt_items.get(k) == v)
    precision = matched / len(pred_items)
    recall = matched / len(gt_items)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def _is_correct(predicted: dict | None, ground_truth: dict) -> bool:
    """Check if predicted intent matches ground truth intent.

    Slots are not evaluated — intent classification accuracy is the primary metric.

    Args:
        predicted: Parsed model output dict (may be None).
        ground_truth: Ground truth dict with 'intent' and 'slots' keys.

    Returns:
        True if intent matches exactly.
    """
    if predicted is None:
        return False
    return predicted.get("intent") == ground_truth.get("intent")


# ---------------------------------------------------------------------------
# Power measurement (optional, requires sudo)
# ---------------------------------------------------------------------------

# Match the combined summary line that powermetrics emits once per sample block,
# after all individual CPU/GPU/ANE lines. Reading this single line gives a
# correctly paired total without the off-by-one pairing bug that arises when
# matching CPU Power and GPU Power separately (CPU Power appears before GPU
# Power in each block, so a per-CPU-line trigger always uses the previous
# block's GPU value).
_COMBINED_POWER_RE = re.compile(r"Combined Power \(CPU \+ GPU \+ ANE\):\s+(\d+)\s+mW")


@contextmanager
def _power_monitor(interval_ms: int = 500) -> Generator[list[float], None, None]:
    """Context manager that runs powermetrics in the background and collects samples.

    Uses a reader thread to consume powermetrics stdout in real time — avoids the
    pipe-buffer loss that occurs when reading only after process termination.

    Yields a list that is populated live with combined CPU+GPU+ANE power readings
    (one per powermetrics sample block). On exit, the subprocess is terminated
    and the reader thread joined.

    Requires sudo. If powermetrics is unavailable or fails to start, yields an
    empty list silently — power columns will be empty in results, no crash.

    Args:
        interval_ms: Sampling interval in milliseconds (default: 500).

    Yields:
        List of per-sample total power readings in Watts (populated during context).
    """
    samples: list[float] = []
    proc = None
    reader_thread = None

    def _reader(pipe) -> None:
        """Read powermetrics stdout, appending one sample per block."""
        for line in pipe:
            m = _COMBINED_POWER_RE.search(line)
            if m:
                samples.append(float(m.group(1)) / 1000)  # mW → W

    try:
        proc = subprocess.Popen(
            [
                "sudo",
                "powermetrics",
                "--samplers",
                "gpu_power,cpu_power",
                "-i",
                str(interval_ms),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        reader_thread = threading.Thread(
            target=_reader, args=(proc.stdout,), daemon=True
        )
        reader_thread.start()
    except (FileNotFoundError, PermissionError) as exc:
        logger.warning("powermetrics unavailable: %s — power metrics skipped", exc)

    yield samples

    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if reader_thread is not None:
        reader_thread.join(timeout=2)


# ---------------------------------------------------------------------------
# Per-variant benchmark
# ---------------------------------------------------------------------------


@contextmanager
def _null_context() -> Generator[list, None, None]:
    """No-op context manager used when measure_power=False."""
    yield []


def run_benchmark(
    variant_key: str,
    n_samples: int | None = None,
    processed_dir: Path | None = None,
    models_dir: Path | None = None,
    measure_power: bool = False,
) -> dict:
    """Benchmark a single model variant.

    Args:
        variant_key: Key from the variant registry (e.g. "smollm2-4bit").
        n_samples: Number of test examples to evaluate on. None (default)
            uses all available examples in test.jsonl.
        processed_dir: Path to processed data dir. Defaults to data/processed/.
        models_dir: Path to models dir. Defaults to models/.
        measure_power: If True, run powermetrics during benchmark to capture GPU+CPU
            power draw. Requires sudo. Adds power_w and energy_per_token_mwh to result.

    Returns:
        Dict with keys: variant, size_mb, ttft_ms, tps, ram_mb,
        accuracy_pct, slot_accuracy_pct, slot_f1_pct, slot_f1_filtered_pct,
        output_tokens_avg, power_w, energy_per_token_mwh.
    """
    if models_dir is None:
        models_dir = get_models_dir()
    if processed_dir is None:
        processed_dir = get_processed_dir()

    variants = build_variants(models_dir)
    if variant_key not in variants:
        raise KeyError(
            f"Unknown variant '{variant_key}'. Choose from: {list(variants)}"
        )

    model_path = variants[variant_key]
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    all_examples = _load_test_examples(processed_dir, n_samples)
    warmup_examples = all_examples[: min(WARMUP_RUNS, len(all_examples))]
    eval_examples = all_examples[len(warmup_examples) :]  # non-overlapping eval pool

    size_mb = dir_size_mb(model_path)

    logger.info("=== Benchmarking: %s (%.0f MB) ===", variant_key, size_mb)
    model, tokenizer = load(str(model_path))

    # Warm up Metal shader cache — discard these results
    for ex in warmup_examples:
        _infer(model, tokenizer, ex["prompt"])
    logger.info("Warmup done (%d runs)", len(warmup_examples))

    ttft_list: list[float] = []
    tps_list: list[float] = []
    ram_list: list[float] = []
    n_tokens_list: list[int] = []
    correct = 0
    predictions: list[dict] = []

    t_bench_start = time.perf_counter()

    with _power_monitor() if measure_power else _null_context() as power_samples:
        for i, ex in enumerate(eval_examples):
            ttft_ms, tps, ram_mb, output, n_tokens = _infer(
                model, tokenizer, ex["prompt"]
            )
            predicted = parse_action(output)
            intent_correct = _is_correct(predicted, ex["ground_truth"])

            gt_slots = ex["ground_truth"].get("slots") or {}
            pred_slots_raw = (predicted.get("slots") or {}) if predicted else {}
            pred_intent = predicted.get("intent") if predicted else None

            # Schema-filtered slots: drop keys hallucinated beyond intent schema
            pred_slots_filtered = (
                filter_slots(pred_intent, pred_slots_raw)
                if pred_intent
                else pred_slots_raw
            )

            # Exact-match slot accuracy (original metric — strict)
            slots_correct = predicted is not None and pred_slots_raw == gt_slots

            # Slot F1 — raw (unfiltered) and schema-filtered
            _, _, slot_f1_raw = _slot_f1(pred_slots_raw, gt_slots)
            _, _, slot_f1_filtered = _slot_f1(pred_slots_filtered, gt_slots)

            if intent_correct:
                correct += 1
            ttft_list.append(ttft_ms)
            tps_list.append(tps)
            ram_list.append(ram_mb)
            n_tokens_list.append(n_tokens)

            # Extract utterance from prompt ("Command: X\nAction: " → "X")
            utterance = (
                ex["prompt"]
                .removeprefix("Command: ")
                .removesuffix("\nAction: ")
                .strip()
            )
            predictions.append(
                {
                    "idx": i,
                    "utterance": utterance,
                    "gt_intent": ex["ground_truth"].get("intent"),
                    "gt_slots": gt_slots,
                    "raw_output": output.strip(),
                    "pred_intent": pred_intent,
                    "pred_slots": pred_slots_raw,
                    "pred_slots_filtered": pred_slots_filtered,
                    "intent_correct": intent_correct,
                    "slots_correct": slots_correct,
                    "slot_f1": round(slot_f1_raw, 4),
                    "slot_f1_filtered": round(slot_f1_filtered, 4),
                    "parse_failed": predicted is None,
                    "output_tokens": n_tokens,
                }
            )

            if (i + 1) % 10 == 0:
                logger.info(
                    "  [%s] %d/%d | acc so far: %.1f%%",
                    variant_key,
                    i + 1,
                    len(eval_examples),
                    100 * correct / (i + 1),
                )

    bench_duration_s = time.perf_counter() - t_bench_start

    avg_ttft = sum(ttft_list) / len(ttft_list)
    avg_tps = sum(tps_list) / len(tps_list)
    avg_ram = sum(ram_list) / len(ram_list)
    accuracy = 100 * correct / len(eval_examples)
    avg_output_tokens = sum(n_tokens_list) / len(n_tokens_list)

    # Power and energy metrics (only populated when measure_power=True)
    power_w: float | None = None
    energy_per_token_mwh: float | None = None
    if measure_power and power_samples:
        power_w = round(sum(power_samples) / len(power_samples), 1)
        total_tokens = sum(n_tokens_list)
        if total_tokens > 0:
            # energy (Wh) = avg_power_W × duration_hours; convert to mWh/token
            energy_wh = power_w * (bench_duration_s / 3600)
            energy_per_token_mwh = round((energy_wh / total_tokens) * 1000, 3)

    slots_correct_count = sum(1 for p in predictions if p["slots_correct"])
    slot_accuracy = 100 * slots_correct_count / len(eval_examples)
    avg_slot_f1 = 100 * sum(p["slot_f1"] for p in predictions) / len(predictions)
    avg_slot_f1_filtered = (
        100 * sum(p["slot_f1_filtered"] for p in predictions) / len(predictions)
    )

    result = {
        "variant": variant_key,
        "size_mb": round(size_mb),
        "ttft_ms": round(avg_ttft, 1),
        "tps": round(avg_tps, 1),
        "ram_mb": round(avg_ram),
        "accuracy_pct": round(accuracy, 1),
        "slot_accuracy_pct": round(slot_accuracy, 1),
        "slot_f1_pct": round(avg_slot_f1, 1),
        "slot_f1_filtered_pct": round(avg_slot_f1_filtered, 1),
        "output_tokens_avg": round(avg_output_tokens, 1),
        "power_w": power_w,
        "energy_per_token_mwh": energy_per_token_mwh,
    }
    logger.info(
        "Done: %s | size=%.0fMB TTFT=%.1fms TPS=%.1f RAM=%.0fMB "
        "intent_acc=%.1f%% slot_acc=%.1f%% "
        "slot_f1=%.1f%% slot_f1_filtered=%.1f%% tokens_avg=%.1f",
        variant_key,
        size_mb,
        avg_ttft,
        avg_tps,
        avg_ram,
        accuracy,
        slot_accuracy,
        avg_slot_f1,
        avg_slot_f1_filtered,
        avg_output_tokens,
    )

    _save_predictions(predictions, variant_key)

    # Free memory before loading next model
    del model
    gc.collect()
    mx.clear_cache()

    return result


# ---------------------------------------------------------------------------
# Full benchmark sweep
# ---------------------------------------------------------------------------


def benchmark_all(
    n_samples: int | None = None,
    processed_dir: Path | None = None,
    models_dir: Path | None = None,
    measure_power: bool = False,
) -> list[dict]:
    """Benchmark all variants and save results to CSV.

    Args:
        n_samples: Number of test examples per variant. None (default)
            uses all available examples in test.jsonl.
        processed_dir: Path to processed data dir.
        models_dir: Path to models dir.
        measure_power: Enable GPU+CPU power measurement via powermetrics (needs sudo).

    Returns:
        List of result dicts, one per variant.
    """
    if models_dir is None:
        models_dir = get_models_dir()
    if processed_dir is None:
        processed_dir = get_processed_dir()

    variants = build_variants(models_dir)
    results = []

    for variant_key in variants:
        result = run_benchmark(
            variant_key=variant_key,
            n_samples=n_samples,
            processed_dir=processed_dir,
            models_dir=models_dir,
            measure_power=measure_power,
        )
        results.append(result)

    _save_csv(results, get_data_dir() / "results" / "comparison_table.csv")
    return results


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "variant",
    "size_mb",
    "ttft_ms",
    "tps",
    "ram_mb",
    "accuracy_pct",
    "slot_accuracy_pct",
    "slot_f1_pct",
    "slot_f1_filtered_pct",
    "output_tokens_avg",
    "power_w",
    "energy_per_token_mwh",
]


def _save_csv(results: list[dict], output_path: Path) -> None:
    """Write benchmark results to CSV.

    Args:
        results: List of result dicts from run_benchmark.
        output_path: Path to write CSV to.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    logger.info("Results saved to %s", output_path)


def _save_predictions(predictions: list[dict], variant_key: str) -> None:
    """Save per-example predictions to data/results/predictions/<variant_key>.jsonl.

    Each line is a JSON object with utterance, ground truth, raw model output,
    parsed prediction, correctness flags, and output token count.

    Args:
        predictions: List of per-example result dicts from run_benchmark.
        variant_key: Variant name used as the filename stem.
    """
    out_dir = get_data_dir() / "results" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{variant_key}.jsonl"
    with out_path.open("w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    parse_failures = sum(1 for p in predictions if p["parse_failed"])
    slots_correct = sum(1 for p in predictions if p["slots_correct"])
    logger.info(
        "Predictions saved to %s | parse failures: %d | slots correct: %d/%d",
        out_path,
        parse_failures,
        slots_correct,
        len(predictions),
    )


def _upsert_csv(result: dict, csv_path: Path) -> None:
    """Insert or update a single variant row in the CSV.

    Reads existing rows, replaces any row with the same variant key,
    then writes back. Creates the file if it doesn't exist.

    Args:
        result: Single result dict from run_benchmark.
        csv_path: Path to the comparison CSV.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if csv_path.exists():
        with csv_path.open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["variant"] != result["variant"]]
    rows.append(result)
    rows.sort(key=lambda r: r["variant"])
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Upserted %s into %s", result["variant"], csv_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    models_dir = get_models_dir()
    variants = build_variants(models_dir)

    parser = argparse.ArgumentParser(
        description="Benchmark fine-tuned and quantized models"
    )
    # Run variants individually to get accurate per-variant RAM readings.
    # MLX peak_memory is a monotonically increasing high-water mark that never
    # resets between model loads in a single process — running all variants in
    # one invocation inflates RAM figures for every model after the first.
    # Use scripts/run_benchmark.sh to run all 9 variants sequentially.
    parser.add_argument(
        "--variant",
        choices=list(variants.keys()),
        required=True,
        help="Variant to benchmark. Run scripts/run_benchmark.sh to benchmark all.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of test examples to evaluate on (default: all)",
    )
    parser.add_argument(
        "--measure-power",
        action="store_true",
        help=(
            "Measure GPU+CPU power draw via powermetrics (macOS, requires sudo). "
            "Adds power_w and energy_per_token_mwh to results."
        ),
    )
    args = parser.parse_args()

    result = run_benchmark(
        variant_key=args.variant,
        n_samples=args.n_samples,
        measure_power=args.measure_power,
    )
    print(json.dumps(result, indent=2))
    # Upsert this variant's row in the CSV so spot-checks persist
    csv_path = get_data_dir() / "results" / "comparison_table.csv"
    _upsert_csv(result, csv_path)
