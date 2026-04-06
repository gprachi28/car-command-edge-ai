"""Benchmark all 9 fine-tuned model variants: TTFT, TPS, RAM, accuracy.

Variants benchmarked:
    - 3 fine-tuned BF16 (finetuned/{smollm2,qwen,llama}-mlx/)
    - 6 quantized (quantized/{smollm2,qwen,llama}-{4,8}bit/)

Metrics per variant:
    - TTFT (ms)              : wall-clock ms from prompt submission to first token
    - TTFT pass              : True if TTFT < 200 ms (real-time threshold)
    - TPS                    : generation tokens/sec (from mlx_lm GenerationResponse)
    - RAM (MB)               : peak memory during inference (mlx_lm GenerationResponse)
    - Accuracy (%)           : intent classification accuracy on 50 test examples
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

from src.utils import (
    dir_size_mb,
    get_data_dir,
    get_logger,
    get_models_dir,
    get_processed_dir,
)

logger = get_logger(__name__)

# Edge deployment context (from edge-ai-automotive-context.md):
# TTFT < 200 ms is the real-time target for in-car voice assistants.
# This is noted in results but not enforced as a hard filter here.


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------


def _build_variants(models_dir: Path) -> dict[str, Path]:
    """Return ordered dict of variant_key -> model_path."""
    finetuned = models_dir / "finetuned"
    quantized = models_dir / "quantized"
    return {
        "smollm2-finetuned": finetuned / "smollm2-mlx",
        "smollm2-4bit": quantized / "smollm2-4bit",
        "smollm2-8bit": quantized / "smollm2-8bit",
        "qwen-finetuned": finetuned / "qwen-mlx",
        "qwen-4bit": quantized / "qwen-4bit",
        "qwen-8bit": quantized / "qwen-8bit",
        "llama-finetuned": finetuned / "llama-mlx-v2",
        "llama-4bit": quantized / "llama-4bit-v2",
        "llama-8bit": quantized / "llama-8bit-v2",
    }


# ---------------------------------------------------------------------------
# Test data loading
# ---------------------------------------------------------------------------


def _load_test_examples(processed_dir: Path, n_samples: int) -> list[dict]:
    """Load n_samples + WARMUP_RUNS examples from test.jsonl.

    Returns WARMUP_RUNS extra examples so the warmup pool and the evaluation
    pool are non-overlapping. Callers slice accordingly.

    Each example returned as:
        {"prompt": "Command: ...\\nAction: ", "ground_truth": {...}}

    Args:
        processed_dir: Directory containing test.jsonl.
        n_samples: Number of evaluation examples (warmup examples are additional).

    Returns:
        List of example dicts (length: min(n_samples + WARMUP_RUNS, available)).
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
            if len(examples) >= n_samples + WARMUP_RUNS:
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
    tps = 0.0
    peak_ram_mb = 0.0
    output_tokens: list[str] = []

    t_start = time.perf_counter()
    for response in stream_generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS):
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - t_start) * 1000
        output_tokens.append(response.text)
        tps = response.generation_tps
        peak_ram_mb = (
            response.peak_memory * 1024
        )  # mlx_lm reports in GB → convert to MB
        if response.finish_reason is not None:
            break

    return ttft_ms or 0.0, tps, peak_ram_mb, "".join(output_tokens), len(output_tokens)


def _parse_action(text: str) -> dict | None:
    """Extract the first valid JSON object from generated text.

    Args:
        text: Raw model output.

    Returns:
        Parsed dict or None if no valid JSON found.
    """
    text = text.strip()
    # Find first '{' and last '}' to handle trailing tokens
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


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

# Regex to extract GPU and CPU power samples from powermetrics text output.
_GPU_POWER_RE = re.compile(r"GPU Power:\s+(\d+)\s+mW")
_CPU_POWER_RE = re.compile(r"CPU Power:\s+(\d+)\s+mW")


@contextmanager
def _power_monitor(interval_ms: int = 500) -> Generator[list[float], None, None]:
    """Context manager that runs powermetrics in the background and collects samples.

    Uses a reader thread to consume powermetrics stdout in real time — avoids the
    pipe-buffer loss that occurs when reading only after process termination.

    Yields a list that is populated live with (gpu_w + cpu_w) readings. On exit,
    the subprocess is terminated and the reader thread joined.

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
        """Read powermetrics stdout line by line, pairing GPU+CPU per sample block."""
        gpu_w = 0.0
        for line in pipe:
            gm = _GPU_POWER_RE.search(line)
            if gm:
                gpu_w = float(gm.group(1)) / 1000  # mW → W
            cm = _CPU_POWER_RE.search(line)
            if cm:
                cpu_w = float(cm.group(1)) / 1000
                samples.append(gpu_w + cpu_w)

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


def run_benchmark(
    variant_key: str,
    n_samples: int = 50,
    processed_dir: Path | None = None,
    models_dir: Path | None = None,
    measure_power: bool = False,
) -> dict:
    """Benchmark a single model variant.

    Args:
        variant_key: Key from the variant registry (e.g. "smollm2-4bit").
        n_samples: Number of test examples to evaluate on.
        processed_dir: Path to processed data dir. Defaults to data/processed/.
        models_dir: Path to models dir. Defaults to models/.
        measure_power: If True, run powermetrics during benchmark to capture GPU+CPU
            power draw. Requires sudo. Adds power_w and energy_per_token_mwh to result.

    Returns:
        Dict with keys: variant, size_mb, ttft_ms, ttft_pass, tps, ram_mb,
        accuracy_pct, output_tokens_avg, power_w, energy_per_token_mwh.
    """
    if models_dir is None:
        models_dir = get_models_dir()
    if processed_dir is None:
        processed_dir = get_processed_dir()

    variants = _build_variants(models_dir)
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
            predicted = _parse_action(output)
            intent_correct = _is_correct(predicted, ex["ground_truth"])
            slots_correct = predicted is not None and predicted.get("slots") == ex[
                "ground_truth"
            ].get("slots")
            if intent_correct:
                correct += 1
            ttft_list.append(ttft_ms)
            tps_list.append(tps)
            ram_list.append(ram_mb)
            n_tokens_list.append(n_tokens)

            # Extract utterance from prompt ("Command: X\nAction: " → "X")
            utterance = (
                ex["prompt"].replace("Command: ", "").replace("\nAction: ", "").strip()
            )
            predictions.append(
                {
                    "idx": i,
                    "utterance": utterance,
                    "gt_intent": ex["ground_truth"].get("intent"),
                    "gt_slots": ex["ground_truth"].get("slots"),
                    "raw_output": output.strip(),
                    "pred_intent": predicted.get("intent") if predicted else None,
                    "pred_slots": predicted.get("slots") if predicted else None,
                    "intent_correct": intent_correct,
                    "slots_correct": slots_correct,
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

    result = {
        "variant": variant_key,
        "size_mb": round(size_mb),
        "ttft_ms": round(avg_ttft, 1),
        "tps": round(avg_tps, 1),
        "ram_mb": round(avg_ram),
        "accuracy_pct": round(accuracy, 1),
        "output_tokens_avg": round(avg_output_tokens, 1),
        "power_w": power_w,
        "energy_per_token_mwh": energy_per_token_mwh,
    }
    logger.info(
        "Done: %s | size=%.0fMB TTFT=%.1fms TPS=%.1f RAM=%.0fMB "
        "acc=%.1f%% tokens_avg=%.1f",
        variant_key,
        size_mb,
        avg_ttft,
        avg_tps,
        avg_ram,
        accuracy,
        avg_output_tokens,
    )

    _save_predictions(predictions, variant_key)

    # Free memory before loading next model
    del model
    gc.collect()
    mx.clear_cache()

    return result


@contextmanager
def _null_context() -> Generator[list, None, None]:
    """No-op context manager used when measure_power=False."""
    yield []


# ---------------------------------------------------------------------------
# Full benchmark sweep
# ---------------------------------------------------------------------------


def benchmark_all(
    n_samples: int = 50,
    processed_dir: Path | None = None,
    models_dir: Path | None = None,
    measure_power: bool = False,
) -> list[dict]:
    """Benchmark all variants and save results to CSV.

    Args:
        n_samples: Number of test examples per variant.
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

    variants = _build_variants(models_dir)
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
    variants = _build_variants(models_dir)

    parser = argparse.ArgumentParser(
        description="Benchmark fine-tuned and quantized models"
    )
    parser.add_argument(
        "--variant",
        choices=[*variants.keys(), "all"],
        default="all",
        help="Which variant to benchmark (default: all)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of test examples to evaluate on (default: 50)",
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

    if args.variant == "all":
        benchmark_all(n_samples=args.n_samples, measure_power=args.measure_power)
    else:
        result = run_benchmark(
            variant_key=args.variant,
            n_samples=args.n_samples,
            measure_power=args.measure_power,
        )
        print(json.dumps(result, indent=2))
        # Append/update this variant in the CSV so spot-checks persist
        csv_path = get_data_dir() / "results" / "comparison_table.csv"
        _upsert_csv(result, csv_path)
