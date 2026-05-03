"""Interactive CLI demo for car command inference.

Loads a fine-tuned or quantized MLX model, then runs an interactive loop
that maps natural-language car commands to structured intent+slot JSON.

Usage:
    python src/demo_cli.py --model smollm2-4bit

Exit with Ctrl+C, an empty input line, or by typing 'exit' / 'quit'.

Public API:
    - main() -> None
"""

import argparse
import json
import time

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from src.utils import (
    build_variants,
    dir_size_mb,
    filter_slots,
    get_logger,
    get_models_dir,
    get_project_root,
    parse_action,
)

logger = get_logger(__name__)

MAX_TOKENS = 80  # car commands average 21-27 output tokens; 80 is a safe ceiling


def _infer(model, tokenizer, utterance: str) -> tuple[dict | None, float, str]:
    """Run one inference pass and return the parsed action, TTFT, and raw output.

    Prompt format matches training: "Command: <utterance>\nAction: "

    Args:
        model: Loaded MLX model.
        tokenizer: Loaded tokenizer.
        utterance: User-supplied car command string.

    Returns:
        Tuple of (parsed_action dict or None, TTFT in ms, raw generated text).
    """
    prompt = f"Command: {utterance}\nAction: "
    ttft_ms: float | None = None
    output_tokens: list[str] = []

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
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - t_start) * 1000
        output_tokens.append(response.text)
        if response.finish_reason is not None:
            break

    raw = "".join(output_tokens)
    return parse_action(raw), ttft_ms or 0.0, raw


def main() -> None:
    """Entry point for the interactive CLI demo."""
    models_dir = get_models_dir()
    variants = build_variants(models_dir)

    parser = argparse.ArgumentParser(
        description=(
            "Car Command Assistant — maps natural language to structured intent+slots"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available variants:\n" + "\n".join(f"  {k}" for k in variants),
    )
    parser.add_argument(
        "--model",
        choices=list(variants),
        default="smollm2-4bit",
        help="Model variant to load (default: smollm2-4bit)",
    )
    args = parser.parse_args()

    model_path = variants[args.model]
    if not model_path.exists():
        print(f"Error: model not found at {model_path}")
        print("Run python src/quantize.py first, or check the models/ directory.")
        return

    size_mb = dir_size_mb(model_path)
    print("\nCar Command Assistant")
    print(f"Model : {args.model}")
    print(f"Size  : {size_mb:.0f} MB")
    print(f"Path  : {model_path.relative_to(get_project_root())}")
    print("\nLoading model...")

    model, tokenizer = load(str(model_path))

    print("Ready. Type a car command. Type 'exit' or press Ctrl+C to quit.\n")

    while True:
        try:
            utterance = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not utterance or utterance.lower() in {"exit", "quit", "/quit", "q"}:
            print("Goodbye.")
            break

        parsed, ttft_ms, raw = _infer(model, tokenizer, utterance)

        if parsed is None:
            print(f"  [parse error] Raw output: {raw!r}")
        else:
            intent = parsed.get("intent", "unknown")
            slots = filter_slots(intent, parsed.get("slots") or {})
            print(
                f"  Intent: {intent} | Slots: {json.dumps(slots)}  "
                f"[TTFT: {ttft_ms:.0f} ms]"
            )


if __name__ == "__main__":
    main()
