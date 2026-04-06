"""Shared utilities: logging, config, paths, and constants.

Public API:
    - get_logger(name: str) -> logging.Logger
    - load_config() -> dict[str, str]
    - get_project_root() -> Path
    - get_data_dir() -> Path
    - get_processed_dir() -> Path
    - get_models_dir() -> Path
    - MODEL_IDS: dict[str, str]
    - LORA_CONFIG: dict[str, float | int | str | list[str]]
    - TRAINING_CONFIG: dict[str, float | int]
"""

import logging
import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv


def get_logger(name: str) -> logging.Logger:
    """Configure and return a logger with consistent formatting.

    Args:
        name: Logger name (usually __name__).

    Returns:
        Configured logger with StreamHandler and consistent format.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if logger already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger


def load_config() -> dict[str, str]:
    """Load environment variables from .env and return config dict.

    Returns:
        Dict with keys: hf_token (str), gemini_api_key (str, may be empty).

    Raises:
        EnvironmentError: If HF_TOKEN is missing or empty.
    """
    load_dotenv(get_project_root() / ".env")

    hf_token = os.getenv("HF_TOKEN")
    gemini_api_key = os.getenv(
        "GEMINI_API_KEY"
    )  # optional — only needed for --backend gemini

    if not hf_token:
        raise EnvironmentError(
            "Missing required environment variable: HF_TOKEN. "
            "Please set it in .env or OS environment."
        )

    return {
        "hf_token": hf_token,
        "gemini_api_key": gemini_api_key or "",
    }


def get_project_root() -> Path:
    """Return the project root directory (parent of src/).

    Returns:
        Path object pointing to the project root.
    """
    return Path(__file__).parent.parent


def get_data_dir() -> Path:
    """Return the data directory (project_root/data/).

    Returns:
        Path object pointing to data directory.
    """
    return get_project_root() / "data"


def get_processed_dir() -> Path:
    """Return the processed data directory (project_root/data/processed/).

    Returns:
        Path object pointing to processed data directory.
    """
    return get_data_dir() / "processed"


def get_models_dir() -> Path:
    """Return the models directory (project_root/models/).

    Returns:
        Path object pointing to models directory.
    """
    return get_project_root() / "models"


def dir_size_mb(path: Path) -> float:
    """Return total size of all files in a directory in MB.

    Args:
        path: Directory to measure.

    Returns:
        Total file size in megabytes.
    """
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024**2)


# Shared constants

MODEL_IDS: Final[dict[str, str]] = {
    "llama": "meta-llama/Llama-3.2-3B",
    "qwen": "Qwen/Qwen2.5-3B",
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B",
}

LORA_CONFIG: Final[dict[str, float | int | str | list[str]]] = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "bias": "none",
    "target_modules": ["q_proj", "v_proj"],
}

TRAINING_CONFIG: Final[dict[str, float | int]] = {
    "learning_rate": 2e-4,
    "batch_size": 8,
    "num_epochs": 3,
    "max_seq_length": 256,
    "warmup_steps": 100,
    "weight_decay": 0.01,
}
