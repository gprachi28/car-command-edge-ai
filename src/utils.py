"""Shared utilities: logging, config, paths, and constants.

Public API:
    - get_logger(name: str) -> logging.Logger
    - load_config() -> dict[str, str]
    - get_project_root() -> Path
    - get_data_dir() -> Path
    - get_processed_dir() -> Path
    - get_models_dir() -> Path
    - MODEL_IDS: dict[str, str]
    - DATASET_PATH: str
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

    Loads variables from .env (or .env.local) using python-dotenv and returns
    a dict with keys: hf_token, kaggle_username, kaggle_key.

    Empty strings are treated as absent and will raise EnvironmentError.

    Returns:
        Dict with keys: hf_token, kaggle_username, kaggle_key.

    Raises:
        EnvironmentError: If any required env var is missing or empty.
    """
    load_dotenv(get_project_root() / ".env")

    hf_token = os.getenv("HF_TOKEN")
    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")

    missing = []
    if not hf_token:
        missing.append("HF_TOKEN")
    if not kaggle_username:
        missing.append("KAGGLE_USERNAME")
    if not kaggle_key:
        missing.append("KAGGLE_KEY")

    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Please set them in .env or OS environment."
        )

    return {
        "hf_token": hf_token,
        "kaggle_username": kaggle_username,
        "kaggle_key": kaggle_key,
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


# Shared constants

MODEL_IDS: Final[dict[str, str]] = {
    "llama": "meta-llama/Llama-3.2-3B",
    "qwen": "Qwen/Qwen2.5-3B",
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B",
}

DATASET_PATH: Final[str] = "oortdatahub/car-command"

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
