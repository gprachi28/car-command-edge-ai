"""Tests for src/finetune.py — mock-based, no real model downloads.

All HuggingFace model/tokenizer calls are patched; tests verify the
orchestration logic (LoRA attachment, training invocation, merge + save)
without touching the network or GPU.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.finetune import (
    build_lora_model,
    finetune_all,
    load_base_model,
    load_train_dataset,
    run_finetune,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_train_jsonl(path: Path, n: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i in range(n):
            f.write(json.dumps({"text": f"Command: cmd {i}\nAction: intent_{i}"}) + "\n")


# ---------------------------------------------------------------------------
# load_base_model
# ---------------------------------------------------------------------------


def test_load_base_model_unknown_key() -> None:
    """Raises KeyError for unrecognised model keys."""
    with pytest.raises(KeyError, match="bad_key"):
        load_base_model("bad_key", hf_token="tok")


@patch("src.finetune.AutoModelForCausalLM.from_pretrained")
@patch("src.finetune.AutoTokenizer.from_pretrained")
def test_load_base_model_returns_model_and_tokenizer(mock_tok, mock_model) -> None:
    """load_base_model returns (model, tokenizer) for a valid key."""
    mock_tok.return_value = MagicMock(pad_token="<pad>")
    mock_model.return_value = MagicMock()

    model, tokenizer = load_base_model("smollm2", hf_token="fake-token")

    assert model is mock_model.return_value
    assert tokenizer is mock_tok.return_value


@patch("src.finetune.AutoModelForCausalLM.from_pretrained")
@patch("src.finetune.AutoTokenizer.from_pretrained")
def test_load_base_model_sets_pad_token_when_missing(mock_tok, mock_model) -> None:
    """pad_token is set to eos_token when absent."""
    fake_tok = MagicMock()
    fake_tok.pad_token = None
    fake_tok.eos_token = "<eos>"
    mock_tok.return_value = fake_tok
    mock_model.return_value = MagicMock()

    _, tokenizer = load_base_model("smollm2", hf_token="tok")

    assert tokenizer.pad_token == "<eos>"


# ---------------------------------------------------------------------------
# build_lora_model
# ---------------------------------------------------------------------------


@patch("src.finetune.get_peft_model")
def test_build_lora_model_calls_get_peft_model(mock_get_peft) -> None:
    """build_lora_model wraps the model with LoRA config."""
    fake_model = MagicMock()
    mock_get_peft.return_value = MagicMock()

    result = build_lora_model(fake_model)

    mock_get_peft.assert_called_once()
    assert result is mock_get_peft.return_value


@patch("src.finetune.get_peft_model")
def test_build_lora_model_uses_correct_config(mock_get_peft) -> None:
    """LoRA config has the spec-defined rank, alpha, and target modules."""
    from peft import LoraConfig

    mock_get_peft.return_value = MagicMock()
    build_lora_model(MagicMock())

    lora_cfg: LoraConfig = mock_get_peft.call_args[0][1]
    assert lora_cfg.r == 8
    assert lora_cfg.lora_alpha == 16
    assert set(lora_cfg.target_modules) == {"q_proj", "v_proj"}


# ---------------------------------------------------------------------------
# load_train_dataset
# ---------------------------------------------------------------------------


def test_load_train_dataset_raises_when_missing(tmp_path: Path) -> None:
    """FileNotFoundError if train.jsonl does not exist."""
    with pytest.raises(FileNotFoundError, match="train.jsonl"):
        load_train_dataset(tmp_path)


@patch("src.finetune.load_dataset")
def test_load_train_dataset_passes_correct_path(mock_load, tmp_path: Path) -> None:
    """load_dataset is called with the correct jsonl path."""
    _write_train_jsonl(tmp_path / "train.jsonl")
    mock_load.return_value = MagicMock(__len__=lambda self: 5)

    load_train_dataset(tmp_path)

    mock_load.assert_called_once_with(
        "json", data_files=str(tmp_path / "train.jsonl"), split="train"
    )


# ---------------------------------------------------------------------------
# run_finetune
# ---------------------------------------------------------------------------


@patch("src.finetune.SFTTrainer")
@patch("src.finetune.SFTConfig")
@patch("src.finetune.build_lora_model")
@patch("src.finetune.load_train_dataset")
@patch("src.finetune.load_base_model")
def test_run_finetune_saves_merged_model(
    mock_load_base,
    mock_load_data,
    mock_build_lora,
    mock_sft_config,
    mock_sft_trainer,
    tmp_path: Path,
) -> None:
    """run_finetune merges LoRA and saves to the expected path."""
    fake_model = MagicMock()
    fake_tokenizer = MagicMock()
    mock_load_base.return_value = (fake_model, fake_tokenizer)
    mock_load_data.return_value = MagicMock()

    lora_model = MagicMock()
    merged_model = MagicMock()
    lora_model.merge_and_unload.return_value = merged_model
    mock_build_lora.return_value = lora_model

    mock_trainer_instance = MagicMock()
    mock_sft_trainer.return_value = mock_trainer_instance

    output_dir = tmp_path / "finetuned"
    result = run_finetune("smollm2", hf_token="tok", output_dir=output_dir)

    expected_save_path = output_dir / "smollm2-finetuned"

    mock_trainer_instance.train.assert_called_once()
    lora_model.merge_and_unload.assert_called_once()
    merged_model.save_pretrained.assert_called_once_with(expected_save_path)
    fake_tokenizer.save_pretrained.assert_called_once_with(expected_save_path)

    assert result == expected_save_path


@patch("src.finetune.SFTTrainer")
@patch("src.finetune.SFTConfig")
@patch("src.finetune.build_lora_model")
@patch("src.finetune.load_train_dataset")
@patch("src.finetune.load_base_model")
def test_run_finetune_sft_config_hyperparameters(
    mock_load_base,
    mock_load_data,
    mock_build_lora,
    mock_sft_config,
    mock_sft_trainer,
    tmp_path: Path,
) -> None:
    """SFTConfig receives the spec-defined hyperparameters."""
    mock_load_base.return_value = (MagicMock(), MagicMock())
    mock_load_data.return_value = MagicMock()
    lora_model = MagicMock()
    lora_model.merge_and_unload.return_value = MagicMock()
    mock_build_lora.return_value = lora_model
    mock_sft_trainer.return_value = MagicMock()

    run_finetune("smollm2", hf_token="tok", output_dir=tmp_path / "ft")

    kwargs = mock_sft_config.call_args.kwargs
    assert kwargs["num_train_epochs"] == 3
    assert kwargs["learning_rate"] == 2e-4
    assert kwargs["max_seq_length"] == 256
    assert kwargs["dataset_text_field"] == "text"


@patch("src.finetune.SFTTrainer")
@patch("src.finetune.SFTConfig")
@patch("src.finetune.build_lora_model")
@patch("src.finetune.load_train_dataset")
@patch("src.finetune.load_base_model")
def test_run_finetune_unknown_key_raises(
    mock_load_base, mock_load_data, mock_build_lora, mock_sft_config, mock_sft_trainer
) -> None:
    """run_finetune propagates KeyError for bad model keys."""
    mock_load_base.side_effect = KeyError("bad_key")

    with pytest.raises(KeyError):
        run_finetune("bad_key", hf_token="tok")


# ---------------------------------------------------------------------------
# finetune_all
# ---------------------------------------------------------------------------


@patch("src.finetune.run_finetune")
def test_finetune_all_runs_all_three_models(mock_run, tmp_path: Path) -> None:
    """finetune_all calls run_finetune for each of the three model keys."""
    mock_run.side_effect = lambda key, **kwargs: tmp_path / f"{key}-finetuned"

    results = finetune_all(hf_token="tok", output_dir=tmp_path)

    assert set(results.keys()) == {"llama", "qwen", "smollm2"}
    assert mock_run.call_count == 3


@patch("src.finetune.run_finetune")
def test_finetune_all_returns_correct_paths(mock_run, tmp_path: Path) -> None:
    """finetune_all returns a dict mapping each key to its checkpoint path."""
    mock_run.side_effect = lambda key, **kwargs: tmp_path / f"{key}-finetuned"

    results = finetune_all(hf_token="tok", output_dir=tmp_path)

    for key in ("llama", "qwen", "smollm2"):
        assert results[key] == tmp_path / f"{key}-finetuned"
