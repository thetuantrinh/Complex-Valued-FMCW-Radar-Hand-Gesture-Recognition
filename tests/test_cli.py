"""Argument parsing and config-override resolution for the CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from cvradar.cli import _resolve_config, build_parser


def parse(argv):
    return build_parser().parse_args(argv)


def test_train_requires_no_arguments():
    args = parse(["train"])
    assert args.command == "train"


def test_missing_subcommand_exits():
    with pytest.raises(SystemExit):
        parse([])


def test_data_root_override_reaches_config():
    config = _resolve_config(parse(["train", "--data-root", "/mnt/radar"]))
    assert config.data.resolved_root() == Path("/mnt/radar")


def test_training_overrides_reach_config():
    config = _resolve_config(
        parse(["train", "--epochs", "7", "--learning-rate", "0.01", "--batch-size", "4"])
    )
    assert config.training.epochs == 7
    assert config.training.learning_rate == pytest.approx(0.01)
    assert config.data.batch_size == 4


def test_uncertainty_overrides_reach_config():
    config = _resolve_config(
        parse(["evaluate", "--method", "deep_ensemble", "--mc-samples", "5"])
    )
    assert config.uncertainty.method == "deep_ensemble"
    assert config.uncertainty.num_samples == 5


def test_unset_options_leave_defaults_untouched():
    config = _resolve_config(parse(["evaluate"]))
    assert config.uncertainty.method == "mc_dropout"
    assert config.training.epochs == 100


def test_invalid_method_is_rejected():
    with pytest.raises(SystemExit):
        parse(["evaluate", "--method", "bayes_by_backprop"])


def test_cli_overrides_beat_config_file(tmp_path):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"training": {"epochs": 1}}))
    config = _resolve_config(parse(["train", "--config", str(path), "--epochs", "9"]))
    assert config.training.epochs == 9
