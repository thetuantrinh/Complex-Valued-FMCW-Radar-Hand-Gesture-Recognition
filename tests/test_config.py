"""Configuration loading, validation and override behaviour."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cvradar.config import DATA_ROOT_ENV, DataConfig, ExperimentConfig, RadarConfig


def test_default_input_shape_matches_publication():
    assert RadarConfig().input_shape == (20, 128, 64, 8)
    assert RadarConfig().channels == 8


def test_from_mapping_builds_nested_sections():
    config = ExperimentConfig.from_mapping(
        {"model": {"num_classes": 7}, "training": {"epochs": 3}}
    )
    assert config.model.num_classes == 7
    assert config.training.epochs == 3
    assert config.radar.frames == 20  # untouched defaults survive


def test_from_mapping_converts_lists_to_tuples():
    config = ExperimentConfig.from_mapping({"model": {"block_filters": [4, 8, 16]}})
    assert config.model.block_filters == (4, 8, 16)


def test_from_mapping_coerces_path_fields():
    config = ExperimentConfig.from_mapping({"data": {"root": "/tmp/radar"}})
    assert isinstance(config.data.root, Path)


@pytest.mark.parametrize(
    "payload",
    [
        {"model": {"no_such_key": 1}},
        {"no_such_section": {}},
    ],
)
def test_unknown_keys_are_rejected(payload):
    with pytest.raises(ValueError, match="unknown configuration key"):
        ExperimentConfig.from_mapping(payload)


def test_data_root_falls_back_to_environment(monkeypatch):
    monkeypatch.setenv(DATA_ROOT_ENV, "/mnt/radar")
    assert DataConfig().resolved_root() == Path("/mnt/radar")


def test_explicit_root_beats_environment(monkeypatch):
    monkeypatch.setenv(DATA_ROOT_ENV, "/mnt/radar")
    assert DataConfig(root=Path("/explicit")).resolved_root() == Path("/explicit")


def test_data_root_defaults_to_local_data_dir(monkeypatch):
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    assert DataConfig().resolved_root() == Path("data")


def test_split_dir_rejects_unknown_split():
    with pytest.raises(ValueError, match="unknown split"):
        DataConfig().split_dir("holdout")


def test_config_is_frozen_but_replaceable():
    config = ExperimentConfig()
    updated = replace(config, data=replace(config.data, batch_size=8))
    assert updated.data.batch_size == 8
    assert config.data.batch_size == 32  # original untouched


def test_to_dict_is_serialisable():
    import json

    payload = ExperimentConfig().to_dict()
    json.dumps(payload)  # must not raise
    assert payload["training"]["checkpoint_dir"] == "checkpoints"


def test_yaml_roundtrip(tmp_path):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"model": {"num_classes": 4}}))
    assert ExperimentConfig.from_yaml(path).model.num_classes == 4


def test_shipped_configs_load():
    yaml = pytest.importorskip("yaml")
    del yaml
    for name in ("default.yaml", "deep_ensemble.yaml"):
        path = Path(__file__).resolve().parents[1] / "configs" / name
        if path.exists():
            ExperimentConfig.from_yaml(path)
