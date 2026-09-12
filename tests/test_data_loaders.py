"""Dataset discovery and the ``tf.data`` pipelines."""

from __future__ import annotations

import pytest

from cvradar.config import DataConfig, RadarConfig
from cvradar.data.labels import GESTURE_CLASSES
from tests.conftest import requires_tensorflow

pytestmark = requires_tensorflow


def test_discover_samples_finds_every_recording(dataset_root):
    from cvradar.data.loaders import discover_samples

    paths, labels = discover_samples(dataset_root / "train")
    assert len(paths) == len(GESTURE_CLASSES) * 2  # two subjects
    assert set(labels) == set(range(len(GESTURE_CLASSES)))


def test_discover_samples_is_deterministic(dataset_root):
    from cvradar.data.loaders import discover_samples

    assert discover_samples(dataset_root / "train") == discover_samples(
        dataset_root / "train"
    )


def test_missing_split_raises_actionable_error(tmp_path):
    from cvradar.data.loaders import discover_samples

    with pytest.raises(FileNotFoundError, match="CVRADAR_DATA_ROOT"):
        discover_samples(tmp_path / "absent")


def test_empty_split_raises(tmp_path):
    from cvradar.data.loaders import discover_samples

    (tmp_path / "train").mkdir()
    with pytest.raises(FileNotFoundError, match="no samples matching"):
        discover_samples(tmp_path / "train")


def test_incomplete_noise_split_keeps_canonical_indices(dataset_root):
    """The noise split holds only 3 gestures; indices must not be renumbered."""
    from cvradar.data.loaders import discover_samples

    _, labels = discover_samples(dataset_root / "noise" / "AWGN_SNR_-5")
    expected = [GESTURE_CLASSES.index(name) for name in GESTURE_CLASSES[:3]]
    assert sorted(set(labels)) == sorted(expected)


def test_pipeline_yields_static_shapes(dataset_root):
    from cvradar.data.loaders import load_split

    radar = RadarConfig(frames=2, chirps=4, samples=4, rx_antennas=4)
    config = DataConfig(root=dataset_root, batch_size=4)

    dataset = load_split("valid", config, radar)
    features, labels = next(iter(dataset))

    assert features.shape[1:] == radar.input_shape
    assert features.shape[0] == 4
    assert labels.shape[0] == 4


def test_shape_mismatch_is_reported(dataset_root):
    from cvradar.data.loaders import load_split

    # Declare the wrong geometry; decoding must fail loudly, not silently.
    config = DataConfig(root=dataset_root, batch_size=1)
    dataset = load_split("valid", config, RadarConfig(frames=99))
    with pytest.raises(Exception, match="expected shape"):
        next(iter(dataset))
