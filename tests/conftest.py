"""Shared pytest fixtures and TensorFlow availability gating."""

from __future__ import annotations

import importlib.util

import pytest


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


TENSORFLOW_AVAILABLE = _module_available("tensorflow")
TFP_AVAILABLE = _module_available("tensorflow_probability")
COMPLEXNN_AVAILABLE = _module_available("complexnn")

requires_tensorflow = pytest.mark.skipif(
    not TENSORFLOW_AVAILABLE, reason="requires tensorflow"
)
requires_tfp = pytest.mark.skipif(
    not (TENSORFLOW_AVAILABLE and TFP_AVAILABLE),
    reason="requires tensorflow and tensorflow-probability",
)
requires_complexnn = pytest.mark.skipif(
    not (TENSORFLOW_AVAILABLE and COMPLEXNN_AVAILABLE),
    reason="requires tensorflow and keras-complex",
)


@pytest.fixture
def dataset_root(tmp_path):
    """Create a miniature dataset tree mirroring the real layout.

    Layout: ``<root>/<split>/<subject>/<gesture>/<sample>.npy`` with tiny
    tensors, so the loader logic can be exercised without the real recordings.
    """
    import numpy as np

    from cvradar.data.labels import GESTURE_CLASSES

    shape = (2, 4, 4, 8)
    for split in ("train", "valid"):
        for subject in ("Person_1", "Person_2"):
            for gesture in GESTURE_CLASSES:
                directory = tmp_path / split / subject / gesture
                directory.mkdir(parents=True, exist_ok=True)
                np.save(directory / "data_1.npy", np.zeros(shape, dtype=np.float32))

    noise_dir = tmp_path / "noise" / "AWGN_SNR_-5"
    for subject in ("Person_9",):
        for gesture in GESTURE_CLASSES[:3]:  # deliberately incomplete
            directory = noise_dir / subject / gesture
            directory.mkdir(parents=True, exist_ok=True)
            np.save(directory / "data_1.npy", np.zeros(shape, dtype=np.float32))

    return tmp_path
