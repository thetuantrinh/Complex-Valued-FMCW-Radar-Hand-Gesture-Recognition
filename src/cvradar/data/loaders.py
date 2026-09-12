"""``tf.data`` input pipelines over raw FMCW ``.npy`` recordings.

Dataset layout::

    <root>/<split>/<subject>/<gesture>/<sample>.npy

Every sample is a real-valued array of shape ``(frames, chirps, samples,
channels)``; the trailing axis interleaves the I and Q components of each
receive antenna. No FFT is applied at any point -- the network consumes the
time-domain tensor directly.

Nothing in this module runs at import time. The original implementation globbed
a hardcoded cluster path (``/work/...``) while the module was being imported,
which made it unusable on any other machine and impossible to unit test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf

from cvradar.config import DataConfig, RadarConfig
from cvradar.data.labels import GESTURE_CLASSES, encode_labels, validate_classes

__all__ = [
    "build_dataset",
    "discover_samples",
    "load_noise_split",
    "load_split",
    "load_train_valid",
]

AUTOTUNE = tf.data.AUTOTUNE


def discover_samples(
    split_dir: str | Path,
    sample_glob: str = "*/*/*.npy",
    classes: Sequence[str] = GESTURE_CLASSES,
) -> tuple[list[str], list[int]]:
    """Enumerate the recordings of one split and encode their labels.

    The gesture name is taken from each file's parent directory and mapped
    through the canonical vocabulary in :mod:`cvradar.data.labels`, so every
    split shares one label space.

    Args:
        split_dir: Directory of the split to scan.
        sample_glob: Glob relative to ``split_dir``.
        classes: Gesture vocabulary defining the class indices.

    Returns:
        A ``(file_paths, labels)`` pair, sorted by path for reproducibility.

    Raises:
        FileNotFoundError: if the directory is missing or contains no samples.
        ValueError: if a gesture directory is outside ``classes``.
    """
    root = Path(split_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(
            f"dataset split not found: {root}\n"
            "Set the dataset location via the CVRADAR_DATA_ROOT environment "
            "variable or the data.root key of your config file."
        )

    file_paths = sorted(str(path) for path in root.glob(sample_glob))
    if not file_paths:
        raise FileNotFoundError(
            f"no samples matching {sample_glob!r} under {root}; "
            "expected <subject>/<gesture>/<sample>.npy"
        )

    label_names = [Path(path).parent.name for path in file_paths]
    validate_classes(sorted(set(label_names)), classes)
    return file_paths, encode_labels(label_names, classes)


def build_dataset(
    file_paths: Sequence[str],
    labels: Sequence[int],
    radar: RadarConfig,
    batch_size: int = 32,
    shuffle: bool = False,
    shuffle_buffer: int | None = None,
    seed: int = 1337,
    reshuffle_each_iteration: bool = True,
    drop_remainder: bool = False,
) -> tf.data.Dataset:
    """Assemble a batched, prefetched pipeline over ``(path, label)`` pairs.

    Shuffling happens over the *path* dataset before decoding, so the shuffle
    buffer holds filenames rather than multi-megabyte radar tensors.

    Args:
        file_paths: Sample paths, as returned by :func:`discover_samples`.
        labels: Integer class index per path.
        radar: Acquisition geometry, used to restore static tensor shapes.
        batch_size: Samples per batch.
        shuffle: Whether to shuffle the split.
        shuffle_buffer: Buffer size; ``None`` shuffles over the full split.
        seed: Shuffle seed.
        reshuffle_each_iteration: Reshuffle between epochs. Keep ``False`` for
            evaluation splits so that predictions align across runs.
        drop_remainder: Drop a trailing partial batch.

    Returns:
        A ``tf.data.Dataset`` yielding ``(features, label)`` batches with fully
        defined static shapes.
    """
    if len(file_paths) != len(labels):
        raise ValueError(
            f"file_paths and labels differ in length: {len(file_paths)} vs {len(labels)}"
        )

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.constant(file_paths, dtype=tf.string), tf.constant(labels, dtype=tf.int32))
    )
    if shuffle:
        dataset = dataset.shuffle(
            shuffle_buffer or len(file_paths),
            seed=seed,
            reshuffle_each_iteration=reshuffle_each_iteration,
        )
    return (
        dataset.map(_make_loader(radar), num_parallel_calls=AUTOTUNE)
        .batch(batch_size, drop_remainder=drop_remainder, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )


def load_split(
    split: str,
    config: DataConfig | None = None,
    radar: RadarConfig | None = None,
    classes: Sequence[str] = GESTURE_CLASSES,
    shuffle: bool | None = None,
) -> tf.data.Dataset:
    """Build the pipeline for a named split (``train``, ``valid`` or ``noise``).

    Training shuffles and reshuffles each epoch; evaluation splits keep their
    deterministic path order.
    """
    config = config or DataConfig()
    radar = radar or RadarConfig()
    is_train = split == "train"

    file_paths, labels = discover_samples(
        config.split_dir(split), config.sample_glob, classes
    )
    return build_dataset(
        file_paths,
        labels,
        radar=radar,
        batch_size=config.batch_size,
        shuffle=is_train if shuffle is None else shuffle,
        shuffle_buffer=config.shuffle_buffer,
        seed=config.seed,
        reshuffle_each_iteration=is_train,
    )


def load_train_valid(
    config: DataConfig | None = None,
    radar: RadarConfig | None = None,
    classes: Sequence[str] = GESTURE_CLASSES,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Return the ``(train, valid)`` pipelines used for training."""
    config = config or DataConfig()
    radar = radar or RadarConfig()
    return (
        load_split("train", config, radar, classes),
        load_split("valid", config, radar, classes),
    )


def load_noise_split(
    noise_type: str = "AWGN_SNR_-5",
    config: DataConfig | None = None,
    radar: RadarConfig | None = None,
    classes: Sequence[str] = GESTURE_CLASSES,
) -> tf.data.Dataset:
    """Return the pipeline for one noise-corrupted validation condition.

    Args:
        noise_type: Subdirectory of the noise split, e.g. ``"AWGN_SNR_-5"``.
        config: Data configuration; defaults to :class:`DataConfig`.
        radar: Acquisition geometry.
        classes: Gesture vocabulary. Shared with the clean splits, so a noise
            condition missing a gesture cannot shift the class indices.
    """
    config = config or DataConfig()
    radar = radar or RadarConfig()

    file_paths, labels = discover_samples(
        config.split_dir("noise") / noise_type, config.sample_glob, classes
    )
    return build_dataset(
        file_paths,
        labels,
        radar=radar,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
        reshuffle_each_iteration=False,
    )


def _make_loader(radar: RadarConfig):
    """Return a ``tf.data`` map function that decodes one ``.npy`` file.

    ``tf.numpy_function`` erases static shape information, so the loaded tensor
    is re-annotated from the radar geometry. Without this the downstream model
    cannot infer its input signature and graph-mode tracing fails.
    """
    expected_shape = radar.input_shape

    def _read_npy(path: bytes) -> np.ndarray:
        array = np.load(path.decode("utf-8"))
        if array.shape != expected_shape:
            raise ValueError(
                f"{path.decode('utf-8')}: expected shape {expected_shape}, "
                f"got {array.shape}"
            )
        return array.astype(np.float32)

    def _load(path: tf.Tensor, label: tf.Tensor):
        features = tf.numpy_function(_read_npy, [path], tf.float32, name="read_npy")
        features.set_shape(expected_shape)
        return features, label

    return _load
