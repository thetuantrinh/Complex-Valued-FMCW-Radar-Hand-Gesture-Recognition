"""Epistemic uncertainty estimation: Monte Carlo Dropout and Deep Ensembles.

Both estimators approximate the predictive distribution by averaging several
stochastic or independent forward passes:

* **Monte Carlo Dropout (MC-D)** keeps the dropout head active at inference and
  draws ``num_samples`` passes from a single trained network -- cheap, and the
  variant that meets the real-time budget on the Jetson Nano.
* **Deep Ensemble Learning (DEL)** averages independently trained networks,
  which captures a broader hypothesis space at a proportionally higher cost.

Averaging is done over **probabilities**, not logits: the mean of softmaxes is
the Monte Carlo estimate of the predictive distribution, whereas averaging
logits would silently change the quantity being estimated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tensorflow as tf

__all__ = [
    "deep_ensemble_predict",
    "iterate_predictions",
    "load_ensemble",
    "mc_dropout_predict",
    "mutual_information",
    "predictive_entropy",
]

_EPSILON = 1e-12


def mc_dropout_predict(
    model: tf.keras.Model,
    inputs: tf.Tensor,
    num_samples: int = 50,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Estimate the predictive distribution by Monte Carlo Dropout.

    Args:
        model: A network whose dropout head stays stochastic at inference.
        inputs: A batch of raw radar tensors.
        num_samples: Stochastic forward passes to average.

    Returns:
        ``(mean_probabilities, per_sample_probabilities)`` with shapes
        ``(batch, num_classes)`` and ``(num_samples, batch, num_classes)``.

    Raises:
        ValueError: if ``num_samples`` is not positive.
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")

    samples = tf.stack(
        [tf.nn.softmax(model(inputs, training=True), axis=-1) for _ in range(num_samples)]
    )
    return tf.reduce_mean(samples, axis=0), samples


def deep_ensemble_predict(
    models: Sequence[tf.keras.Model],
    inputs: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Estimate the predictive distribution by averaging ensemble members.

    Args:
        models: Independently trained networks.
        inputs: A batch of raw radar tensors.

    Returns:
        ``(mean_probabilities, per_member_probabilities)`` with shapes
        ``(batch, num_classes)`` and ``(num_members, batch, num_classes)``.

    Raises:
        ValueError: if ``models`` is empty.
    """
    if not models:
        raise ValueError("deep_ensemble_predict requires at least one model")

    members = tf.stack(
        [tf.nn.softmax(model(inputs, training=False), axis=-1) for model in models]
    )
    return tf.reduce_mean(members, axis=0), members


def load_ensemble(ensemble_dir: str | Path) -> list[tf.keras.Model]:
    """Load every SavedModel directory found directly under ``ensemble_dir``.

    Args:
        ensemble_dir: Directory holding one subdirectory per ensemble member.

    Returns:
        The members, ordered by directory name for reproducibility.

    Raises:
        FileNotFoundError: if the directory is missing or holds no SavedModel.
    """
    root = Path(ensemble_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"ensemble directory not found: {root}")

    member_dirs = sorted(
        path for path in root.iterdir() if (path / "saved_model.pb").is_file()
    )
    if not member_dirs:
        raise FileNotFoundError(
            f"no SavedModel members under {root}; "
            "expected subdirectories each containing saved_model.pb"
        )
    return [tf.keras.models.load_model(str(path)) for path in member_dirs]


def predictive_entropy(probabilities: tf.Tensor) -> tf.Tensor:
    """Total predictive uncertainty: the entropy of the mean distribution.

    Args:
        probabilities: Mean probabilities of shape ``(batch, num_classes)``.

    Returns:
        Per-sample entropy in nats, shape ``(batch,)``.
    """
    probabilities = tf.convert_to_tensor(probabilities)
    return -tf.reduce_sum(probabilities * tf.math.log(probabilities + _EPSILON), axis=-1)


def mutual_information(samples: tf.Tensor) -> tf.Tensor:
    """Epistemic uncertainty: total entropy minus mean per-sample entropy.

    This isolates *model* uncertainty from the irreducible data noise, and is
    the quantity that rises on out-of-distribution or heavily corrupted input.

    Args:
        samples: Per-pass probabilities of shape
            ``(num_samples, batch, num_classes)``.

    Returns:
        Per-sample mutual information in nats, shape ``(batch,)``.
    """
    samples = tf.convert_to_tensor(samples)
    mean_probabilities = tf.reduce_mean(samples, axis=0)
    total = predictive_entropy(mean_probabilities)
    aleatoric = tf.reduce_mean(
        -tf.reduce_sum(samples * tf.math.log(samples + _EPSILON), axis=-1), axis=0
    )
    return total - aleatoric


def iterate_predictions(
    dataset: Iterable[tuple[tf.Tensor, tf.Tensor]],
    predict_fn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ``predict_fn`` over a dataset and concatenate the results.

    Args:
        dataset: Iterable of ``(features, labels)`` batches.
        predict_fn: Callable mapping a feature batch to
            ``(mean_probabilities, per_sample_probabilities)``.

    Returns:
        ``(probabilities, labels, epistemic)`` as NumPy arrays over the whole
        dataset, where ``epistemic`` is the per-sample mutual information.
    """
    all_probabilities, all_labels, all_epistemic = [], [], []
    for features, labels in dataset:
        mean_probabilities, samples = predict_fn(features)
        all_probabilities.append(np.asarray(mean_probabilities))
        all_labels.append(np.asarray(labels))
        all_epistemic.append(np.asarray(mutual_information(samples)))

    return (
        np.concatenate(all_probabilities, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_epistemic, axis=0),
    )
