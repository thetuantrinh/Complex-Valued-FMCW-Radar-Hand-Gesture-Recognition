"""Evaluation driver: accuracy, calibration and epistemic uncertainty.

Evaluates a trained (2+1)D CVNet on a clean or noise-corrupted split using
either Monte Carlo Dropout or a Deep Ensemble, and reports the metrics from the
paper: top-1 accuracy, Expected Calibration Error and negative log-likelihood,
alongside the mean epistemic uncertainty.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

from cvradar.config import ExperimentConfig
from cvradar.data.loaders import load_noise_split, load_split
from cvradar.evaluation.metrics import UncertaintyMetrics
from cvradar.evaluation.uncertainty import (
    deep_ensemble_predict,
    iterate_predictions,
    load_ensemble,
    mc_dropout_predict,
)

__all__ = ["evaluate"]

LOGGER = logging.getLogger(__name__)


def evaluate(
    config: ExperimentConfig | None = None,
    model_path: str | Path | None = None,
    split: str = "valid",
    noise_type: str | None = None,
) -> dict[str, float]:
    """Evaluate a trained model and return its metric summary.

    Args:
        config: Full experiment configuration.
        model_path: SavedModel directory. Required for ``mc_dropout``; ignored
            for ``deep_ensemble``, which loads every member of
            :attr:`~cvradar.config.UncertaintyConfig.ensemble_dir`.
        split: ``"valid"`` for the clean split, ``"noise"`` for a corrupted one.
        noise_type: Noise condition, e.g. ``"AWGN_SNR_-5"``. Required when
            ``split`` is ``"noise"``.

    Returns:
        ``{"accuracy", "ece", "nll", "epistemic", "num_samples"}``.

    Raises:
        ValueError: on an unknown uncertainty method, a missing ``model_path``
            for MC Dropout, or a missing ``noise_type`` for the noise split.
    """
    config = config or ExperimentConfig()
    method = config.uncertainty.method

    if split == "noise":
        if noise_type is None:
            raise ValueError("split='noise' requires a noise_type, e.g. 'AWGN_SNR_-5'")
        dataset = load_noise_split(noise_type, config.data, config.radar)
    else:
        dataset = load_split(split, config.data, config.radar)

    if method == "mc_dropout":
        if model_path is None:
            raise ValueError("uncertainty.method='mc_dropout' requires a model_path")
        LOGGER.info("Loading MC Dropout model from %s", model_path)
        model = tf.keras.models.load_model(str(model_path))
        num_passes = config.uncertainty.num_samples

        def predict_fn(features):
            return mc_dropout_predict(model, features, num_passes)

    elif method == "deep_ensemble":
        LOGGER.info("Loading ensemble from %s", config.uncertainty.ensemble_dir)
        members = load_ensemble(config.uncertainty.ensemble_dir)
        LOGGER.info("Loaded %d ensemble members", len(members))

        def predict_fn(features):
            return deep_ensemble_predict(members, features)

    else:
        raise ValueError(
            f"unknown uncertainty method {method!r}; "
            "expected 'mc_dropout' or 'deep_ensemble'"
        )

    probabilities, labels, epistemic = iterate_predictions(dataset, predict_fn)

    # The metric helpers consume logits; log-probabilities are an equivalent
    # parameterisation of the same softmax distribution.
    metrics = UncertaintyMetrics(np.log(probabilities + 1e-12), labels)
    summary = metrics.summary(config.uncertainty.calibration_bins)
    summary["epistemic"] = float(np.mean(epistemic))
    summary["num_samples"] = int(labels.shape[0])

    LOGGER.info(
        "accuracy=%.4f  ece=%.4f  nll=%.4f  epistemic=%.4f  (n=%d)",
        summary["accuracy"],
        summary["ece"],
        summary["nll"],
        summary["epistemic"],
        summary["num_samples"],
    )
    return summary
