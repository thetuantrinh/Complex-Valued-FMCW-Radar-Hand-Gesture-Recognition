"""End-to-end training entry point for the (2+1)D CVNet."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import tensorflow as tf

from cvradar.config import ExperimentConfig
from cvradar.data.loaders import load_train_valid
from cvradar.models.cv_net import build_cv_net
from cvradar.training.callbacks import build_callbacks

__all__ = ["compile_model", "train"]

LOGGER = logging.getLogger(__name__)


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-3) -> tf.keras.Model:
    """Compile a CVNet for sparse-label classification over logits.

    ``from_logits=True`` matches the network head, which deliberately emits
    unnormalised scores so that the calibration metrics stay numerically stable.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def train(
    config: ExperimentConfig | None = None,
    run_name: str | None = None,
) -> tuple[tf.keras.Model, dict[str, list[float]], Path]:
    """Train a (2+1)D CVNet and persist the run.

    Writes the SavedModel, the resolved configuration and the training history
    into a timestamped run directory beneath
    :attr:`~cvradar.config.TrainingConfig.checkpoint_dir`.

    Args:
        config: Full experiment configuration.
        run_name: Run directory name; defaults to ``run_<timestamp>``.

    Returns:
        A ``(model, history, run_dir)`` triple.
    """
    config = config or ExperimentConfig()

    run_name = run_name or f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path(config.training.checkpoint_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Run directory: %s", run_dir)

    train_ds, valid_ds = load_train_valid(config.data, config.radar)

    model = compile_model(
        build_cv_net(config.radar, config.model), config.training.learning_rate
    )
    LOGGER.info(
        "Model %s: %d trainable tensors", model.name, len(model.trainable_variables)
    )

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=config.training.epochs,
        callbacks=build_callbacks(config.training, run_dir / "weights.ckpt"),
        verbose=config.training.verbose,
    )

    model.save(str(run_dir / "saved_model"))
    _write_json(run_dir / "config.json", config.to_dict())
    serialisable_history = {
        key: [float(value) for value in values] for key, values in history.history.items()
    }
    _write_json(run_dir / "history.json", serialisable_history)
    LOGGER.info("Saved model and run metadata to %s", run_dir)

    return model, history.history, run_dir


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
