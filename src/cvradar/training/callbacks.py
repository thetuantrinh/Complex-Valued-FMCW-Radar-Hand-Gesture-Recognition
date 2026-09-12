"""Keras callbacks assembled from :class:`~cvradar.config.TrainingConfig`."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from cvradar.config import TrainingConfig

__all__ = ["build_callbacks"]


def build_callbacks(
    config: TrainingConfig | None = None,
    checkpoint_path: str | Path | None = None,
) -> list[tf.keras.callbacks.Callback]:
    """Build the callback list for a training run.

    Always includes ``ReduceLROnPlateau``. A ``ModelCheckpoint`` is added when
    ``checkpoint_path`` is given, ``EarlyStopping`` when
    :attr:`TrainingConfig.early_stopping` is set, and ``TensorBoard`` when
    :attr:`TrainingConfig.tensorboard_dir` is set.

    Args:
        config: Training configuration; defaults to :class:`TrainingConfig`.
        checkpoint_path: Destination for the best-validation-accuracy weights.

    Returns:
        The configured callbacks, ready to pass to ``Model.fit``.
    """
    config = config or TrainingConfig()

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.lr_factor,
            patience=max(1, config.patience // 2),
            min_lr=config.min_learning_rate,
            verbose=config.verbose,
        )
    ]

    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_sparse_categorical_accuracy",
                mode="max",
                save_best_only=True,
                save_weights_only=True,
                verbose=config.verbose,
            )
        )

    if config.early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.patience,
                restore_best_weights=True,
                verbose=config.verbose,
            )
        )

    if config.tensorboard_dir is not None:
        log_dir = Path(config.tensorboard_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1)
        )

    return callbacks
