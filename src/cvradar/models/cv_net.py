"""(2+1)D Complex-Valued Network ((2+1)D CVNet).

An end-to-end classifier over raw time-domain FMCW radar tensors. The network
learns in the native complex domain and contains no FFT stage, which is what
removes the Range-Doppler / micro-Doppler preprocessing bottleneck on embedded
targets.

Topology::

    raw I/Q tensor (frames, chirps, samples, channels)
        -> (2+1)D complex stem + complex batch norm + ReLU
        -> 4 complex residual blocks (4, 4, 8, 8 complex filters)
        -> 3D max pooling -> ReLU -> global average pooling
        -> Monte Carlo Dropout -> dense classifier (logits)

The head emits **logits**, not probabilities: the calibration metrics in
:mod:`cvradar.evaluation.metrics` and the sparse categorical loss both expect
unnormalised scores.
"""

from __future__ import annotations

import complexnn
import tensorflow as tf

from cvradar.config import ModelConfig, RadarConfig
from cvradar.models.layers import add_residual_block, complex_conv_2plus1d

__all__ = ["build_cv_net"]


def build_cv_net(
    radar: RadarConfig | None = None,
    model: ModelConfig | None = None,
    name: str = "cv_2plus1d_net",
) -> tf.keras.Model:
    """Build the (2+1)D CVNet classifier.

    Args:
        radar: Acquisition geometry defining the input shape. Defaults to the
            published configuration, ``(20, 128, 64, 8)``.
        model: Architecture hyperparameters.
        name: Keras model name.

    Returns:
        An uncompiled ``tf.keras.Model`` mapping a raw radar tensor to
        ``model.num_classes`` logits.

    Example:
        >>> model = build_cv_net()  # doctest: +SKIP
        >>> model.output_shape  # doctest: +SKIP
        (None, 10)
    """
    radar = radar or RadarConfig()
    model = model or ModelConfig()

    inputs = tf.keras.Input(shape=radar.input_shape, name="raw_iq")

    # Stem: strided (2+1)D complex convolution.
    x = complex_conv_2plus1d(
        inputs,
        filters=model.stem_filters,
        kernel_size=model.kernel_size,
        padding="same",
        strides=model.stem_strides,
    )
    x = complexnn.bn.ComplexBatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # Complex residual trunk.
    for filters in model.block_filters:
        x = add_residual_block(x, filters, model.kernel_size)

    x = tf.keras.layers.MaxPool3D(pool_size=model.pool_size)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)

    # Monte Carlo Dropout head: with `mc_dropout` the layer stays stochastic at
    # inference, so repeated forward passes sample the epistemic predictive
    # distribution instead of collapsing to a point estimate.
    x = tf.keras.layers.Dropout(model.dropout_rate, name="mc_dropout")(
        x, training=True if model.mc_dropout else None
    )
    outputs = tf.keras.layers.Dense(model.num_classes, name="logits")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
