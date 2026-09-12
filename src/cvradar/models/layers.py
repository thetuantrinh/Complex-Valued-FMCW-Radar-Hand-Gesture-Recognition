"""Complex-valued building blocks of the (2+1)D CVNet.

All convolutions come from `keras-complex <https://github.com/JesperDramsch/
keras-complex>`_ (imported as ``complexnn``) and operate on tensors whose
trailing axis stacks the real and imaginary planes. A ``ComplexConv3D`` with
``filters=f`` therefore emits ``2 * f`` real channels.

The (2+1)D factorisation replaces a full ``(K_t, K_r, K_d)`` complex
convolution with a spatial convolution ``(1, K_r, K_d)`` across the chirp and
fast-time sample axes followed by a temporal convolution ``(K_t, 1, 1)`` across
radar frames. This cuts parameters and multiply-accumulate operations while
still capturing intra-frame phase structure and inter-frame kinematics.
"""

from __future__ import annotations

from typing import Sequence

import complexnn
import tensorflow as tf

__all__ = [
    "add_residual_block",
    "complex_conv_2plus1d",
    "projection_shortcut",
    "residual_block",
]


def complex_conv_2plus1d(
    inputs: tf.Tensor,
    filters: int,
    kernel_size: Sequence[int],
    padding: str = "same",
    strides: int | Sequence[int] = 1,
) -> tf.Tensor:
    """Apply a (2+1)D complex convolution: spatial first, then temporal.

    Args:
        inputs: Tensor of shape ``(batch, frames, chirps, samples, channels)``.
        filters: Complex filters; the output carries ``2 * filters`` channels.
        kernel_size: ``(time, chirp, sample)`` kernel extents.
        padding: ``"same"`` or ``"valid"``.
        strides: Stride specification.

    Returns:
        The convolved tensor.

    Note:
        ``strides`` is applied to *both* factors, matching the published model
        and its released checkpoints. A stem stride of ``(1, 2, 1)`` thus
        decimates the chirp axis twice in total, once per factor.
    """
    if len(kernel_size) != 3:
        raise ValueError(f"kernel_size must have 3 entries, got {tuple(kernel_size)}")

    # Spatial decomposition across the chirp and fast-time sample axes.
    x = complexnn.conv.ComplexConv3D(
        filters=filters,
        kernel_size=(1, kernel_size[1], kernel_size[2]),
        padding=padding,
        strides=strides,
    )(inputs)
    # Temporal decomposition across radar frames.
    return complexnn.conv.ComplexConv3D(
        filters=filters,
        kernel_size=(kernel_size[0], 1, 1),
        padding=padding,
        strides=strides,
    )(x)


def residual_block(
    inputs: tf.Tensor, filters: int, kernel_size: Sequence[int]
) -> tf.Tensor:
    """Two (2+1)D complex convolutions with complex batch norm, pre-shortcut."""
    x = complex_conv_2plus1d(
        inputs, filters=filters, kernel_size=kernel_size, padding="same", strides=1
    )
    x = complexnn.bn.ComplexBatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = complex_conv_2plus1d(
        x, filters=filters, kernel_size=kernel_size, padding="same", strides=1
    )
    return complexnn.bn.ComplexBatchNormalization()(x)


def projection_shortcut(inputs: tf.Tensor, channels: int) -> tf.Tensor:
    """Match a shortcut's channel count to the residual branch.

    Args:
        inputs: Shortcut tensor.
        channels: Target number of *real* channels. ``ComplexConv3D`` doubles
            its filter count, so ``channels // 2`` complex filters are used.
    """
    x = complexnn.conv.ComplexConv3D(
        filters=channels // 2,
        kernel_size=(1, 1, 2),
        padding="same",
        strides=(1, 1, 1),
    )(inputs)
    return complexnn.bn.ComplexBatchNormalization()(x)


def add_residual_block(
    inputs: tf.Tensor, filters: int, kernel_size: Sequence[int]
) -> tf.Tensor:
    """Residual block with an identity or projection shortcut, then addition."""
    residual_branch = residual_block(inputs, filters, kernel_size)
    shortcut = inputs
    if residual_branch.shape[-1] != inputs.shape[-1]:
        shortcut = projection_shortcut(shortcut, residual_branch.shape[-1])
    return tf.keras.layers.add([shortcut, residual_branch])
