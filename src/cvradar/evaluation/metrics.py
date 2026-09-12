"""Calibration metrics for the uncertainty-aware classifier.

A model can be accurate and still badly calibrated -- confident when it is
wrong. These metrics quantify that gap, and are the ones reported in the paper:

* **Expected Calibration Error (ECE)** -- the average absolute gap between
  confidence and accuracy, computed over equal-width confidence bins.
* **Negative Log-Likelihood (NLL)** -- the proper scoring rule that penalises
  confident mistakes.

Both consume **logits**, matching the head built by
:func:`cvradar.models.cv_net.build_cv_net`.
"""

from __future__ import annotations

import tensorflow as tf

__all__ = [
    "UncertaintyMetrics",
    "expected_calibration_error",
    "negative_log_likelihood",
]


def expected_calibration_error(
    logits: tf.Tensor, labels: tf.Tensor, num_bins: int = 10
) -> tf.Tensor:
    """Compute the Expected Calibration Error.

    Args:
        logits: Unnormalised scores of shape ``(batch, num_classes)``.
        labels: Integer ground-truth labels of shape ``(batch,)``.
        num_bins: Number of equal-width confidence bins.

    Returns:
        A scalar tensor; lower is better, ``0`` means perfect calibration.
    """
    import tensorflow_probability as tfp

    logits = tf.convert_to_tensor(logits)
    labels = tf.cast(tf.convert_to_tensor(labels), tf.int32)
    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)

    return tfp.stats.expected_calibration_error(
        num_bins=num_bins,
        logits=logits,
        labels_true=labels,
        labels_predicted=predictions,
    )


def negative_log_likelihood(logits: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    """Compute the mean negative log-likelihood of the true labels.

    Args:
        logits: Unnormalised scores of shape ``(batch, num_classes)``.
        labels: Integer ground-truth labels of shape ``(batch,)``.

    Returns:
        A scalar tensor; lower is better.
    """
    logits = tf.convert_to_tensor(logits)
    labels = tf.cast(tf.convert_to_tensor(labels), tf.int32)
    log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits
    )
    return tf.reduce_mean(log_probs)


class UncertaintyMetrics:
    """Convenience accessor bundling the calibration metrics for one batch.

    Example:
        >>> metrics = UncertaintyMetrics(logits, labels)  # doctest: +SKIP
        >>> metrics.summary()  # doctest: +SKIP
        {'accuracy': 0.99, 'ece': 0.01, 'nll': 0.03}
    """

    def __init__(self, logits: tf.Tensor, labels: tf.Tensor) -> None:
        """Store one batch of predictions.

        Args:
            logits: Unnormalised scores of shape ``(batch, num_classes)``.
            labels: Integer ground-truth labels of shape ``(batch,)``.
        """
        self.logits = tf.convert_to_tensor(logits)
        self.labels = tf.cast(tf.convert_to_tensor(labels), tf.int32)

    @property
    def probabilities(self) -> tf.Tensor:
        """Softmax probabilities of shape ``(batch, num_classes)``."""
        return tf.nn.softmax(self.logits, axis=-1)

    @property
    def predictions(self) -> tf.Tensor:
        """Arg-max class predictions of shape ``(batch,)``."""
        return tf.argmax(self.logits, axis=-1, output_type=tf.int32)

    def accuracy(self) -> tf.Tensor:
        """Top-1 accuracy as a scalar tensor."""
        correct = tf.cast(tf.equal(self.predictions, self.labels), tf.float32)
        return tf.reduce_mean(correct)

    def ece(self, num_bins: int = 10) -> tf.Tensor:
        """Expected Calibration Error. See :func:`expected_calibration_error`."""
        return expected_calibration_error(self.logits, self.labels, num_bins)

    def nll(self) -> tf.Tensor:
        """Negative log-likelihood. See :func:`negative_log_likelihood`."""
        return negative_log_likelihood(self.logits, self.labels)

    def summary(self, num_bins: int = 10) -> dict[str, float]:
        """Return accuracy, ECE and NLL as plain Python floats."""
        return {
            "accuracy": float(self.accuracy().numpy()),
            "ece": float(self.ece(num_bins).numpy()),
            "nll": float(self.nll().numpy()),
        }
