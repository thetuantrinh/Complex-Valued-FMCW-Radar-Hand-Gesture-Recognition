"""Figures for training curves, confusion matrices and calibration.

Every function returns its ``matplotlib.figure.Figure`` instead of calling
``plt.show()``, so figures can be saved headlessly in CI or on a cluster and
composed by the caller. Pass ``save_path`` to write the figure to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

__all__ = [
    "plot_confusion_matrix",
    "plot_reliability_diagram",
    "plot_training_history",
]


def plot_training_history(
    history: Mapping[str, Sequence[float]],
    title: str | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot loss and accuracy curves on twin axes.

    Args:
        history: Keras history dictionary, or the ``history.json`` of a run.
        title: Optional figure title.
        save_path: If given, write the figure here.

    Returns:
        The figure, for further customisation by the caller.
    """
    accuracy_key = _first_present(history, ("sparse_categorical_accuracy", "accuracy"))
    val_accuracy_key = _first_present(
        history, ("val_sparse_categorical_accuracy", "val_accuracy")
    )

    figure, loss_axis = plt.subplots(figsize=(8, 5))
    loss_axis.set_xlabel("Epoch", fontsize=12)
    loss_axis.set_ylabel("Loss", fontsize=12)
    loss_axis.plot(history["loss"], "-.", color="tab:red", label="Train loss")
    if "val_loss" in history:
        loss_axis.plot(history["val_loss"], "-.", color="tab:blue", label="Valid loss")
    loss_axis.grid(linestyle="-.", alpha=0.4)

    if accuracy_key is not None:
        accuracy_axis = loss_axis.twinx()
        accuracy_axis.set_ylabel("Accuracy", fontsize=12)
        accuracy_axis.plot(history[accuracy_key], color="tab:red", label="Train accuracy")
        if val_accuracy_key is not None:
            accuracy_axis.plot(
                history[val_accuracy_key], color="tab:blue", label="Valid accuracy"
            )
        _merge_legends(figure, loss_axis, accuracy_axis)
    else:
        loss_axis.legend(fontsize=11)

    if title:
        figure.suptitle(title)
    figure.tight_layout()
    return _finalise(figure, save_path)


def plot_confusion_matrix(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    class_names: Sequence[str],
    normalize: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot a confusion matrix as a heatmap.

    Rows are true classes and columns predicted classes, so a normalised row
    sums to 100%.

    Args:
        true_labels: Ground-truth class indices.
        predicted_labels: Predicted class indices.
        class_names: Axis tick labels, in class-index order.
        normalize: Show row-wise percentages rather than raw counts.
        save_path: If given, write the figure here.

    Returns:
        The figure.
    """
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(
        y_true=true_labels,
        y_pred=predicted_labels,
        labels=np.arange(len(class_names)),
    ).astype(float)

    if normalize:
        row_totals = matrix.sum(axis=1, keepdims=True)
        # Guard against a class that is absent from this split.
        matrix = (
            np.divide(
                matrix, row_totals, out=np.zeros_like(matrix), where=row_totals != 0
            )
            * 100.0
        )

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=100 if normalize else None)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fmt = "{:.1f}" if normalize else "{:.0f}"
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                fmt.format(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if matrix[row, column] > threshold else "black",
            )

    axis.set_xticks(np.arange(len(class_names)))
    axis.set_yticks(np.arange(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    axis.set_yticklabels(class_names, fontsize=9)
    axis.set_xlabel("Predicted label", fontsize=12)
    axis.set_ylabel("True label", fontsize=12)
    figure.tight_layout()
    return _finalise(figure, save_path)


def plot_reliability_diagram(
    probabilities: np.ndarray,
    labels: Sequence[int],
    num_bins: int = 10,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot a reliability diagram: accuracy against confidence per bin.

    A perfectly calibrated model traces the diagonal; bars below it mark
    overconfidence, bars above it underconfidence.

    Args:
        probabilities: Predicted probabilities, shape ``(n, num_classes)``.
        labels: Ground-truth class indices, shape ``(n,)``.
        num_bins: Equal-width confidence bins.
        save_path: If given, write the figure here.

    Returns:
        The figure.
    """
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    confidence = probabilities.max(axis=-1)
    correct = probabilities.argmax(axis=-1) == labels

    edges = np.linspace(0.0, 1.0, num_bins + 1)
    centres, accuracies = [], []
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        centres.append((lower + upper) / 2.0)
        accuracies.append(correct[in_bin].mean() if in_bin.any() else 0.0)

    figure, axis = plt.subplots(figsize=(5.5, 5))
    axis.bar(
        centres,
        accuracies,
        width=1.0 / num_bins * 0.9,
        edgecolor="black",
        color="tab:blue",
        label="Accuracy",
    )
    axis.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    axis.set_xlabel("Confidence", fontsize=12)
    axis.set_ylabel("Accuracy", fontsize=12)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(fontsize=10)
    axis.grid(linestyle="-.", alpha=0.4)
    figure.tight_layout()
    return _finalise(figure, save_path)


def _first_present(mapping: Mapping[str, object], keys: Sequence[str]) -> str | None:
    return next((key for key in keys if key in mapping), None)


def _merge_legends(figure: Figure, *axes) -> None:
    handles, labels = [], []
    for axis in axes:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        handles.extend(axis_handles)
        labels.extend(axis_labels)
    axes[0].legend(handles, labels, fontsize=10, loc="center right")


def _finalise(figure: Figure, save_path: str | Path | None) -> Figure:
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=300, bbox_inches="tight")
    return figure
