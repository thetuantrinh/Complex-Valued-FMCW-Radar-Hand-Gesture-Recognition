"""Canonical gesture vocabulary and label encoding.

The class index of a gesture is fixed by this module rather than being inferred
per split. The original pipeline fitted a separate ``sklearn.LabelEncoder`` on
the directory names found in each split, which silently produced *different*
index assignments whenever a split did not contain every gesture -- for example
a noise-corrupted subset missing one class. Pinning the vocabulary here keeps
training, clean validation and noise validation on one shared label space.

The ordering is the alphabetical order of the on-disk gesture directory names,
which is exactly what ``LabelEncoder`` produced for a complete split; published
checkpoints therefore remain compatible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "GESTURE_CLASSES",
    "class_to_index",
    "encode_labels",
    "index_to_class",
    "infer_classes_from_directory",
    "validate_classes",
]

#: Gesture directory names, in canonical (alphabetical) class-index order.
GESTURE_CLASSES: tuple[str, ...] = (
    "clock-wise",
    "counter_clock-wise",
    "empty",
    "pull-up",
    "push-down",
    "to_left",
    "to_right",
    "unknown",
    "zoom-in",
    "zoom-out",
)

_CLASS_TO_INDEX = {name: index for index, name in enumerate(GESTURE_CLASSES)}


def class_to_index(name: str, classes: Sequence[str] = GESTURE_CLASSES) -> int:
    """Map a gesture directory name to its integer class index.

    Raises:
        KeyError: if ``name`` is not part of the vocabulary.
    """
    lookup = _CLASS_TO_INDEX if classes is GESTURE_CLASSES else _index_map(classes)
    try:
        return lookup[name]
    except KeyError:
        raise KeyError(
            f"unknown gesture {name!r}; expected one of {list(classes)}"
        ) from None


def index_to_class(index: int, classes: Sequence[str] = GESTURE_CLASSES) -> str:
    """Map an integer class index back to its gesture name."""
    try:
        return classes[index]
    except IndexError:
        raise IndexError(
            f"class index {index} out of range for {len(classes)} classes"
        ) from None


def encode_labels(
    names: Iterable[str], classes: Sequence[str] = GESTURE_CLASSES
) -> list[int]:
    """Encode an iterable of gesture names into class indices."""
    lookup = _CLASS_TO_INDEX if classes is GESTURE_CLASSES else _index_map(classes)
    encoded = []
    for name in names:
        try:
            encoded.append(lookup[name])
        except KeyError:
            raise KeyError(
                f"unknown gesture {name!r}; expected one of {list(classes)}"
            ) from None
    return encoded


def infer_classes_from_directory(
    split_dir: str | Path, depth: int = 2
) -> tuple[str, ...]:
    """Return the sorted gesture directory names found beneath ``split_dir``.

    The dataset is laid out as ``<split>/<subject>/<gesture>/<sample>.npy``, so
    gesture directories sit at ``depth`` 2 by default.
    """
    root = Path(split_dir)
    pattern = "/".join(["*"] * depth)
    names = {path.name for path in root.glob(pattern) if path.is_dir()}
    return tuple(sorted(names))


def validate_classes(
    found: Sequence[str], expected: Sequence[str] = GESTURE_CLASSES
) -> None:
    """Warn-free strict check that a split's directories match the vocabulary.

    Raises:
        ValueError: if ``found`` contains a gesture outside ``expected``.
    """
    unexpected = sorted(set(found) - set(expected))
    if unexpected:
        raise ValueError(
            f"dataset contains unknown gesture directories {unexpected}; "
            f"configured vocabulary is {list(expected)}"
        )


def _index_map(classes: Sequence[str]) -> dict[str, int]:
    return {name: index for index, name in enumerate(classes)}
