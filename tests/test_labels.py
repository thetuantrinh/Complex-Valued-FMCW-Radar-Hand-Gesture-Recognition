"""Canonical gesture vocabulary and label encoding."""

from __future__ import annotations

import pytest

from cvradar.data.labels import (
    GESTURE_CLASSES,
    class_to_index,
    encode_labels,
    index_to_class,
    infer_classes_from_directory,
    validate_classes,
)

# The vocabulary the original pipeline fed to sklearn's LabelEncoder.
LEGACY_LABELS = [
    "empty",
    "counter_clock-wise",
    "clock-wise",
    "push-down",
    "pull-up",
    "zoom-in",
    "zoom-out",
    "to_left",
    "to_right",
    "unknown",
]


def test_vocabulary_has_ten_unique_gestures():
    assert len(GESTURE_CLASSES) == 10
    assert len(set(GESTURE_CLASSES)) == 10


def test_ordering_matches_legacy_label_encoder():
    """LabelEncoder assigned indices alphabetically; checkpoints depend on it."""
    assert tuple(sorted(LEGACY_LABELS)) == GESTURE_CLASSES


def test_index_roundtrip():
    for index, name in enumerate(GESTURE_CLASSES):
        assert class_to_index(name) == index
        assert index_to_class(index) == name


def test_encode_labels_preserves_order():
    assert encode_labels(["empty", "zoom-out", "clock-wise"]) == [2, 9, 0]


def test_unknown_gesture_is_rejected():
    with pytest.raises(KeyError, match="unknown gesture"):
        class_to_index("somersault")
    with pytest.raises(KeyError, match="unknown gesture"):
        encode_labels(["empty", "somersault"])


def test_out_of_range_index_is_rejected():
    with pytest.raises(IndexError):
        index_to_class(len(GESTURE_CLASSES))


def test_encoding_is_split_independent():
    """A split missing gestures must not shift the remaining class indices.

    This is the regression guard for the per-split LabelEncoder the original
    loaders used, which renumbered classes whenever a split was incomplete.
    """
    partial = ["empty", "zoom-out"]
    assert encode_labels(partial) == [class_to_index(n) for n in partial]
    assert encode_labels(partial) == [2, 9]


def test_validate_classes_accepts_subsets_and_rejects_strangers():
    validate_classes(["empty", "zoom-in"])  # a subset is fine
    with pytest.raises(ValueError, match="unknown gesture directories"):
        validate_classes(["empty", "moonwalk"])


def test_infer_classes_from_directory(tmp_path):
    for gesture in ("zoom-in", "empty"):
        (tmp_path / "Person_1" / gesture).mkdir(parents=True)
    assert infer_classes_from_directory(tmp_path) == ("empty", "zoom-in")
