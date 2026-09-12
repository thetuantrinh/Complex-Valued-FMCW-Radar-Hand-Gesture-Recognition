"""Dataset discovery and ``tf.data`` input pipelines for raw FMCW recordings."""

from cvradar.data.labels import (
    GESTURE_CLASSES,
    class_to_index,
    encode_labels,
    index_to_class,
    infer_classes_from_directory,
)

__all__ = [
    "GESTURE_CLASSES",
    "build_dataset",
    "class_to_index",
    "discover_samples",
    "encode_labels",
    "index_to_class",
    "infer_classes_from_directory",
    "load_noise_split",
    "load_split",
    "load_train_valid",
]


def __getattr__(name: str):
    """Defer the TensorFlow-backed loaders until they are actually requested."""
    if name in {
        "build_dataset",
        "discover_samples",
        "load_noise_split",
        "load_split",
        "load_train_valid",
    }:
        from cvradar.data import loaders

        return getattr(loaders, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
