"""Complex-valued (2+1)D convolutional networks for FMCW radar hand-gesture recognition.

Reference implementation for:

    T. T. Trinh et al., "Complex-Valued (2+1)D Convolutional Neural Networks for
    Real-Time Hand Gesture Recognition on Edge Devices With FMCW Radar,"
    IEEE Transactions on Aerospace and Electronic Systems, vol. 62,
    pp. 13147-13156, 2026. doi:10.1109/TAES.2026.3709269

The package operates directly on raw time-domain FMCW ADC recordings; no
Range-Doppler or micro-Doppler FFT stage is applied anywhere in the pipeline.

Submodules are imported lazily so that ``import cvradar`` stays cheap and does
not pull in TensorFlow until a component that actually needs it is touched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "1.0.0"

__all__ = [
    "ExperimentConfig",
    "GESTURE_CLASSES",
    "__version__",
    "build_cv_net",
]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cvradar.config import ExperimentConfig
    from cvradar.data.labels import GESTURE_CLASSES
    from cvradar.models.cv_net import build_cv_net

_LAZY_ATTRS = {
    "ExperimentConfig": ("cvradar.config", "ExperimentConfig"),
    "GESTURE_CLASSES": ("cvradar.data.labels", "GESTURE_CLASSES"),
    "build_cv_net": ("cvradar.models.cv_net", "build_cv_net"),
}


def __getattr__(name: str) -> Any:
    """Resolve top-level re-exports on first access (PEP 562)."""
    try:
        module_name, attr = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    from importlib import import_module

    return getattr(import_module(module_name), attr)


def __dir__() -> list[str]:
    return sorted(__all__)
