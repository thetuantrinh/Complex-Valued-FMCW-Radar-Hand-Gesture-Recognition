"""Typed configuration objects for the (2+1)D CVNet experiments.

Every tunable in the pipeline lives here rather than being hardcoded inside a
module, so an experiment is fully described by a single YAML file plus the code
revision. Filesystem locations additionally honour environment variables, which
keeps cluster paths out of version control.
"""

from __future__ import annotations

import os
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "ModelConfig",
    "RadarConfig",
    "TrainingConfig",
    "UncertaintyConfig",
]

#: Environment variable consulted when ``DataConfig.root`` is left unset.
DATA_ROOT_ENV = "CVRADAR_DATA_ROOT"


@dataclass(frozen=True)
class RadarConfig:
    """Geometry of one raw FMCW acquisition.

    The sensor is a Texas Instruments AWR1243BOOST (77 GHz) captured through a
    DCA1000EVM. Each recording is a real-valued tensor of shape
    ``(frames, chirps, samples, channels)`` where the trailing axis interleaves
    the in-phase and quadrature components of every receive antenna.
    """

    frames: int = 20
    """Slow-time radar frames per gesture sample."""

    chirps: int = 128
    """Chirps per frame."""

    samples: int = 64
    """Fast-time ADC samples per chirp."""

    rx_antennas: int = 4
    """Physical receive antennas on the sensor."""

    @property
    def channels(self) -> int:
        """Trailing tensor axis: one real and one imaginary plane per antenna."""
        return self.rx_antennas * 2

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        """Per-sample input shape, excluding the batch axis."""
        return (self.frames, self.chirps, self.samples, self.channels)


@dataclass(frozen=True)
class DataConfig:
    """Where the dataset lives and how it is fed to the model."""

    root: Path | None = None
    """Dataset root. Falls back to ``$CVRADAR_DATA_ROOT`` then to ``./data``."""

    train_subdir: str = "train"
    valid_subdir: str = "valid"
    noise_subdir: str = "noise"
    """Split directories beneath :attr:`root`."""

    sample_glob: str = "*/*/*.npy"
    """Glob relative to a split directory: ``<subject>/<gesture>/<sample>.npy``."""

    batch_size: int = 32
    shuffle_buffer: int | None = None
    """Shuffle buffer size; ``None`` shuffles over the full split."""

    seed: int = 1337

    def resolved_root(self) -> Path:
        """Return the dataset root, resolving the env-var and default fallbacks."""
        if self.root is not None:
            return Path(self.root).expanduser()
        env_root = os.environ.get(DATA_ROOT_ENV)
        if env_root:
            return Path(env_root).expanduser()
        return Path("data")

    def split_dir(self, split: str) -> Path:
        """Return the directory for ``split`` (``train``/``valid``/``noise``)."""
        subdirs = {
            "train": self.train_subdir,
            "valid": self.valid_subdir,
            "noise": self.noise_subdir,
        }
        try:
            return self.resolved_root() / subdirs[split]
        except KeyError:
            raise ValueError(
                f"unknown split {split!r}; expected one of {sorted(subdirs)}"
            ) from None


@dataclass(frozen=True)
class ModelConfig:
    """Architecture of the (2+1)D complex-valued network."""

    num_classes: int = 10

    stem_filters: int = 4
    """Complex filters in the stem convolution."""

    stem_strides: tuple[int, int, int] = (1, 2, 1)
    """Stem stride, applied to both factors of the (2+1)D decomposition."""

    block_filters: tuple[int, ...] = (4, 4, 8, 8)
    """Complex filters per residual block, in order."""

    kernel_size: tuple[int, int, int] = (3, 3, 2)
    """``(time, chirp, sample)`` kernel, factorised into a spatial and a
    temporal convolution."""

    pool_size: tuple[int, int, int] = (2, 2, 2)
    """Max-pooling window applied before global average pooling."""

    dropout_rate: float = 0.05
    """Dropout probability of the Monte Carlo Dropout head."""

    mc_dropout: bool = True
    """Keep dropout active at inference so that repeated forward passes sample
    the epistemic predictive distribution."""


@dataclass(frozen=True)
class TrainingConfig:
    """Optimisation schedule."""

    epochs: int = 100
    learning_rate: float = 1e-3
    patience: int = 5
    """Drives the ReduceLROnPlateau / EarlyStopping schedules."""

    min_learning_rate: float = 1e-9
    lr_factor: float = 0.1
    early_stopping: bool = False
    checkpoint_dir: Path = Path("checkpoints")
    tensorboard_dir: Path | None = None
    verbose: int = 1


@dataclass(frozen=True)
class UncertaintyConfig:
    """Epistemic uncertainty estimation."""

    method: str = "mc_dropout"
    """``mc_dropout`` or ``deep_ensemble``."""

    num_samples: int = 50
    """Stochastic forward passes for Monte Carlo Dropout."""

    ensemble_dir: Path = Path("checkpoints/deep_ensemble")
    """Directory holding one SavedModel per ensemble member."""

    calibration_bins: int = 10
    """Bin count for the Expected Calibration Error."""


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration: one YAML file maps onto one instance."""

    radar: RadarConfig = field(default_factory=RadarConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> ExperimentConfig:
        """Build a config from a nested mapping, ignoring nothing silently.

        Raises:
            ValueError: if the mapping contains an unknown section or key.
        """
        return _from_mapping(cls, mapping, path="")

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load a config from a YAML file."""
        import yaml  # imported lazily: the dataclasses are usable without PyYAML

        with open(path, encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}: expected a YAML mapping at the top level")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, YAML-serialisable dictionary."""
        return _jsonify(asdict(self))


def _from_mapping(cls: type, mapping: Mapping[str, Any], path: str) -> Any:
    """Recursively instantiate nested dataclasses, validating every key."""
    known = {f.name: f for f in fields(cls)}
    unknown = set(mapping) - set(known)
    if unknown:
        where = path or "<root>"
        raise ValueError(
            f"unknown configuration key(s) {sorted(unknown)} in {where}; "
            f"valid keys are {sorted(known)}"
        )

    kwargs: dict[str, Any] = {}
    for name, value in mapping.items():
        field_type = known[name].type
        nested = _nested_dataclass(cls, name)
        if nested is not None and isinstance(value, Mapping):
            child_path = f"{path}.{name}".lstrip(".")
            kwargs[name] = _from_mapping(nested, value, path=child_path)
        elif isinstance(value, list):
            kwargs[name] = tuple(value)
        elif isinstance(field_type, str) and "Path" in field_type and value is not None:
            kwargs[name] = Path(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _nested_dataclass(cls: type, name: str) -> type | None:
    """Return the dataclass type of field ``name``, if it is itself a dataclass."""
    factories = {f.name: f.default_factory for f in fields(cls)}
    default_factory = factories.get(name)
    if default_factory is None or default_factory is MISSING:
        return None
    try:
        candidate = default_factory()
    except TypeError:  # pragma: no cover - defensive
        return None
    return type(candidate) if is_dataclass(candidate) else None


def _jsonify(value: Any) -> Any:
    """Convert Paths and tuples into YAML/JSON-friendly primitives."""
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
