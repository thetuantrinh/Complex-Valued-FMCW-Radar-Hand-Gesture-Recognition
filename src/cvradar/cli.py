"""Command-line interface: ``cvradar {train,evaluate,summary}``.

Examples::

    cvradar summary
    cvradar train --config configs/default.yaml
    cvradar evaluate --model checkpoints/mc_dropout/run_20250510_101611
    cvradar evaluate --method deep_ensemble --split noise --noise-type AWGN_SNR_-5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from cvradar import __version__
from cvradar.config import ExperimentConfig

__all__ = ["build_parser", "main"]

LOGGER = logging.getLogger("cvradar")


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="cvradar",
        description=(
            "Complex-valued (2+1)D CNNs for real-time FMCW radar hand-gesture "
            "recognition."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"cvradar {__version__}")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable debug logging"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config", type=Path, default=None, help="path to a YAML experiment config"
    )
    common.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="dataset root; overrides the config and $CVRADAR_DATA_ROOT",
    )
    common.add_argument(
        "--batch-size", type=int, default=None, help="override batch size"
    )

    train_parser = subparsers.add_parser(
        "train",
        parents=[common],
        help="train a (2+1)D CVNet from scratch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser.add_argument("--epochs", type=int, default=None, help="override epochs")
    train_parser.add_argument(
        "--learning-rate", type=float, default=None, help="override learning rate"
    )
    train_parser.add_argument(
        "--run-name", default=None, help="run directory name (default: run_<timestamp>)"
    )
    train_parser.set_defaults(handler=_command_train)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        parents=[common],
        help="evaluate accuracy, calibration and epistemic uncertainty",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    evaluate_parser.add_argument(
        "--model", type=Path, default=None, help="SavedModel directory (MC Dropout)"
    )
    evaluate_parser.add_argument(
        "--method",
        choices=("mc_dropout", "deep_ensemble"),
        default=None,
        help="uncertainty estimator",
    )
    evaluate_parser.add_argument(
        "--ensemble-dir",
        type=Path,
        default=None,
        help="directory of ensemble members (deep_ensemble)",
    )
    evaluate_parser.add_argument(
        "--split", choices=("valid", "train", "noise"), default="valid", help="split"
    )
    evaluate_parser.add_argument(
        "--noise-type", default=None, help="noise condition, e.g. AWGN_SNR_-5"
    )
    evaluate_parser.add_argument(
        "--mc-samples", type=int, default=None, help="stochastic forward passes"
    )
    evaluate_parser.add_argument(
        "--output", type=Path, default=None, help="write the metrics to this JSON file"
    )
    evaluate_parser.set_defaults(handler=_command_evaluate)

    summary_parser = subparsers.add_parser(
        "summary",
        parents=[common],
        help="print the model architecture and parameter count",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    summary_parser.set_defaults(handler=_command_summary)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit status."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        return args.handler(args, _resolve_config(args))
    except (FileNotFoundError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1


def _resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    """Load the YAML config, then apply command-line overrides on top."""
    config = (
        ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()
    )

    data_overrides = {}
    if getattr(args, "data_root", None) is not None:
        data_overrides["root"] = args.data_root
    if getattr(args, "batch_size", None) is not None:
        data_overrides["batch_size"] = args.batch_size
    if data_overrides:
        config = replace(config, data=replace(config.data, **data_overrides))

    training_overrides = {}
    if getattr(args, "epochs", None) is not None:
        training_overrides["epochs"] = args.epochs
    if getattr(args, "learning_rate", None) is not None:
        training_overrides["learning_rate"] = args.learning_rate
    if training_overrides:
        config = replace(config, training=replace(config.training, **training_overrides))

    uncertainty_overrides = {}
    if getattr(args, "method", None) is not None:
        uncertainty_overrides["method"] = args.method
    if getattr(args, "ensemble_dir", None) is not None:
        uncertainty_overrides["ensemble_dir"] = args.ensemble_dir
    if getattr(args, "mc_samples", None) is not None:
        uncertainty_overrides["num_samples"] = args.mc_samples
    if uncertainty_overrides:
        config = replace(
            config, uncertainty=replace(config.uncertainty, **uncertainty_overrides)
        )

    return config


def _command_train(args: argparse.Namespace, config: ExperimentConfig) -> int:
    from cvradar.training.trainer import train

    _, _, run_dir = train(config, run_name=args.run_name)
    print(f"Training complete. Artifacts written to {run_dir}")
    return 0


def _command_evaluate(args: argparse.Namespace, config: ExperimentConfig) -> int:
    from cvradar.evaluation.evaluate import evaluate

    summary = evaluate(
        config,
        model_path=args.model,
        split=args.split,
        noise_type=args.noise_type,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        LOGGER.info("Metrics written to %s", args.output)

    return 0


def _command_summary(args: argparse.Namespace, config: ExperimentConfig) -> int:
    from cvradar.models.cv_net import build_cv_net

    model = build_cv_net(config.radar, config.model)
    model.summary()
    print(f"\nTrainable tensors: {len(model.trainable_variables)}")
    print(f"Input shape:       {(None, *config.radar.input_shape)}")
    print(f"Output classes:    {config.model.num_classes}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
