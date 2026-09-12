"""Classification metrics and epistemic uncertainty quantification."""

from cvradar.evaluation.evaluate import evaluate
from cvradar.evaluation.metrics import (
    UncertaintyMetrics,
    expected_calibration_error,
    negative_log_likelihood,
)
from cvradar.evaluation.uncertainty import (
    deep_ensemble_predict,
    load_ensemble,
    mc_dropout_predict,
    mutual_information,
    predictive_entropy,
)

__all__ = [
    "UncertaintyMetrics",
    "deep_ensemble_predict",
    "evaluate",
    "expected_calibration_error",
    "load_ensemble",
    "mc_dropout_predict",
    "mutual_information",
    "negative_log_likelihood",
    "predictive_entropy",
]
