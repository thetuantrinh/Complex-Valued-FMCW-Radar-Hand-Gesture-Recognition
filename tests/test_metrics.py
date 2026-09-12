"""Calibration metrics and uncertainty decomposition."""

from __future__ import annotations

import pytest

from tests.conftest import requires_tensorflow, requires_tfp

pytestmark = requires_tensorflow


def test_nll_is_near_zero_for_confident_correct_predictions():
    import numpy as np

    from cvradar.evaluation.metrics import negative_log_likelihood

    logits = np.array([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0]], dtype=np.float32)
    labels = np.array([0, 1])
    assert float(negative_log_likelihood(logits, labels)) == pytest.approx(0.0, abs=1e-5)


def test_nll_is_large_for_confident_wrong_predictions():
    import numpy as np

    from cvradar.evaluation.metrics import negative_log_likelihood

    logits = np.array([[20.0, 0.0, 0.0]], dtype=np.float32)
    assert float(negative_log_likelihood(logits, np.array([1]))) > 10.0


@requires_tfp
def test_summary_reports_accuracy_ece_and_nll():
    import numpy as np

    from cvradar.evaluation.metrics import UncertaintyMetrics

    logits = np.array([[5.0, 0.0], [0.0, 5.0], [5.0, 0.0]], dtype=np.float32)
    summary = UncertaintyMetrics(logits, np.array([0, 1, 1])).summary()

    assert set(summary) == {"accuracy", "ece", "nll"}
    assert summary["accuracy"] == pytest.approx(2 / 3)


def test_predictive_entropy_bounds():
    import numpy as np

    from cvradar.evaluation.uncertainty import predictive_entropy

    certain = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    uniform = np.full((1, 3), 1 / 3, dtype=np.float32)

    assert float(predictive_entropy(certain)[0]) == pytest.approx(0.0, abs=1e-5)
    assert float(predictive_entropy(uniform)[0]) == pytest.approx(np.log(3), abs=1e-5)


def test_mutual_information_is_zero_when_members_agree():
    import numpy as np

    from cvradar.evaluation.uncertainty import mutual_information

    agreeing = np.stack([np.array([[0.7, 0.3]], dtype=np.float32)] * 4)
    assert float(mutual_information(agreeing)[0]) == pytest.approx(0.0, abs=1e-5)


def test_mutual_information_is_positive_when_members_disagree():
    import numpy as np

    from cvradar.evaluation.uncertainty import mutual_information

    disagreeing = np.array(
        [[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32
    )  # two confident but opposed members
    assert float(mutual_information(disagreeing)[0]) > 0.5


def test_mc_dropout_predict_rejects_zero_samples():
    from cvradar.evaluation.uncertainty import mc_dropout_predict

    with pytest.raises(ValueError, match="num_samples"):
        mc_dropout_predict(model=None, inputs=None, num_samples=0)


def test_deep_ensemble_predict_rejects_empty_ensemble():
    from cvradar.evaluation.uncertainty import deep_ensemble_predict

    with pytest.raises(ValueError, match="at least one model"):
        deep_ensemble_predict([], inputs=None)
