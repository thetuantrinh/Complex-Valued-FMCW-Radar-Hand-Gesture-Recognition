"""Architecture construction and forward-pass behaviour."""

from __future__ import annotations

from cvradar.config import ModelConfig, RadarConfig
from tests.conftest import requires_complexnn

pytestmark = requires_complexnn

# A miniature geometry keeps the test fast while exercising the real topology.
TINY_RADAR = RadarConfig(frames=4, chirps=8, samples=8, rx_antennas=4)


def test_model_builds_with_expected_signature():
    from cvradar.models.cv_net import build_cv_net

    model = build_cv_net(TINY_RADAR, ModelConfig(num_classes=10))
    assert model.input_shape == (None, *TINY_RADAR.input_shape)
    assert model.output_shape == (None, 10)


def test_forward_pass_returns_logits():
    import numpy as np

    from cvradar.models.cv_net import build_cv_net

    model = build_cv_net(TINY_RADAR, ModelConfig(num_classes=10))
    batch = np.zeros((2, *TINY_RADAR.input_shape), dtype=np.float32)
    logits = model(batch, training=False)

    assert logits.shape == (2, 10)
    # Logits, not probabilities: rows must not already sum to one.
    assert not np.allclose(np.sum(np.asarray(logits), axis=-1), 1.0)


def test_mc_dropout_makes_inference_stochastic():
    import numpy as np

    from cvradar.models.cv_net import build_cv_net

    model = build_cv_net(
        TINY_RADAR, ModelConfig(num_classes=10, dropout_rate=0.5, mc_dropout=True)
    )
    batch = (
        np.random.default_rng(0)
        .normal(size=(4, *TINY_RADAR.input_shape))
        .astype("float32")
    )
    first = np.asarray(model(batch, training=False))
    second = np.asarray(model(batch, training=False))
    assert not np.allclose(first, second), "MC dropout head collapsed to a point estimate"


def test_block_filters_control_depth():
    from cvradar.models.cv_net import build_cv_net

    shallow = build_cv_net(TINY_RADAR, ModelConfig(block_filters=(4,)))
    deep = build_cv_net(TINY_RADAR, ModelConfig(block_filters=(4, 4, 8, 8)))
    assert len(deep.layers) > len(shallow.layers)
