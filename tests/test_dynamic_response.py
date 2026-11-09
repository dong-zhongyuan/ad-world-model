import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/dynamic_response"))

from headline_metrics import delta_pearson, relative_mse_reduction, roc_auc, top_k_overlap
from dynamic_model import fit_readouts, predict_readout


def test_headline_metric_definitions():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    observed = np.array([1.0, 2.0])
    predicted = np.array([1.5, 2.0])
    baseline = np.array([0.0, 2.0])
    assert relative_mse_reduction(observed, predicted, baseline) == pytest.approx(0.75)
    assert top_k_overlap(observed, predicted, k=2) == pytest.approx(1.0)
    assert delta_pearson(observed, observed * 0.5) == pytest.approx(1.0)


def test_dynamic_batch_regression_and_feature_order():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(6, 4, 2))
    y = np.einsum("ngf,f->ng", x, np.array([0.7, -0.2])) + 0.3
    source, world = x[:, :, 0], x[:, :, 1]
    fitted = fit_readouts(source, world, y, alpha=1e-8)
    predicted = predict_readout(source, world, fitted["combined_linear"])
    assert predicted.shape == y.shape
    np.testing.assert_allclose(predicted, y, atol=1e-6)
    assert not np.allclose(predict_readout(world, source, fitted["combined_linear"]), y)
