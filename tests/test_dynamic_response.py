import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/dynamic_response"))

from headline_metrics import delta_pearson, relative_mse_reduction, roc_auc, top_k_overlap
from tune_dynamic_lps import feature_array, fit_batch, predict_batch


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
    fit = fit_batch(x, y, alpha=1e-8)
    predicted = predict_batch(x, fit)
    assert predicted.shape == y.shape
    assert np.mean((predicted - y) ** 2) == pytest.approx(0.0, abs=1e-10)
    mono = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    world = pd.DataFrame({"a": [5.0, 6.0], "b": [7.0, 8.0]})
    features = feature_array(mono, world, ["a", "b"], np.array([0, 1]), "combined")
    assert features.shape == (2, 2, 2)
    assert np.array_equal(features[1, 1], [4.0, 8.0])
