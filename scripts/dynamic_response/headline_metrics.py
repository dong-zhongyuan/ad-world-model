import numpy as np
from scipy.stats import rankdata


def roc_auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    keep = np.isfinite(scores)
    labels = labels[keep]
    scores = scores[keep]
    n_positive = int(labels.sum())
    n_negative = int((~labels).sum())
    if n_positive == 0 or n_negative == 0:
        raise ValueError("AUC requires both positive and negative labels")
    ranks = rankdata(scores, method="average")
    return float((ranks[labels].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def relative_mse_reduction(observed, predicted, baseline_predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    baseline_predicted = np.asarray(baseline_predicted, dtype=float)
    if observed.shape != predicted.shape or observed.shape != baseline_predicted.shape:
        raise ValueError("Observed and predicted arrays must have identical shapes")
    keep = np.isfinite(observed) & np.isfinite(predicted) & np.isfinite(baseline_predicted)
    if not keep.any():
        raise ValueError("No jointly measured values")
    model_mse = float(np.mean((predicted[keep] - observed[keep]) ** 2))
    baseline_mse = float(np.mean((baseline_predicted[keep] - observed[keep]) ** 2))
    if baseline_mse <= 0:
        raise ValueError("Baseline MSE must be positive")
    return 1.0 - model_mse / baseline_mse


def top_k_overlap(observed_delta, predicted_delta, k=100):
    observed_delta = np.asarray(observed_delta, dtype=float)
    predicted_delta = np.asarray(predicted_delta, dtype=float)
    if observed_delta.shape != predicted_delta.shape:
        raise ValueError("Delta arrays must have identical shapes")
    keep = np.isfinite(observed_delta) & np.isfinite(predicted_delta)
    observed_delta = observed_delta[keep]
    predicted_delta = predicted_delta[keep]
    k = min(int(k), len(observed_delta))
    if k <= 0:
        raise ValueError("k must be positive")
    observed_top = set(np.argsort(-np.abs(observed_delta), kind="mergesort")[:k])
    predicted_top = set(np.argsort(-np.abs(predicted_delta), kind="mergesort")[:k])
    return len(observed_top & predicted_top) / k


def delta_pearson(observed_delta, predicted_delta):
    observed_delta = np.asarray(observed_delta, dtype=float)
    predicted_delta = np.asarray(predicted_delta, dtype=float)
    keep = np.isfinite(observed_delta) & np.isfinite(predicted_delta)
    observed_delta = observed_delta[keep]
    predicted_delta = predicted_delta[keep]
    if len(observed_delta) < 2 or np.std(observed_delta) == 0 or np.std(predicted_delta) == 0:
        return float("nan")
    return float(np.corrcoef(observed_delta, predicted_delta)[0, 1])
