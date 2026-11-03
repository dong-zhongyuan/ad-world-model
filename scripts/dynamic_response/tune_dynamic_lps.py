import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_lps_brain_v2 import SAMPLES
from headline_metrics import delta_pearson, top_k_overlap

WORK = Path(__file__).resolve().parent
DATA = WORK / "results/lps_dynamic_adaptation_v2/state_means.npz"
OUT = WORK / "results/lps_dynamic_adaptation_tuned_v1"
ALPHAS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
METHODS = {"source": [0], "world": [1], "combined": [0, 1]}


def fit_batch(x, y, alpha):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    z = (x - mean) / scale
    y_mean = y.mean(axis=0)
    a = np.einsum("ngf,ngk->gfk", z, z)
    a += alpha * np.eye(x.shape[2])[None, :, :]
    b = np.einsum("ngf,ng->gf", z, y - y_mean)
    coefficient = np.linalg.solve(a, b[..., None])[..., 0]
    return mean, scale, y_mean, coefficient


def predict_batch(x, fit):
    mean, scale, y_mean, coefficient = fit
    return y_mean + np.einsum("ngf,gf->ng", (x - mean) / scale, coefficient)


def feature_array(mono, world, samples, nodes, method):
    base = np.stack([mono[samples].to_numpy().T, world[samples].to_numpy().T], axis=-1)
    return base[:, nodes][:, :, METHODS[method]]


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(DATA, allow_pickle=True) as data:
        nodes = data["nodes"].astype(str)
        micro = pd.DataFrame({gsm: data[f"micro__{gsm}"] for gsm in SAMPLES})
        mono = pd.DataFrame({gsm: data[f"mono__{gsm}"] for gsm in SAMPLES})
        world = pd.DataFrame({gsm: data[f"world__{gsm}"] for gsm in SAMPLES})
        detection = np.stack([data[f"detection__{gsm}"] for gsm in SAMPLES])
    genes = pd.read_csv("/public/home/mengxl/dzy/ad_omics_v3/data/processed/genes.csv")
    mask = (detection.max(axis=0) >= 0.01) & genes.kind.eq("shared_og").to_numpy()
    node_ids = np.flatnonzero(mask)

    cv_rows = []
    test_rows = []
    selections = {}
    for train_processing, test_processing in [("dmso", "inhibitors"), ("inhibitors", "dmso")]:
        train = [gsm for gsm, value in SAMPLES.items() if value[1] == train_processing]
        train_lps = [gsm for gsm in train if SAMPLES[gsm][0] == "lps"]
        train_saline = [gsm for gsm in train if SAMPLES[gsm][0] == "saline"]
        for method in METHODS:
            for alpha in ALPHAS:
                for lps_animal in train_lps:
                    for saline_animal in train_saline:
                        fit_animals = [gsm for gsm in train if gsm not in [lps_animal, saline_animal]]
                        x_train = feature_array(mono, world, fit_animals, node_ids, method)
                        y_train = micro[fit_animals].to_numpy().T[:, node_ids]
                        fit = fit_batch(x_train, y_train, alpha)
                        x_test = feature_array(mono, world, [lps_animal, saline_animal], node_ids, method)
                        predicted = predict_batch(x_test, fit)
                        predicted_delta = predicted[0] - predicted[1]
                        observed_delta = micro[lps_animal].to_numpy()[node_ids] - micro[saline_animal].to_numpy()[node_ids]
                        mse = float(np.mean((predicted_delta - observed_delta) ** 2))
                        zero_mse = float(np.mean(observed_delta**2))
                        cv_rows.append({
                            "train_processing": train_processing,
                            "method": method,
                            "alpha": alpha,
                            "lps_animal": lps_animal,
                            "saline_animal": saline_animal,
                            "mse": mse,
                            "zero_response_mse": zero_mse,
                            "relative_mse_reduction": 1.0 - mse / zero_mse,
                        })
        cv = pd.DataFrame([row for row in cv_rows if row["train_processing"] == train_processing])
        selection = cv.groupby(["method", "alpha"]).mse.mean().reset_index().sort_values("mse").iloc[0]
        method = selection.method
        alpha = float(selection.alpha)
        selections[train_processing] = {"method": method, "alpha": alpha, "inner_cv_mse": float(selection.mse)}

        test = [gsm for gsm, value in SAMPLES.items() if value[1] == test_processing]
        test_lps = [gsm for gsm in test if SAMPLES[gsm][0] == "lps"]
        test_saline = [gsm for gsm in test if SAMPLES[gsm][0] == "saline"]
        fit = fit_batch(
            feature_array(mono, world, train, node_ids, method),
            micro[train].to_numpy().T[:, node_ids],
            alpha,
        )
        predicted = predict_batch(feature_array(mono, world, test, node_ids, method), fit)
        predicted_frame = pd.DataFrame(predicted, index=test, columns=nodes[node_ids])
        for gsm in test_lps:
            predicted_delta = predicted_frame.loc[gsm].to_numpy() - predicted_frame.loc[test_saline].mean(axis=0).to_numpy()
            observed_delta = micro[gsm].to_numpy()[node_ids] - micro[test_saline].mean(axis=1).to_numpy()[node_ids]
            mse = float(np.mean((predicted_delta - observed_delta) ** 2))
            zero_mse = float(np.mean(observed_delta**2))
            test_rows.append({
                "train_processing": train_processing,
                "test_processing": test_processing,
                "selected_method": method,
                "selected_alpha": alpha,
                "gsm": gsm,
                "mse": mse,
                "zero_response_mse": zero_mse,
                "relative_mse_reduction": 1.0 - mse / zero_mse,
                "delta_pearson": delta_pearson(observed_delta, predicted_delta),
                "top100_overlap": top_k_overlap(observed_delta, predicted_delta, k=100),
            })
    cv_frame = pd.DataFrame(cv_rows)
    test_frame = pd.DataFrame(test_rows)
    cv_frame.to_csv(OUT / "inner_cv_metrics.csv", index=False)
    test_frame.to_csv(OUT / "test_metrics_by_animal.csv", index=False)
    (OUT / "selected_models.json").write_text(json.dumps(selections, indent=2) + "\n")
    summary = test_frame.groupby("train_processing").agg(
        mse=("mse", "mean"),
        zero_response_mse=("zero_response_mse", "mean"),
        relative_mse_reduction=("relative_mse_reduction", "mean"),
        delta_pearson=("delta_pearson", "mean"),
        top100_overlap=("top100_overlap", "mean"),
    ).reset_index()
    summary.to_csv(OUT / "test_metrics_summary.csv", index=False)
    result = {
        "status": "complete",
        "selection": "global method and alpha minimizing four paired leave-two-out folds within training animals",
        "test_used_for_selection": False,
        "selected_models": selections,
        "target_relative_mse_reduction": 0.50,
        "both_directions_met": bool((summary.relative_mse_reduction >= 0.50).all()),
        "runtime_seconds": time.time() - started,
    }
    (OUT / "run_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(selections, indent=2), flush=True)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
