import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_lps_brain_v2 import SAMPLES
from tune_dynamic_lps import feature_array, fit_batch, predict_batch

WORK = Path(__file__).resolve().parent
DATA = WORK / "results/lps_dynamic_adaptation_v2/state_means.npz"
OUT = WORK / "results/lps_dynamic_scene_v1"


def main():
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
    selected_nodes = nodes[node_ids]

    observed = []
    predicted = []
    sample_rows = []
    for train_processing, test_processing in [("dmso", "inhibitors"), ("inhibitors", "dmso")]:
        train = [gsm for gsm, value in SAMPLES.items() if value[1] == train_processing]
        test = [gsm for gsm, value in SAMPLES.items() if value[1] == test_processing]
        test_lps = [gsm for gsm in test if SAMPLES[gsm][0] == "lps"]
        test_saline = [gsm for gsm in test if SAMPLES[gsm][0] == "saline"]
        fit = fit_batch(feature_array(mono, world, train, node_ids, "combined"), micro[train].to_numpy().T[:, node_ids], 1.0)
        absolute = predict_batch(feature_array(mono, world, test, node_ids, "combined"), fit)
        absolute = pd.DataFrame(absolute, index=test, columns=selected_nodes)
        for gsm in test_lps:
            observed_delta = micro[gsm].to_numpy()[node_ids] - micro[test_saline].mean(axis=1).to_numpy()[node_ids]
            predicted_delta = absolute.loc[gsm].to_numpy() - absolute.loc[test_saline].mean(axis=0).to_numpy()
            observed.append(observed_delta)
            predicted.append(predicted_delta)
            sample_rows.append({"gsm": gsm, "train_processing": train_processing, "test_processing": test_processing})
    observed = np.stack(observed)
    predicted = np.stack(predicted)
    sample_frame = pd.DataFrame(sample_rows)
    np.savez_compressed(
        OUT / "dynamic_response_vectors.npz",
        nodes=selected_nodes,
        observed=observed,
        predicted=predicted,
        gsm=sample_frame.gsm.to_numpy().astype(str),
        train_processing=sample_frame.train_processing.to_numpy().astype(str),
        test_processing=sample_frame.test_processing.to_numpy().astype(str),
    )
    gene_table = pd.DataFrame({
        "node": selected_nodes,
        "symbols": genes.set_index("node_id").loc[selected_nodes].symbols.astype(str).to_numpy(),
        "observed_mean": observed.mean(axis=0),
        "predicted_mean": predicted.mean(axis=0),
        "observed_abs_mean": np.abs(observed).mean(axis=0),
        "predicted_abs_mean": np.abs(predicted).mean(axis=0),
        "response_pearson": [np.corrcoef(observed[:, j], predicted[:, j])[0, 1] if np.std(observed[:, j]) > 0 and np.std(predicted[:, j]) > 0 else np.nan for j in range(len(selected_nodes))],
    })
    gene_table["max_abs_response"] = gene_table[["observed_mean", "predicted_mean"]].abs().max(axis=1)
    gene_table.sort_values("max_abs_response", ascending=False).to_csv(OUT / "response_gene_table.csv", index=False)
    gene_table.head(200).to_csv(OUT / "top200_response_genes.csv", index=False)

    original = pd.read_csv(WORK / "results/lps_dynamic_adaptation_v1/metrics_by_test_animal.csv")
    original = original.loc[original.method.eq("combined_linear")].set_index("gsm")
    recomputed_mse = np.mean((observed - predicted) ** 2, axis=1)
    expected_mse = original.loc[sample_frame.gsm].mse.to_numpy()
    max_difference = float(np.max(np.abs(recomputed_mse - expected_mse)))
    if max_difference > 1e-10:
        raise ValueError(f"Response export changed MSE: {max_difference}")
    zero_mse = np.mean(observed**2)
    result = {
        "status": "complete",
        "n_nodes": len(selected_nodes),
        "n_test_lps_animals": len(sample_frame),
        "max_mse_difference_from_original": max_difference,
        "relative_mse_reduction": 1.0 - float(np.mean((observed - predicted) ** 2)) / float(zero_mse),
        "mean_delta_pearson": float(np.mean([np.corrcoef(observed[i], predicted[i])[0, 1] for i in range(len(observed))])),
    }
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
