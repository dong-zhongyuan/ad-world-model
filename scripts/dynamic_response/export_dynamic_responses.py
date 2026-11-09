import json
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import CONFIG, RESULTS, configured_path

DATA = RESULTS / 'adaptation'
OUT = RESULTS / 'scene'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(DATA/'state_means.npz', allow_pickle=False) as data:
        nodes = data["nodes"].astype(str)
    genes = pd.read_csv(configured_path('genes_file'))
    with np.load(DATA/'heads.npz',allow_pickle=False) as heads:
        node_ids=heads['node_ids']
    selected_nodes = nodes[node_ids]
    folds=json.loads((DATA/'folds.json').read_text())
    responses=dict(np.load(DATA/'responses.npz',allow_pickle=False))

    observed = []
    predicted = []
    sample_rows = []
    for train_processing, info in folds.items():
        test_processing=info['test_processing']
        for gsm in info['test_lps']:
            observed_delta = responses[f'{train_processing}__{gsm}__observed']
            predicted_delta = responses[f'{train_processing}__{gsm}__combined_linear']
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
    gene_table=gene_table.sort_values("max_abs_response", ascending=False)
    gene_table.to_csv(OUT / "response_gene_table.csv", index=False)
    gene_table.head(200).to_csv(OUT / "top200_response_genes.csv", index=False)

    original = pd.read_csv(DATA / "metrics_by_test_animal.csv")
    original = original.loc[original.method.eq("combined_linear")].set_index("gsm")
    recomputed_mse = np.mean((observed - predicted) ** 2, axis=1)
    expected_mse = original.loc[sample_frame.gsm].mse.to_numpy()
    max_difference = float(np.max(np.abs(recomputed_mse - expected_mse)))
    if max_difference > 1e-10:
        raise ValueError(f"Response export changed MSE: {max_difference}")
    zero_mse = np.mean(observed**2)
    result = {
        "status": "complete",
        "model": "official_v3_1500_plus_fixed_ridge",
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
