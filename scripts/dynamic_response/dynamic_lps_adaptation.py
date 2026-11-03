import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.mixture import GaussianMixture

from evaluate_lps_brain_v2 import MICROGLIA_MARKERS, SAMPLES, load_sample, mouse_mapping
from headline_metrics import delta_pearson, top_k_overlap

ROOT = Path(os.environ.get("AD_OMICS_ROOT", "/public/home/mengxl/dzy/ad_omics_v3"))
WORK = Path(__file__).resolve().parent
OUT = WORK / "results/lps_dynamic_adaptation_v1"
MONOCYTE_MARKERS = ["Ly6c2", "Plac8", "Ccr2", "Fn1", "S100a8", "S100a9", "Lyz2", "Vcan", "F13a1"]


def normalized_log(matrix):
    total = np.asarray(matrix.sum(axis=1)).ravel()
    normalizer = sp.diags(np.divide(1e4, total, out=np.zeros(len(total)), where=total > 0))
    log = (normalizer @ matrix).tocsr()
    log.data = np.log1p(log.data)
    return log


def ridge_fit(x, y, alpha=1.0):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    z = (x - mean) / scale
    y_mean = y.mean()
    coefficient = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - y_mean))
    return mean, scale, y_mean, coefficient


def ridge_predict(x, fit):
    mean, scale, y_mean, coefficient = fit
    return y_mean + ((x - mean) / scale) @ coefficient


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(16)
    samples = {gsm: load_sample(gsm) for gsm in SAMPLES}
    all_scores = np.concatenate([sample["marker_score"] for sample in samples.values()])
    micro_mixture = GaussianMixture(n_components=2, random_state=42).fit(all_scores[:, None])
    micro_component = int(np.argmax(micro_mixture.means_.ravel()))

    genes = pd.read_csv(ROOT / "data/processed/genes.csv")
    members = pd.read_csv(ROOT / "data/reference/v2/mappings/orthogroups_member_level.csv")
    nodes = np.array(genes.node_id)
    micro_mean = {}
    mono_mean = {}
    mono_cells = {}
    detection = {}
    qc_rows = []
    for gsm, sample in samples.items():
        matrix = sample["matrix"]
        log = normalized_log(matrix)
        micro = micro_mixture.predict(sample["marker_score"][:, None]) == micro_component
        mono_score = np.asarray(log[:, [i for i, gene in enumerate(sample["genes"]) if gene.lower() in {m.lower() for m in MONOCYTE_MARKERS}]].mean(axis=1)).ravel()
        mono_mixture = GaussianMixture(n_components=2, random_state=42).fit(mono_score[~micro, None])
        mono_component = int(np.argmax(mono_mixture.means_.ravel()))
        mono = np.zeros(matrix.shape[0], dtype=bool)
        mono[~micro] = mono_mixture.predict(mono_score[~micro, None]) == mono_component
        mapping = mouse_mapping(genes, members, sample["genes"])
        for label, selected in [("micro", micro), ("mono", mono)]:
            selected_matrix = matrix[selected]
            og = (selected_matrix @ mapping).tocsr()
            log_og = normalized_log(og)
            mean = np.asarray(log_og.mean(axis=0)).ravel()
            if label == "micro":
                micro_mean[gsm] = mean
                detection[gsm] = np.asarray(og.getnnz(axis=0)).ravel() / max(og.shape[0], 1)
            else:
                mono_mean[gsm] = mean
                if len(mono_cells) == 0:
                    pass
                mono_cells[gsm] = og
        treatment, processing, animal = SAMPLES[gsm]
        qc_rows.append({"gsm": gsm, "treatment": treatment, "processing": processing, "animal": animal, "micro_cells": int(micro.sum()), "mono_cells": int(mono.sum())})
        print(f"CELLS {gsm} micro={micro.sum()} mono={mono.sum()}", flush=True)
    pd.DataFrame(qc_rows).to_csv(OUT / "sample_cell_counts.csv", index=False)

    v2 = Path("/public/home/mengxl/dzy/ad_omics_v2")
    sys.path.insert(0, str(v2 / "scripts/step1_world_model/cross_species"))
    from unified_model_v3 import Config, WorldModelV3
    import __main__
    __main__.Config = Config
    checkpoint = v2 / "models/world_model_v5_2_joint_member/model_real.pt"
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = WorldModelV3(len(ckpt["gene_names"]), ckpt["esm_matrix"], ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    brain_labels = ckpt["cond_layout"]["brain_align"]
    blood_labels = ckpt["cond_layout"]["blood_align"]
    species_labels = ckpt["cond_layout"]["species"]

    @torch.no_grad()
    def predict(expression, n):
        condition = np.zeros((n, len(brain_labels) + len(blood_labels) + len(species_labels)), dtype=np.float32)
        condition[:, brain_labels.index("Micro")] = 1
        condition[:, len(brain_labels) + blood_labels.index("Mono")] = 1
        condition[:, len(brain_labels) + len(blood_labels) + species_labels.index("mouse")] = 1
        expression = torch.as_tensor(expression, dtype=torch.float32)
        condition = torch.as_tensor(condition)
        latent, _ = model.encoder((expression - model.gene_mean) @ model.esm_matrix)
        transported = model.forward_ode(latent, "blood_to_brain", condition)
        return (model.dec_out(model.decoder_z(transported)) + model.gene_mean).numpy()

    world_mean = {}
    for gsm in SAMPLES:
        og = mono_cells[gsm]
        if og.shape[0] > 128:
            ids = np.random.default_rng(42).choice(og.shape[0], 128, replace=False)
            og = og[ids]
        log = normalized_log(og).toarray().astype(np.float32)
        world_mean[gsm] = predict(log, len(log)).mean(axis=0)
        print(f"WORLD {gsm}", flush=True)

    mask = (np.max(np.stack(list(detection.values())), axis=0) >= 0.01) & genes.kind.eq("shared_og").to_numpy()
    rows = []
    coefficient_records = {}
    for train_processing, test_processing in [("dmso", "inhibitors"), ("inhibitors", "dmso")]:
        train = [gsm for gsm, value in SAMPLES.items() if value[1] == train_processing]
        test = [gsm for gsm, value in SAMPLES.items() if value[1] == test_processing]
        test_lps = [gsm for gsm in test if SAMPLES[gsm][0] == "lps"]
        test_saline = [gsm for gsm in test if SAMPLES[gsm][0] == "saline"]
        node_ids = np.flatnonzero(mask)
        predictions = {method: {} for method in ["training_mean", "source_linear", "world_linear", "combined_linear"]}
        for node in node_ids:
            y = np.array([micro_mean[gsm][node] for gsm in train])
            source = np.array([mono_mean[gsm][node] for gsm in train])[:, None]
            world = np.array([world_mean[gsm][node] for gsm in train])[:, None]
            fits = {
                "source_linear": ridge_fit(source, y),
                "world_linear": ridge_fit(world, y),
                "combined_linear": ridge_fit(np.column_stack([source[:, 0], world[:, 0]]), y),
            }
            coefficient_records[train_processing, int(node)] = fits
            for method in predictions:
                if method == "training_mean":
                    values = {gsm: y.mean() for gsm in test}
                else:
                    if method == "source_linear":
                        test_x = np.array([mono_mean[gsm][node] for gsm in test])[:, None]
                    elif method == "world_linear":
                        test_x = np.array([world_mean[gsm][node] for gsm in test])[:, None]
                    else:
                        test_x = np.array([[mono_mean[gsm][node], world_mean[gsm][node]] for gsm in test])
                    values = dict(zip(test, ridge_predict(test_x, fits[method])))
                predictions[method][node] = values
        for method in predictions:
            saline_predicted = np.mean([predictions[method][node][gsm] for gsm in test_saline for node in []]) if False else None
            for gsm in test_lps:
                predicted_delta = np.zeros(len(nodes))
                observed_delta = np.zeros(len(nodes))
                for node in node_ids:
                    predicted_delta[node] = predictions[method][node][gsm] - np.mean([predictions[method][node][control] for control in test_saline])
                    observed_delta[node] = micro_mean[gsm][node] - np.mean([micro_mean[control][node] for control in test_saline])
                mse = float(np.mean((predicted_delta[mask] - observed_delta[mask]) ** 2))
                zero_mse = float(np.mean(observed_delta[mask] ** 2))
                rows.append({
                    "train_processing": train_processing,
                    "test_processing": test_processing,
                    "gsm": gsm,
                    "method": method,
                    "mse": mse,
                    "zero_response_mse": zero_mse,
                    "relative_mse_reduction": 1.0 - mse / zero_mse,
                    "delta_pearson": delta_pearson(observed_delta[mask], predicted_delta[mask]),
                    "top100_overlap": top_k_overlap(observed_delta[mask], predicted_delta[mask], k=100),
                })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "metrics_by_test_animal.csv", index=False)
    summary = metrics.groupby(["train_processing", "method"]).agg(
        mse=("mse", "mean"),
        zero_response_mse=("zero_response_mse", "mean"),
        relative_mse_reduction=("relative_mse_reduction", "mean"),
        delta_pearson=("delta_pearson", "mean"),
        top100_overlap=("top100_overlap", "mean"),
    ).reset_index()
    summary.to_csv(OUT / "metrics_summary.csv", index=False)
    primary = float(summary.loc[(summary.train_processing == "dmso") & (summary.method == "combined_linear"), "relative_mse_reduction"].item())
    result = {
        "status": "complete",
        "n_response_nodes": int(mask.sum()),
        "primary_training": "four DMSO animals",
        "primary_test": "four inhibitor-treated animals; MSE evaluated in the two LPS animals versus matched saline",
        "target_relative_mse_reduction": 0.50,
        "primary_combined_relative_mse_reduction": primary,
        "target_met": primary >= 0.50,
        "runtime_seconds": time.time() - started,
    }
    (OUT / "run_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
