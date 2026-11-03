import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread

from evaluate_lps_brain_v2 import mouse_mapping
from headline_metrics import delta_pearson, top_k_overlap

ROOT = Path(os.environ.get("AD_OMICS_ROOT", "/public/home/mengxl/dzy/ad_omics_v3"))
WORK = Path(__file__).resolve().parent
RAW = WORK / "data/grn_brain/raw"
OUT = WORK / "results/grn_brain_response_v1"


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    config = pd.read_csv(WORK / "data/grn_brain/archives/GSE272377_feature_config.csv.gz")
    config["is_ko"] = config.genotype.str.contains("Grn KO")
    genes = pd.read_csv(ROOT / "data/processed/genes.csv")
    members = pd.read_csv(ROOT / "data/reference/v2/mappings/orthogroups_member_level.csv")
    nodes = np.array(genes.node_id)
    node_index = {node: index for index, node in enumerate(nodes)}
    mean_log = {}
    detection = {}
    qc_rows = []

    for row in config.itertuples():
        sample = row.sample_id
        features = pd.read_csv(RAW / sample / "features.tsv.gz", sep="\t", header=None)
        source_genes = features.iloc[:, 1].astype(str).to_numpy()
        barcodes = pd.read_csv(RAW / sample / "barcodes.tsv.gz", sep="\t", header=None).iloc[:, 0].astype(str)
        matrix = mmread(RAW / sample / "matrix.mtx.gz").tocsr().T.tocsr()
        total = np.asarray(matrix.sum(axis=1)).ravel()
        n_genes = np.asarray(matrix.getnnz(axis=1)).ravel()
        mt = np.flatnonzero(pd.Series(source_genes).str.upper().str.startswith("MT-").to_numpy())
        mt_counts = np.asarray(matrix[:, mt].sum(axis=1)).ravel()
        mt_percent = np.divide(mt_counts * 100, total, out=np.zeros(matrix.shape[0]), where=total > 0)
        quality = (n_genes >= 200) & (mt_percent <= 20) & (total > 0)
        matrix = matrix[quality].tocsr()
        mapping = mouse_mapping(genes, members, source_genes)
        og = (matrix @ mapping).tocsr()
        totals = np.asarray(og.sum(axis=1)).ravel()
        normalizer = sp.diags(np.divide(1e4, totals, out=np.zeros(len(totals)), where=totals > 0))
        log = (normalizer @ og).tocsr()
        log.data = np.log1p(log.data)
        mean_log[sample] = np.asarray(log.mean(axis=0)).ravel()
        detection[sample] = np.asarray(og.getnnz(axis=0)).ravel() / max(og.shape[0], 1)
        qc_rows.append({"sample": sample, "sex": row.sex, "is_ko": row.is_ko, "input_cells": len(barcodes), "quality_cells": int(quality.sum())})
        print(f"LOADED {sample} KO={row.is_ko} cells={int(quality.sum())}", flush=True)
    pd.DataFrame(qc_rows).to_csv(OUT / "sample_qc.csv", index=False)

    observed = {}
    for row in config.loc[config.is_ko].itertuples():
        controls = config.loc[(~config.is_ko) & (config.sex == row.sex)].sample_id.tolist()
        control_mean = np.mean([mean_log[sample] for sample in controls], axis=0)
        observed[row.sample_id] = mean_log[row.sample_id] - control_mean
    detected = np.max(np.stack(list(detection.values())), axis=0) >= 0.01
    mask = detected & genes.kind.eq("shared_og").to_numpy()
    np.savez_compressed(OUT / "observed_grn_ko_responses.npz", nodes=nodes, mask=mask, **{f"delta__{sample}": value for sample, value in observed.items()})

    with np.load(WORK / "results/cd4_real_action_brain_v1/brain_response_predictions.npz", allow_pickle=True) as predictions:
        model_nodes = predictions["nodes"].astype(str)
        targets = predictions["targets"].astype(str)
        conditions = predictions["conditions"].astype(str)
        model_response = predictions["response__Micro"][list(targets).index("GRN")]
    if list(model_nodes) != list(nodes):
        aligned = np.zeros((model_response.shape[0], len(nodes)), dtype=np.float32)
        for j, node in enumerate(model_nodes):
            aligned[:, node_index[node]] = model_response[:, j]
        model_response = aligned
    model_mean = model_response.mean(axis=0)

    rows = []
    for sample, truth in observed.items():
        sex = config.loc[config.sample_id.eq(sample), "sex"].item()
        for condition_index, condition in enumerate(conditions):
            for mode, predicted in [("condition", model_response[condition_index]), ("condition_mean", model_mean)]:
                mse = float(np.mean((predicted[mask] - truth[mask]) ** 2))
                zero_mse = float(np.mean(truth[mask] ** 2))
                rows.append({
                    "sample": sample,
                    "sex": sex,
                    "condition": condition,
                    "mode": mode,
                    "mse": mse,
                    "zero_response_mse": zero_mse,
                    "relative_mse_reduction": 1.0 - mse / zero_mse,
                    "delta_pearson": delta_pearson(truth[mask], predicted[mask]),
                    "top100_overlap": top_k_overlap(truth[mask], predicted[mask], k=100),
                    "n_nodes": int(mask.sum()),
                })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "metrics_by_mouse_condition.csv", index=False)
    summary = metrics.groupby(["condition", "mode"]).agg(
        mse=("mse", "mean"),
        zero_response_mse=("zero_response_mse", "mean"),
        relative_mse_reduction=("relative_mse_reduction", "mean"),
        delta_pearson=("delta_pearson", "mean"),
        top100_overlap=("top100_overlap", "mean"),
    ).reset_index()
    summary.to_csv(OUT / "metrics_summary.csv", index=False)
    primary_reduction = float(summary.loc[summary["mode"].eq("condition_mean"), "relative_mse_reduction"].mean())
    result = {
        "status": "complete",
        "n_ko_mice": len(observed),
        "n_control_mice": int((~config.is_ko).sum()),
        "n_response_nodes": int(mask.sum()),
        "comparison": "sex-matched observed Grn-KO microglia response versus measured peripheral CD4 GRN-action prediction",
        "target_relative_mse_reduction": 0.50,
        "condition_mean_relative_mse_reduction": primary_reduction,
        "target_met": primary_reduction >= 0.50,
        "runtime_seconds": time.time() - started,
    }
    (OUT / "run_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
