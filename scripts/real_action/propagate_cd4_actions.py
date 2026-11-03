import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

ROOT = Path(os.environ.get("AD_OMICS_ROOT", "/public/home/mengxl/dzy/ad_omics_v3"))
WORK = Path(__file__).resolve().parent
OUT = WORK / "results/cd4_real_action_brain_v1"
sys.path.insert(0, str(ROOT / "scripts/step1_world_model"))
from data_io import Dataset
from inference import Predictor
from propagate_blood_actions import apply_log2fc

CANDIDATES = ["BIN1", "GRN", "PILRB", "BST2", "ST13", "ACP5", "CD55", "CR1", "GNAS"]
CONDITIONS = ["Rest", "Stim8hr", "Stim48hr"]


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(16)
    genes = pd.read_csv(ROOT / "data/processed/genes.csv")
    members = pd.read_csv(ROOT / "data/reference/v2/mappings/orthogroups_member_level.csv")
    og_to_nodes = genes.reset_index().groupby("og_id").indices
    gene_nodes = defaultdict(set)
    for row in members.itertuples():
        for symbol in str(row.human_symbols).split(";"):
            if symbol and symbol != "nan":
                gene_nodes[symbol.upper()].update(og_to_nodes.get(row.og_id, []))
    for index, row in enumerate(genes.itertuples()):
        if row.kind == "human_specific":
            for symbol in str(row.symbols).split("|"):
                gene_nodes[symbol.upper()].add(index)

    actions = {}
    with h5py.File(WORK / "data/cd4/GWCD4i.DE_stats.h5ad", "r") as handle:
        obs = ad.io.read_elem(handle["obs"])
        var = ad.io.read_elem(handle["var"])
        response_genes = np.asarray(var.gene_name).astype(str)
        log_fc = handle["layers/log_fc"]
        base_mean = handle["layers/baseMean"]
        for condition in CONDITIONS:
            rows = np.flatnonzero(np.asarray(obs.culture_condition == condition))
            for target in CANDIDATES:
                selected = [row for row in rows if obs.target_contrast_gene_name.iloc[row] == target]
                if len(selected) != 1:
                    raise ValueError(f"Expected one row for {target} {condition}, found {len(selected)}")
                row = selected[0]
                fc = log_fc[row, :].astype(float)
                base = base_mean[row, :].astype(float)
                matrix_rows = []
                matrix_columns = []
                weights = []
                for j, gene in enumerate(response_genes):
                    if not np.isfinite(fc[j]) or not np.isfinite(base[j]):
                        continue
                    nodes = gene_nodes.get(gene.upper(), set())
                    for node in nodes:
                        matrix_rows.append(j)
                        matrix_columns.append(node)
                        weights.append(max(base[j], 0.0))
                mapping = sp.csr_matrix((weights, (matrix_rows, matrix_columns)), shape=(len(response_genes), len(genes)))
                mapping = mapping @ sp.diags(1.0 / np.maximum(np.asarray(mapping.sum(axis=0)).ravel(), 1e-12))
                actions[target, condition] = np.asarray(fc @ mapping).ravel().astype(np.float32)

    data = Dataset()
    blood_obs = data.data["human_blood"]["obs"]
    ids = np.flatnonzero((blood_obs.celltype == "CD4T") & (blood_obs.disease == "CTRL") & (blood_obs.split == "train"))
    if len(ids) > 128:
        ids = np.random.default_rng(42).choice(ids, 128, replace=False)
    x = data.data["human_blood"]["x"][ids].toarray().astype(np.float32)
    assays = blood_obs.iloc[ids].assay.unique()
    if len(assays) != 1:
        raise ValueError("Multiple source assays")
    assay = assays[0]
    checkpoint = ROOT / "runs/refinement/joint_full/seed_42/last.pt"
    predictor = Predictor(checkpoint, device="cpu")
    save = {
        "nodes": np.array(predictor.genes),
        "targets": np.array(CANDIDATES),
        "conditions": np.array(CONDITIONS),
    }
    for target_type in ["Micro", "Oligo"]:
        baseline = predictor.predict(x, predictor.genes, "human", "CTRL", "CD4T", target_type, assay)["brain_log1p_cp10k"]
        result = np.zeros((len(CANDIDATES), len(CONDITIONS), len(predictor.genes)), dtype=np.float32)
        for target_index, target in enumerate(CANDIDATES):
            for condition_index, condition in enumerate(CONDITIONS):
                shifted = apply_log2fc(x, actions[target, condition])
                predicted = predictor.predict(shifted, predictor.genes, "human", "CTRL", "CD4T", target_type, assay)["brain_log1p_cp10k"]
                result[target_index, condition_index] = (predicted - baseline).mean(axis=0)
        save[f"response__{target_type}"] = result
    np.savez_compressed(OUT / "brain_response_predictions.npz", **save)
    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "source": "GSE181279 control CD4 T cells with measured GSE314342 target log2 fold changes",
        "n_source_cells": len(ids),
        "source_assay": assay,
        "targets": CANDIDATES,
        "conditions": CONDITIONS,
        "routes": ["CD4T_to_Micro", "CD4T_to_Oligo"],
        "uses_trained_action_model_prediction": False,
        "runtime_seconds": time.time() - started,
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
