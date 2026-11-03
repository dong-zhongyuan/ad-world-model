import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(os.environ.get("AD_OMICS_ROOT", "/public/home/mengxl/dzy/ad_omics_v3"))
WORK = Path(__file__).resolve().parent
DATA = WORK / "data/myeloid/processed"
OUT = WORK / "results/blood_actions_v1"
LANES = ["Mac_L01", "Mac_L02", "Mac_L03", "Mac_L04"]


def log2fc(case, control):
    case = np.asarray(case, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    n = case.shape[-1]
    case_probability = (case + 0.5) / (case.sum(axis=-1, keepdims=True) + 0.5 * n)
    control_probability = (control + 0.5) / (control.sum(axis=-1, keepdims=True) + 0.5 * n)
    return np.log2(case_probability) - np.log2(control_probability)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
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

    first = np.load(DATA / "Mac_L01_Ctrl_pseudobulk.npz", allow_pickle=True)
    human_genes = first["genes"].astype(str)
    first.close()
    rows = []
    columns = []
    for j, gene in enumerate(human_genes):
        for node in gene_nodes.get(gene.upper(), set()):
            rows.append(j)
            columns.append(node)
    mapping = sp.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(human_genes), len(genes)))

    targets = sorted(set().union(*[set(np.load(DATA / f"{lane}_{condition}_pseudobulk.npz", allow_pickle=True)["targets"].astype(str)) for lane in LANES for condition in ["Ctrl", "LPS"]]))
    target_index = {target: index for index, target in enumerate(targets)}
    action_untreated = np.full((len(LANES), len(targets), len(genes)), np.nan, dtype=np.float32)
    action_lps = np.full_like(action_untreated, np.nan)
    lps_response = np.full((len(LANES), len(genes)), np.nan, dtype=np.float32)
    n_cells = np.zeros((len(LANES), len(targets), 2), dtype=np.int32)

    for lane_index, lane in enumerate(LANES):
        samples = {}
        for condition in ["Ctrl", "LPS"]:
            with np.load(DATA / f"{lane}_{condition}_pseudobulk.npz", allow_pickle=True) as data:
                sample_targets = data["targets"].astype(str)
                counts = data["sum_counts"]
                cells = data["n_cells"]
                og_counts = counts @ mapping
                og_counts = np.asarray(og_counts)
                samples[condition] = {"targets": sample_targets, "counts": og_counts, "cells": cells}
        for condition_index, condition in enumerate(["Ctrl", "LPS"]):
            sample = samples[condition]
            ntc_position = list(sample["targets"]).index("NTC")
            ntc_counts = sample["counts"][ntc_position]
            for target in sample["targets"]:
                i = target_index[target]
                j = list(sample["targets"]).index(target)
                values = log2fc(sample["counts"][j], ntc_counts).astype(np.float32)
                if condition == "Ctrl":
                    action_untreated[lane_index, i] = values
                else:
                    action_lps[lane_index, i] = values
                n_cells[lane_index, i, condition_index] = sample["cells"][j]
        ctrl_ntc = samples["Ctrl"]["counts"][list(samples["Ctrl"]["targets"]).index("NTC")]
        lps_ntc = samples["LPS"]["counts"][list(samples["LPS"]["targets"]).index("NTC")]
        lps_response[lane_index] = log2fc(lps_ntc, ctrl_ntc).astype(np.float32)

    np.savez_compressed(
        OUT / "blood_action_log2fc.npz",
        nodes=np.array(genes.node_id),
        targets=np.array(targets),
        lanes=np.array(LANES),
        action_untreated=action_untreated,
        action_lps=action_lps,
        lps_response=lps_response,
        n_cells=n_cells,
    )
    lane_correlations = np.corrcoef(lps_response)
    summary = {
        "status": "complete",
        "n_nodes": len(genes),
        "n_targets": len(targets),
        "n_lanes": len(LANES),
        "scale": "log2 aggregated-count fold change versus matched NTC",
        "pseudocount": 0.5,
        "lps_lane_pearson_min": float(lane_correlations[np.triu_indices(len(LANES), 1)].min()),
        "lps_lane_pearson_max": float(lane_correlations[np.triu_indices(len(LANES), 1)].max()),
        "lps_lane_pearson_mean": float(lane_correlations[np.triu_indices(len(LANES), 1)].mean()),
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
