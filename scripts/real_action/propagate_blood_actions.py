import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("AD_OMICS_ROOT", "/public/home/mengxl/dzy/ad_omics_v3"))
WORK = Path(__file__).resolve().parent
OUT = WORK / "results/real_action_brain_v1"
sys.path.insert(0, str(ROOT / "scripts/step1_world_model"))
from data_io import Dataset
from inference import Predictor

CANDIDATES = ["ACP5", "CR1", "GNAS"]
CONDITIONS = ["untreated", "LPS"]


def apply_log2fc(x, delta):
    shifted = np.expm1(x, dtype=np.float64) * np.exp2(np.asarray(delta, dtype=np.float64))
    if not np.isfinite(shifted).all():
        raise ValueError("Non-finite shifted expression")
    return np.log1p(shifted).astype(np.float32)


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(16)
    data = Dataset()
    obs = data.data["human_blood"]["obs"]
    ids = np.flatnonzero((obs.celltype == "Mono") & (obs.disease == "CTRL") & (obs.split == "train"))
    if len(ids) > 128:
        ids = np.random.default_rng(42).choice(ids, 128, replace=False)
    x = data.data["human_blood"]["x"][ids].toarray().astype(np.float32)
    assays = obs.iloc[ids].assay.unique()
    if len(assays) != 1:
        raise ValueError("Multiple source assays")
    assay = assays[0]
    checkpoint = ROOT / "runs/refinement/joint_full/seed_42/last.pt"
    predictor = Predictor(checkpoint, device="cpu")
    with np.load(WORK / "results/blood_actions_v1/blood_action_log2fc.npz", allow_pickle=True) as actions:
        nodes = actions["nodes"]
        targets = actions["targets"].astype(str)
        lanes = actions["lanes"].astype(str)
        action_untreated = actions["action_untreated"]
        action_lps = actions["action_lps"]
        lps_response = actions["lps_response"]
    if list(nodes) != predictor.genes:
        raise ValueError("Action node order differs from the frozen model")
    target_index = {target: index for index, target in enumerate(targets)}

    predictions = {target_type: {"lps": [], "candidate": []} for target_type in ["Micro", "Oligo"]}
    for target_type in ["Micro", "Oligo"]:
        for lane_index, lane in enumerate(lanes):
            baseline = predictor.predict(x, predictor.genes, "human", "CTRL", "Mono", target_type, assay)["brain_log1p_cp10k"]
            lps = predictor.predict(apply_log2fc(x, lps_response[lane_index]), predictor.genes, "human", "CTRL", "Mono", target_type, assay)["brain_log1p_cp10k"]
            predictions[target_type]["lps"].append((lps - baseline).mean(axis=0))
        candidate_array = np.zeros((len(CANDIDATES), len(lanes), len(CONDITIONS), len(nodes)), dtype=np.float32)
        for candidate_index, candidate in enumerate(CANDIDATES):
            target = target_index[candidate]
            for lane_index, lane in enumerate(lanes):
                for condition_index, condition in enumerate(CONDITIONS):
                    if condition == "untreated":
                        baseline_delta = np.zeros(len(nodes), dtype=np.float32)
                        action = action_untreated[lane_index, target]
                    else:
                        baseline_delta = lps_response[lane_index]
                        action = action_lps[lane_index, target]
                    baseline = predictor.predict(apply_log2fc(x, baseline_delta), predictor.genes, "human", "CTRL", "Mono", target_type, assay)["brain_log1p_cp10k"]
                    perturbed = predictor.predict(apply_log2fc(x, baseline_delta + action), predictor.genes, "human", "CTRL", "Mono", target_type, assay)["brain_log1p_cp10k"]
                    candidate_array[candidate_index, lane_index, condition_index] = (perturbed - baseline).mean(axis=0)
        predictions[target_type]["candidate"] = candidate_array
        predictions[target_type]["lps"] = np.stack(predictions[target_type]["lps"])

    save = {
        "nodes": nodes,
        "lanes": lanes,
        "candidates": np.array(CANDIDATES),
        "conditions": np.array(CONDITIONS),
    }
    for target_type in ["Micro", "Oligo"]:
        save[f"lps__{target_type}"] = predictions[target_type]["lps"]
        save[f"candidate__{target_type}"] = predictions[target_type]["candidate"]
    np.savez_compressed(OUT / "brain_response_predictions.npz", **save)
    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "source": "GSE181279 control monocytes with measured myeloid LPS/perturbation log2 fold changes",
        "n_source_cells": len(ids),
        "source_assay": assay,
        "disease_condition": "CTRL for both baseline and action",
        "targets": CANDIDATES,
        "routes": ["Mono_to_Micro", "Mono_to_Oligo"],
        "runtime_seconds": time.time() - started,
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
