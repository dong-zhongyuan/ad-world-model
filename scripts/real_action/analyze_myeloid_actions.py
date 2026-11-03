import json
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent
DATA = WORK / "data/myeloid/processed"
OUT = WORK / "results/myeloid_action_v1"
LANES = ["Mac_L01", "Mac_L02", "Mac_L03", "Mac_L04"]
CANDIDATES = ["CR1", "ACP5", "GNAS"]


def load_sample(lane, condition):
    with np.load(DATA / f"{lane}_{condition}_pseudobulk.npz", allow_pickle=True) as data:
        return {
            "targets": data["targets"],
            "genes": data["genes"],
            "mean_log": data["mean_log"],
            "sum_counts": data["sum_counts"],
            "n_cells": data["n_cells"],
        }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    samples = {(lane, condition): load_sample(lane, condition) for lane in LANES for condition in ["Ctrl", "LPS"]}
    genes = samples[LANES[0], "Ctrl"]["genes"]
    for sample in samples.values():
        assert np.array_equal(sample["genes"], genes)
    all_targets = sorted(set().union(*[set(sample["targets"]) for sample in samples.values()]))
    target_index = {target: index for index, target in enumerate(all_targets)}
    gene_index = {gene: index for index, gene in enumerate(genes)}

    action_untreated = np.full((len(LANES), len(all_targets), len(genes)), np.nan, dtype=np.float32)
    action_lps = np.full_like(action_untreated, np.nan)
    lps_response = np.full((len(LANES), len(genes)), np.nan, dtype=np.float32)
    n_cells = np.zeros((len(LANES), len(all_targets), 2), dtype=np.int32)
    coverage = np.zeros((len(LANES), len(all_targets), 2), dtype=bool)

    for lane_index, lane in enumerate(LANES):
        for condition_index, condition in enumerate(["Ctrl", "LPS"]):
            sample = samples[lane, condition]
            if "NTC" not in sample["targets"]:
                raise ValueError(f"NTC missing in {lane} {condition}")
            ntc = sample["mean_log"][list(sample["targets"]).index("NTC")]
            for target in sample["targets"]:
                target_position = list(sample["targets"]).index(target)
                global_position = target_index[target]
                values = sample["mean_log"][target_position] - ntc
                if condition == "Ctrl":
                    action_untreated[lane_index, global_position] = values
                else:
                    action_lps[lane_index, global_position] = values
                n_cells[lane_index, global_position, condition_index] = sample["n_cells"][target_position]
                coverage[lane_index, global_position, condition_index] = True
            ntc_lps = samples[lane, "LPS"]["mean_log"][list(samples[lane, "LPS"]["targets"]).index("NTC")]
            ntc_ctrl = samples[lane, "Ctrl"]["mean_log"][list(samples[lane, "Ctrl"]["targets"]).index("NTC")]
            lps_response[lane_index] = ntc_lps - ntc_ctrl

    np.savez_compressed(
        OUT / "myeloid_action_vectors.npz",
        targets=np.array(all_targets),
        genes=genes,
        lanes=np.array(LANES),
        action_untreated=action_untreated,
        action_lps=action_lps,
        lps_response=lps_response,
        n_cells=n_cells,
        coverage=coverage,
    )

    rows = []
    for target in all_targets:
        target_gene_position = gene_index.get(target)
        for condition, vectors in [("untreated", action_untreated), ("LPS", action_lps)]:
            for lane_index, lane in enumerate(LANES):
                values = vectors[lane_index, target_index[target]]
                if not np.isfinite(values).any():
                    continue
                finite = np.isfinite(values)
                rows.append({
                    "target": target,
                    "condition": condition,
                    "lane": lane,
                    "n_cells": int(n_cells[lane_index, target_index[target], 0 if condition == "untreated" else 1]),
                    "action_rms": float(np.sqrt(np.nanmean(values**2))),
                    "n_response_genes_abs_gt_0.5": int(np.sum(np.abs(values) > 0.5)),
                    "on_target_delta": float(values[target_gene_position]) if target_gene_position is not None and finite[target_gene_position] else np.nan,
                })
    actions = pd.DataFrame(rows)
    actions.to_csv(OUT / "action_by_lane.csv", index=False)
    candidate = actions.loc[actions.target.isin(CANDIDATES)].copy()
    candidate.to_csv(OUT / "candidate_actions_by_lane.csv", index=False)

    lps_rows = []
    for lane_index, lane in enumerate(LANES):
        for gene_index_position, gene in enumerate(genes):
            lps_rows.append({"lane": lane, "gene": gene, "delta_log_expression": float(lps_response[lane_index, gene_index_position])})
    pd.DataFrame(lps_rows).to_csv(OUT / "ntc_lps_response_by_lane.csv.gz", index=False, compression="gzip")

    summary = {
        "status": "complete",
        "n_lanes": len(LANES),
        "n_targets": len(all_targets),
        "n_genes": len(genes),
        "candidate_targets_present": {target: bool(target in target_index) for target in CANDIDATES},
        "biological_replicate_claim": False,
        "unit": "lane-level response; lanes are not relabeled as donors",
        "lps_response_rms_by_lane": {lane: float(np.sqrt(np.mean(lps_response[i] ** 2))) for i, lane in enumerate(LANES)},
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(candidate.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
