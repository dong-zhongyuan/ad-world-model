import json
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent
OUT = WORK / "results/integrated_target_map_v1"
FIG = Path("/public/home/mengxl/dzy/ad_omics_v3/analysis/figure_build_20261031/data")
TARGETS = ["GRN", "BIN1", "BLNK", "CR1", "PILRB", "CD55", "ACP5", "ST13", "GNAS", "BST2"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    molecules = pd.read_csv(FIG / "representative_molecules.csv").set_index("target")
    gates = pd.read_csv(FIG / "evidence_gates.csv")
    cd4 = pd.read_csv(WORK / "cd4_target_coverage.csv")
    cd4_response_path = WORK / "results/cd4_real_action_brain_v1/brain_response_predictions.npz"
    with np.load(cd4_response_path, allow_pickle=True) as data:
        cd4_targets = data["targets"].astype(str)
        cd4_conditions = data["conditions"].astype(str)
        cd4_response = data["response__Micro"]
    myeloid = pd.read_csv(WORK / "results/myeloid_action_v1/candidate_actions_by_lane.csv")
    grn_markers = pd.read_csv(WORK / "results/grn_route_v1/grn_marker_response.csv")
    bin1_markers = pd.read_csv(WORK / "results/bin1_route_v1/bin1_marker_responses.csv")

    rows = []
    for target in TARGETS:
        target_gates = gates.loc[gates.protein.eq(target)]
        target_cd4 = cd4.loc[cd4.target_contrast_gene_name.eq(target)]
        target_myeloid = myeloid.loc[myeloid.target.eq(target)]
        row = {
            "target": target,
            "story_role": "lead" if target == "GRN" else "complementary_mechanism" if target == "BIN1" else "intervention_map",
            "network_hub_top_percent_min": target_gates.hub_top_percent.min(),
            "mr_bell_pass": bool(target_gates.mr_bell.eq("pass").any()),
            "mr_finn_pass": bool(target_gates.mr_finn.eq("pass").any()),
            "direction_outcome_pass": bool(target_gates.direction_outcome.eq("pass").any()),
            "coloc_bell_max": target_gates.bell_h4.max(),
            "coloc_finn_max": target_gates.finn_h4.max(),
            "drugclip_pass": bool(target_gates.drugclip.eq("pass").any()),
            "docking_qc_pass": bool(target_gates.docking_qc.eq("pass").any()),
            "cd4_perturbation_present": not target_cd4.empty,
            "cd4_on_target_significant_any": bool(target_cd4.ontarget_significant.any()) if not target_cd4.empty else False,
            "cd4_max_abs_on_target_logfc": target_cd4.ontarget_effect_size.abs().max() if not target_cd4.empty else np.nan,
            "cd4_max_total_de_genes": target_cd4.n_total_de_genes.max() if not target_cd4.empty else np.nan,
            "myeloid_perturbation_present": not target_myeloid.empty,
            "myeloid_n_lanes": target_myeloid.lane.nunique() if not target_myeloid.empty else 0,
            "myeloid_mean_on_target_delta": target_myeloid.on_target_delta.mean() if not target_myeloid.empty else np.nan,
            "independent_brain_dataset": "GSE272377" if target == "GRN" else "Mendeley:whgkcy72yv" if target == "BIN1" else "",
        }
        if target in set(cd4_targets):
            values = cd4_response[list(cd4_targets).index(target)]
            row["cd4_real_action_brain_rms_by_condition"] = ";".join(f"{condition}:{float(np.sqrt(np.mean(values[i] ** 2))):.6g}" for i, condition in enumerate(cd4_conditions))
        if target == "GRN":
            row["brain_GPNMB_response"] = grn_markers.loc[grn_markers.marker.eq("Gpnmb"), "mean_ko_response"].mean()
            row["brain_CTSB_response"] = grn_markers.loc[grn_markers.marker.eq("Ctsb"), "mean_ko_response"].mean()
            row["brain_APOE_response"] = grn_markers.loc[grn_markers.marker.eq("Apoe"), "mean_ko_response"].mean()
        if target == "BIN1":
            row["brain_CX3CR1_response"] = bin1_markers.loc[bin1_markers.marker.eq("Cx3cr1"), "Bin1cKO_baseline"].mean()
            row["brain_TREM2_response"] = bin1_markers.loc[bin1_markers.marker.eq("Trem2"), "Bin1cKO_baseline"].mean()
            row["brain_CST3_response"] = bin1_markers.loc[bin1_markers.marker.eq("Cst3"), "Bin1cKO_baseline"].mean()
        if target in molecules.index:
            molecule = molecules.loc[target]
            row.update({
                "compound": molecule["name"],
                "compound_chembl": molecule.chembl_id,
                "compound_rank": molecule["rank"],
                "compound_folds_top1pct": molecule.folds_in_top_1_percent,
                "compound_median_fold_rank_percent": molecule.median_fold_rank_percent,
                "pocket_mean_pLDDT": molecule.pocket_mean_pLDDT,
            })
            docking = target_gates.vina_score.dropna()
            row["vina_score"] = docking.iloc[0] if not docking.empty else np.nan
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "integrated_target_map.csv", index=False)
    summary = {
        "status": "complete",
        "n_targets": len(result),
        "lead_target": "GRN",
        "complementary_target": "BIN1",
        "n_with_cd4_real_action": int(result.cd4_perturbation_present.sum()),
        "n_with_myeloid_real_action": int(result.myeloid_perturbation_present.sum()),
        "n_with_candidate_compound": int(result.compound.notna().sum()),
        "n_with_independent_brain_data": int(result.independent_brain_dataset.ne("").sum()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(result.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
