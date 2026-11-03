import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_lps_programs import GMT, enrich, read_gmt

WORK = Path(__file__).resolve().parent
OUT = WORK / "results/grn_route_v1"
MARKERS = ["Gpnmb", "Ctsd", "Ctsb", "Ctsl", "Lamp1", "Lamp2", "Tfeb", "Tfe3", "Mitf", "C1qa", "C1qb", "C1qc", "Grn", "Hexb", "Cst3", "Cst7", "Lpl", "Apoe", "Trem2", "Tyrobp"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(WORK / "results/grn_brain_response_v1/observed_grn_ko_responses.npz", allow_pickle=True) as data:
        nodes = data["nodes"].astype(str)
        mask = data["mask"]
        observed = {key[7:]: data[key] for key in data.files if key.startswith("delta__")}
    with np.load(WORK / "results/cd4_real_action_brain_v1/brain_response_predictions.npz", allow_pickle=True) as data:
        targets = data["targets"].astype(str)
        conditions = data["conditions"].astype(str)
        model_nodes = data["nodes"].astype(str)
        model = data["response__Micro"][list(targets).index("GRN")]
    if list(nodes) != list(model_nodes):
        raise ValueError("Observed and model node order differ")
    genes = pd.read_csv("/public/home/mengxl/dzy/ad_omics_v3/data/processed/genes.csv").set_index("node_id")
    members = pd.read_csv("/public/home/mengxl/dzy/ad_omics_v3/data/reference/v2/mappings/orthogroups_member_level.csv").set_index("og_id")
    selected = nodes[mask]
    observed_matrix = np.stack([value[mask] for value in observed.values()])
    table = pd.DataFrame({
        "node": selected,
        "symbols": genes.loc[selected].symbols.astype(str).to_numpy(),
        "observed_mean_ko_response": observed_matrix.mean(axis=0),
        "observed_sd_ko_response": observed_matrix.std(axis=0, ddof=1),
        "ko_sign_consistency": np.mean(np.sign(observed_matrix) == np.sign(observed_matrix.mean(axis=0)), axis=0),
    })
    for index, condition in enumerate(conditions):
        table[f"model_{condition}"] = model[index, mask]
    table["abs_observed_mean"] = table.observed_mean_ko_response.abs()
    table.sort_values("abs_observed_mean", ascending=False).to_csv(OUT / "grn_observed_gene_response.csv", index=False)

    marker_rows = []
    for marker in MARKERS:
        matched_nodes = []
        for node in selected:
            symbols = str(members.mouse_symbols.get(node, "")) if node in members.index else str(genes.loc[node].symbols)
            if marker.upper() in [x.upper() for x in symbols.replace("|", ";").split(";")]:
                matched_nodes.append(node)
        for node in matched_nodes:
            values = np.array([value[nodes.tolist().index(node)] for value in observed.values()])
            marker_rows.append({"marker": marker, "node": node, "mean_ko_response": values.mean(), "sd_ko_response": values.std(ddof=1), "n_ko_mice": len(values), "sign_consistency": np.mean(np.sign(values) == np.sign(values.mean()))})
    marker_frame = pd.DataFrame(marker_rows).drop_duplicates(["marker", "node"])
    marker_frame.to_csv(OUT / "grn_marker_response.csv", index=False)

    top = table.nlargest(100, "abs_observed_mean")
    background = sorted({symbol.upper() for node in selected for symbol in str(members.human_symbols.get(node, "")).split(";") if symbol})
    query = sorted({symbol.upper() for node in top.node for symbol in str(members.human_symbols.get(node, "")).split(";") if symbol})
    enrichment = enrich(query, background, read_gmt(GMT), "Grn_KO_top100")
    enrichment.to_csv(OUT / "grn_go_enrichment.csv", index=False)

    summary = {
        "status": "complete",
        "n_ko_mice": len(observed),
        "n_response_nodes": len(selected),
        "markers": {marker: marker_frame.loc[marker_frame.marker.eq(marker), "mean_ko_response"].mean() for marker in MARKERS if marker in set(marker_frame.marker)},
        "n_go_terms_q05": int((enrichment.q <= 0.05).sum()),
        "top_go_terms": enrichment.loc[enrichment.q <= 0.05].head(15)[["term", "q"]].to_dict("records"),
    }
    (OUT / "grn_route_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
