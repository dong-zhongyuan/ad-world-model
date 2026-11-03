import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from build_lps_programs import GMT, enrich, read_gmt
from evaluate_lps_brain_v2 import mouse_mapping
from headline_metrics import delta_pearson, top_k_overlap

ROOT = Path(os.environ.get("AD_OMICS_ROOT", "/public/home/mengxl/dzy/ad_omics_v3"))
WORK = Path(__file__).resolve().parent
OUT = WORK / "results/bin1_route_v1"
CONDITIONS = ["Control_PBS", "Bin1cKO_PBS", "Control_LPS", "Bin1cKO_LPS"]
MARKERS = ["Bin1", "Ifitm3", "Cx3cr1", "Apoe", "Trem2", "Tyrobp", "Spp1", "Grn", "Lamp1", "Cst3", "Cst7", "Ctsb", "Ctsd", "Il1b", "C3", "Irak3"]


def normalized_log(matrix):
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    normalizer = sp.diags(np.divide(1e4, totals, out=np.zeros(len(totals)), where=totals > 0))
    log = (normalizer @ matrix).tocsr()
    log.data = np.log1p(log.data)
    return log


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(WORK / "data/bin1_brain/metadata_Bin1cKO.csv", sep=";")
    metadata["normalized_id"] = metadata.cell_id.astype(str).str.replace("-", ".", regex=False).str.replace(":", ".", regex=False)
    micro = metadata.loc[metadata["Main.Groups"].eq("Micro")].copy()
    counts_path = WORK / "data/bin1_brain/counts_Bin1cKO.csv"
    header = pd.read_csv(counts_path, nrows=0)
    columns = list(header.columns)
    selected_columns = [column for column in columns[1:] if column in set(micro.normalized_id)]
    if len(selected_columns) != len(micro):
        missing = len(micro) - len(selected_columns)
        raise ValueError(f"Missing {missing} annotated microglial columns")
    counts = pd.read_csv(counts_path, usecols=[columns[0]] + selected_columns, index_col=0)
    source_genes = counts.index.astype(str).to_numpy()
    matrix = sp.csr_matrix(counts.to_numpy(dtype=np.int32).T)
    cell_meta = micro.set_index("normalized_id").loc[selected_columns].copy()
    cell_meta[["condition", "sample_id", "Main.Groups"]].to_csv(OUT / "microglia_cell_metadata.csv", index=False)
    cell_meta.groupby(["condition", "sample_id"]).size().to_csv(OUT / "cell_group_counts.csv")

    genes = pd.read_csv(ROOT / "data/processed/genes.csv")
    members = pd.read_csv(ROOT / "data/reference/v2/mappings/orthogroups_member_level.csv")
    mapping = mouse_mapping(genes, members, source_genes)
    og = (matrix @ mapping).tocsr()
    log = normalized_log(og)
    nodes = np.array(genes.node_id)
    node_index = {node: index for index, node in enumerate(nodes)}
    means = {}
    detection = {}
    for condition in CONDITIONS:
        selected = np.flatnonzero(cell_meta.condition.to_numpy() == condition)
        means[condition] = np.asarray(log[selected].mean(axis=0)).ravel()
        detection[condition] = np.asarray(og[selected].getnnz(axis=0)).ravel() / len(selected)
    observed = {
        "Bin1cKO_baseline": means["Bin1cKO_PBS"] - means["Control_PBS"],
        "Control_LPS_response": means["Control_LPS"] - means["Control_PBS"],
        "Bin1cKO_LPS_response": means["Bin1cKO_LPS"] - means["Bin1cKO_PBS"],
        "Bin1cKO_by_LPS_interaction": (means["Bin1cKO_LPS"] - means["Bin1cKO_PBS"]) - (means["Control_LPS"] - means["Control_PBS"]),
    }
    mask = (np.max(np.stack(list(detection.values())), axis=0) >= 0.01) & genes.kind.eq("shared_og").to_numpy()
    np.savez_compressed(OUT / "observed_bin1_responses.npz", nodes=nodes, mask=mask, **{f"delta__{key}": value for key, value in observed.items()})

    with np.load(WORK / "results/cd4_real_action_brain_v1/brain_response_predictions.npz", allow_pickle=True) as predictions:
        model_nodes = predictions["nodes"].astype(str)
        targets = predictions["targets"].astype(str)
        conditions = predictions["conditions"].astype(str)
        model = predictions["response__Micro"][list(targets).index("BIN1")]
    if list(model_nodes) != list(nodes):
        aligned = np.zeros((model.shape[0], len(nodes)), dtype=np.float32)
        for j, node in enumerate(model_nodes):
            aligned[:, node_index[node]] = model[:, j]
        model = aligned
    truth = observed["Bin1cKO_baseline"]
    model_mean = model.mean(axis=0)
    rows = []
    for index, condition in enumerate(conditions):
        for mode, predicted in [("condition", model[index]), ("condition_mean", model_mean)]:
            mse = float(np.mean((predicted[mask] - truth[mask]) ** 2))
            zero_mse = float(np.mean(truth[mask] ** 2))
            rows.append({"condition": condition, "mode": mode, "mse": mse, "zero_response_mse": zero_mse, "relative_mse_reduction": 1.0 - mse / zero_mse, "delta_pearson": delta_pearson(truth[mask], predicted[mask]), "top100_overlap": top_k_overlap(truth[mask], predicted[mask], k=100)})
    pd.DataFrame(rows).to_csv(OUT / "model_observed_metrics.csv", index=False)

    member_index = members.set_index("og_id")
    gene_index = genes.set_index("node_id")
    member_nodes = set(members.og_id)
    marker_rows = []
    for marker in MARKERS:
        for node in nodes[mask]:
            symbols = str(member_index.mouse_symbols.get(node, "")) if node in member_nodes else str(gene_index.loc[node].symbols)
            if marker.upper() in [value.upper() for value in symbols.replace("|", ";").split(";")]:
                marker_rows.append({"marker": marker, "node": node, **{key: float(value[node_index[node]]) for key, value in observed.items()}})
    marker_frame = pd.DataFrame(marker_rows).drop_duplicates(["marker", "node"])
    marker_frame.to_csv(OUT / "bin1_marker_responses.csv", index=False)

    top = pd.DataFrame({"node": nodes[mask], "response": truth[mask]})
    top["abs_response"] = top.response.abs()
    top = top.nlargest(100, "abs_response")
    top.to_csv(OUT / "top100_observed_bin1cko_genes.csv", index=False)
    background = sorted({symbol.upper() for node in nodes[mask] for symbol in str(member_index.human_symbols.get(node, "")).split(";") if symbol})
    query = sorted({symbol.upper() for node in top.node for symbol in str(member_index.human_symbols.get(node, "")).split(";") if symbol})
    enrichment = enrich(query, background, read_gmt(GMT), "Bin1cKO_baseline_top100")
    enrichment.to_csv(OUT / "bin1_go_enrichment.csv", index=False)

    result = {
        "status": "complete",
        "n_microglia_cells": len(micro),
        "cell_counts_by_condition": cell_meta.condition.value_counts().to_dict(),
        "n_response_nodes": int(mask.sum()),
        "observed_contrasts": list(observed),
        "n_go_terms_q05": int((enrichment.q <= 0.05).sum()),
        "runtime_seconds": time.time() - started,
    }
    (OUT / "run_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(marker_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
