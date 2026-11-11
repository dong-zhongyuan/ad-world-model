import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread
from sklearn.mixture import GaussianMixture

from headline_metrics import delta_pearson, top_k_overlap

ROOT = Path(os.environ.get("AD_OMICS_ROOT", "/public/home/mengxl/dzy/ad_omics_v3"))
WORK = Path(__file__).resolve().parent
RAW = WORK / "data/lps_brain/raw"
OUT = WORK / "results/lps_brain_response_v2"
SAMPLES = {
    "GSM7840548": ("saline", "dmso", 1),
    "GSM7840549": ("lps", "dmso", 1),
    "GSM7840550": ("saline", "inhibitors", 1),
    "GSM7840551": ("lps", "inhibitors", 1),
    "GSM7840552": ("saline", "dmso", 2),
    "GSM7840553": ("lps", "dmso", 2),
    "GSM7840554": ("saline", "inhibitors", 2),
    "GSM7840555": ("lps", "inhibitors", 2),
}
MICROGLIA_MARKERS = ["C1qa", "C1qb", "C1qc", "Hexb", "Csf1r", "Cx3cr1", "Fcrls", "Tyrobp", "Aif1", "Spi1"]


def load_sample(gsm, raw_dir=None):
    raw_dir = RAW if raw_dir is None else Path(raw_dir)
    feature_path = next(raw_dir.glob(f"{gsm}_*_features.tsv.gz"))
    suffix = feature_path.name[len(gsm) + 1:].removesuffix("_features.tsv.gz")
    features = pd.read_csv(feature_path, sep="\t", header=None)
    gene_names = features.iloc[:, 1].astype(str).to_numpy() if features.shape[1] >= 2 else features.iloc[:, 0].astype(str).to_numpy()
    matrix = mmread(raw_dir / f"{gsm}_{suffix}_matrix.mtx.gz").tocsr().T.tocsr()
    barcodes = pd.read_csv(raw_dir / f"{gsm}_{suffix}_barcodes.tsv.gz", sep="\t", header=None).iloc[:, 0].astype(str).to_numpy()
    if matrix.shape != (len(barcodes), len(gene_names)):
        raise ValueError(f"Matrix dimensions do not match metadata for {gsm}")
    input_cells = matrix.shape[0]
    total = np.asarray(matrix.sum(axis=1)).ravel()
    n_genes = np.asarray(matrix.getnnz(axis=1)).ravel()
    mt = np.flatnonzero(pd.Series(gene_names).str.upper().str.startswith("MT-").to_numpy())
    mt_counts = np.asarray(matrix[:, mt].sum(axis=1)).ravel() if len(mt) else np.zeros(matrix.shape[0])
    mt_percent = np.divide(mt_counts * 100, total, out=np.zeros(matrix.shape[0]), where=total > 0)
    quality = (n_genes >= 200) & (mt_percent <= 20) & (total > 0)
    matrix = matrix[quality].tocsr()
    barcodes = barcodes[quality]
    total = total[quality]
    normalizer = sp.diags(np.divide(1e4, total, out=np.zeros(len(total)), where=total > 0))
    log = (normalizer @ matrix).tocsr()
    log.data = np.log1p(log.data)
    marker_indices = [i for i, gene in enumerate(gene_names) if gene.lower() in {marker.lower() for marker in MICROGLIA_MARKERS}]
    marker_score = np.asarray(log[:, marker_indices].mean(axis=1)).ravel() if marker_indices else np.zeros(matrix.shape[0])
    return {"matrix": matrix, "genes": gene_names, "barcodes": barcodes, "total": total, "marker_score": marker_score, "input_cells": input_cells, "quality_cells": int(quality.sum())}


def mouse_mapping(genes, members, source_genes):
    og_to_nodes = genes.reset_index().groupby("og_id").indices
    mapping = defaultdict(set)
    for row in members.itertuples():
        for symbol in str(row.mouse_symbols).split(";"):
            if symbol and symbol != "nan":
                mapping[symbol.upper()].update(og_to_nodes.get(row.og_id, []))
    for index, row in enumerate(genes.itertuples()):
        if row.kind == "mouse_specific":
            for symbol in str(row.symbols).split("|"):
                mapping[symbol.upper()].add(index)
    rows = []
    columns = []
    for j, gene in enumerate(source_genes):
        for node in mapping.get(gene.upper(), set()):
            rows.append(j)
            columns.append(node)
    return sp.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(source_genes), len(genes)))


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    samples = {}
    for gsm in SAMPLES:
        samples[gsm] = load_sample(gsm)
        print(f"LOADED {gsm} {samples[gsm]['matrix'].shape}", flush=True)
    all_scores = np.concatenate([sample["marker_score"] for sample in samples.values()])
    mixture = GaussianMixture(n_components=2, random_state=42).fit(all_scores[:, None])
    microglia_component = int(np.argmax(mixture.means_.ravel()))

    genes = pd.read_csv(ROOT / "data/processed/genes.csv")
    members = pd.read_csv(ROOT / "data/reference/v2/mappings/orthogroups_member_level.csv")
    nodes = np.array(genes.node_id)
    node_index = {node: index for index, node in enumerate(nodes)}
    mean_log = {}
    detection = {}
    qc_rows = []
    for gsm, sample in samples.items():
        treatment, processing, animal = SAMPLES[gsm]
        scores = sample["marker_score"]
        microglia = mixture.predict(scores[:, None]) == microglia_component
        matrix = sample["matrix"][microglia].tocsr()
        mapping = mouse_mapping(genes, members, sample["genes"])
        og = (matrix @ mapping).tocsr()
        totals = np.asarray(og.sum(axis=1)).ravel()
        normalizer = sp.diags(np.divide(1e4, totals, out=np.zeros(len(totals)), where=totals > 0))
        log = (normalizer @ og).tocsr()
        log.data = np.log1p(log.data)
        mean_log[gsm] = np.asarray(log.mean(axis=0)).ravel()
        detection[gsm] = np.asarray(og.getnnz(axis=0)).ravel() / max(og.shape[0], 1)
        qc_rows.append({
            "gsm": gsm,
            "treatment": treatment,
            "processing": processing,
            "animal": animal,
            "quality_cells": sample["matrix"].shape[0],
            "microglia_cells": int(microglia.sum()),
            "microglia_fraction": float(microglia.mean()),
            "microglia_marker_mean": float(scores[microglia].mean()),
            "other_marker_mean": float(scores[~microglia].mean()),
        })
        print(f"MICROGLIA {gsm} {int(microglia.sum())}", flush=True)
    pd.DataFrame(qc_rows).to_csv(OUT / "microglia_sample_qc.csv", index=False)
    np.savez_compressed(OUT / "observed_microglia_means.npz", nodes=nodes, **{f"mean__{gsm}": mean_log[gsm] for gsm in SAMPLES}, **{f"detection__{gsm}": detection[gsm] for gsm in SAMPLES})

    detected = np.max(np.stack(list(detection.values())), axis=0) >= 0.01
    shared = genes.kind.eq("shared_og").to_numpy()
    mask = detected & shared
    with np.load(WORK / "results/real_action_brain_v1/brain_response_predictions.npz", allow_pickle=True) as predictions:
        model_nodes = predictions["nodes"].astype(str)
        model_lps = predictions["lps__Micro"]
        lanes = predictions["lanes"].astype(str)
    if list(model_nodes) != list(nodes):
        aligned = np.zeros((model_lps.shape[0], len(nodes)), dtype=np.float32)
        for j, node in enumerate(model_nodes):
            aligned[:, node_index[node]] = model_lps[:, j]
        model_lps = aligned
    model_mean = model_lps.mean(axis=0)

    observed = {}
    for processing in ["dmso", "inhibitors"]:
        saline = [gsm for gsm, value in SAMPLES.items() if value[0] == "saline" and value[1] == processing]
        lps = [gsm for gsm, value in SAMPLES.items() if value[0] == "lps" and value[1] == processing]
        saline_mean = np.mean([mean_log[gsm] for gsm in saline], axis=0)
        for gsm in lps:
            observed[gsm] = mean_log[gsm] - saline_mean
    np.savez_compressed(OUT / "observed_lps_responses.npz", nodes=nodes, mask=mask, **{f"delta__{gsm}": value for gsm, value in observed.items()})

    rows = []
    for gsm, truth in observed.items():
        processing = SAMPLES[gsm][1]
        for lane_index, lane in enumerate(lanes):
            for mode, predicted in [("lane", model_lps[lane_index]), ("lane_mean", model_mean)]:
                mse = float(np.mean((predicted[mask] - truth[mask]) ** 2))
                zero_mse = float(np.mean(truth[mask] ** 2))
                rows.append({
                    "gsm": gsm,
                    "processing": processing,
                    "lane": lane,
                    "mode": mode,
                    "mse": mse,
                    "zero_response_mse": zero_mse,
                    "relative_mse_reduction": 1.0 - mse / zero_mse,
                    "delta_pearson": delta_pearson(truth[mask], predicted[mask]),
                    "top100_overlap": top_k_overlap(truth[mask], predicted[mask], k=100),
                    "n_nodes": int(mask.sum()),
                })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "metrics_by_animal_lane.csv", index=False)
    summary = metrics.groupby(["processing", "mode"]).agg(
        mse=("mse", "mean"),
        zero_response_mse=("zero_response_mse", "mean"),
        relative_mse_reduction=("relative_mse_reduction", "mean"),
        delta_pearson=("delta_pearson", "mean"),
        top100_overlap=("top100_overlap", "mean"),
    ).reset_index()
    summary.to_csv(OUT / "metrics_summary.csv", index=False)
    result = {
        "status": "complete",
        "n_response_nodes": int(mask.sum()),
        "primary_scope": "shared orthogroups detected in at least 1% of microglia in at least one sample",
        "primary_processing": "dmso",
        "target_relative_mse_reduction": 0.50,
        "primary_lane_mean_relative_mse_reduction": float(summary.loc[(summary.processing == "dmso") & (summary["mode"] == "lane_mean"), "relative_mse_reduction"].item()),
        "runtime_seconds": time.time() - started,
    }
    result["target_met"] = result["primary_lane_mean_relative_mse_reduction"] >= 0.50
    (OUT / "run_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
