import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

WORK = Path(__file__).resolve().parent
RAW = WORK / "data/myeloid/raw"
OUT = WORK / "data/myeloid/processed"
MIN_GENES = 200
MAX_MT_PERCENT = 20.0
MIN_GUIDE_UMI = 3
MIN_GUIDE_FRACTION = 0.80


def process_file(path, guide_metadata):
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    match = re.search(r"(GSM\d+)_(Mac_L\d+)_Mac_(Ctrl|LPS)_", path.name)
    if not match:
        raise ValueError(f"Unrecognized sample name: {path.name}")
    gsm, lane, condition = match.groups()
    sample = f"{lane}_{condition}"
    data = sc.read_10x_h5(path, gex_only=False)
    data.var_names_make_unique()
    x = data.X.tocsr()
    feature_types = data.var.feature_types.astype(str).to_numpy()
    gene_indices = np.flatnonzero(feature_types == "Gene Expression")
    guide_indices = np.flatnonzero(feature_types == "CRISPR Guide Capture")
    gene_names = data.var_names.to_numpy()[gene_indices]
    guide_ids = data.var_names.to_numpy()[guide_indices]
    if len(gene_indices) != 18129 or len(guide_indices) != 2925:
        raise ValueError(f"Unexpected feature counts in {path.name}")

    gene_x = x[:, gene_indices].tocsr()
    guide_x = x[:, guide_indices].tocsr()
    total_counts = np.asarray(gene_x.sum(axis=1)).ravel()
    n_genes = np.asarray(gene_x.getnnz(axis=1)).ravel()
    mt_indices = np.flatnonzero(pd.Series(gene_names).str.upper().str.startswith("MT-").to_numpy())
    mt_counts = np.asarray(gene_x[:, mt_indices].sum(axis=1)).ravel() if len(mt_indices) else np.zeros(x.shape[0])
    mt_percent = np.divide(mt_counts * 100, total_counts, out=np.zeros_like(mt_counts, dtype=float), where=total_counts > 0)

    n_cells = x.shape[0]
    top_umi = np.zeros(n_cells, dtype=np.int32)
    second_umi = np.zeros(n_cells, dtype=np.int32)
    top_index = np.full(n_cells, -1, dtype=np.int32)
    guide_total = np.zeros(n_cells, dtype=np.int32)
    for start in range(0, n_cells, 5000):
        end = min(start + 5000, n_cells)
        dense = guide_x[start:end].toarray()
        guide_total[start:end] = dense.sum(axis=1)
        if dense.shape[1] > 1:
            top_two = np.partition(dense, -2, axis=1)[:, -2:]
            second_umi[start:end] = top_two[:, 0]
            top_umi[start:end] = top_two[:, 1]
        top_index[start:end] = np.argmax(dense, axis=1)
    top_fraction = np.divide(top_umi, guide_total, out=np.zeros(n_cells, dtype=float), where=guide_total > 0)
    quality = (n_genes >= MIN_GENES) & (mt_percent <= MAX_MT_PERCENT) & (total_counts > 0)
    assigned = quality & (top_umi >= MIN_GUIDE_UMI) & (top_fraction >= MIN_GUIDE_FRACTION)
    assigned_guide = np.full(n_cells, "", dtype=object)
    valid_guide = top_index >= 0
    assigned_guide[valid_guide] = guide_ids[top_index[valid_guide]]

    guide_map = guide_metadata.set_index("guide_id")
    assigned_target = pd.Series(assigned_guide).map(guide_map.guide_target_gene_name).fillna("").to_numpy()
    roles = pd.Series(assigned_guide).map(guide_map.guide_role).fillna("").to_numpy()
    assigned_target = np.where(roles == "Control", "NTC", assigned_target)
    assigned_target = np.where(assigned, assigned_target, "")
    selected = np.flatnonzero(assigned & (assigned_target != ""))
    targets = sorted(np.unique(assigned_target[selected]))
    target_index = {target: index for index, target in enumerate(targets)}
    membership = sp.csr_matrix(
        (np.ones(len(selected)), (np.arange(len(selected)), np.array([target_index[target] for target in assigned_target[selected]]))),
        shape=(len(selected), len(targets)),
    )
    selected_gene = gene_x[selected].tocsr()
    normalization = sp.diags(np.divide(1e4, total_counts[selected], out=np.zeros(len(selected)), where=total_counts[selected] > 0))
    log_x = (normalization @ selected_gene).tocsr()
    log_x.data = np.log1p(log_x.data)
    sum_log = (membership.T @ log_x).toarray().astype(np.float32)
    sum_counts = (membership.T @ selected_gene).toarray().astype(np.int64)
    group_cells = np.asarray(membership.sum(axis=0)).ravel().astype(np.int32)
    mean_log = np.divide(sum_log, group_cells[:, None], out=np.zeros_like(sum_log), where=group_cells[:, None] > 0)

    cell_data = pd.DataFrame({
        "cell_index": selected,
        "barcode": data.obs_names.to_numpy()[selected],
        "guide_id": assigned_guide[selected],
        "target": assigned_target[selected],
        "guide_role": roles[selected],
        "total_counts": total_counts[selected],
        "n_genes": n_genes[selected],
        "pct_mt": mt_percent[selected],
        "top_guide_umi": top_umi[selected],
        "second_guide_umi": second_umi[selected],
        "top_guide_fraction": top_fraction[selected],
    })
    cell_data.to_csv(OUT / f"{sample}_cells.csv.gz", index=False, compression="gzip")
    np.savez_compressed(
        OUT / f"{sample}_pseudobulk.npz",
        targets=np.array(targets),
        genes=gene_names,
        mean_log=mean_log,
        sum_counts=sum_counts,
        n_cells=group_cells,
    )
    summary = {
        "gsm": gsm,
        "lane": lane,
        "condition": condition,
        "input_cells": n_cells,
        "quality_pass": int(quality.sum()),
        "assigned_cells": len(selected),
        "n_targets": len(targets),
        "n_targeting_cells": int((assigned_target[selected] != "NTC").sum()),
        "n_NTC_cells": int((assigned_target[selected] == "NTC").sum()),
        "min_genes": MIN_GENES,
        "max_mt_percent": MAX_MT_PERCENT,
        "min_guide_umi": MIN_GUIDE_UMI,
        "min_guide_fraction": MIN_GUIDE_FRACTION,
        "runtime_seconds": time.time() - started,
    }
    (OUT / f"{sample}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    guide_metadata = pd.read_csv(WORK / "data/myeloid/guide_template.txt.gz", sep="\t")
    summaries = []
    for path in sorted(RAW.glob("*.h5")):
        sample_match = re.search(r"(Mac_L\d+)_Mac_(Ctrl|LPS)_", path.name)
        output = OUT / f"{sample_match.group(1)}_{sample_match.group(2)}_summary.json"
        if output.exists():
            summaries.append(json.loads(output.read_text()))
            print(f"SKIP {path.name}", flush=True)
            continue
        summaries.append(process_file(path, guide_metadata))
    pd.DataFrame(summaries).to_csv(WORK / "myeloid_processing_summary.csv", index=False)
    print("MYELOID PROCESSING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
