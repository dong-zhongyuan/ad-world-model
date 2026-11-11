#!/usr/bin/env python3
"""鼠脑 + BM12 骨髓的细胞类型标注 (marker 打分法)。

输出:
  data/transcriptomics/brain_celltypes.csv     (obs_name, celltype, score 明细)
  data/transcriptomics/bm12_celltypes.csv
基因是人风格大写 symbol (1:1 同源), marker 直接用人 symbol。
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

PROJECT = Path(os.environ.get("AD_PROJECT_ROOT", os.path.expanduser("~/dzy/ad_omics_v2")))

BRAIN_MARKERS = {
    "Neuron":   ["SNAP25", "RBFOX3", "SYT1", "SLC17A7", "GAD1"],
    "Oligo":    ["MBP", "MOG", "PLP1", "OLIG1"],
    "Astro":    ["GFAP", "AQP4", "ALDH1L1", "SLC1A2"],
    "OPC":      ["PDGFRA", "CSPG4", "OLIG2"],
    "Micro":    ["CX3CR1", "P2RY12", "TMEM119", "AIF1", "C1QA"],
    "Endo":     ["CLDN5", "FLT1", "PECAM1"],
    "Pericyte": ["PDGFRB", "RGS5", "ACTA2"],
}

BLOOD_MARKERS = {
    "HSPC":      ["KIT", "FLT3", "CD34", "PROM1"],
    "MyeloidP":  ["MPO", "ELANE", "CEBPA", "PRTN3"],
    "Mono":      ["LYZ2", "LYZ1", "CD14", "CSF1R", "FCGR3"],
    "B":         ["CD79A", "MS4A1", "CD19"],
    "T":         ["CD3D", "CD3E", "CD8A"],
    "NK":        ["NCR1", "KLRD1", "GNLY"],
    "Ery":       ["HBB", "HBA1", "HBA2", "ALAS2"],
    "Mega":      ["PF4", "PPBP", "ITGA2B"],
    "pDC":       ["SIGLECH", "TCF4", "CLEC4C"],
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def annotate(h5ad_path, markers, out_csv, filter_obs=None):
    log(f"标注: {h5ad_path.name}")
    adata = sc.read_h5ad(h5ad_path)
    if filter_obs is not None:
        adata = adata[filter_obs(adata.obs)].copy()
        log(f"  过滤后: {adata.shape[0]} 细胞")
    avail = set(adata.var_names.astype(str))
    markers = {k: [g for g in v if g in avail] for k, v in markers.items()}
    for k, v in markers.items():
        if len(v) < 2:
            log(f"  警告: {k} 可用 marker 不足 ({v})")
        sc.tl.score_genes(adata, v, score_name=f"score_{k}", use_raw=False)
    score_cols = [f"score_{k}" for k in markers]
    S = adata.obs[score_cols].to_numpy()
    best = S.argmax(axis=1)
    labels = list(markers.keys())
    ct = np.array([labels[i] for i in best])
    S_sorted = np.sort(S, axis=1)
    margin = S_sorted[:, -1] - S_sorted[:, -2] if S.shape[1] > 1 else S[:, 0]
    out = pd.DataFrame({"celltype": ct, "margin": margin}, index=adata.obs_names)
    out.to_csv(out_csv)
    log(f"  分布: {out['celltype'].value_counts().to_dict()}")
    log(f"  margin 中位数: {np.median(margin):.3f}, margin<0.05 比例: {(margin < 0.05).mean():.1%}")
    log(f"  已保存: {out_csv}")


def main():
    annotate(PROJECT / "data/transcriptomics/5xFAD_expression_matrix_for_step4.h5ad",
             BRAIN_MARKERS,
             PROJECT / "data/transcriptomics/brain_celltypes.csv",
             filter_obs=lambda obs: obs["tissue"].astype(str) == "Brain")
    annotate(PROJECT / "data/transcriptomics/BM12_blood_genotype_labeled.h5ad",
             BLOOD_MARKERS,
             PROJECT / "data/transcriptomics/bm12_celltypes.csv")


if __name__ == "__main__":
    main()
