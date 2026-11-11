#!/usr/bin/env python3
"""BM3/BM4(股骨骨髓 髓系+HSPC) → 带基因型/鼠个体/细胞类型标签的 h5ad。

输入: filtered_feature_bc_matrix.h5 (Gene Expression) + demux_bm34.py 的分形表
输出: data/transcriptomics/BM34_blood_genotype_labeled.h5ad
  X: log 归一化, var=11982 目标基因(同序)
  obs: genotype, mouse(5xFAD_1..4/WT_1..4), lane(BM3/BM4), cmo(Hashtag_*), celltype
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

PROJECT = Path(os.environ.get("AD_PROJECT_ROOT", os.path.expanduser("~/dzy/ad_omics_v2")))
RAW = PROJECT / "data" / "single-cell" / "GSE329430_raw"
MAP_DIR = PROJECT / "data" / "mappings"
REF_H5AD = PROJECT / "data" / "transcriptomics" / "5xFAD_expression_matrix_for_step4.h5ad"
OUT = PROJECT / "data" / "transcriptomics" / "BM34_blood_genotype_labeled.h5ad"

LANES = {"BM3": "GSM9703842_BM3", "BM4": "GSM9703843_BM4"}
MIN_CONF = 0.8
MIN_COUNTS, MIN_GENES = 500, 200

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


def main():
    ref = sc.read_h5ad(REF_H5AD, backed="r")
    target_genes = [str(g) for g in ref.var_names]
    target_set = {g.upper() for g in target_genes}
    log(f"目标基因: {len(target_genes)}")

    m = pd.read_csv(MAP_DIR / "orthologs_mgi_1to1.csv")
    m2h = dict(zip(m["mouse_symbol"].astype(str), m["human_symbol"].astype(str)))
    m2h_upper = {k.upper(): v for k, v in m2h.items()}

    adatas = []
    for lane, prefix in LANES.items():
        a = sc.read_10x_h5(RAW / f"{prefix}_filtered_feature_bc_matrix.h5")
        a.var_names_make_unique()
        log(f"{lane}: 原始 {a.shape}")

        dem = pd.read_csv(RAW / "hto" / f"{lane}_demux_assignments.csv")
        dem = dem[(dem["genotype"] != "") & (dem["assignment_confidence"] >= MIN_CONF)]
        dem = dem.set_index("barcode")
        bc = pd.Index(a.obs_names.astype(str))
        hit = bc.isin(dem.index)
        a = a[hit].copy()
        a.obs["genotype"] = dem.loc[a.obs_names, "genotype"].values
        a.obs["mouse"] = dem.loc[a.obs_names, "mouse"].values
        a.obs["cmo"] = dem.loc[a.obs_names, "hashtag"].values
        a.obs["lane"] = lane
        log(f"{lane}: 分形交集 {hit.sum()} 细胞 "
            f"(5xFAD={(a.obs['genotype']=='5xFAD').sum()}, WT={(a.obs['genotype']=='WT').sum()})")
        adatas.append(a)

    adata = adatas[0].concatenate(adatas[1], join="outer", fill_value=0)
    log(f"合并: {adata.shape}")

    X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    counts = np.asarray(X.sum(axis=1)).ravel()
    ngenes = np.asarray((X > 0).sum(axis=1)).ravel()
    keep = (counts >= MIN_COUNTS) & (ngenes >= MIN_GENES)
    adata = adata[keep].copy()
    log(f"QC: {keep.sum()}/{len(keep)} 保留")

    # 细胞类型 (marker 打分, 用鼠→人映射前的鼠 symbol? 用 mapped 人 symbol)
    mouse_syms = pd.Index(adata.var_names.astype(str))
    human = mouse_syms.map(lambda s: m2h.get(s) or m2h_upper.get(s.upper()))
    col_ok = np.array([(h is not None and str(h).upper() in target_set) for h in human])
    log(f"基因映射: {col_ok.sum()}/{len(mouse_syms)} 鼠基因命中目标集")

    X = sp.csr_matrix(adata.X)[:, col_ok]
    human_ok = pd.Series([str(h) for h in human[col_ok]])
    grp = pd.factorize(human_ok)[0]
    S = sp.csr_matrix((np.ones(X.shape[1]), (np.arange(X.shape[1]), grp)),
                      shape=(X.shape[1], human_ok.nunique()))
    X = X @ S
    human_uniq = human_ok.drop_duplicates().values
    pos = {g.upper(): i for i, g in enumerate(target_genes)}
    col_idx = [pos[str(g).upper()] for g in human_uniq]
    P = sp.csr_matrix((np.ones(len(col_idx)), (np.arange(len(col_idx)), col_idx)),
                      shape=(len(col_idx), len(target_genes)))
    Xf = (X @ P).tocsr()
    log(f"最终矩阵: {Xf.shape}, 缺失补零: {len(target_genes) - len(set(col_idx))}")

    out = sc.AnnData(X=Xf,
                     obs=adata.obs[["genotype", "mouse", "lane", "cmo"]].reset_index(drop=True),
                     var=pd.DataFrame(index=target_genes))
    sc.pp.normalize_total(out, target_sum=1e4)
    sc.pp.log1p(out)

    # 细胞类型打分 (人 symbol marker, 目标集内)
    avail = set(out.var_names.astype(str))
    for k, v in BLOOD_MARKERS.items():
        v2 = [g for g in v if g in avail]
        sc.tl.score_genes(out, v2, score_name=f"score_{k}")
    cols = [f"score_{k}" for k in BLOOD_MARKERS]
    S_ = out.obs[cols].to_numpy()
    labels = list(BLOOD_MARKERS.keys())
    out.obs["celltype"] = [labels[i] for i in S_.argmax(axis=1)]
    out.obs = out.obs[["genotype", "mouse", "lane", "cmo", "celltype"]]

    out.write_h5ad(OUT)
    log(f"已保存: {OUT}")
    log(f"celltype: {out.obs['celltype'].value_counts().to_dict()}")
    log(f"lane×genotype: {out.obs.groupby(['lane','genotype']).size().to_dict()}")


if __name__ == "__main__":
    main()
