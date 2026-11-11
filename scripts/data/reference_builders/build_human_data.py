#!/usr/bin/env python3
"""构建人侧世界模型数据: 人脑 GSE138852 + 人血 GSE181279 → v4 兼容 h5ad。

输出 (data/single-cell/human/):
  human_brain.h5ad: 13214 细胞 EC, obs=[disease(AD/CTRL), celltype, donor_pair]
  human_blood.h5ad: 36849 细胞 PBMC, obs=[disease, celltype, donor]
  两者 var 均为 11982 目标基因(与鼠侧 5xFAD h5ad 同序, 人 symbol 直接匹配)

泳道核验: GSE181279 无逐细胞元数据, 泳道顺序用 XIST(女高) 验证:
  期望 AD2(66F) 与 NC1(77F) 为女性泳道 (假定顺序 1=AD1,2=AD2,3=AD3,4=NC1,5=NC2)
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

PROJECT = Path(os.environ.get("AD_PROJECT_ROOT", os.path.expanduser("~/dzy/ad_omics_v2")))
HUMAN = PROJECT / "data" / "single-cell" / "human"
REF_H5AD = PROJECT / "data" / "transcriptomics" / "5xFAD_expression_matrix_for_step4.h5ad"

LANE2DONOR = {"1": "AD1", "2": "AD2", "3": "AD3", "4": "NC1", "5": "NC2"}  # 已弃用: 现按供体文件分别读取
DONOR2DISEASE = {"AD1": "AD", "AD2": "AD", "AD3": "AD", "NC1": "CTRL", "NC2": "CTRL"}
FEMALE_DONORS = {"AD2", "NC1"}  # XIST 核验期望

PBMC_MARKERS = {
    "CD4T":  ["CD3D", "CD3E", "CD4", "IL7R"],
    "CD8T":  ["CD3D", "CD3E", "CD8A", "CD8B"],
    "NK":    ["NCR1", "KLRD1", "KLRF1", "PRF1"],
    "B":     ["CD79A", "MS4A1", "CD19", "BANK1"],
    "Mono":  ["CD14", "LYZ", "VCAN", "S100A9", "FCN1"],
    "DC":    ["CLEC4C", "TCF4", "LILRA4", "CLEC10A"],
    "Mega":  ["PF4", "PPBP", "ITGA2B"],
    "Ery":   ["HBB", "HBA1", "HBA2", "ALAS2"],
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def reindex_to_target(X, symbols, target_genes):
    """把表达矩阵列(任意人 symbol 列表)重排到 target_genes 顺序, 缺补零, 重复列求和。"""
    target_set = {g.upper() for g in target_genes}
    sym = pd.Series([str(s).upper() for s in symbols])
    col_ok = sym.isin(target_set).values
    log(f"  基因映射: {col_ok.sum()}/{len(sym)} 列命中目标集")
    X = sp.csr_matrix(X)[:, col_ok]
    sym_ok = sym[col_ok]
    grp = pd.factorize(sym_ok)[0]
    S = sp.csr_matrix((np.ones(X.shape[1]), (np.arange(X.shape[1]), grp)),
                      shape=(X.shape[1], sym_ok.nunique()))
    X = X @ S
    sym_uniq = sym_ok.drop_duplicates().values
    pos = {g.upper(): i for i, g in enumerate(target_genes)}
    col_idx = [pos[g] for g in sym_uniq]
    P = sp.csr_matrix((np.ones(len(col_idx)), (np.arange(len(col_idx)), col_idx)),
                      shape=(len(col_idx), len(target_genes)))
    Xf = (X @ P).tocsr()
    log(f"  最终: {Xf.shape}, 缺失补零: {len(target_genes) - len(set(col_idx))}")
    return Xf


def build_brain(target_genes):
    log("== 人脑 GSE138852 ==")
    counts = pd.read_csv(HUMAN / "GSE138852_counts.csv.gz", index_col=0)
    log(f"  counts: {counts.shape} (基因×细胞)")
    cov = pd.read_csv(HUMAN / "GSE138852_covariates.csv.gz", index_col=0)
    cov.columns = [c.split(".")[-1] for c in cov.columns]  # batchCond/cellType/...
    cov["disease"] = cov["batchCond"].map({"AD": "AD", "ct": "CTRL"})
    cov["donor_pair"] = [i.split("_", 1)[1] if "_" in i else "NA" for i in cov.index]

    common = cov.index.intersection(counts.columns)
    log(f"  counts∩covariates: {len(common)}/{len(cov.index)} 细胞")
    counts = counts[common]
    cov = cov.loc[common]

    keep = ~cov["cellType"].isin(["doublet", "unID"])
    cov = cov[keep]
    counts = counts[cov.index]
    log(f"  去 doublet/unID 后: {counts.shape[1]} 细胞")

    X = sp.csr_matrix(counts.T.values.astype(np.float32))  # 细胞×基因
    Xf = reindex_to_target(X, counts.index, target_genes)

    obs = cov[["disease", "cellType", "donor_pair"]].rename(columns={"cellType": "celltype"})
    out = sc.AnnData(X=Xf, obs=obs, var=pd.DataFrame(index=target_genes))
    sc.pp.normalize_total(out, target_sum=1e4)
    sc.pp.log1p(out)
    out.write_h5ad(HUMAN / "human_brain.h5ad")
    log(f"  已保存 human_brain.h5ad: {out.shape}, "
        f"disease={out.obs['disease'].value_counts().to_dict()}")
    log(f"  celltype={out.obs['celltype'].value_counts().to_dict()}")


def build_blood(target_genes):
    log("== 人血 GSE181279 (按供体分别读取, 无泳道歧义) ==")
    RAWD = HUMAN / "GSE181279_raw"
    parts = []
    for donor in ["AD1", "AD2", "AD3", "NC1", "NC2"]:
        a1 = sc.read_10x_mtx(RAWD, var_names="gene_symbols",
                             prefix=f"GSM{ {'AD1':'5494107','AD2':'5494110','AD3':'5494113','NC1':'5494116','NC2':'5494119'}[donor] }_{donor}_")
        a1.obs["donor"] = donor
        a1.obs["disease"] = DONOR2DISEASE[donor]
        # XIST 核验打印(每供体)
        xi = [i for i, g in enumerate(a1.var_names) if str(g).upper() == "XIST"]
        if xi:
            xv = np.asarray(a1.X.tocsr()[:, xi[0]].todense()).ravel()
            log(f"  {donor}: {a1.shape[0]} 细胞, XIST 均值={xv.mean():.3f} "
                f"({'女' if donor in FEMALE_DONORS else '男'})")
        parts.append(a1)
    a = parts[0].concatenate(*parts[1:], join="outer", fill_value=0)
    log(f"  合并: {a.shape}")

    obs = a.obs[["disease", "donor"]].copy()

    # 细胞类型 (marker 打分)
    avail = set(a.var_names.astype(str))
    for k, v in PBMC_MARKERS.items():
        v2 = [g for g in v if g in avail]
        sc.tl.score_genes(a, v2, score_name=f"score_{k}")
    cols = [f"score_{k}" for k in PBMC_MARKERS]
    S = a.obs[cols].to_numpy()
    labels = list(PBMC_MARKERS.keys())
    obs["celltype"] = [labels[i] for i in S.argmax(axis=1)]

    # QC
    Xd = a.X.tocsr()
    counts = np.asarray(Xd.sum(axis=1)).ravel()
    ngenes = np.asarray((Xd > 0).sum(axis=1)).ravel()
    keep = (counts >= 500) & (ngenes >= 200)
    log(f"  QC: {keep.sum()}/{len(keep)} 保留")

    Xf = reindex_to_target(Xd[keep], a.var_names, target_genes)
    out = sc.AnnData(X=Xf, obs=obs[keep].reset_index(drop=True),
                     var=pd.DataFrame(index=target_genes))
    sc.pp.normalize_total(out, target_sum=1e4)
    sc.pp.log1p(out)
    out.write_h5ad(HUMAN / "human_blood.h5ad")
    log(f"  已保存 human_blood.h5ad: {out.shape}, "
        f"disease={out.obs['disease'].value_counts().to_dict()}")
    log(f"  celltype={out.obs['celltype'].value_counts().to_dict()}")


def main():
    ref = sc.read_h5ad(REF_H5AD, backed="r")
    target_genes = [str(g) for g in ref.var_names]
    log(f"目标基因: {len(target_genes)}")
    if (HUMAN / "human_brain.h5ad").exists():
        log("human_brain.h5ad 已存在, 跳过(删除后可重建)")
    else:
        build_brain(target_genes)
    build_blood(target_genes)


if __name__ == "__main__":
    main()
