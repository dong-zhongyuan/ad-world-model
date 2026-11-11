#!/usr/bin/env python3
"""从 BM1/BM2 + CMO 分形表重建带基因型标签的血端(骨髓 HSPC)矩阵。

背景: 现有 h5ad 的血端 (BM3/BM4) 全部标 'mixed', 导致跨组织预测任务无监督信号
(模型退化为常数预测器, mse 25.89 ≈ 最优常数基线 25.85)。BM1/BM2 有 CMO 分形表,
可恢复每个细胞的基因型, 支持基因型配对训练。

输出: data/transcriptomics/BM12_blood_genotype_labeled.h5ad
  X:   log 归一化 (normalize_total(1e4) + log1p, 与现有 5xFAD h5ad 一致)
  var: 与 5xFAD_expression_matrix_for_step4.h5ad 相同的 11982 基因, 同序
  obs: genotype(5xFAD/WT), origin(femur/skull), mouse, lane(BM1/BM2), cmo
过滤: Assignment 有效(非 Multiplet/Blank/Unassigned) + Assignment_Probability>=0.8
      + counts>=500 且 genes>=200
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
OUT = PROJECT / "data" / "transcriptomics" / "BM12_blood_genotype_labeled.h5ad"

LANES = {"BM1": "GSM9703838_BM1", "BM2": "GSM9703839_BM2"}
MIN_PROB = 0.8
MIN_COUNTS, MIN_GENES = 500, 200


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_cmo_map():
    """feature reference → {CMOxxx: (genotype, origin, mouse)}。"""
    fr = pd.read_csv(MAP_DIR / "GSE329430_feature_reference.csv.gz")
    fr = fr[fr["feature_type"] == "Multiplexing Capture"]
    cmap = {}
    for _, r in fr.iterrows():
        parts = str(r["multiplexed_sample"]).split("_")
        cmap[r["id"]] = (parts[0], parts[-1], r["multiplexed_sample"])
    log(f"CMO 映射: {len(cmap)} 个 tag")
    return cmap


def main():
    cmo_map = load_cmo_map()

    # 目标基因集 (与现有 h5ad 同序; var 索引用原始大小写, 匹配时用大写)
    ref = sc.read_h5ad(REF_H5AD, backed="r")
    target_genes = [str(g) for g in ref.var_names]
    log(f"目标基因: {len(target_genes)}")

    # 鼠 symbol → 人 symbol
    m = pd.read_csv(MAP_DIR / "orthologs_mgi_1to1.csv")
    m2h = dict(zip(m["mouse_symbol"].astype(str), m["human_symbol"].astype(str)))
    m2h_upper = {k.upper(): v for k, v in m2h.items()}
    log(f"同源映射: {len(m2h)} 对")

    adatas = []
    for lane, prefix in LANES.items():
        # 表达矩阵
        a = sc.read_10x_mtx(RAW, var_names="gene_symbols", prefix=f"{prefix}_")
        log(f"{lane}: 原始矩阵 {a.shape[0]} barcodes × {a.shape[1]} genes")

        # 分形表
        tbl = pd.read_csv(RAW / f"{prefix}_assignment_confidence_table.csv.gz")
        valid = (~tbl["Assignment"].isin(["Multiplet", "Blank", "Unassigned"])) & \
                (tbl["Assignment_Probability"] >= MIN_PROB)
        tbl = tbl[valid].copy()
        tbl["genotype"] = tbl["Assignment"].map(lambda c: cmo_map[c][0])
        tbl["origin"] = tbl["Assignment"].map(lambda c: cmo_map[c][1])
        tbl["mouse"] = tbl["Assignment"].map(lambda c: cmo_map[c][2])
        log(f"{lane}: 有效分形 {len(tbl)} 细胞 "
            f"(5xFAD={ (tbl['genotype']=='5xFAD').sum() }, WT={ (tbl['genotype']=='WT').sum() })")

        # 交集: 矩阵 barcodes ∩ 分形 barcodes
        bc = pd.Index(a.obs_names.astype(str))
        tbl = tbl[tbl["Barcode"].isin(bc)]
        a = a[tbl["Barcode"].values, :]
        a.obs = tbl.set_index("Barcode")[["Assignment", "genotype", "origin", "mouse"]].loc[a.obs_names]
        a.obs = a.obs.rename(columns={"Assignment": "cmo"})
        a.obs["lane"] = lane
        adatas.append(a)
        log(f"{lane}: 交集后 {a.shape[0]} 细胞")

    adata = adatas[0].concatenate(adatas[1], join="outer", fill_value=0)
    log(f"合并: {adata.shape}")

    # QC
    X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    counts = np.asarray(X.sum(axis=1)).ravel()
    ngenes = np.asarray((X > 0).sum(axis=1)).ravel()
    keep = (counts >= MIN_COUNTS) & (ngenes >= MIN_GENES)
    adata = adata[keep].copy()
    log(f"QC (counts>={MIN_COUNTS}, genes>={MIN_GENES}): {keep.sum()}/{len(keep)} 保留")

    # 鼠 symbol → 人 symbol → 目标 11982 基因 (同序, 大小写不敏感匹配)
    mouse_syms = pd.Index(adata.var_names.astype(str))
    human = mouse_syms.map(lambda s: m2h.get(s) or m2h_upper.get(s.upper()))
    target_set = {g.upper() for g in target_genes}
    col_ok = np.array([(h is not None and str(h).upper() in target_set) for h in human])
    log(f"基因映射: {col_ok.sum()}/{len(mouse_syms)} 鼠基因可映射到目标集")

    X = adata.X.tocsr()[:, col_ok]
    human_ok = pd.Series(human[col_ok].values)
    # 同一人基因多列合并 (求和)
    dup = human_ok.duplicated(keep=False).values
    if dup.any():
        log(f"  合并重复人基因列: {dup.sum()} 列 → {human_ok[dup].nunique()} 基因")
        grp = pd.factorize(human_ok)[0]
        S = sp.csr_matrix((np.ones(X.shape[1]), (np.arange(X.shape[1]), grp)),
                          shape=(X.shape[1], human_ok.nunique()))
        X = X @ S
        human_uniq = human_ok.drop_duplicates().values
    else:
        human_uniq = human_ok.values

    # reindex 到目标顺序 (匹配大写, 落位用原始大小写的目标顺序)
    pos_in_target = {g.upper(): i for i, g in enumerate(target_genes)}
    col_idx = [pos_in_target[str(g).upper()] for g in human_uniq]
    P = sp.csr_matrix((np.ones(len(human_uniq)), (np.arange(len(human_uniq)), col_idx)),
                      shape=(len(human_uniq), len(target_genes)))
    X_final = (X @ P).tocsr()
    missing = len(target_genes) - len(set(col_idx))
    log(f"最终矩阵: {X_final.shape}, 缺失基因(补零): {missing}")

    out = sc.AnnData(X=X_final,
                     obs=adata.obs[["genotype", "origin", "mouse", "lane", "cmo"]],
                     var=pd.DataFrame(index=target_genes))

    # 与 v1 一致的 log 归一化
    sc.pp.normalize_total(out, target_sum=1e4)
    sc.pp.log1p(out)
    log(f"归一化完成: max={out.X.max():.2f}")

    out.write_h5ad(OUT)
    log(f"已保存: {OUT}")
    log(f"obs 分布:\n{out.obs.groupby(['lane','genotype']).size().to_string()}")


if __name__ == "__main__":
    main()
