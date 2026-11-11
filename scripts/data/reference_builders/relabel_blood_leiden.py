#!/usr/bin/env python3
"""人血细胞类型升级: Leiden 聚类 + 簇级 marker 共识注释 (替代 per-cell argmax 粗标签)。

流程: HVG(3000) → PCA(50) → neighbors(15) → Leiden(resolution=1.0) → UMAP
      → 簇级 marker 均值打分 → 每簇取最高分细胞类型 (top2 差距过小记 ambiguous)
产出:
  - human_blood.h5ad 增加 obs 列: leiden, celltype_leiden (原 celltype 保留不动; 先备份 .bak)
  - data/member_level/human_blood.h5ad 同步 celltype_leiden (按细胞序校验 donor 一致后写入)
  - results/figures/umap_human_blood_leiden.png (左: 新细标签, 右: disease)
  - results/figures/blood_leiden_concordance.csv (旧标签 × 新标签交叉表)
  - results/figures/blood_leiden_report.txt (簇大小/打分/ambiguity)
注意: 本步只落标签, 不改模型 cond 空间 (是否细化 BLOOD_ALIGN 另议)。
"""
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

PROJECT = Path.home() / "dzy" / "ad_omics_v2"
H5 = PROJECT / "data/single-cell/human/human_blood.h5ad"
H5_MEMBER = PROJECT / "data/member_level/human_blood.h5ad"
FIG = PROJECT / "results/figures"
FIG.mkdir(parents=True, exist_ok=True)

MARKERS = {
    "CD4T":      ["CD3D", "CD3E", "CD4", "IL7R", "LTB"],
    "CD8T":      ["CD3D", "CD3E", "CD8A", "CD8B", "GZMK"],
    "NK":        ["NCR1", "KLRD1", "KLRF1", "PRF1", "GNLY"],
    "B":         ["CD79A", "MS4A1", "CD19", "BANK1"],
    "CD14_Mono": ["CD14", "LYZ", "VCAN", "S100A9"],
    "CD16_Mono": ["FCGR3A", "MS4A7", "LST1", "CDKN1C"],
    "pDC":       ["CLEC4C", "TCF4", "LILRA4"],
    "cDC":       ["CLEC10A", "CD1C", "FCER1A"],
    "Mega":      ["PF4", "PPBP", "ITGA2B"],
    "Ery":       ["HBB", "HBA1", "HBA2", "ALAS2"],
    "Prolif":    ["MKI67", "TOP2A", "STMN1"],
}


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def main():
    log("加载 human_blood.h5ad")
    a = sc.read_h5ad(H5)
    log(f"  {a.shape}, 旧标签: {a.obs['celltype'].value_counts().to_dict()}")

    log("HVG → PCA → Leiden ...")
    sc.pp.highly_variable_genes(a, n_top_genes=3000, flavor="seurat")
    sc.pp.pca(a, n_comps=50, use_highly_variable=True)
    sc.pp.neighbors(a, n_neighbors=15)
    sc.tl.leiden(a, resolution=1.0, key_added="leiden")
    sc.tl.umap(a)
    nc = a.obs["leiden"].nunique()
    log(f"  Leiden 簇数: {nc}")

    # 簇级 marker 共识打分
    avail = {g: i for i, g in enumerate(a.var_names.astype(str))}
    X = a.X.tocsr()
    clusters = a.obs["leiden"].values
    cl_ids = sorted(clusters.unique(), key=int)
    score_mat = {}
    for ct, genes in MARKERS.items():
        cols = [avail[g] for g in genes if g in avail]
        sub = X[:, cols]
        per_cell = np.asarray(sub.mean(axis=1)).ravel()
        per_cl = np.array([per_cell[clusters == c].mean() for c in cl_ids])
        score_mat[ct] = per_cl
    S = pd.DataFrame(score_mat, index=cl_ids)
    Sz = (S - S.mean()) / S.std().clip(lower=1e-9)  # 按 marker 集跨簇 z 化

    assign, report_lines = {}, []
    for c in cl_ids:
        z = Sz.loc[c].sort_values(ascending=False)
        top1, top2 = z.index[0], z.index[1]
        gap = z.iloc[0] - z.iloc[1]
        if z.iloc[0] < 0.5:
            # 无 marker 超背景: 低信号/环境 RNA 簇, 不硬安细胞类型 (如原"Ery"簇 z≈0)
            assign[c] = "Unresolved"
            report_lines.append(
                f"簇 {c:>2}: n={int((clusters == c).sum()):>6} → Unresolved "
                f"(最高 z={z.iloc[0]:.2f} 无 marker 超背景, 疑似低质量/环境RNA簇)")
            continue
        flag = "" if gap >= 0.3 else " [AMBIGUOUS]"
        assign[c] = top1
        report_lines.append(
            f"簇 {c:>2}: n={int((clusters == c).sum()):>6} → {top1:<10} "
            f"(z={z.iloc[0]:.2f}, 次席 {top2} {z.iloc[1]:.2f}, gap={gap:.2f}){flag}")
        if flag:
            report_lines.append(f"       打分明细: {S.loc[c].round(2).to_dict()}")

    a.obs["celltype_leiden"] = [assign[c] for c in clusters]
    new_counts = a.obs["celltype_leiden"].value_counts().to_dict()
    log(f"  新标签分布: {new_counts}")

    # 一致性对照
    conc = pd.crosstab(a.obs["celltype"], a.obs["celltype_leiden"])
    conc.to_csv(FIG / "blood_leiden_concordance.csv")
    log(f"  对照表:\n{conc.to_string()}")

    # 写回 (先备份)
    if not (H5.parent / "human_blood.h5ad.bak").exists():
        shutil.copy(H5, H5.parent / "human_blood.h5ad.bak")
    a.write_h5ad(H5)
    log(f"  已写回 {H5.name} (备份 human_blood.h5ad.bak)")

    # 同步 member-level (校验细胞序)
    m = sc.read_h5ad(H5_MEMBER)
    assert m.n_obs == a.n_obs, f"member 文件细胞数不一致 {m.n_obs} vs {a.n_obs}"
    same = (m.obs["donor"].astype(str).values == a.obs["donor"].astype(str).values).all()
    assert same, "member 文件细胞序与 symbol 版不一致, 拒绝写入"
    m.obs["leiden"] = a.obs["leiden"].values
    m.obs["celltype_leiden"] = a.obs["celltype_leiden"].values
    m.write_h5ad(H5_MEMBER)
    log("  member-level human_blood.h5ad 已同步 (细胞序校验通过)")

    # 报告
    (FIG / "blood_leiden_report.txt").write_text(
        f"Leiden(resolution=1.0) 簇数 {nc}\n新标签分布: {new_counts}\n\n" +
        "\n".join(report_lines) + "\n\n旧×新对照见 blood_leiden_concordance.csv\n")

    # 图
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    xy = a.obsm["X_umap"]
    for ax, col, title in ((axes[0], "celltype_leiden", "human_blood — Leiden cluster annotation"),
                           (axes[1], "disease", "human_blood — disease")):
        cats = pd.Categorical(a.obs[col].astype(str))
        pal = plt.get_cmap("tab20")
        for i, cname in enumerate(cats.categories):
            msk = (cats == cname)
            ax.scatter(xy[msk, 0], xy[msk, 1], s=1.5, color=pal(i % 20),
                       label=str(cname), rasterized=True)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.legend(markerscale=4, fontsize=7, frameon=False,
                  loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(FIG / "umap_human_blood_leiden.png", dpi=300, bbox_inches="tight")
    log(f"  图已保存 {FIG}/umap_human_blood_leiden.png")


if __name__ == "__main__":
    main()
