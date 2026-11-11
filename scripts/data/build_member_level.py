#!/usr/bin/env python3
"""member-level 数据重建 (原则一第二步; 2026-08-08 拍板: 聚合规则=求和)。

从 RAW 按全符号重建四个数据集到同源组(OG)列空间:
  鼠脑  GSM9703846_955-958 h5 (限于 brain_celltypes.csv 的 5000 细胞, 标签 barcode 直挂)
  鼠血  BM1/BM2 mtx (CMO 分形) + BM3/BM4 h5 (hto demux), 同原 builder 的 QC
  人脑  GSE138852_counts/covariates (仅 10850 预过滤基因, 如实记录)
  人血  GSE181279_raw 五供体 mtx

列空间 (node):
  shared OG: gene2og 中 n_mouse>=1 且 n_human>=1 的组 → 列名=og_id, 组内成员原始计数求和
  私有基因: 不在 shared OG 的基因(含 mouse_only/human_only 组成员与非 HOM 基因),
            单独成列, 列名 'M:<sym>'/'H:<sym>', 显式 kind 标注;
            仅保留在 >=1 个同物种数据集中检出率 >= --priv-min-det 的(默认 0.005)
聚合顺序与原流程一致: 原始计数先求和到列空间 → normalize_total(1e4) → log1p。

产出 (data/member_level/):
  mouse_brain.h5ad / mouse_blood_bm12.h5ad / mouse_blood_bm34.h5ad
  human_brain.h5ad / human_blood.h5ad   (var=node 表, obs 含训练所需全部标签)
  columns.csv                           (node_id, kind, og_id, symbols, n_mouse, n_human, esm_covered)
  build_report.json                     (各数据集维度/映射/私有基因统计)
  cache/esm2_embeddings/esm2_member_level_dim640.pt  (OG=成员均值, 私有=自身, 缺失=0)
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

PROJECT = Path(os.environ.get("AD_PROJECT_ROOT", os.path.expanduser("~/dzy/ad_omics_v2")))
RAW = PROJECT / "data" / "single-cell" / "GSE329430_raw"
HUMAN = PROJECT / "data" / "single-cell" / "human"
MAP_DIR = PROJECT / "data" / "mappings"
OUT_DIR = PROJECT / "data" / "member_level"
ESM_CACHE = PROJECT / "cache" / "esm2_embeddings" / "esm2_all_genes_t30_150M_dim640.pt"

LABEL_DIR = PROJECT / "data/transcriptomics"
HUMAN_BRAIN = HUMAN
HUMAN_BLOOD = HUMAN / "GSE181279_raw"


MIN_COUNTS, MIN_GENES = 500, 200
SAMPLE2MOUSE = {"955": "WT_1", "956": "5xFAD_1", "957": "WT_2", "958": "5xFAD_2"}
SAMPLE2GENO = {"955": "WT", "956": "5xFAD", "957": "WT", "958": "5xFAD"}
DONOR2DISEASE = {"AD1": "AD", "AD2": "AD", "AD3": "AD", "NC1": "CTRL", "NC2": "CTRL"}
DONOR_GSM = {"AD1": "5494107", "AD2": "5494110", "AD3": "5494113",
             "NC1": "5494116", "NC2": "5494119"}

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
BLOOD_MARKERS_BM34 = {
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


def qc_mask(X):
    counts = np.asarray(X.sum(axis=1)).ravel()
    ngenes = np.asarray((X > 0).sum(axis=1)).ravel()
    return (counts >= MIN_COUNTS) & (ngenes >= MIN_GENES)


class ColumnSpace:
    """node 列空间: shared OG + 私有基因(检出率过滤)。"""

    def __init__(self, priv_min_det):
        self.priv_min_det = priv_min_det
        og = pd.read_csv(MAP_DIR / "orthogroups_member_level.csv", dtype=str)
        g2 = pd.read_csv(MAP_DIR / "gene2og.csv", dtype=str)
        og["n_mouse"] = og["n_mouse"].astype(int)
        og["n_human"] = og["n_human"].astype(int)
        self.og = og.set_index("og_id")
        shared = set(self.og[(self.og.n_mouse > 0) & (self.og.n_human > 0)].index)
        log(f"shared OG: {len(shared)} / 总 {len(self.og)}")
        # 物种 symbol → og_id (仅 shared; 其余为私有候选)
        self.sym2og = {"mouse": {}, "human": {}}
        for sp in ("mouse", "human"):
            sub = g2[g2.species == sp]
            for s, o in zip(sub["symbol"], sub["og_id"]):
                if o in shared:
                    self.sym2og[sp][s] = o
        log(f"symbol→shared OG: 鼠 {len(self.sym2og['mouse'])}, 人 {len(self.sym2og['human'])}")
        self.priv_det = {"mouse": {}, "human": {}}   # sym → 各数据集最大检出率
        self.nodes = []                               # 最终列 (build 后)
        self.node_index = {}

    def scan_dataset(self, species, symbols, X_qc):
        """登记私有候选基因的检出率 (在 QC 后细胞上)。X_qc: csr (cells×genes)。"""
        det = np.asarray((X_qc > 0).mean(axis=0)).ravel()
        cand = 0
        for i, s in enumerate(symbols):
            if s not in self.sym2og[species]:
                d = self.priv_det[species].get(s, 0.0)
                if det[i] > d:
                    self.priv_det[species][s] = float(det[i])
                cand += 1
        log(f"  私有候选扫描 ({species}): {cand} 个非 shared 列")

    def build(self):
        """确定最终列: shared OG (全保留) + 检出达标的私有基因。"""
        nodes = []
        for og_id in sorted(self.og[(self.og.n_mouse > 0) & (self.og.n_human > 0)].index):
            nodes.append((og_id, "shared_og"))
        n_priv = {"mouse": 0, "human": 0}
        for sp, tag in (("mouse", "M"), ("human", "H")):
            for s, d in sorted(self.priv_det[sp].items()):
                if d >= self.priv_min_det:
                    nodes.append((f"{tag}:{s}", f"private_{sp}"))
                    n_priv[sp] += 1
        log(f"列空间: {len(nodes)} = shared {len(nodes)-sum(n_priv.values())} "
            f"+ 鼠私有 {n_priv['mouse']} + 人私有 {n_priv['human']}")
        self.nodes = nodes
        self.node_index = {n: i for i, (n, _) in enumerate(nodes)}
        return nodes

    def mapping_matrix(self, species, symbols):
        """(n_raw × n_nodes) 稀疏映射: 基因→shared OG 或私有列 (求和聚合)。"""
        rows, cols = [], []
        n_drop = 0
        for i, s in enumerate(symbols):
            og_id = self.sym2og[species].get(s)
            if og_id is not None:
                j = self.node_index.get(og_id)
            else:
                tag = "M" if species == "mouse" else "H"
                j = self.node_index.get(f"{tag}:{s}")
            if j is None:
                n_drop += 1
                continue
            rows.append(i)
            cols.append(j)
        S = sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                          shape=(len(symbols), len(self.nodes)))
        log(f"  映射 ({species}): {len(rows)}/{len(symbols)} 基因落列, 丢弃 {n_drop} (低检出私有)")
        return S

    def var_table(self):
        rows = []
        for node, kind in self.nodes:
            if kind == "shared_og":
                r = self.og.loc[node]
                rows.append({"node_id": node, "kind": kind, "og_id": node,
                             "symbols": f"{r['mouse_symbols']}|{r['human_symbols']}",
                             "n_mouse": r["n_mouse"], "n_human": r["n_human"]})
            else:
                sp = "mouse" if kind == "private_mouse" else "human"
                sym = node.split(":", 1)[1]
                rows.append({"node_id": node, "kind": kind, "og_id": "",
                             "symbols": sym, "n_mouse": int(sp == "mouse"),
                             "n_human": int(sp == "human")})
        return pd.DataFrame(rows).set_index("node_id")


def aggregate(X, S):
    """原始计数 → node 空间求和 → normalize_total(1e4) → log1p (与原流程同序)。"""
    Xn = (X @ S).tocsr()
    a = sc.AnnData(X=Xn)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a.X.tocsr()


def build_mouse_brain(cs, report):
    log("== 鼠脑 (955-958 h5, 限 brain_celltypes 5000 细胞) ==")
    ctbl = pd.read_csv(LABEL_DIR / "brain_celltypes.csv", index_col=0)["celltype"]
    parts, meta = [], []
    for s in ["955", "956", "957", "958"]:
        a = sc.read_10x_h5(RAW / f"GSM9703846_{s}_sample_filtered_feature_bc_matrix.h5")
        keep = a.obs_names.astype(str).isin(ctbl.index)
        a = a[keep].copy()
        meta.append(pd.DataFrame({"sample": s, "genotype": SAMPLE2GENO[s],
                                  "mouse": SAMPLE2MOUSE[s]}, index=a.obs_names.astype(str)))
        parts.append(a)
        log(f"  {s}: 交集 {keep.sum()} 细胞 × {a.shape[1]} 基因")
    symbols = parts[0].var_names.astype(str)
    for p in parts[1:]:
        assert list(p.var_names.astype(str)) == list(symbols), "鼠脑样本间基因不一致"
    X = sp.vstack([p.X.tocsr() for p in parts], format="csr")
    obs = pd.concat(meta)
    obs["tissue"] = "Brain"
    obs["celltype"] = ctbl.reindex(obs.index).astype(str)
    assert (obs["celltype"] != "nan").all(), "脑细胞类型存在未匹配!"
    cs.scan_dataset("mouse", list(symbols), X)
    S = cs.mapping_matrix("mouse", list(symbols))
    Xn = aggregate(X, S)
    out = sc.AnnData(X=Xn, obs=obs[["sample", "genotype", "mouse", "tissue", "celltype"]],
                     var=cs.var_table())
    out.write_h5ad(OUT_DIR / "mouse_brain.h5ad")
    report["mouse_brain"] = {"cells": int(Xn.shape[0]), "raw_genes": len(symbols)}
    log(f"  已保存 mouse_brain.h5ad: {Xn.shape}, 类型 {obs['celltype'].value_counts().to_dict()}")


def build_mouse_bm12(cs, report):
    log("== 鼠血 BM1/BM2 (CMO 分形) ==")
    fr = pd.read_csv(MAP_DIR / "GSE329430_feature_reference.csv.gz")
    fr = fr[fr["feature_type"] == "Multiplexing Capture"]
    cmap = {r["id"]: (str(r["multiplexed_sample"]).split("_")[0],
                      str(r["multiplexed_sample"]).split("_")[-1],
                      r["multiplexed_sample"]) for _, r in fr.iterrows()}
    ctbl = pd.read_csv(LABEL_DIR / "bm12_celltypes.csv", index_col=0)["celltype"]
    parts, metas = [], []
    for bidx, (lane, prefix) in enumerate({"BM1": "GSM9703838_BM1", "BM2": "GSM9703839_BM2"}.items()):
        a = sc.read_10x_mtx(RAW, var_names="gene_symbols", prefix=f"{prefix}_")
        tbl = pd.read_csv(RAW / f"{prefix}_assignment_confidence_table.csv.gz")
        valid = (~tbl["Assignment"].isin(["Multiplet", "Blank", "Unassigned"])) & \
                (tbl["Assignment_Probability"] >= 0.8)
        tbl = tbl[valid].copy()
        tbl = tbl[tbl["Barcode"].isin(pd.Index(a.obs_names.astype(str)))]
        a = a[tbl["Barcode"].values, :]
        obs = tbl.set_index("Barcode")[["Assignment"]].copy()
        obs.columns = ["cmo"]
        obs["genotype"] = [cmap[c][0] for c in obs["cmo"]]
        obs["origin"] = [cmap[c][1] for c in obs["cmo"]]
        obs["mouse"] = [cmap[c][2] for c in obs["cmo"]]
        obs["lane"] = lane
        obs.index = [f"{bc}-{bidx}" for bc in obs.index]  # 与 bm12_celltypes 对齐
        parts.append(a)
        metas.append(obs)
        log(f"  {lane}: 分形有效 {a.shape[0]} 细胞")
    symbols = parts[0].var_names.astype(str)
    assert list(parts[1].var_names.astype(str)) == list(symbols)
    X = sp.vstack([p.X.tocsr() for p in parts], format="csr")
    obs = pd.concat(metas)
    keep = qc_mask(X)
    X, obs = X[keep], obs[keep]
    log(f"  QC: {keep.sum()}/{len(keep)} 保留")
    obs["celltype"] = ctbl.reindex(obs.index).astype(str)
    n_nan = int((obs["celltype"] == "nan").sum())
    if n_nan:
        log(f"  [警告] {n_nan} 细胞无 bm12_celltypes 标签, 记 'nan'")
    cs.scan_dataset("mouse", list(symbols), X)
    S = cs.mapping_matrix("mouse", list(symbols))
    Xn = aggregate(X, S)
    out = sc.AnnData(X=Xn, obs=obs[["genotype", "origin", "mouse", "lane", "cmo", "celltype"]],
                     var=cs.var_table())
    out.write_h5ad(OUT_DIR / "mouse_blood_bm12.h5ad")
    report["mouse_blood_bm12"] = {"cells": int(Xn.shape[0]), "raw_genes": len(symbols),
                                  "ct_unmatched": n_nan}
    log(f"  已保存 mouse_blood_bm12.h5ad: {Xn.shape}")


def build_mouse_bm34(cs, report):
    log("== 鼠血 BM3/BM4 (hto demux, marker 打分) ==")
    parts, metas = [], []
    for lane, prefix in {"BM3": "GSM9703842_BM3", "BM4": "GSM9703843_BM4"}.items():
        a = sc.read_10x_h5(RAW / f"{prefix}_filtered_feature_bc_matrix.h5")
        dem = pd.read_csv(RAW / "hto" / f"{lane}_demux_assignments.csv")
        dem = dem[(dem["genotype"] != "") & (dem["assignment_confidence"] >= 0.8)]
        dem = dem.set_index("barcode")
        hit = pd.Index(a.obs_names.astype(str)).isin(dem.index)
        a = a[hit].copy()
        obs = pd.DataFrame({"genotype": dem.loc[a.obs_names, "genotype"].values,
                            "mouse": dem.loc[a.obs_names, "mouse"].values,
                            "cmo": dem.loc[a.obs_names, "hashtag"].values,
                            "lane": lane}, index=a.obs_names.astype(str))
        parts.append(a)
        metas.append(obs)
        log(f"  {lane}: 分形交集 {a.shape[0]} 细胞")
    symbols = parts[0].var_names.astype(str)
    X = sp.vstack([p.X.tocsr() for p in parts], format="csr")
    obs = pd.concat(metas)
    keep = qc_mask(X)
    X, obs = X[keep], obs[keep]
    log(f"  QC: {keep.sum()}/{len(keep)} 保留")
    cs.scan_dataset("mouse", list(symbols), X)
    S = cs.mapping_matrix("mouse", list(symbols))
    Xn = aggregate(X, S)
    # 细胞类型: 在聚合后矩阵上按 marker→列 打分 (marker 基本 1:1, 与原 11982 空间打分等价)
    var = cs.var_table()
    tmp = sc.AnnData(X=Xn, var=var)
    sym2node = {}
    for node, kind in cs.nodes:
        if kind == "shared_og":
            r = cs.og.loc[node]
            for s in str(r["human_symbols"]).split(";"):
                if s:
                    sym2node.setdefault(s.upper(), node)
        else:
            sym2node.setdefault(node.split(":", 1)[1].upper(), node)
    avail = set(sym2node)
    ren = {}
    for k, v in BLOOD_MARKERS_BM34.items():
        cols = [sym2node[s.upper()] for s in v if s.upper() in avail]
        if not cols:
            log(f"  [警告] marker 组 {k} 无可用列")
            continue
        sc.tl.score_genes(tmp, cols, score_name=f"score_{k}")
        ren[k] = f"score_{k}"
    labels = list(ren.keys())
    Sc = tmp.obs[[ren[k] for k in labels]].to_numpy()
    obs["celltype"] = [labels[i] for i in Sc.argmax(axis=1)]
    out = sc.AnnData(X=Xn, obs=obs[["genotype", "mouse", "lane", "cmo", "celltype"]], var=var)
    out.write_h5ad(OUT_DIR / "mouse_blood_bm34.h5ad")
    report["mouse_blood_bm34"] = {"cells": int(Xn.shape[0]), "raw_genes": len(symbols)}
    log(f"  已保存 mouse_blood_bm34.h5ad: {Xn.shape}, 类型 {obs['celltype'].value_counts().to_dict()}")


def build_human_brain(cs, report):
    log("== 人脑 GSE138852 (注意: 仅 10850 预过滤基因) ==")
    counts = pd.read_csv(HUMAN_BRAIN / "GSE138852_counts.csv.gz", index_col=0)
    cov = pd.read_csv(HUMAN_BRAIN / "GSE138852_covariates.csv.gz", index_col=0)
    cov.columns = [c.split(".")[-1] for c in cov.columns]
    cov["disease"] = cov["batchCond"].map({"AD": "AD", "ct": "CTRL"})
    cov["donor_pair"] = [i.split("_", 1)[1] if "_" in i else "NA" for i in cov.index]
    common = cov.index.intersection(counts.columns)
    counts = counts[common]
    cov = cov.loc[common]
    keep = ~cov["cellType"].isin(["doublet", "unID"])
    cov = cov[keep]
    counts = counts[cov.index]
    log(f"  去 doublet/unID: {counts.shape[1]} 细胞 × {counts.shape[0]} 基因")
    symbols = pd.Index(counts.index.astype(str))
    X = sp.csr_matrix(counts.T.values.astype(np.float32))
    obs = cov[["disease", "cellType", "donor_pair"]].rename(columns={"cellType": "celltype"})
    cs.scan_dataset("human", list(symbols), X)
    S = cs.mapping_matrix("human", list(symbols))
    Xn = aggregate(X, S)
    out = sc.AnnData(X=Xn, obs=obs, var=cs.var_table())
    out.write_h5ad(OUT_DIR / "human_brain.h5ad")
    report["human_brain"] = {"cells": int(Xn.shape[0]), "raw_genes": len(symbols)}
    log(f"  已保存 human_brain.h5ad: {Xn.shape}, 疾病 {obs['disease'].value_counts().to_dict()}")


def build_human_blood(cs, report):
    log("== 人血 GSE181279 (五供体) ==")
    parts = []
    for donor in ["AD1", "AD2", "AD3", "NC1", "NC2"]:
        a = sc.read_10x_mtx(HUMAN_BLOOD, var_names="gene_symbols",
                            prefix=f"GSM{DONOR_GSM[donor]}_{donor}_")
        a.obs["donor"] = donor
        parts.append(a)
        log(f"  {donor}: {a.shape[0]} 细胞")
    a = parts[0].concatenate(*parts[1:], join="outer", fill_value=0)
    symbols = a.var_names.astype(str)
    avail = set(symbols)
    for k, v in PBMC_MARKERS.items():
        sc.tl.score_genes(a, [g for g in v if g in avail], score_name=f"score_{k}")
    labels = list(PBMC_MARKERS.keys())
    Sc = a.obs[[f"score_{k}" for k in labels]].to_numpy()
    a.obs["celltype"] = [labels[i] for i in Sc.argmax(axis=1)]
    X = a.X.tocsr()
    keep = qc_mask(X)
    X = X[keep]
    obs = a.obs[["donor", "celltype"]][keep].copy()
    obs["disease"] = obs["donor"].map(DONOR2DISEASE)
    log(f"  QC: {keep.sum()}/{len(keep)} 保留")
    cs.scan_dataset("human", list(symbols), X)
    S = cs.mapping_matrix("human", list(symbols))
    Xn = aggregate(X, S)
    out = sc.AnnData(X=Xn, obs=obs[["disease", "donor", "celltype"]].reset_index(drop=True),
                     var=cs.var_table())
    out.write_h5ad(OUT_DIR / "human_blood.h5ad")
    report["human_blood"] = {"cells": int(Xn.shape[0]), "raw_genes": len(symbols)}
    log(f"  已保存 human_blood.h5ad: {Xn.shape}, 类型 {obs['celltype'].value_counts().to_dict()}")


def build_esm_cache(cs):
    log("== ESM2 member-level 缓存 (OG=成员均值) ==")
    import torch
    per_gene = torch.load(ESM_CACHE, map_location="cpu", weights_only=False)
    log(f"  单基因缓存: {len(per_gene)} 条")
    cache, n_cov = {}, 0
    for node, kind in cs.nodes:
        if kind == "shared_og":
            r = cs.og.loc[node]
            syms = [s for s in str(r["human_symbols"]).split(";") if s] + \
                   [s for s in str(r["mouse_symbols"]).split(";") if s]
            vecs = [np.asarray(per_gene[s.upper()], np.float32) for s in syms
                    if s.upper() in per_gene]
        else:
            sym = node.split(":", 1)[1]
            vecs = [np.asarray(per_gene[sym.upper()], np.float32)] if sym.upper() in per_gene else []
        if vecs:
            cache[node] = np.mean(vecs, axis=0).astype(np.float32)
            n_cov += 1
        else:
            cache[node] = np.zeros(640, np.float32)
    torch.save(cache, PROJECT / "cache/esm2_embeddings/esm2_member_level_dim640.pt")
    log(f"  已保存 esm2_member_level_dim640.pt: {len(cache)} 节点, 覆盖 {n_cov} ({n_cov/len(cache):.1%})")
    return n_cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priv-min-det", type=float, default=0.005,
                    help="私有基因保留阈值: 至少一个同物种数据集中的检出率 (默认 0.005)")
    ap.add_argument("--skip-esm", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cs = ColumnSpace(args.priv_min_det)
    report = {"priv_min_det": args.priv_min_det}

    # 两遍: 先扫描私有候选检出率, 再建列空间, 再聚合
    # 为省内存, 每个数据集 扫描→(build 后)聚合 分两段进行; 这里按数据集顺序处理,
    # 但 mapping 依赖最终列空间 → 先对所有数据集做 scan, 再统一 aggregate。
    # 实现: scan 在 build_* 的第一遍里完成代价高, 故拆成两个循环。
    import gc

    def scan_all():
        # 鼠脑
        ctbl = pd.read_csv(LABEL_DIR / "brain_celltypes.csv", index_col=0)["celltype"]
        for s in ["955", "956", "957", "958"]:
            a = sc.read_10x_h5(RAW / f"GSM9703846_{s}_sample_filtered_feature_bc_matrix.h5")
            keep = a.obs_names.astype(str).isin(ctbl.index)
            cs.scan_dataset("mouse", list(a.var_names.astype(str)), a.X.tocsr()[keep])
            del a
            gc.collect()
        # BM12
        for prefix in ["GSM9703838_BM1", "GSM9703839_BM2"]:
            a = sc.read_10x_mtx(RAW, var_names="gene_symbols", prefix=f"{prefix}_")
            tbl = pd.read_csv(RAW / f"{prefix}_assignment_confidence_table.csv.gz")
            valid = (~tbl["Assignment"].isin(["Multiplet", "Blank", "Unassigned"])) & \
                    (tbl["Assignment_Probability"] >= 0.8)
            tbl = tbl[valid]
            tbl = tbl[tbl["Barcode"].isin(pd.Index(a.obs_names.astype(str)))]
            X = a[tbl["Barcode"].values, :].X.tocsr()
            keep = qc_mask(X)
            cs.scan_dataset("mouse", list(a.var_names.astype(str)), X[keep])
            del a, X
            gc.collect()
        # BM34
        for lane, prefix in {"BM3": "GSM9703842_BM3", "BM4": "GSM9703843_BM4"}.items():
            a = sc.read_10x_h5(RAW / f"{prefix}_filtered_feature_bc_matrix.h5")
            dem = pd.read_csv(RAW / "hto" / f"{lane}_demux_assignments.csv")
            dem = dem[(dem["genotype"] != "") & (dem["assignment_confidence"] >= 0.8)]
            hit = pd.Index(a.obs_names.astype(str)).isin(set(dem["barcode"]))
            X = a.X.tocsr()[hit]
            keep = qc_mask(X)
            cs.scan_dataset("mouse", list(a.var_names.astype(str)), X[keep])
            del a, X
            gc.collect()
        # 人脑
        counts = pd.read_csv(HUMAN_BRAIN / "GSE138852_counts.csv.gz", index_col=0)
        cov = pd.read_csv(HUMAN_BRAIN / "GSE138852_covariates.csv.gz", index_col=0)
        common = cov.index.intersection(counts.columns)
        X = sp.csr_matrix(counts[common].T.values.astype(np.float32))
        cs.scan_dataset("human", list(counts.index.astype(str)), X)
        del counts, X
        gc.collect()
        # 人血
        for donor in ["AD1", "AD2", "AD3", "NC1", "NC2"]:
            a = sc.read_10x_mtx(HUMAN_BLOOD, var_names="gene_symbols",
                                prefix=f"GSM{DONOR_GSM[donor]}_{donor}_")
            X = a.X.tocsr()
            keep = qc_mask(X)
            cs.scan_dataset("human", list(a.var_names.astype(str)), X[keep])
            del a, X
            gc.collect()

    log("── 第一遍: 私有基因检出率扫描 ──")
    scan_all()
    log(f"私有候选: 鼠 {len(cs.priv_det['mouse'])}, 人 {len(cs.priv_det['human'])}")
    cs.build()

    log("── 第二遍: 聚合写出 ──")
    build_mouse_brain(cs, report)
    build_mouse_bm12(cs, report)
    build_mouse_bm34(cs, report)
    build_human_brain(cs, report)
    build_human_blood(cs, report)

    cs.var_table().to_csv(OUT_DIR / "columns.csv")
    if not args.skip_esm:
        report["esm_covered"] = int(build_esm_cache(cs))
    report["n_nodes"] = len(cs.nodes)
    report["n_private_mouse"] = int(sum(1 for _, k in cs.nodes if k == "private_mouse"))
    report["n_private_human"] = int(sum(1 for _, k in cs.nodes if k == "private_human"))
    with open(OUT_DIR / "build_report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log(f"完成: {json.dumps({k: v for k, v in report.items() if k != 'mouse_brain'}, ensure_ascii=False)[:400]}")


if __name__ == "__main__":
    main()
