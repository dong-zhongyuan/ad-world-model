#!/usr/bin/env python3
"""鼠血 T 细胞细分 CD4T/CD8T (v5.2 cond 拆分前置, member-level OG 空间版)。

旧 1:1 symbol 空间缺 CD8A/CD8B(落多基因同源组), 故在 member-level OG 空间打分:
  s4 = OG(CD4) 表达; s8 = mean(OG(CD8A), OG(CD8B)) 表达
  T 细胞中: s4>s8 → CD4T; s8>s4 → CD8T; max(s4,s8)<0.1 → 保持 "T"(不进 cond)
写回 member-level h5ad 的 celltype 列 (bm12/bm34 两版)。
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

MIN_EXPR = 0.1


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def og_node_for(og_flat, human_symbol):
    hit = og_flat[og_flat["human_symbols"].str.contains(
        rf"(?:^|;){human_symbol}(?:;|$)", na=False, regex=True)]
    hit = hit[(hit.n_mouse > 0) & (hit.n_human > 0)]
    return hit["og_id"].tolist()


def split_mouse_t(mapping, member_files):
    og = pd.read_csv(mapping,
                     dtype=str)
    og["n_mouse"] = og["n_mouse"].astype(int)
    og["n_human"] = og["n_human"].astype(int)
    n4 = og_node_for(og, "CD4")
    n8a = og_node_for(og, "CD8A")
    n8b = og_node_for(og, "CD8B")
    log(f"OG 节点: CD4={n4}, CD8A={n8a}, CD8B={n8b}")
    assert n4 and n8a, "CD4/CD8A 不在跨物种 OG 中?"

    for path in member_files:
        a = sc.read_h5ad(path)
        vn = list(a.var_names.astype(str))
        g2i = {g: i for i, g in enumerate(vn)}
        i4 = [g2i[g] for g in n4 if g in g2i]
        i8 = [g2i[g] for g in (n8a + n8b) if g in g2i]
        assert i4 and i8, f"{path.name} 缺 CD4/CD8 OG 列"
        X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
        ct = a.obs["celltype"].astype(str).values.copy()
        is_t = ct == "T"
        log(f"{path.name}: T 细胞 {is_t.sum()} / {a.n_obs}")
        if not is_t.any():
            continue
        s4 = np.asarray(X[is_t][:, i4].mean(axis=1)).ravel()
        s8 = np.asarray(X[is_t][:, i8].mean(axis=1)).ravel()
        new = np.where(np.maximum(s4, s8) < MIN_EXPR, "T",
                       np.where(s4 > s8, "CD4T", "CD8T"))
        ct[is_t] = new
        log(f"  拆分: {pd.Series(new).value_counts().to_dict()} "
            f"(s4均值={s4.mean():.3f}, s8均值={s8.mean():.3f})")
        a.obs["celltype"] = ct
        a.write_h5ad(path)
        log(f"  已写回 {path.name}")
    log("完成")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    member = root / "data/reference/v2/member_level"
    split_mouse_t(root / "data/mappings/orthogroups_member_level.csv",
                  [member / "mouse_blood_bm12.h5ad", member / "mouse_blood_bm34.h5ad"])
