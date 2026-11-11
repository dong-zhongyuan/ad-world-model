#!/usr/bin/env python3
"""BM3/BM4 HTO 分形: 直接用 h5 内置 Antibody Capture 计数 (无需 fastq)。

每泳道 4 个 tag, 逐 tag 在 log10(count+1) 上做 2 类 kmeans 门控,
singlet=恰好 1 个阳性 tag → 赋基因型/鼠个体; ≥2 阳性 → multiplet; 0 阳性 → unassigned。
产出: data/single-cell/GSE329430_raw/hto/BM{3,4}_demux_assignments.csv
  列: barcode,hashtag,genotype,mouse,hto_counts,assignment_confidence
"""
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans

BASE = Path.home() / "dzy" / "ad_omics_v2" / "data" / "single-cell" / "GSE329430_raw"

LANES = {
    "BM3": dict(h5="GSM9703842_BM3_filtered_feature_bc_matrix.h5",
                tags={"Hashtag_3": ("5xFAD", "5xFAD_1"), "Hashtag_4": ("WT", "WT_1"),
                      "Hashtag_5": ("WT", "WT_2"), "Hashtag_8": ("5xFAD", "5xFAD_2")}),
    "BM4": dict(h5="GSM9703843_BM4_filtered_feature_bc_matrix.h5",
                tags={"Hashtag_6": ("WT", "WT_3"), "Hashtag_7": ("WT", "WT_4"),
                      "Hashtag_9": ("5xFAD", "5xFAD_3"), "Hashtag_10": ("5xFAD", "5xFAD_4")}),
}


def gate(vals):
    """log10(count+1) 2-类 kmeans, 返回阈值中点。"""
    x = np.log10(vals.astype(float) + 1)
    nz = x[vals > 0]
    if len(nz) < 50 or nz.max() == nz.min():
        return np.inf
    init = np.array([[np.percentile(nz, 10)], [np.percentile(nz, 99)]])
    km = KMeans(n_clusters=2, n_init=5, init=init, random_state=0).fit(nz.reshape(-1, 1))
    c = sorted(km.cluster_centers_.flatten())
    return (c[0] + c[1]) / 2.0


def main():
    for lane, cfg in LANES.items():
        f = h5py.File(BASE / cfg["h5"], "r")
        feat = f["matrix/features"]
        names = feat["name"][:].astype(str)
        types = feat["feature_type"][:].astype(str)
        barcodes = f["matrix/barcodes"][:].astype(str)
        import scipy.sparse as sp
        M = sp.csc_matrix((f["matrix/data"][:], f["matrix/indices"][:], f["matrix/indptr"][:]),
                          shape=f["matrix/shape"][:])
        tag_names = list(cfg["tags"].keys())
        idx = [int(np.where(names == f"Ms.{t}")[0][0]) for t in tag_names]
        AB = np.asarray(M[idx, :].T.todense())  # cells × 4 tags
        print(f"[{lane}] {AB.shape[0]} 细胞 × {AB.shape[1]} tags, 总 UMI={AB.sum():.3e}", flush=True)

        # 逐 tag 门控
        thr = {t: gate(AB[:, j]) for j, t in enumerate(tag_names)}
        print(f"[{lane}] 门控阈值(log10): " + ", ".join(f"{t}={thr[t]:.2f}" for t in tag_names), flush=True)
        logx = np.log10(AB.astype(float) + 1)
        pos = np.column_stack([logx[:, j] >= thr[t] for j, t in enumerate(tag_names)])
        npos = pos.sum(axis=1)

        total_counts = AB.sum(axis=1).clip(min=1)
        best = AB.argmax(axis=1)
        recs = []
        for i in range(AB.shape[0]):
            bc = barcodes[i]
            if npos[i] == 1:
                t = tag_names[int(np.where(pos[i])[0][0])]
                g, m = cfg["tags"][t]
                conf = AB[i, int(np.where(pos[i])[0][0])] / total_counts[i]
                recs.append((bc, t, g, m, int(AB[i].sum()), float(conf)))
            elif npos[i] >= 2:
                recs.append((bc, "Multiplet", "", "", int(AB[i].sum()), np.nan))
            else:
                recs.append((bc, "Unassigned", "", "", int(AB[i].sum()), np.nan))
        df = pd.DataFrame(recs, columns=["barcode", "hashtag", "genotype", "mouse",
                                         "hto_counts", "assignment_confidence"])
        out = BASE / "hto" / f"{lane}_demux_assignments.csv"
        df.to_csv(out, index=False)
        sing = df["hashtag"].value_counts()
        print(f"[{lane}] 分形: {sing.to_dict()}", flush=True)
        if "Multiplet" in sing or "Unassigned" in sing:
            pass
        sub = df[df["genotype"] != ""]
        print(f"[{lane}] singlet 基因型: {sub['genotype'].value_counts().to_dict()}, "
              f"鼠个体: {sub['mouse'].value_counts().to_dict()}, "
              f"confidence 中位={sub['assignment_confidence'].median():.3f}", flush=True)
        print(f"[{lane}] 已保存: {out}", flush=True)


if __name__ == "__main__":
    main()
