"""Build the five atlas inputs on the exact released molecular coordinates."""
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import anndata as ad

import build_member_level as builder

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'


def main():
    builder.RAW = DATA / 'raw/GSE329430'
    builder.HUMAN_BRAIN = DATA / 'raw/GSE138852'
    builder.HUMAN_BLOOD = DATA / 'raw/GSE181279'
    builder.MAP_DIR = DATA / 'reference/v2/mappings'
    builder.OUT_DIR = DATA / 'reference/v2/member_level'
    builder.LABEL_DIR = DATA / 'reference/mouse_labels'
    builder.OUT_DIR.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(DATA / 'mappings/orthogroups_member_level.csv',
                    builder.MAP_DIR / 'orthogroups_member_level.csv')
    coordinates = DATA / 'member_level/columns.csv'
    table = pd.read_csv(coordinates)
    cs = builder.ColumnSpace(priv_min_det=0.005)
    cs.nodes = list(zip(table.node_id, table.kind))
    cs.node_index = {name: i for i, (name, _) in enumerate(cs.nodes)}
    report = {}
    for build in (builder.build_mouse_brain, builder.build_mouse_bm12, builder.build_mouse_bm34,
                  builder.build_human_brain, builder.build_human_blood):
        build(cs, report)
    shutil.copyfile(coordinates, builder.OUT_DIR / 'columns.csv')
    # Leiden annotations are tied to the frozen GSE181279 row order.
    # Match identifiers, donors and marker labels before attaching them.
    path = builder.OUT_DIR / 'human_blood.h5ad'
    atlas = ad.read_h5ad(path)
    labels = pd.read_csv(DATA / 'reference/human_blood_labels.csv.gz', dtype=str).set_index('row')
    np.testing.assert_array_equal(atlas.obs_names, labels.index)
    for key in ('donor', 'celltype'):
        np.testing.assert_array_equal(atlas.obs[key].astype(str), labels[key])
    for key in ('leiden', 'celltype_leiden'):
        atlas.obs[key] = labels[key].to_numpy()
    atlas.write_h5ad(path)
    cache = DATA / 'intermediate/restored_full_inputs'
    cache.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(DATA / 'processed/protein_annotations.csv', cache / 'protein_annotations.csv')
    print('Built five reference atlases on the released coordinate set.')


if __name__ == '__main__':
    main()
