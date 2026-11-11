"""Rebuild the released GSE245309 model inputs and target states from the GEO count matrices."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.mixture import GaussianMixture

from dynamic_model import V3DynamicPredictor
from evaluate_lps_brain_v2 import SAMPLES, load_sample, mouse_mapping
from project_paths import configured_path


MONOCYTE = {"ly6c2", "plac8", "ccr2", "fn1", "s100a8", "s100a9", "lyz2", "vcan", "f13a1"}


def normalized_log(matrix):
    total = np.asarray(matrix.sum(axis=1)).ravel()
    factors = np.divide(1e4, total, out=np.zeros(len(total)), where=total > 0)
    result = (sp.diags(factors) @ matrix).tocsr()
    result.data = np.log1p(result.data)
    return result


def upper_component(scores):
    mixture = GaussianMixture(n_components=2, random_state=42).fit(scores[:, None])
    return mixture, int(mixture.means_.argmax())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True, type=Path, help='Directory of extracted GSM 10x files')
    parser.add_argument('--out', required=True, type=Path, help='New directory for reconstructed inputs')
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('--out must be a new or empty directory')
    args.out.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(8)
    model = V3DynamicPredictor(configured_path('model_package')).backbone
    genes = pd.read_csv(configured_path('genes_file'))
    members = pd.read_csv(configured_path('members_file'))
    nodes = genes.node_id.to_numpy(dtype=str)
    if list(nodes) != model.genes:
        raise ValueError('Model and mapping coordinate orders differ')
    samples = {gsm: load_sample(gsm, args.raw) for gsm in SAMPLES}
    micro_fit, micro_high = upper_component(np.concatenate([s['marker_score'] for s in samples.values()]))
    inputs, states, counts = {'nodes': nodes}, {'nodes': nodes}, []
    for gsm, sample in samples.items():
        matrix = sample['matrix']
        micro = micro_fit.predict(sample['marker_score'][:, None]) == micro_high
        marker_ids = [i for i, g in enumerate(sample['genes']) if g.lower() in MONOCYTE]
        scores = np.asarray(normalized_log(matrix)[:, marker_ids].mean(axis=1)).ravel()
        mono_fit, mono_high = upper_component(scores[~micro])
        mono = np.zeros(len(micro), dtype=bool)
        mono[~micro] = mono_fit.predict(scores[~micro, None]) == mono_high
        mapping = mouse_mapping(genes, members, sample['genes'])
        target = (matrix[micro] @ mapping).tocsr()
        source = (matrix[mono] @ mapping).tocsr()
        states[f'micro__{gsm}'] = np.asarray(normalized_log(target).mean(axis=0)).ravel()
        states[f'detection__{gsm}'] = target.getnnz(axis=0) / target.shape[0]
        states[f'mono__{gsm}'] = np.asarray(normalized_log(source).mean(axis=0)).ravel()
        if source.shape[0] > 128:
            selected = np.random.default_rng(42).choice(source.shape[0], 128, replace=False)
            source = source[selected]
        inputs[gsm] = normalized_log(source).toarray().astype(np.float32)
        states[f'world__{gsm}'] = model.predict_cells(inputs[gsm], nodes).mean(axis=0)
        treatment, processing, animal = SAMPLES[gsm]
        counts.append(dict(gsm=gsm, treatment=treatment, processing=processing, animal=animal,
                           micro_cells=int(micro.sum()), mono_cells=int(mono.sum())))
    pd.DataFrame(counts).to_csv(args.out / 'sample_cell_counts.csv', index=False)
    for filename, arrays in [('selected_world_inputs.npz', inputs), ('state_means.npz', states)]:
        np.savez_compressed(args.out / filename, **arrays)
        with np.load(configured_path('benchmark_inputs') / filename, allow_pickle=False) as ref:
            if set(arrays) != set(ref.files):
                raise ValueError(f'Coordinate or sample keys differ in {filename}')
            for key, value in arrays.items():
                if key == 'nodes':
                    np.testing.assert_array_equal(value, ref[key])
                else:
                    atol = 3e-6 if key.startswith('world__') else 1e-7
                    np.testing.assert_allclose(value, ref[key], atol=atol, rtol=1e-6)
    print('GSE245309 raw-count reconstruction verified for all eight samples.')


if __name__ == '__main__':
    main()
