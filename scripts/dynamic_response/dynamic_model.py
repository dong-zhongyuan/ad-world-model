"""Frozen v3 backbone plus explicitly fitted, fold-specific dynamic readouts."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ridge_fit(x, y, alpha=1.0):
    """Same training-only standardization and ridge as the historical benchmark."""
    x = np.asarray(x)
    y = np.asarray(y)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    z = (x - mean) / scale
    intercept = y.mean()
    coefficient = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - intercept))
    return mean, scale, intercept, coefficient


def fit_readouts(source, world, target, alpha=1.0):
    """Inputs are training animals x response genes. No test target accepted."""
    result = {}
    for method in ('source_linear', 'world_linear', 'combined_linear'):
        means, scales, intercepts, coefficients = [], [], [], []
        for j in range(target.shape[1]):
            if method == 'source_linear':
                x = source[:, j:j+1]
            elif method == 'world_linear':
                x = world[:, j:j+1]
            else:
                x = np.column_stack([source[:, j], world[:, j]])
            mean, scale, intercept, coef = ridge_fit(x, target[:, j], alpha)
            means.append(mean); scales.append(scale)
            intercepts.append(intercept); coefficients.append(coef)
        result[method] = dict(mean=np.asarray(means), scale=np.asarray(scales),
                              intercept=np.asarray(intercepts), coef=np.asarray(coefficients))
    return result


def predict_readout(source, world, params, method='combined_linear'):
    source, world = np.asarray(source), np.asarray(world)
    if method == 'source_linear':
        features = source[..., None]
    elif method == 'world_linear':
        features = world[..., None]
    elif method == 'combined_linear':
        features = np.stack([source, world], axis=-1)
    else:
        raise ValueError(method)
    return params['intercept'] + (((features - params['mean']) / params['scale']) * params['coef']).sum(-1)


class FrozenV3:
    def __init__(self, checkpoint, code_dir):
        sys.path.insert(0, str(code_dir))
        from world_model import WorldModel, V3Config
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if ck['step'] != 1500:
            raise ValueError('Expected the official 1500-step checkpoint')
        self.genes = list(ck['gene_names'])
        self.layout = ck['condition_layout']
        self.config = ck['config']
        s = ck['model']
        self.model = WorldModel(s['esm_matrix'].numpy(), s['gene_mean'].numpy(),
                                s['protein_mean'].numpy(), s['protein_sd'].numpy(), V3Config(**ck['config']))
        self.model.load_state_dict(s, strict=True)
        self.model.eval().requires_grad_(False)
        supported = any(r['status'] == 'active' and r['species'] == 'mouse' and
                        r['disease'] == 'CTRL' and r['source_type'] == 'Mono' and r['target_type'] == 'Micro'
                        for r in ck['route_registry'])
        if not supported:
            raise ValueError('Requested route absent from the official checkpoint')
        self.release_checkpoint = {k: ck[k] for k in ('model', 'config', 'step', 'gene_names',
                                                     'protein_features', 'condition_layout', 'route_registry',
                                                     'assay_names', 'assay_masks')}

    @torch.no_grad()
    def predict_cells(self, expression, genes):
        if list(genes) != self.genes:
            raise ValueError('Node identities/order differ from checkpoint')
        x = np.asarray(expression, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != len(self.genes) or not len(x) or not np.isfinite(x).all():
            raise ValueError('Expected finite cells x all model coordinates')
        values = dict(species='mouse', disease='CTRL', blood='Mono', brain='Micro')
        c = torch.zeros((len(x), sum(map(len, self.layout.values()))), dtype=torch.float32)
        start = 0
        for key, labels in self.layout.items():
            c[:, start + labels.index(values[key])] = 1
            start += len(labels)
        # Preserve the historical external benchmark's complete input vector and
        # posterior-mean response inference. No new sampling/mask selection.
        y, _ = self.model.predict_brain(torch.from_numpy(x), c, sample=False, source_mask=None)
        return y.numpy()


class V3DynamicPredictor:
    """Relocatable release; caller explicitly chooses the training-processing fold.

    source_mean: mean log1p(CP10K) over all source cells.
    world_cells: the original fixed-seed 128-cell input used by the benchmark.
    The package contains two cross-validation heads, not an all-data refit.
    """
    def __init__(self, package):
        self.package = Path(package)
        self.metadata = json.loads((self.package / 'metadata.json').read_text())
        for name, expected in self.metadata['file_hashes'].items():
            if sha256(self.package / name) != expected:
                raise ValueError(f'Package hash mismatch: {name}')
        self.backbone = FrozenV3(self.package / 'backbone.pt', self.package / 'model_code')
        self.heads = dict(np.load(self.package / 'heads.npz', allow_pickle=False))
        self.node_ids = self.heads['node_ids']

    def predict_state(self, source_mean, world_cells, genes, train_processing):
        if train_processing not in self.metadata['folds']:
            raise ValueError('Choose an exported training-processing fold')
        source_mean = np.asarray(source_mean)
        if source_mean.shape != (len(self.backbone.genes),) or not np.isfinite(source_mean).all():
            raise ValueError('Expected one finite full-coordinate source mean')
        world = self.backbone.predict_cells(world_cells, genes).mean(axis=0)
        params = {name: self.heads[f'{train_processing}__{name}']
                  for name in ('mean', 'scale', 'intercept', 'coef')}
        return predict_readout(source_mean[self.node_ids], world[self.node_ids], params)

    def predict_response(self, source_mean, world_cells, controls, genes, train_processing):
        """Controls contain source inputs only; no observed brain labels are used."""
        if not controls:
            raise ValueError('At least one control source state required')
        state = self.predict_state(source_mean, world_cells, genes, train_processing)
        reference = np.mean([self.predict_state(m, x, genes, train_processing) for m, x in controls], axis=0)
        return state - reference
