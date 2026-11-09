"""Reproduce the fixed v3 dynamic benchmark from archived, verified source inputs.

No backbone training or parameter search. Refit only alpha=1 readouts within
each original training fold. Raw-input preparation remains in the frozen
2026-11-08 execution archive referenced by the configuration.
"""
import json
import time
import numpy as np
import pandas as pd
import torch
from dynamic_model import V3DynamicPredictor, fit_readouts, predict_readout, sha256
from headline_metrics import delta_pearson, top_k_overlap
from project_paths import CONFIG, RESULTS, configured_path


def main():
    started = time.time()
    torch.set_num_threads(8)
    package = V3DynamicPredictor(configured_path('model_package'))
    inputs_path = configured_path('benchmark_inputs')
    for name, expected in CONFIG['input_hashes'].items():
        if sha256(inputs_path/name) != expected:
            raise ValueError(f'Frozen input changed: {name}')
    inputs = dict(np.load(inputs_path/'selected_world_inputs.npz',allow_pickle=False))
    states = dict(np.load(inputs_path/'state_means.npz',allow_pickle=False))
    nodes = inputs['nodes']
    if list(nodes)!=list(states['nodes']) or list(nodes)!=package.backbone.genes:
        raise ValueError('Checkpoint/input node order mismatch')
    folds = package.metadata['folds']
    gsms = sorted({g for info in folds.values() for g in info['train']+info['test']})
    world = {g:package.backbone.predict_cells(inputs[g],nodes).mean(0) for g in gsms}
    for g in gsms:
        np.testing.assert_allclose(world[g],states[f'world__{g}'],atol=1e-7,rtol=1e-6)
    idx = package.node_ids
    rows, responses, head_arrays = [], {}, {'node_ids':idx}
    for fold, info in folds.items():
        train,test = info['train'],info['test']
        if set(train)&set(test):raise ValueError('Overlapping train/test samples')
        source=np.stack([states[f'mono__{g}'][idx] for g in train])
        latent=np.stack([world[g][idx] for g in train])
        target=np.stack([states[f'micro__{g}'][idx] for g in train])
        heads=fit_readouts(source,latent,target,alpha=CONFIG['ridge_alpha'])
        for name,value in heads['combined_linear'].items():
            np.testing.assert_array_equal(value,package.heads[f'{fold}__{name}'])
            head_arrays[f'{fold}__{name}']=value
        predictions={method:{g:predict_readout(states[f'mono__{g}'][idx],world[g][idx],params,method)
                             for g in test} for method,params in heads.items()}
        predictions['training_mean']={g:target.mean(0) for g in test}
        predictions['v3_frozen_direct']={g:world[g][idx] for g in test}
        observed_control=np.mean([states[f'micro__{g}'][idx] for g in info['test_saline']],axis=0)
        for method, values in predictions.items():
            predicted_control=np.mean([values[g] for g in info['test_saline']],axis=0)
            for g in info['test_lps']:
                delta=values[g]-predicted_control
                observed=states[f'micro__{g}'][idx]-observed_control
                mse=float(np.mean((delta-observed)**2));zero=float(np.mean(observed**2))
                rows.append(dict(train_processing=fold,test_processing=info['test_processing'],gsm=g,method=method,
                                 mse=mse,zero_response_mse=zero,relative_mse_reduction=1-mse/zero,
                                 delta_pearson=delta_pearson(observed,delta),top100_overlap=top_k_overlap(observed,delta,k=100)))
                responses[f'{fold}__{g}__{method}']=delta
                responses[f'{fold}__{g}__observed']=observed
    table=pd.DataFrame(rows)
    expected=pd.read_csv(inputs_path/'verified_metrics.csv')
    expected['method']=expected.method.replace({'zero_response':'training_mean'})
    cols=['mse','zero_response_mse','relative_mse_reduction','delta_pearson','top100_overlap']
    a=table.sort_values(['method','gsm']).reset_index(drop=True)
    b=expected.sort_values(['method','gsm']).reset_index(drop=True)
    np.testing.assert_array_equal(a[['method','gsm']],b[['method','gsm']])
    np.testing.assert_allclose(a[cols],b[cols],atol=1e-12,rtol=1e-10,equal_nan=True)
    out=RESULTS/'adaptation';out.mkdir(parents=True,exist_ok=True)
    table.to_csv(out/'metrics_by_test_animal.csv',index=False)
    comparison=pd.concat([table,pd.read_csv(inputs_path/'v2_reference.csv')],ignore_index=True)
    comparison.to_csv(out/'comparison_by_sample.csv',index=False)
    summary=table.groupby(['train_processing','method']).agg(mse=('mse','mean'),zero_response_mse=('zero_response_mse','mean'),relative_mse_reduction=('relative_mse_reduction','mean'),delta_pearson=('delta_pearson','mean'),top100_overlap=('top100_overlap','mean')).reset_index()
    summary.to_csv(out/'metrics_summary.csv',index=False)
    np.savez_compressed(out/'responses.npz',**responses)
    np.savez_compressed(out/'heads.npz',**head_arrays)
    np.savez_compressed(out/'state_means.npz',**{**states,**{f'world__{g}':world[g] for g in gsms}})
    (out/'folds.json').write_text(json.dumps(folds,indent=2)+'\n')
    (out/'run_summary.json').write_text(json.dumps(dict(status='complete_verified',backbone='official_v3_1500',
        backbone_sha256=package.metadata['backbone_original_sha256'],ridge_alpha=CONFIG['ridge_alpha'],
        matched_verified_20261108_results=True,backbone_training=False,runtime_seconds=time.time()-started,
        scope=CONFIG['claim_scope'],input_policy=package.metadata['input_policy']),indent=2)+'\n')
    from summarize_dynamic_headline import main as summarize
    from export_dynamic_responses import main as export
    from build_lps_programs import main as programs
    summarize();export();programs()
    print('OFFICIAL_V3_DYNAMIC_BENCHMARK_VERIFIED',flush=True)


if __name__=='__main__':main()
