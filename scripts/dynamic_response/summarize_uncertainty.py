"""Summarize biological-unit variability for released prediction benchmarks."""
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/source_data'
FIG = ROOT / 'figure_work/data'


def summary(frame, group_cols, unit, metrics, dataset):
    rows = []
    for group, part in frame.groupby(group_cols, sort=True):
        if part[unit].duplicated().any():
            raise ValueError(f'Duplicate biological unit within {dataset} group {group}')
        for metric in metrics:
            values = part[metric].dropna().to_numpy()
            if len(values) < 2:
                continue
            ids = np.asarray(list(product(range(len(values)), repeat=len(values))))
            means = values[ids].mean(axis=1)
            lo, hi = np.quantile(means, [0.025, 0.975])
            rows.append(dict(benchmark=dataset, method=group[0], stratum=group[1] if len(group)>1 else 'all',
                             metric=metric, unit=unit, n=len(values), mean=values.mean(), sd=values.std(ddof=1),
                             lower_95=lo, upper_95=hi))
    return rows


def brain_tables():
    calibrated = pd.read_csv(SOURCE / 'brain_donors.csv')
    candidates = pd.read_csv(SOURCE / 'brain_native_donors.csv')
    native = candidates[candidates.candidate == 'scalar_1_b0']
    return [('brain calibrated', calibrated), ('brain native', native)]


def main():
    brain_metrics = ['energy', 'mmd_squared', 'mean_mse', 'mean_pearson', 'pred_var']
    lps = pd.read_csv(SOURCE / 'lps_animals.csv')
    rows = summary(lps, ['method'], 'gsm', lps.select_dtypes('number').columns, 'LPS external comparisons')
    frozen = pd.read_csv(ROOT / 'results/dynamic_response/adaptation/comparison_by_sample.csv')
    rows += summary(frozen, ['method'], 'gsm', ['mse', 'zero_response_mse', 'relative_mse_reduction',
                                               'delta_pearson', 'top100_overlap'], 'LPS backbone and readouts')
    for label, frame in brain_tables():
        rows += summary(frame, ['method'], 'unit', brain_metrics, label)
    brain = pd.read_csv(FIG / 'brain_donor_summary.csv')
    rows += summary(brain, ['arm'], 'unit', ['energy','mmd','mean_mse','mean_pearson','pred_var'], 'brain ablations')
    protein = pd.read_csv(FIG / 'protein_donor_summary.csv')
    rows += summary(protein, ['arm','celltype'], 'unit', ['mse','median_pearson'], 'CITE-seq ablations')
    pd.DataFrame(rows).to_csv(SOURCE / 'metric_uncertainty.csv', index=False)

    by_channel = pd.read_csv(FIG / 'protein_all_rows.csv')
    keys = ['arm', 'celltype', 'protein_index']
    if by_channel.duplicated(keys + ['unit']).any():
        raise ValueError('A protein/donor/stratum has more than one score')
    channel_stats = by_channel.groupby(keys)[['mse','pearson']].agg(['count','mean','std'])
    channel_stats.columns = ['_'.join(name) for name in channel_stats.columns]
    channel_stats.to_csv(SOURCE / 'antibody_channel_uncertainty.csv.gz',
                         compression={'method': 'gzip', 'mtime': 0})

    paired = []
    def reduction(label, method, comparator, metric, a, b, unit):
        # Percent reduction is calculated after averaging the paired biological units.
        if not a.index.equals(b.index) or a.index.has_duplicates:
            raise ValueError(f'Mismatched units for {label}/{method}/{comparator}')
        x, y = a.to_numpy(), b.to_numpy()
        ix = np.asarray(list(product(range(len(x)), repeat=len(x))))
        boot = 100 * (1 - x[ix].mean(axis=1) / y[ix].mean(axis=1))
        lo, hi = np.quantile(boot, [0.025, 0.975])
        paired.append(dict(benchmark=label,method=method,comparator=comparator,metric=metric,
                           unit=unit,n=len(x),reduction_pct=100*(1-x.mean()/y.mean()),
                           lower_95_pct=lo,upper_95_pct=hi))
    for label, frame in brain_tables():
        for metric in ['energy','mmd_squared','mean_mse']:
            values = frame.pivot(index='unit',columns='method',values=metric)
            for method in values.columns.difference(['world_model']):
                reduction(label, 'world_model', method, metric, values.world_model, values[method], 'donor')
    for metric in ['mmd','energy','mean_mse']:
        values = brain.pivot(index='unit',columns='arm',values=metric)
        for method in values.columns.difference(['joint_full']):
            reduction('brain ablations','joint_full',method,metric,values.joint_full,values[method],'donor')
    protein_mse = protein[protein.celltype.eq('ALL')].pivot(index='unit',columns='arm',values='mse')
    for method in protein_mse.columns.difference(['joint_full']):
        reduction('CITE-seq ablations','joint_full',method,'mse',protein_mse.joint_full,protein_mse[method],'donor')
    for method, part in lps.groupby('method'):
        part=part.set_index('gsm').sort_index()
        for metric, baseline in [('mse','zero_response_mse'),('program_mse','program_zero_mse')]:
            reduction('LPS external comparisons',method,'zero response',metric,part[metric],part[baseline],'animal')
    lps_mse = lps.pivot(index='gsm',columns='method',values='mse')
    for method in lps_mse.columns.difference(['world_model_combined_ridge','zero_response']):
        reduction('LPS external comparisons','world_model_combined_ridge',method,'mse',
                  lps_mse.world_model_combined_ridge,lps_mse[method],'animal')
    # The adjusted AD-control contrast uses all test donors jointly, so report
    # delete-one-donor sensitivity and jackknife SE rather than resampling genes.
    contrast = pd.read_csv(SOURCE / 'covariate_LODO_comparisons.csv').set_index('excluded_donor')
    all_donors = contrast.loc['all_test_donors']
    deleted = contrast.drop('all_test_donors')
    cols = ['full_mse','noBlood_mse','optimized_brain_only_mse',
            'full_relative_MSE_reduction_vs_noBlood','full_relative_MSE_reduction_vs_optimized_brain_only']
    variability = pd.DataFrame({'estimate':all_donors[cols].astype(float),
                                'delete_one_min':deleted[cols].min(),'delete_one_max':deleted[cols].max(),
                                'jackknife_se':deleted[cols].std(ddof=0)*np.sqrt(len(deleted)-1),
                                'n_donors':len(deleted)})
    variability.to_csv(SOURCE / 'adjusted_contrast_uncertainty.csv',index_label='metric')
    pd.DataFrame(paired).to_csv(SOURCE / 'paired_reductions.csv', index=False)


if __name__ == '__main__':
    main()
