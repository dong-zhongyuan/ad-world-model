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
    for method, part in lps.groupby('method'):
        part=part.set_index('gsm').sort_index()
        for metric, baseline in [('mse','zero_response_mse'),('program_mse','program_zero_mse')]:
            reduction('LPS external comparisons',method,'zero response',metric,part[metric],part[baseline],'animal')
    pd.DataFrame(paired).to_csv(SOURCE / 'paired_reductions.csv', index=False)


if __name__ == '__main__':
    main()
