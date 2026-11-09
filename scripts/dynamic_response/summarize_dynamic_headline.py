import json
import numpy as np
import pandas as pd
from project_paths import CONFIG, RESULTS


def main():
    table=pd.read_csv(RESULTS/'adaptation/comparison_by_sample.csv')
    out=RESULTS/'headline';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for method,frame in table.groupby('method'):
        rows.append(dict(method=method,n_test_lps_animals=len(frame),mse=float(frame.mse.mean()),
                         zero_response_mse=float(frame.zero_response_mse.mean()),
                         relative_mse_reduction=float(1-frame.mse.mean()/frame.zero_response_mse.mean()),
                         mean_delta_pearson=float(frame.delta_pearson.mean()),mean_top100_overlap=float(frame.top100_overlap.mean())))
    summary=pd.DataFrame(rows).sort_values('relative_mse_reduction',ascending=False)
    summary.to_csv(out/'headline_by_method.csv',index=False)
    combined=table[table.method.eq('combined_linear')].sort_values('gsm')
    combined.to_csv(out/'combined_by_test_animal.csv',index=False)
    m,z=combined.mse.to_numpy(),combined.zero_response_mse.to_numpy()
    indices=np.random.default_rng(42).integers(0,len(m),size=(10000,len(m)))
    interval=np.quantile(1-m[indices].mean(1)/z[indices].mean(1),[.025,.975])
    row=summary[summary.method.eq('combined_linear')].iloc[0]
    result=dict(status='complete',model='official_v3_1500_plus_fixed_ridge',
                protocol='fixed source-plus-v3 ridge alpha=1; reciprocal processing-held-out benchmark',
                n_test_lps_animals=len(m),n_test_lps_samples=len(m),
                relative_mse_reduction=float(row.relative_mse_reduction),
                animal_bootstrap_95_low=float(interval[0]),animal_bootstrap_95_high=float(interval[1]),
                mean_delta_pearson=float(row.mean_delta_pearson),mean_top100_overlap=float(row.mean_top100_overlap),
                target_relative_mse_reduction=.5328,target_met=bool(row.relative_mse_reduction>=.5328),
                uncertainty_scope='Descriptive four-library bootstrap; shared controls and reciprocal fits limit independence.',
                claim_scope=CONFIG['claim_scope'])
    (out/'headline_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
