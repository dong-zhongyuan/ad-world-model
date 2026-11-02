#!/usr/bin/env python3
"""Fail loudly if any original matrix row/node/configuration scope was reduced."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,anndata as ad
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/step1_world_model'))
from data_io import Dataset

def main():
 data=Dataset();contract=json.loads((ROOT/'docs/SCOPE_CONTRACT.json').read_text());report={}
 assert data.genes==contract['reference_gene_names'] and len(data.genes)==25790
 assert data.layout['brain'][:7]==contract['reference_config']['ct_brain_labels']
 assert data.layout['blood'][:11]==contract['reference_config']['ct_blood_labels']
 for name in contract['required_datasets']:
  legacy=ad.read_h5ad(ROOT/'data/reference/v2/member_level'/(name+'.h5ad'))
  dest='mouse_blood' if name.startswith('mouse_blood') else name
  d=data.data[dest];o=d['obs'];ids=np.flatnonzero(o.source_dataset=='v2:'+name)
  assert len(ids)==legacy.n_obs,(name,len(ids),legacy.n_obs)
  # The original v2 loader explicitly converts every matrix to float32 before training.
  z=(d['x'][ids]-legacy.X.astype(np.float32)).tocsr();z.eliminate_zeros();assert z.nnz==0,name
  assert o.iloc[ids].legacy_celltype.tolist()==legacy.obs.celltype.astype(str).tolist()
  if name=='human_blood':assert o.iloc[ids].legacy_celltype_leiden.tolist()==legacy.obs.celltype_leiden.astype(str).tolist()
  report[name]=dict(legacy_cells=legacy.n_obs,retained_cells=len(ids),model_input_float32_identical=True,labels_preserved=True)
 for name,d in data.data.items():
  o=d['obs'];x=d['x'];assert x.shape==(len(o),25790) and np.isfinite(x.data).all()
  assert (o.groupby('unit').split.nunique()==1).all()
  assert not o[['species','unit','disease','split','celltype','assay']].isin(['']).any().any()
  for assay,idx in o.groupby('assay',observed=True).indices.items():
   off=~data.masks[assay]
   assert x[idx][:,off].nnz==0,(name,assay,'nonzero in unavailable coordinate')
  report[name+'_full']=dict(cells=len(o),types=sorted(o.celltype.unique()),units=o.unit.nunique(),split_units=o.groupby('split').unit.nunique().to_dict())
 h=data.data['human_brain']['obs'];h=h[h.assay=='human_cortex_GSE157827'];assert h.unit.nunique()==21
 assert h[h.split=='test'].unit.nunique()==4 and h[h.split=='val'].unit.nunique()==4
 c=data.data['cite'];o=c['obs'];assert c['y'].shape==(len(o),224) and np.isfinite(c['y']).all()
 for split,want in [('train',{'P1','P2','P3','P4'}),('val',{'P5'}),('test',{'P6','P7','P8'})]:assert set(o[o.split==split].unit)==want
 rng=np.random.default_rng(982);coverage=[];swap_missing=[]
 for species in data.layout['species']:
  for route in data.routes(species):
   label,src,tgt=route['disease'],route['source_type'],route['target_type']
   for side,ct in [('blood',src),('brain',tgt)]:
    x,m,_=data.sample(species+'_'+side,label,ct,4,rng);assert x.shape==(4,25790) and m.shape==(25790,)
   cc=data.condition(4,species,label,src,tgt,'cpu');assert cc.shape==(4,data.cond_dim) and (cc.sum(1)==4).all()
   coverage.append(route)
   if species=='mouse' and ('mouse_brain','train','AD' if label=='CTRL' else 'CTRL',tgt) not in data.pools:swap_missing.append(route)
 assert not swap_missing,swap_missing
 data.registry.to_csv(ROOT/'data/processed/route_registry.csv',index=False)
 eval_coverage=[]
 for row in coverage:
  for split in ['val','test']:
   pool=data.pools.get((row['species']+'_brain',split,row['disease'],row['target_type']),{})
   eval_coverage.append(dict(**row,split=split,evaluation_status='observable' if pool else 'no_observed_target_in_split',target_evaluation_units=sum(len(v) for v in pool.values()),target_evaluation_cells=sum(len(i) for v in pool.values() for i in v.values())))
 pd.DataFrame(eval_coverage).to_csv(ROOT/'results/evaluation_route_coverage.csv',index=False)
 # Registry is a validation artifact; preserve frozen matrix manifest and hash this separately at launch.
 protocol=json.loads((ROOT/'configs/ablation_manifest.json').read_text())
 report.update(status='passed',nodes=25790,original_batch=1024,formal_updates=protocol['minimum_formal_updates'],condition_dim=data.cond_dim,
   active_strata=len(coverage),active_routes=len({(r['species'],r['source_type'],r['target_type']) for r in coverage}),
   active_strata_by_species=pd.DataFrame(coverage).species.value_counts().to_dict(),all_active_strata_sampled=True,
   no_legacy_cell_or_node_removed=True,diagnostic_mouse_label_swap_supported=True)
 (ROOT/'results/full_scope_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
