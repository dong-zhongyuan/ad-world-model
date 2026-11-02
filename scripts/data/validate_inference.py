#!/usr/bin/env python3
"""Exercise every registered stratum through the real full-size checkpoint, validation data only."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/step1_world_model'))
from data_io import Dataset
from inference import Predictor
from evaluation import evaluate_protein

def main():
 torch.set_num_threads(4);data=Dataset();ck=ROOT/'runs/smoke_real/joint_full/seed_42/best.pt';p=Predictor(ck)
 rng=np.random.default_rng(943);covered=[]
 for species in ['human','mouse']:
  for route in data.routes(species):
   label,src,tgt=route['disease'],route['source_type'],route['target_type']
   x,mask,meta=data.sample(species+'_blood',label,src,4,rng)
   v=p.predict(x,data.genes,species,label,src,tgt,meta['assay'])
   assert v['brain_log1p_cp10k'].shape==(4,25790) and np.isfinite(v['brain_log1p_cp10k']).all()
   if species=='human':assert v['adt_clr'].shape==(4,224) and np.isfinite(v['adt_clr']).all()
   else:assert v['adt_clr'] is None
   covered.append(route)
 route=data.routes('human')[0];label,src,tgt=route['disease'],route['source_type'],route['target_type']
 x,mask,meta=data.sample('human_blood',label,src,8,rng);j=int(np.flatnonzero(mask)[0]);x[:,j]=0
 z=p.perturb(x,data.genes,data.genes[j],0.,'human',label,src,tgt,meta['assay'])
 assert np.max(np.abs(z['brain_delta']))==0 and np.max(np.abs(z['adt_delta']))==0
 try:p.predict(x,list(reversed(data.genes)),'human',label,src,tgt,meta['assay'])
 except ValueError:pass
 else:raise AssertionError('gene-order protection failed')
 rows=evaluate_protein(p.model,data,'val','cpu',ncap=512)
 assert {r['unit'] for r in rows}=={'P5'} and all(np.isfinite(r['mse']) for r in rows)
 report=dict(status='passed',checkpoint=str(ck.relative_to(ROOT)),nodes=25790,protein_channels=224,
   tested_strata=len(covered),tested_routes=len({(r['species'],r['source_type'],r['target_type']) for r in covered}),
   zero_change_exact=True,gene_order_guard=True,mouse_adt_suppressed=True,protein_evaluation_split='P5',heldout_test_evaluated=False)
 (ROOT/'results/inference_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
