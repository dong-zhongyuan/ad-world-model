#!/usr/bin/env python3
"""Extend verify_drugclip_ready.py: independent matrix verification and ten-target rankings.
All parents remain in each target table. Rank = maximum pocket MAD-normalized score
across all states of that parent, after averaging the six official model cosines.
Fold agreement is descriptive and never an additional candidate inclusion gate.
"""
import csv,gzip,hashlib,json,math,time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from paths import DRUG_WORK as ROOT
OUT=ROOT/'results/drugclip_ten'
INP=ROOT/'inputs/ten_targets'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2))
def robust_z(a):
 median=np.median(a,axis=-1,keepdims=True)
 mad=np.median(np.abs(a-median),axis=-1,keepdims=True)
 return .6745*(a-median)/(mad+1e-6)
def table(path,rows):
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)

def main():
 progress=json.loads((OUT/'progress.json').read_text());assert progress['status']=='complete'
 row_check=json.loads((OUT/'row_identity_verification.json').read_text())
 assert row_check['status'] in ['corrected_and_verified','verified_direct_loader_row_order']
 with gzip.open(INP/'verification/acceleration_inputs/drugclip_ten_ready.json.gz','rt') as f:source=json.load(f)
 with gzip.open(INP/'pockets/all_ten_pockets.json.gz','rt') as f:pockets=json.load(f)
 identity=json.loads((OUT/'input_identity.json').read_text())
 registry=json.loads((INP/'verification/acceleration_inputs/drugclip_ten_target_identity.json').read_text())
 assert identity['pockets']==pockets
 molecules=source['molecules'];N=len(molecules);P=len(pockets)
 assert len({m['state_id'] for m in molecules})==N
 assert identity['molecules']==[{k:m[k] for k in ['state_id','chembl_id','smi','name','source_sha256']} for m in molecules]
 raw=np.load(OUT/'six_fold_similarity.npy');assert raw.shape==(6,P,N) and np.isfinite(raw).all()
 largest=0.
 for fold in range(6):
  m=np.load(OUT/f'embedding_cache/fold_{fold}_molecules.npy')
  p=np.load(OUT/f'embedding_cache/fold_{fold}_pockets.npy')
  assert m.shape==(N,128) and p.shape==(P,128)
  # Independent float64 matrix product in bounded chunks, without reusing scoring code.
  for start in range(0,N,2048):
   expected=np.einsum('pk,mk->pm',p.astype(np.float64),m[start:start+2048].astype(np.float64))
   largest=max(largest,float(np.max(np.abs(expected-raw[fold,:,start:start+2048]))))
  assert largest<5e-6,largest
 mean=np.load(OUT/'six_fold_mean_similarity.npy')
 np.testing.assert_allclose(mean,raw.astype(np.float64).mean(axis=0),rtol=2e-6,atol=2e-7)
 z=robust_z(mean)
 by_parent={}
 for i,m in enumerate(molecules):by_parent.setdefault(m['chembl_id'],[]).append(i)
 ids=sorted(by_parent);parent_indices=[np.array(by_parent[c]) for c in ids]
 assert len(ids)==source['ready_parents_at_export']
 top=[];summaries=[]
 for target in sorted(registry['targets'],key=lambda r:(r['both_outcomes_supported']!='True',r['gene'])):
  gene=target['gene'];group='intersection' if target['both_outcomes_supported']=='True' else 'union_only'
  pix=np.array([j for j,p in enumerate(pockets) if p['target']==gene]);assert len(pix)>0
  # Fold ranks expose ensemble consistency, independently of the primary average score.
  fold_ranks=[]
  for fold in range(6):
   fz=robust_z(raw[fold,pix]);state_scores=fz.max(axis=0)
   vals=np.array([state_scores[idx].max() for idx in parent_indices])
   fold_ranks.append(rankdata(-vals,method='min'))
  fold_ranks=np.asarray(fold_ranks)
  rows=[]
  for k,cid in enumerate(ids):
   idx=parent_indices[k];block=z[np.ix_(pix,idx)]
   pi,si=np.unravel_index(np.argmax(block),block.shape);pidx=int(pix[pi]);sidx=int(idx[si]);p=pockets[pidx];m=molecules[sidx]
   row={'target':gene,'screening_name':target['screening_name'],'group':group,'uniprot':target['uniprot'],
        'chembl_id':cid,'name':m['name'],'score_max_pocket_z':float(block[pi,si]),'best_state_id':m['state_id'],
        'best_state_smiles':m['smi'],'states_scored':len(idx),'best_pocket':p['pocket'],
        'best_pocket_mean_cosine':float(mean[pidx,sidx]),'folds_in_top_1_percent':int((fold_ranks[:,k]<=math.ceil(len(ids)*.01)).sum()),
        'median_fold_rank_percent':float(np.median(fold_ranks[:,k])/len(ids)*100),
        'pocket_source':p['source_kind'],'pocket_mean_pLDDT':p.get('mean_residue_pLDDT',''),
        'pocket_fraction_pLDDT_ge70':p.get('fraction_residues_pLDDT_ge70','')}
   rows.append(row)
  rows.sort(key=lambda x:(-x['score_max_pocket_z'],x['chembl_id']))
  rows=[{'rank':i,**row} for i,row in enumerate(rows,1)]
  table(OUT/(gene+'_all_parents.tsv'),rows)
  top.extend(rows[:20])
  # Separate state-level table; all microstates retained, no global cross-target maximum.
  state_scores=z[pix].max(axis=0);best=pix[np.argmax(z[pix],axis=0)]
  state_order=np.argsort(-state_scores,kind='stable')
  state_rows=[{'rank':k,'target':gene,'state_id':molecules[int(i)]['state_id'],'chembl_id':molecules[int(i)]['chembl_id'],
               'score_max_pocket_z':float(state_scores[i]),'best_pocket':pockets[int(best[i])]['pocket']} for k,i in enumerate(state_order,1)]
  table(OUT/(gene+'_all_states.tsv'),state_rows)
  summaries.append({'target':gene,'screening_name':target['screening_name'],'group':group,'parents':len(rows),'states':N,'pockets':len(pix),
                    'top_name':rows[0]['name'],'top_chembl_id':rows[0]['chembl_id'],'top_score':rows[0]['score_max_pocket_z'],
                    'parent_table_sha256':sha(OUT/(gene+'_all_parents.tsv'))})
 table(OUT/'top20_per_target.tsv',top);table(OUT/'target_summary.tsv',summaries)
 # The original BIN1 snapshot must remain byte-for-byte unchanged.
 original=ROOT/'results/drugclip_ready';legacy=json.loads((original/'independent_verification.json').read_text())
 assert sha(original/'six_fold_similarity.npy')==legacy['raw_scores_sha256']
 assert sha(original/'all_microstate_scores.tsv')==legacy['score_table_sha256']
 report={'status':'passed','targets':10,'intersection_targets':5,'union_only_targets':5,'parents_per_target':len(ids),
         'states_per_target':N,'pockets':P,'raw_matrix_shape':list(raw.shape),'all_six_models':True,
         'max_independent_float64_product_difference':largest,'all_target_tables_have_every_parent':True,
         'earlier_raw_files_preserved_as_superseded':True,'source_identity_verified':True,'actual_encoder_row_identity_verified':True,'raw_scores_sha256':sha(OUT/'six_fold_similarity.npy'),
         'candidate_scores_are_virtual_screening_rankings':True,'additional_score_inclusion_gate':None}
 write(OUT/'independent_verification.json',report)
 write(OUT/'screening_summary.json',{'status':'complete_for_prepared_library_snapshot','verification':report,'targets':summaries,
       'library_total_parents':2402,'parents_scored':len(ids),'parents_remaining':2402-len(ids),
       'scoring_seconds':progress['total_measured_seconds'],'method':'six official models; mean cosine then per-pocket median/MAD standardization; maximum over target pockets and parent states',
       'all_detected_pockets_retained':True,'md_required':False})
 print(json.dumps(report),flush=True)

def figure_folds():
 import pandas as pd
 from paths import FIGURE_WORK
 fig=FIGURE_WORK
 identity=json.loads((OUT/'input_identity.json').read_text());molecules=identity['molecules'];pockets=identity['pockets']
 raw=np.load(OUT/'six_fold_similarity.npy',mmap_mode='r');assert raw.shape==(6,323,10323)
 by_parent={}
 for i,m in enumerate(molecules):by_parent.setdefault(m['chembl_id'],[]).append(i)
 ids=sorted(by_parent);parent_indices=[np.array(by_parent[c])for c in ids];reps=pd.read_csv(fig/'data/representative_molecules.csv');rows=[]
 for r in reps.to_dict('records'):
  pix=np.array([j for j,p in enumerate(pockets)if p['target']==r['target']]);k=ids.index(r['chembl_id']);ranks=[]
  for fold in range(6):
   fz=robust_z(raw[fold,pix]);state_scores=fz.max(axis=0);vals=np.array([state_scores[idx].max()for idx in parent_indices]);rank=rankdata(-vals,method='min');ranks.append(int(rank[k]));rows.append(dict(target=r['target'],chembl_id=r['chembl_id'],name=r['name'],fold=fold+1,rank=int(rank[k]),rank_percent=100*rank[k]/len(ids)))
  assert sum(x<=math.ceil(len(ids)*.01)for x in ranks)==r['folds_in_top_1_percent']
 pd.DataFrame(rows).to_csv(fig/'data/representative_fold_ranks.csv',index=False)
 print('FOLD_RANKS_COMPLETE',len(rows),flush=True)
if __name__=='__main__':figure_folds()
