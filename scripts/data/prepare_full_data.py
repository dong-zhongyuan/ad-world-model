#!/usr/bin/env python3
"""Full-scope rebuild: preserve all five v2 matrices/rows/nodes, add raw-derived data.
Reuses copied v2 aggregation; no HVG filtering or cell-type pruning.
Missing-assay coordinates have explicit masks, not measured zeros.
"""
from pathlib import Path
import os,sys,json,hashlib,gzip,shutil
os.environ.setdefault('OMP_NUM_THREADS','8')
import numpy as np,pandas as pd,scipy.sparse as sp,anndata as ad,h5py
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'data/processed'
REF=ROOT/'data/reference/v2';OLD=ROOT/'data/intermediate/restored_full_inputs'
sys.path.insert(0,str(ROOT/'scripts/data'))
from build_member_level import aggregate
import torch

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def emit(*x):print(*x,flush=True)
def write_csr(x,d):
 d.mkdir(parents=True,exist_ok=True);x=sp.csr_matrix(x,dtype='f');x.sort_indices()
 for name in ['data','indices','indptr']:np.save(d/(name+'.npy'),getattr(x,name))
 (d/'shape.json').write_text(json.dumps(list(x.shape)))
def read_csr(d):
 return sp.csr_matrix((np.load(d/'data.npy',mmap_mode='r'),np.load(d/'indices.npy',mmap_mode='r'),np.load(d/'indptr.npy',mmap_mode='r')),shape=json.loads((d/'shape.json').read_text()),copy=False)

def main():
 OUT.mkdir(exist_ok=True)
 contract=json.loads((ROOT/'docs/SCOPE_CONTRACT.json').read_text())
 cols=pd.read_csv(REF/'member_level/columns.csv').fillna('');genes=cols.node_id.tolist()
 assert genes==contract['reference_gene_names'] and len(genes)==25790
 cols.to_csv(OUT/'genes.csv',index=False)
 g2o=pd.read_csv(REF/'mappings/gene2og.csv');index={g:i for i,g in enumerate(genes)}
 maps={}
 for species,prefix in [('human','H:'),('mouse','M:')]:
  m={}
  for row in g2o[g2o.species==species].itertuples():
   if row.og_id in index:m.setdefault(row.symbol,set()).add(index[row.og_id])
  for j,g in enumerate(genes):
   if g.startswith(prefix):m.setdefault(g[2:],set()).add(j)
  assert all(len(v)==1 for v in m.values()),'Ambiguous exact node mapping'
  maps[species]={s:next(iter(v)) for s,v in m.items()}
 def mapping(symbols,species):
  rows=[];js=[]
  folded={}
  for sym,j in maps[species].items():folded.setdefault(sym.upper(),set()).add(j)
  for i,s in enumerate(symbols):
   s=str(s);j=maps[species].get(s)
   if j is None:
    candidates=folded.get(s.upper(),set())
    if len(candidates)>1:raise ValueError('Ambiguous gene casing: '+s)
    if candidates:j=next(iter(candidates))
   if j is not None:rows.append(i);js.append(j)
  mat=sp.csr_matrix((np.ones(len(rows),'f'),(rows,js)),shape=(len(symbols),len(genes)))
  return mat,np.asarray(mat.sum(0)).ravel()>0
 def raw_h5_symbols(name):
  with h5py.File(ROOT/'data/raw/GSE329430'/name) as h:
   return [s.decode() for s in h['matrix/features/name'][:]]
 masks={};parts={n:[] for n in ['human_blood','human_brain','mouse_blood','mouse_brain','cite']};obs={n:[] for n in parts};provenance=[]
 def add(name,x,o,assay,mask,source):
  assert x.shape[1]==len(genes) and len(o)==x.shape[0]
  o=o.copy();o['assay']=assay;o['source_dataset']=source;o.index=[name+':'+str(sum(len(i) for i in obs[name])+j) for j in range(len(o))]
  parts[name].append(x);obs[name].append(o)
  if assay in masks:assert np.array_equal(masks[assay],mask)
  masks[assay]=mask
  provenance.append(dict(dataset=name,source=source,cells=len(o),assay=assay,available_nodes=int(mask.sum())))
  emit('added',name,source,x.shape,o.celltype.value_counts().to_dict())
 brain_map={'mg':'Micro','oligo':'Oligo','astro':'Astro','neuron':'Neuron','endo':'Endo'}
 legacy_keys=set()
 for name in contract['required_datasets']:
  a=ad.read_h5ad(REF/'member_level'/(name+'.h5ad'))
  assert list(a.var_names)==genes
  o=a.obs.copy();o['original_cell_id']=o.index.astype(str)
  o['legacy_celltype']=o.celltype.astype(str)
  o['celltype']=o.celltype.astype(str).replace(brain_map).replace({'pDC':'DC'})
  o['annotation']='legacy_v2_preserved_unrevalidated';o['split']='train'
  if name=='human_blood':
   o['legacy_celltype_leiden']=o.celltype_leiden.astype(str)
   o['celltype']=o.celltype_leiden.astype(str).replace({'CD14_Mono':'Mono'})
   o['unit']='GSE181279:'+o.donor.astype(str);o['species']='human'
   symbols=pd.read_csv(next((ROOT/'data/raw/GSE181279').glob('*features.tsv.gz')),sep='\t',header=None)[1].astype(str)
   # v2 sc.read_10x_mtx makes duplicate feature names unique before OG mapping.
   symbols=ad.utils.make_index_unique(pd.Index(symbols))
   assay='human_pbmc_GSE181279';dest='human_blood'
  elif name=='human_brain':
   o['unit']='GSE138852:'+o.donor_pair.astype(str);o['species']='human'
   f=next((ROOT/'data/raw').rglob('GSE138852_counts.csv.gz'))
   symbols=pd.read_csv(f,usecols=[0]).iloc[:,0].astype(str)
   assay='human_cortex_GSE138852';dest='human_brain'
  elif name=='mouse_brain':
   o['unit']='Brain1:'+o['sample'].astype(str);o['species']='mouse';o['disease']=o.genotype.map({'WT':'CTRL','5xFAD':'AD'})
   symbols=raw_h5_symbols('GSM9703846_955_sample_filtered_feature_bc_matrix.h5')
   assay='mouse_Brain1';dest='mouse_brain'
   legacy_keys=set(zip(o['sample'].astype(str),o.original_cell_id))
  else:
   o['species']='mouse';o['disease']=o.genotype.map({'WT':'CTRL','5xFAD':'AD'})
   if name.endswith('bm12'):
    # A library/pool identity is not automatically an independently paired animal.
    o['unit']='BM12:'+o.mouse.astype(str);assay='mouse_BM12'
    symbols=pd.read_csv(ROOT/'data/raw/GSE329430/GSM9703838_BM1_features.tsv.gz',sep='\t',header=None)[1].astype(str)
    symbols=ad.utils.make_index_unique(pd.Index(symbols))
   else:
    o['unit']='BM34:'+o.mouse.astype(str);assay='mouse_BM34'
    symbols=raw_h5_symbols('GSM9703842_BM3_filtered_feature_bc_matrix.h5')
   dest='mouse_blood'
  _,mask=mapping(symbols,str(o.species.iloc[0]))
  add(dest,a.X.tocsr(),o,assay,mask,'v2:'+name)
  del a
 # Add full raw-derived mouse Brain1 cells not present in the preserved 5,000 reference rows.
 a=ad.read_h5ad(OLD/'normalized/mouse_brain.h5ad');o=a.obs.copy()
 sample=o.unit.astype(str).str.split(':').str[-1]
 barcode=pd.Series(a.obs_names,index=a.obs_names).str.split(':').str[1]
 keep=np.array([(s,b) not in legacy_keys for s,b in zip(sample,barcode)])
 a=a[keep].copy();o=o.iloc[np.flatnonzero(keep)].copy()
 o['original_cell_id']=barcode.iloc[np.flatnonzero(keep)].values;o['legacy_celltype']=''
 o['celltype']=o.celltype.astype(str).replace({'ambiguous':'Unresolved'})
 source_case={s.upper():s for s in raw_h5_symbols('GSM9703846_955_sample_filtered_feature_bc_matrix.h5')}
 mat,mask=mapping([source_case.get(s,s) for s in a.var_names],'mouse');x=a.X.copy();x.data=np.expm1(x.data)
 add('mouse_brain',aggregate(x,mat),o,'mouse_Brain1',mask,'raw_Brain1_extension')
 # Preserve the previous held-out donor identities; reinstate all excluded donors as training.
 a=ad.read_h5ad(OLD/'normalized/human_brain.h5ad');o=a.obs.copy()
 o['split']=o.split.astype(str).replace({'ineligible':'train'})
 o['celltype']=o.celltype.astype(str).replace({'ambiguous':'Unresolved'});o['legacy_celltype']='';o['original_cell_id']=o.index
 assert o.groupby('unit').split.nunique().max()==1 and o.unit.nunique()==21
 mat,mask=mapping(a.var_names,'human');x=a.X.copy();x.data=np.expm1(x.data)
 add('human_brain',aggregate(x,mat),o,'human_cortex_GSE157827',mask,'GSE157827_all_celltypes')
 # Same-cell CITE measurements with every retained cell type; no independent encoder/ODE.
 a=ad.read_h5ad(OLD/'normalized/cite.h5ad');o=a.obs.copy();o['legacy_celltype']='';o['original_cell_id']=o.index
 annotations=pd.read_csv(OLD/'protein_annotations.csv').fillna('')
 annotations.to_csv(OUT/'protein_annotations.csv',index=False)
 allp=list(a.uns['proteins']);pids=[allp.index(x) for x in annotations.adt_feature]
 y=a.obsm['adt'][:,pids].astype('f')
 mat,mask=mapping(a.var_names,'human');x=a.X.copy();x.data=np.expm1(x.data)
 add('cite',aggregate(x,mat),o,'human_CITE_GSE164378',mask,'GSE164378_P1_P8')
 (OUT/'cite').mkdir(exist_ok=True);np.save(OUT/'cite/Y.npy',y)
 (OUT/'cite/proteins.json').write_text(json.dumps(annotations.adt_feature.tolist()))
 names=list(masks);np.save(OUT/'assay_masks.npy',np.stack([masks[n] for n in names]))
 (OUT/'assays.json').write_text(json.dumps(names))
 unit_rows=[]
 for name in parts:
  x=sp.vstack(parts[name],format='csr');o=pd.concat(obs[name],axis=0).astype(object).fillna('')
  assert x.shape[0]==len(o) and o.index.is_unique
  write_csr(x,OUT/name);o.to_csv(OUT/name/'obs.csv',index_label='cell')
  for (assay,unit,split,disease,ct),b in o.groupby(['assay','unit','split','disease','celltype'],observed=True):
   unit_rows.append(dict(dataset=name,assay=assay,unit=unit,split=split,disease=disease,celltype=ct,cells=len(b)))
  parts[name]=None;obs[name]=None
 pd.DataFrame(unit_rows).to_csv(OUT/'sample_manifest.csv',index=False)
 # Same full node order and same nonzero copied sequence representation; no hidden 4k subset.
 enc=torch.load(ROOT/'data/reference/esm2_embeddings/esm2_member_level_dim640.pt',map_location='cpu',weights_only=False)
 repairs=OLD/'esm_repairs.pt'
 if repairs.exists():enc.update(torch.load(repairs,map_location='cpu',weights_only=False))
 esm=np.stack([np.asarray(enc.get(g,np.zeros(640)),np.float32) for g in genes])
 covered=np.linalg.norm(esm,axis=1)>0
 pd.DataFrame({'node_id':genes,'esm_nonzero':covered}).to_csv(OUT/'sequence_coverage.csv',index=False)
 esm/=np.maximum(np.linalg.norm(esm,axis=0,keepdims=True),1e-8);np.save(OUT/'esm.npy',esm)
 # Fit only human training donors and CITE P1-P4, accounting for unavailable assay coordinates.
 sm=np.zeros(len(genes));s2=np.zeros(len(genes));den=np.zeros(len(genes))
 for name in ['human_blood','human_brain','cite']:
  x=read_csr(OUT/name);o=pd.read_csv(OUT/name/'obs.csv',index_col=0)
  for (assay,unit),b in o[o.split=='train'].groupby(['assay','unit'],observed=True):
   ix=o.index.get_indexer(b.index);z=x[ix];mask=masks[assay]
   sm+=np.asarray(z.mean(0)).ravel()*mask;s2+=np.asarray(z.power(2).mean(0)).ravel()*mask;den+=mask
 mean=sm/np.maximum(den,1);var=np.maximum(s2/np.maximum(den,1)-mean**2,0)
 scales=np.array([max(2*var[masks[n]].sum(),1.) for n in names],'f')
 co=pd.read_csv(OUT/'cite/obs.csv');pm=[];ps=[]
 for u in ['P1','P2','P3','P4']:
  z=y[np.flatnonzero(co.unit==u)];pm.append(z.mean(0));ps.append((z*z).mean(0))
 pm=np.mean(pm,0);ps=np.sqrt(np.maximum(np.mean(ps,0)-pm**2,1e-4))
 np.savez(OUT/'preprocessing.npz',gene_mean=mean.astype('f'),protein_mean=pm.astype('f'),protein_sd=ps.astype('f'),cost_scales=scales)
 # Persist original condition axes verbatim, then add previously unrepresented observed labels.
 brain=list(contract['reference_config']['ct_brain_labels']);blood=list(contract['reference_config']['ct_blood_labels'])
 for side,labels in [('brain',brain),('blood',blood)]:
  for species in ['human','mouse']:
   o=pd.read_csv(OUT/(species+'_'+side)/'obs.csv')
   labels.extend(sorted(set(o.celltype)-set(labels)))
 layout=dict(species=['mouse','human'],disease=['CTRL','AD'],brain=brain,blood=blood)
 (OUT/'condition_layout.json').write_text(json.dumps(layout,indent=2))
 info=dict(status='prepared_full_scope',n_nodes=len(genes),n_proteins=len(pids),required_legacy_rows_preserved=True,
  sources=provenance,condition_layout=layout,legacy_config=contract['reference_config'],esm_nonzero_nodes=int(covered.sum()),
  esm_note='Original missing-sequence vectors retained as unavailable, not imputed with random sequences. All nodes remain inputs/outputs and are available to learned-projection controls.',
  annotation_note='Original labels retained as legacy computational annotations, not asserted biological ground truth. Mouse Brain1 was myeloid-sorted; non-myeloid labels remain explicitly unvalidated.',
  frozen_holdouts='GSE157827 previous 4 val/4 test preserved; all 21 donors retained; CITE P1-4/P5/P6-8',
  statistical_scope='Unpaired population prediction, no invented donor or animal pairs',
  data_files={str(f.relative_to(OUT)):sha(f) for f in OUT.rglob('*') if f.is_file() and f.name!='data_manifest.json'})
 (OUT/'data_manifest.json').write_text(json.dumps(info,indent=2));emit('COMPLETE',len(genes),'nodes',layout)
if __name__=='__main__':main()
