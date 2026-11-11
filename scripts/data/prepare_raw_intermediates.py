"""Rebuild v3 from copied raw data; donor splits precede feature fitting.
Reads legacy demultiplexing assignments, never uses legacy learned feature matrices.
Marker labels are explicitly provisional and independent of disease labels.
"""
from pathlib import Path
import os,sys,json,gzip,tarfile,shutil,argparse,hashlib,re
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np,pandas as pd,scipy.sparse as sp
from scipy.io import mmread
import anndata as ad,scanpy as sc,torch
ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'data/raw';OUT=ROOT/'data/intermediate/restored_full_inputs';CACHE=OUT/'normalized'
CACHE.mkdir(parents=True,exist_ok=True)
from build_member_level import qc_mask
BLOOD={'Mono':['LYZ','LYZ2','CD14','FCN1','CSF1R','CTSS'],
       'T':['CD3D','CD3E','TRAC','CD247'],'B':['CD79A','MS4A1','CD79B','CD19'],
       'NK':['NKG7','KLRD1','NCR1','GNLY'],'DC':['FCER1A','CLEC10A','CLEC4C','TCF4'],
       'HSPC':['KIT','CD34','PROM1','HLF'],'MyeloidP':['MPO','ELANE','PRTN3','CTSG'],
       'Ery':['HBB','HBA1','ALAS2'],'Mega':['PF4','PPBP','ITGA2B']}
BRAIN={'Micro':['P2RY12','TMEM119','CX3CR1','C1QA','C1QB','HEXB'],
       'MDM':['S100A8','S100A9','FCN1','CCR2','LY6C2'],
       'Astro':['AQP4','GFAP','SLC1A3','ALDH1L1'],'Oligo':['MBP','PLP1','MOG','MAG'],
       'OPC':['PDGFRA','CSPG4','OLIG1'],'Neuron':['SNAP25','SYT1','RBFOX3','SLC17A7','GAD1'],
       'Endo':['CLDN5','PECAM1','VWF'],'Pericyte':['PDGFRB','RGS5','ABCC9'],
       'T':['CD3D','CD3E','TRAC'],'B':['CD79A','MS4A1','CD19']}
TARGETS=['IFNAR1','IFNGR1','IL6R','IL4R','TLR4','CD14','CSF1R','TNFRSF1A','TNFRSF1B','CD40']
def emit(*a):print(*a,flush=True)
def read10(directory,prefix):
 with gzip.open(directory/(prefix+'matrix.mtx.gz'),'rb') as f:x=sp.csr_matrix(mmread(f).T,dtype=np.float32)
 ft=pd.read_csv(directory/(prefix+'features.tsv.gz'),sep='\t',header=None)
 bc=pd.read_csv(directory/(prefix+'barcodes.tsv.gz'),sep='\t',header=None)[0].astype(str)
 return ad.AnnData(x,obs=pd.DataFrame(index=pd.Index(bc.to_numpy(),name=None)),var=pd.DataFrame(index=pd.Index(ft[1].astype(str).to_numpy(),name=None)))
def normalize(a,name):
 x=sp.csr_matrix(a.X,dtype=np.float32)
 symbols=np.asarray([str(s).upper() for s in a.var_names])
 unique,idx=np.unique(symbols,return_inverse=True)
 x=x@sp.csr_matrix((np.ones(len(idx)),(np.arange(len(idx)),idx)),shape=(len(idx),len(unique)))
 x=x.tocsr().astype('f')
 total=np.asarray(x.sum(1)).ravel()
 mito=np.char.startswith(unique,'MT-')
 pct=np.asarray(x[:,mito].sum(1)).ravel()/np.maximum(total,1)
 keep=qc_mask(x)&(pct<0.10)
 obs=a.obs.iloc[np.flatnonzero(keep)].copy()
 obs['n_counts']=total[keep];obs['mito_fraction']=pct[keep]
 x=x[keep].multiply(1e4/np.maximum(total[keep,None],1)).tocsr();x.data=np.log1p(x.data)
 out=ad.AnnData(x,obs=obs,var=pd.DataFrame(index=unique));out.obs_names_make_unique()
 emit('normalized',name,a.shape,'->',out.shape)
 return out
def annotate(a,markers):
 genes={g:i for i,g in enumerate(a.var_names)}
 scores=[];present=[]
 for label,gs in markers.items():
  j=[genes[g] for g in gs if g in genes]
  scores.append(np.asarray(a.X[:,j].mean(1)).ravel() if j else np.zeros(a.n_obs))
  present.append(np.asarray((a.X[:,j]>0).sum(1)).ravel() if j else np.zeros(a.n_obs))
 scores=np.stack(scores,1);present=np.stack(present,1);idx=scores.argmax(1)
 top=scores[np.arange(len(idx)),idx];margin=top-np.partition(scores,-2,axis=1)[:,-2]
 labels=np.array(list(markers))[idx].astype(object)
 labels[(margin<.10)|(present[np.arange(len(idx)),idx]<2)]='ambiguous'
 a.obs['celltype']=labels
 a.obs['annotation']='predefined_marker_provisional'
 a.obs['marker_margin']=margin
 for k,label in enumerate(markers):a.obs['marker_'+label]=scores[:,k]
 return a
def save(a,name):
 a.obs_names=[name+':'+str(i) for i in a.obs_names]
 a.write_h5ad(CACHE/(name+'.h5ad'),compression='lzf')
 a.obs.groupby(['unit','disease','celltype'],observed=True).size().rename('cells').to_csv(OUT/(name+'_counts.csv'))
 a.obs.groupby('celltype',observed=True)[[c for c in a.obs if c.startswith('marker_')]].mean().to_csv(OUT/(name+'_marker_means.csv'))
 emit('saved',name,a.shape,a.obs.celltype.value_counts().to_dict())
def base():
 if not (CACHE/'mouse_brain.h5ad').exists():
  parts=[]
  for sid in ['955','956','957','958']:
   a=sc.read_10x_h5(RAW/'GSE329430'/f'GSM9703846_{sid}_sample_filtered_feature_bc_matrix.h5')
   a.obs['unit']='Brain1:'+sid;a.obs['disease']='AD' if sid in ['956','958'] else 'CTRL'
   parts.append(annotate(normalize(a,sid),BRAIN))
  a=ad.concat(parts,join='inner',index_unique=':');a.obs['species']='mouse';a.obs['split']='train'
  save(a,'mouse_brain')
 if not (CACHE/'cite.h5ad').exists():
  d=RAW/'GSE164378/extracted';d.mkdir(exist_ok=True)
  with tarfile.open(RAW/'GSE164378/GSE164378_RAW.tar') as t:
   for m in t:
    name=Path(m.name).name
    if m.isfile() and name.startswith(('GSM5008737_RNA_3P-','GSM5008738_ADT_3P-')):
     dst=d/name
     if not dst.exists():
      with t.extractfile(m) as fi,dst.open('wb') as fo:shutil.copyfileobj(fi,fo)
  a=read10(d,'GSM5008737_RNA_3P-');b=read10(d,'GSM5008738_ADT_3P-')
  meta=pd.read_csv(RAW/'GSE164378/GSE164378_sc.meta.data_3P.csv.gz',index_col=0)
  assert a.obs_names.is_unique and b.obs_names.is_unique and meta.index.is_unique
  common=a.obs_names.intersection(b.obs_names).intersection(meta.index)
  common=common[~meta.loc[common,'celltype.l2'].str.contains('Doublet',case=False)]
  a=a[common].copy();meta=meta.loc[common]
  assert 'donor' in meta
  a.obs['unit']=meta.donor.astype(str).values;a.obs['disease']='healthy'
  a.obs['subtype']=meta['celltype.l2'].astype(str).values
  a.obs['celltype']=meta['celltype.l1'].astype(str).values
  a=normalize(a,'CITE')
  y=b[a.obs_names].X.toarray().astype('f')
  u=np.log1p(y);u-=u.mean(1,keepdims=True)
  a.obsm['adt']=u;a.uns['proteins']=np.asarray(b.var_names.astype(str),dtype=str)
  a.obs['split']=a.obs.unit.map({**{p:'train' for p in ['P1','P2','P3','P4']},'P5':'val',**{p:'test' for p in ['P6','P7','P8']}})
  assert not a.obs.split.isna().any()
  a.obs['species']='human'
  save(a,'cite')
def brain():
 if (CACHE/'human_brain.h5ad').exists():return
 files=sorted((RAW/'GSE157827').glob('*_matrix.mtx.gz'))
 assert len(files)==21, 'Complete cohort required; no partial replacement'
 splits=json.loads((ROOT/'data/reference/fixed_brain_splits.json').read_text())
 parts=[]
 for f in files:
  donor=f.name.split('_')[1]
  a=read10(f.parent,f.name.replace('matrix.mtx.gz',''))
  a.obs['unit']='GSE157827:'+donor;a.obs['disease']='AD' if donor.startswith('AD') else 'CTRL'
  a.obs['split']=splits['GSE157827:'+donor]
  parts.append(annotate(normalize(a,donor),BRAIN))
 a=ad.concat(parts,join='inner',index_unique=':');a.obs['species']='human'
 assert a.obs.unit.nunique()==21
 save(a,'human_brain')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['base','brain','all'],default='all');args=ap.parse_args()
 if args.stage in ['base','all']:base()
 if args.stage in ['brain','all']:brain()
