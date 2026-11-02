from pathlib import Path
import json
import numpy as np,pandas as pd,scipy.sparse as sp,torch
ROOT=Path(__file__).resolve().parents[2]
def read_csr(d):
 return sp.csr_matrix((np.load(d/'data.npy',mmap_mode='r'),np.load(d/'indices.npy',mmap_mode='r'),np.load(d/'indptr.npy',mmap_mode='r')),shape=json.loads((d/'shape.json').read_text()),copy=False)
def tensor(x,device):
 # Blocking transfer owns the device copy; a second full NumPy allocation is unnecessary.
 return torch.as_tensor(np.asarray(x),dtype=torch.float32,device=device)
class Dataset:
 def __init__(self,root=None):
  self.root=Path(root or ROOT/'data/processed');self.data={};self.pools={}
  self.pre=dict(np.load(self.root/'preprocessing.npz'));self.esm=np.load(self.root/'esm.npy')
  self.genes=pd.read_csv(self.root/'genes.csv').node_id.tolist()
  self.layout=json.loads((self.root/'condition_layout.json').read_text());self.cond_dim=sum(map(len,self.layout.values()))
  self.assays=json.loads((self.root/'assays.json').read_text());mm=np.load(self.root/'assay_masks.npy')
  self.masks={a:mm[i] for i,a in enumerate(self.assays)}
  self.scales={a:float(self.pre['cost_scales'][i]) for i,a in enumerate(self.assays)}
  for name in ['human_blood','human_brain','mouse_blood','mouse_brain','cite']:
   p=self.root/name;o=pd.read_csv(p/'obs.csv',index_col=0,low_memory=False).fillna('');x=read_csr(p)
   self.data[name]=dict(x=x,obs=o)
   if name=='cite':self.data[name]['y']=np.load(p/'Y.npy',mmap_mode='r')
   for key,indices in o.groupby(['split','disease','celltype','assay','unit'],observed=True).indices.items():
    split,label,ct,assay,unit=key
    self.pools.setdefault((name,split,label,ct),{}).setdefault(assay,{})[unit]=np.asarray(indices)
  o=self.data['cite']['obs'];self.cite_pools={}
  for (unit,subtype),idx in o[o.split=='train'].groupby(['unit','subtype'],observed=True).groups.items():
   self.cite_pools.setdefault(unit,{})[subtype]=o.index.get_indexer(idx)
  self.registry=self.build_registry()
 def condition(self,n,species,disease,source_type,target_type,device):
  values=dict(species=species,disease=disease,brain=target_type,blood=source_type)
  c=torch.zeros(n,self.cond_dim,device=device);offset=0
  for key,labels in self.layout.items():
   if values[key] not in labels:raise ValueError('Unrecognized '+key+': '+values[key])
   c[:,offset+labels.index(values[key])]=1;offset+=len(labels)
  return c
 def build_registry(self):
  rows=[]
  for species in self.layout['species']:
   for label in ['CTRL','AD']:
    for brain in self.layout['brain']:
     for blood in self.layout['blood']:
      sk=(species+'_blood','train',label,blood);tk=(species+'_brain','train',label,brain)
      src=self.pools.get(sk,{});tgt=self.pools.get(tk,{})
      legacy_allowed=not(species=='mouse' and brain=='Micro' and blood not in ['HSPC','MyeloidP','Mono','DC'])
      reason='active' if src and tgt and legacy_allowed else ('legacy_mouse_Micro_compartment_rule' if not legacy_allowed else 'no_observed_source' if not src else 'no_observed_target')
      rows.append(dict(species=species,disease=label,source_type=blood,target_type=brain,status=reason,
          source_cells=sum(len(i) for v in src.values() for i in v.values()),target_cells=sum(len(i) for v in tgt.values() for i in v.values()),
          source_units=sum(len(v) for v in src.values()),target_units=sum(len(v) for v in tgt.values())))
  return pd.DataFrame(rows)
 def routes(self,species):
  return self.registry[(self.registry.species==species)&(self.registry.status=='active')].to_dict('records')
 def sample(self,name,label,ct,n,rng,split='train',assay=None,unit=None):
  pool=self.pools[(name,split,label,ct)]
  if assay is None:assay=str(rng.choice(list(pool)))
  pp=pool[assay]
  if unit is not None:ids=rng.choice(pp[unit],n,replace=len(pp[unit])<n)
  else:
   units=list(pp);draws=rng.integers(len(units),size=n);ids=np.empty(n,dtype=np.int64)
   for j,u in enumerate(units):
    loc=np.flatnonzero(draws==j);ids[loc]=rng.choice(pp[u],len(loc),replace=len(pp[u])<len(loc))
  return self.data[name]['x'][ids].toarray(),self.masks[assay],dict(assay=assay,indices=ids)
 def sample_cite(self,n,rng):
  d=self.data['cite'];unit=str(rng.choice(list(self.cite_pools)));subtype=str(rng.choice(list(self.cite_pools[unit])))
  pool=self.cite_pools[unit][subtype];ids=rng.choice(pool,n,replace=len(pool)<n)
  return d['x'][ids].toarray(),np.asarray(d['y'][ids]),self.masks['human_CITE_GSE164378']
