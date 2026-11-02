"""Per-cell-type disease contrasts; mouse training observations are not held-out evidence."""
import numpy as np,pandas as pd,torch
from data_io import tensor
from evaluation import rng_for,corr
@torch.no_grad()
def evaluate(model,data,device,out):
 model.eval();observed={};predicted={};rows=[]
 for species in ['human','mouse']:
  split='test' if species=='human' else 'train';d=data.data[species+'_brain'];o=d['obs']
  for ct in data.layout['brain']:
   z={};masks={}
   for label in ['AD','CTRL']:
    pool=data.pools.get((species+'_brain',split,label,ct),{})
    if not pool:continue
    means=[];mm=[]
    for assay,units in pool.items():
     for unit,ids in units.items():means.append(np.asarray(d['x'][ids].mean(0)).ravel());mm.append(data.masks[assay])
    den=np.stack(mm).sum(0);z[label]=(np.stack(means)*mm).sum(0)/np.maximum(den,1);masks[label]=den>0
   if len(z)==2:observed[species,ct]=(z['AD']-z['CTRL'],masks['AD']&masks['CTRL'])
  for src in data.layout['blood']:
   for ct in data.layout['brain']:
    pr={}
    if (species,ct) not in observed:continue
    available={(r['disease'],r['source_type'],r['target_type']) for r in data.routes(species)}
    for label in ['AD','CTRL']:
     if (label,src,ct) not in available:continue
     x,m,_=data.sample(species+'_blood',label,src,1024,rng_for('species',species,src,ct,label))
     c=data.condition(len(x),species,label,src,ct,device)
     y=model.predict_brain(tensor(x,device),c,source_mask=tensor(m,device))[0].mean(0).cpu().numpy();pr[label]=y
    if len(pr)==2:
     delta=pr['AD']-pr['CTRL'];truth,mask=observed[species,ct];predicted[species,src,ct]=delta
     rows.append(dict(comparison='predicted_vs_observed',species=species,source_type=src,target_type=ct,pearson=corr(delta[mask],truth[mask]),nodes=int(mask.sum()),scope='held-out human' if species=='human' else 'mouse training description'))
 for ct in data.layout['brain']:
  if ('human',ct) not in observed or ('mouse',ct) not in observed:continue
  h,hm=observed['human',ct];m,mm=observed['mouse',ct];mask=hm&mm
  rows.append(dict(comparison='human_vs_mouse_observed',species='both',source_type='',target_type=ct,pearson=corr(h[mask],m[mask]),nodes=int(mask.sum()),scope='observational disease concordance, not conserved intervention proof'))
 pd.DataFrame(rows).to_csv(out/'species_disease_concordance.csv',index=False)
 np.savez_compressed(out/'species_disease_effects.npz',nodes=np.asarray(data.genes),**{'observed__'+'__'.join(k):v[0] for k,v in observed.items()},**{'mask__'+'__'.join(k):v[1] for k,v in observed.items()},**{'predicted__'+'__'.join(k):v for k,v in predicted.items()})
