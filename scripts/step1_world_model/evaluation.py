"""All observable routes, same held-out units; no Micro-only evaluation."""
import hashlib,json
import numpy as np,pandas as pd,torch
from data_io import tensor
from world_model import mmd

def rng_for(*parts):
 seed=int(hashlib.sha256('|'.join(map(str,parts)).encode()).hexdigest()[:8],16)
 return np.random.default_rng(seed)
def corr(a,b):
 return float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-8 and np.std(b)>1e-8 else None
def target_statistics(target,scale):
 norm=target.square().sum(1)
 yy=(norm[:,None]+norm[None,:]-2*target@target.T).clamp_min(0)/scale
 return dict(norm=norm,kernel_mean=sum(torch.exp(-yy/w) for w in [.5,1.,2.]).div(3).mean(),sqrt_mean=yy.sqrt().mean())
@torch.no_grad()
def metrics(pred,target,true_mean,scale,target_stats=None):
 xx=(pred.square().sum(1)[:,None]+pred.square().sum(1)[None,:]-2*pred@pred.T).clamp_min(0)/scale
 stats=target_statistics(target,scale) if target_stats is None else target_stats
 xy=(pred.square().sum(1)[:,None]+stats['norm'][None,:]-2*pred@target.T).clamp_min(0)/scale
 kernel=lambda z:sum(torch.exp(-z/w) for w in [.5,1.,2.])/3
 pm=pred.mean(0).cpu().numpy()
 return dict(mmd=float(kernel(xx).mean()+stats['kernel_mean']-2*kernel(xy).mean()),
   energy=float(2*xy.sqrt().mean()-xx.sqrt().mean()-stats['sqrt_mean']),mean_mse=float(np.mean((pm-true_mean)**2)),
   mean_pearson=corr(pm,true_mean),pred_var=float(pred.var(0,unbiased=False).mean()))
@torch.no_grad()
def evaluate_brain(model,data,split,device,n=1024,species='human',cache=True):
 model.eval();rows=[]
 d=data.data[species+'_brain'];o=d['obs']
 if not hasattr(data,'_brain_eval_cache'):data._brain_eval_cache={}
 saved=data._brain_eval_cache.setdefault((species,split,n,str(device)),dict(sources={},targets={})) if cache else dict(sources={},targets={})
 for route in data.routes(species):
  label,src,tgt=route['disease'],route['source_type'],route['target_type']
  pool=data.pools.get((species+'_brain',split,label,tgt),{})
  if not pool:continue
  key=(label,src)
  if not cache or key not in saved['sources']:
   xb,sm,meta=data.sample(species+'_blood',label,src,n,rng_for('source',species,label,src))
   saved['sources'][key]=(tensor(xb,device),tensor(sm,device))
  xb,sm=saved['sources'][key]
  c=data.condition(n,species,label,src,tgt,device)
  noise=tensor(rng_for('noise',species,label,src,tgt).normal(size=(n,model.config.latent_dim)).astype('f'),device)
  pred=model.predict_brain(xb,c,noise=noise,source_mask=sm)[0]
  for assay,units in pool.items():
   mask=data.masks[assay];pp=pred[:,torch.as_tensor(mask,device=device)]
   for unit,ids in sorted(units.items()):
    key=(assay,unit,tgt)
    if not cache or key not in saved['targets']:
     rng=rng_for('target',species,split,assay,unit,tgt)
     sampled=rng.choice(ids,min(n,len(ids)),replace=False)
     yy=tensor(d['x'][sampled][:,mask].toarray(),device);ym=np.asarray(d['x'][ids][:,mask].mean(0)).ravel()
     saved['targets'][key]=(yy,ym,target_statistics(yy,data.scales[assay]) if cache else None,len(sampled))
    yy,ym,stats,n_target=saved['targets'][key]
    vals=metrics(pp,yy,ym,data.scales[assay],stats)
    rows.append(dict(species=species,assay=assay,unit=unit,disease=label,source_type=src,target_type=tgt,cells=len(ids),metric_sample=n_target,predicted_cells=n,**vals))
 return rows

def validation_score(rows):
 a=pd.DataFrame(rows)
 if a.empty:raise RuntimeError('No validation routes')
 a['score']=a.mmd+.1*a.mean_mse
 return float(a.groupby(['source_type','target_type'],observed=True).score.mean().mean())
@torch.no_grad()
def evaluate_protein(model,data,split,device,out=None,ncap=None):
 model.eval();d=data.data['cite'];o=d['obs'];rows=[];mask=tensor(data.masks['human_CITE_GSE164378'],device)
 for unit in sorted(o.loc[o.split==split,'unit'].unique()):
  ids=np.flatnonzero((o.unit==unit)&(o.split==split))
  if ncap and len(ids)>ncap:ids=rng_for('protein',unit).choice(ids,ncap,False)
  pred=[]
  for st in range(0,len(ids),1024):pred.append(model.predict_adt(tensor(d['x'][ids[st:st+1024]].toarray(),device),mask).cpu().numpy())
  pred=np.vstack(pred);true=np.asarray(d['y'][ids])
  if out is not None:np.savez_compressed(out/(unit+'_protein_predictions.npz'),indices=ids,predicted=pred)
  for ct in ['ALL']+sorted(o.iloc[ids].celltype.unique()):
   keep=np.ones(len(ids),bool) if ct=='ALL' else np.asarray(o.iloc[ids].celltype==ct)
   aa,bb=pred[keep],true[keep]
   for j in range(bb.shape[1]):
    rows.append(dict(unit=unit,celltype=ct,protein_index=j,cells=int(keep.sum()),mse=float(np.mean((aa[:,j]-bb[:,j])**2)),pearson=corr(aa[:,j],bb[:,j])))
 return rows
