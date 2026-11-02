"""Explicit full-scope Step2/3 interface; all node and condition identities are in checkpoint."""
import numpy as np,torch
from world_model import WorldModel,V3Config
from data_io import tensor
class Predictor:
 def __init__(self,checkpoint,device='cpu'):
  ck=torch.load(checkpoint,map_location='cpu',weights_only=False);s=ck['model'];self.device=device
  self.genes=ck['gene_names'];self.proteins=ck['protein_features'];self.layout=ck['condition_layout'];self.masks=ck['assay_masks'];self.metadata=ck
  cfg=V3Config(**ck['config'])
  self.model=WorldModel(s['esm_matrix'].numpy(),s['gene_mean'].numpy(),s['protein_mean'].numpy(),s['protein_sd'].numpy(),cfg).to(device).eval()
  self.model.load_state_dict(s)
 def prepare(self,x,genes,species,disease,source_type,target_type,assay):
  if list(genes)!=self.genes:raise ValueError('Full node order differs from checkpoint')
  if x.ndim!=2 or x.shape[1]!=len(self.genes):raise ValueError('Input shape mismatch')
  if assay not in self.masks:raise ValueError('Explicit measured-feature assay required')
  if not assay.startswith(species+'_'):raise ValueError('Assay species differs from requested species')
  supported=any(r['status']=='active' and r['species']==species and r['disease']==disease and r['source_type']==source_type and r['target_type']==target_type for r in self.metadata['route_registry'])
  if not supported:raise ValueError('This cell-type combination has no observed training support in the checkpoint registry')
  v=dict(species=species,disease=disease,brain=target_type,blood=source_type);c=torch.zeros(len(x),sum(map(len,self.layout.values())),device=self.device);offset=0
  for key,labels in self.layout.items():
   if v[key] not in labels:raise ValueError('Unsupported '+key+': '+v[key])
   c[:,offset+labels.index(v[key])]=1;offset+=len(labels)
  m=tensor(self.masks[assay],self.device)
  return tensor(x,self.device),c,m
 @torch.no_grad()
 def predict(self,x,genes,species,disease,source_type,target_type,assay):
  xt,c,m=self.prepare(x,genes,species,disease,source_type,target_type,assay)
  y=self.model.predict_brain(xt,c,source_mask=m)[0]
  p=self.model.predict_adt(xt,m) if self.model.config.protein_weight and species=='human' else None
  return dict(brain_log1p_cp10k=y.cpu().numpy(),adt_clr=p.cpu().numpy() if p is not None else None)
 @torch.no_grad()
 def perturb(self,x,genes,source_node,value,species,disease,source_type,target_type,assay):
  xt,c,m=self.prepare(x,genes,species,disease,source_type,target_type,assay);j=self.genes.index(source_node)
  if not self.masks[assay][j]:raise ValueError('Source node unavailable in input assay')
  dy,dp=self.model.perturb(xt,c,j,value,source_mask=m)
  return dict(brain_delta=dy.cpu().numpy(),adt_delta=dp.cpu().numpy() if self.model.config.protein_weight and species=='human' else None,
     claim='model input-clamp response, not experimentally established protein intervention')
