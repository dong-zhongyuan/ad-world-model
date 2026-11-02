#!/usr/bin/env python3
"""Correct the ignored latent-noise evaluation bug; evaluate EVERY fixed final checkpoint.

Training and weights are unchanged. Frozen old results remain intact. This amendment
is registered before corrected test evaluation, applies equally to all twenty arms,
and uses the fixed final checkpoint because intermediate validation candidates were
not all retained. It does not select a checkpoint based on corrected test performance.
"""
from pathlib import Path
import os,sys,json,time,hashlib,argparse,datetime,shutil
import numpy as np,pandas as pd,torch
from paths import REPO_ROOT as ROOT
sys.path.insert(0,str(ROOT/'scripts/step1_world_model'))
from world_model import WorldModel,V3Config,representation_matrix
from data_io import Dataset
from evaluation import evaluate_brain,evaluate_protein
from species_diagnostics import evaluate as evaluate_species
OUT=ROOT/'results/compact_repair'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
class DistributionModel(WorldModel):
 def predict_brain(self,x,c,sample=False,noise=None,source_mask=None):
  if noise is None or not self.config.source_expression or not self.config.stochastic:
   return super().predict_brain(x,c,sample=sample,noise=noise,source_mask=source_mask)
  _,mu,lv=self.encode(x,False,source_mask)
  z=mu+noise*torch.exp(.5*lv);c=self.condition(c)
  if self.config.mapping=='mlp':zt=z+self.mlp_mapping(torch.cat([z,c],1))
  elif self.config.mapping=='identity':zt=z
  else:zt=self.forward_ode(z,'blood_to_brain',c)
  return self.decode(zt),dict(mu=mu,lv=lv,z=z,zt=zt)
def checks():
 torch.set_num_threads(4);torch.manual_seed(31)
 cfg=V3Config(latent_dim=8,enc_hidden=16,dec_z_hidden=16,ode_hidden=16,ode_n_steps=2,n_proteins=5,cond_dim=12)
 m=DistributionModel(np.random.default_rng(3).normal(size=(24,12)).astype('f'),np.ones(24,'f'),np.zeros(5,'f'),np.ones(5,'f'),cfg).eval()
 x=torch.rand(7,24);c=torch.zeros(7,12);zero=torch.zeros(7,8);one=torch.ones(7,8)
 with torch.no_grad():
  original=WorldModel.predict_brain(m,x,c)[0]
  assert torch.equal(original,m.predict_brain(x,c,noise=zero)[0])
  assert not torch.equal(m.predict_brain(x,c,noise=zero)[0],m.predict_brain(x,c,noise=one)[0])
  torch.manual_seed(57);sampled=WorldModel.predict_brain(m,x,c,sample=True)[0]
  torch.manual_seed(57);eps=torch.randn(7,8)
  assert torch.equal(sampled,m.predict_brain(x,c,noise=eps)[0])
  assert torch.equal(m.predict_brain(x,c,noise=eps)[0],m.predict_brain(x,c,noise=eps)[0])
  for stoch,source in [(False,True),(True,False)]:
   m.config.stochastic=stoch;m.config.source_expression=source
   assert torch.equal(WorldModel.predict_brain(m,x,c,noise=eps)[0],m.predict_brain(x,c,noise=eps)[0])
 return dict(passed=True,zero_noise_matches_original=True,fixed_noise_matches_training_reparameterization=True,noise_changes_stochastic_prediction=True,deterministic_and_condition_only_controls_unchanged=True)
def main():
 from evaluation import validation_score
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.24,0)
 control=ROOT/'runs/compact_repair/fixed_control'
 cm=json.loads((control/'manifest.json').read_text());assert cm['status']=='complete' and cm['final_step']==1500
 source=ROOT/'runs/refinement/joint_full/seed_42/last.pt';dest=ROOT/'runs/compact_repair/main'
 dest.mkdir(parents=True,exist_ok=False);OUT.mkdir(parents=True,exist_ok=True)
 ck=torch.load(source,map_location='cpu',weights_only=False);fixed=torch.load(control/'last.pt',map_location='cpu',weights_only=False)
 assert ck['step']==fixed['step']==1500 and ck['gene_names']==fixed['gene_names'] and len(ck['gene_names'])==25790
 assert ck['spec']['batch']==fixed['spec']['batch']==1024
 assert ck['refinement_sampler']==fixed['refinement_sampler'], 'Matched data streams differ'
 assert ck['effective_missing_inputs'] and not fixed['effective_missing_inputs']
 old=torch.load(ROOT/'runs/formal/joint_full/seed_42/last.pt',map_location='cpu',weights_only=False)
 nonzero=old['model']['esm_matrix'].norm(dim=1)>0
 assert torch.equal(ck['model']['esm_matrix'][nonzero],old['model']['esm_matrix'][nonzero])
 assert torch.equal(fixed['model']['esm_matrix'],old['model']['esm_matrix'])
 changed=int(((ck['model']['esm_matrix']-old['model']['esm_matrix']).abs().sum(1)>0).sum())
 assert changed>0
 # Freeze validation-only recommendation before evaluating this retained checkpoint on test donors.
 sel=json.loads((ROOT/'results/repair_sampler_selection.json').read_text());v=ck['refinement_metrics']
 score=.5*(v['brain_validation_score']/sel['baseline_brain']+v['protein_P5_mse']/sel['baseline_protein_P5'])
 candidates=[dict(checkpoint='runs/formal/joint_full/seed_42/last.pt',step=1000,score=1.),dict(checkpoint='runs/repair_pilots/mixed_missing/last.pt',step=1250,score=next(x['score'] for x in sel['candidates'] if x['name']=='mixed_missing')),dict(checkpoint=str(source.relative_to(ROOT)),step=1500,score=score)]
 selected=min(candidates,key=lambda x:x['score'])
 recommendation=dict(preferred_checkpoint=selected['checkpoint'],step=selected['step'],candidates=candidates,criterion=sel['criterion'],heldout_test_not_used=True,frozen_at=datetime.datetime.now().astimezone().isoformat())
 (OUT/'validation_based_model_recommendation.json').write_text(json.dumps(recommendation,indent=2))
 (OUT/'matched_integrity.json').write_text(json.dumps(dict(same_total_steps=1500,same_refinement_sampler_and_rng=True,nodes=25790,batch=1024,known_sequence_rows_unchanged=True,learned_RNA_input_rows=changed,fixed_input_control_unchanged=True,original_weights_and_optimizer_resumed=True),indent=2))
 del fixed,old
 status=json.loads((ROOT/'runs/repair_status.json').read_text());status.update(status='compact_finish_running',current='main_heldout_evaluation',current_pid=os.getpid());(ROOT/'runs/repair_status.json').write_text(json.dumps(status,indent=2))
 begin=time.monotonic();data=Dataset();cfg=V3Config(**ck['config'])
 model=DistributionModel(representation_matrix(data.esm,cfg.representation),data.pre['gene_mean'],data.pre['protein_mean'],data.pre['protein_sd'],cfg).cuda().eval();model.load_state_dict(ck['model'])
 m=dict(status='running',arm='joint_full_repaired',mode='mixed',final_step=1500,checkpoint=str(source),checkpoint_sha256=sha(source),script_sha256=sha(Path(__file__)),nodes=25790,batch=1024,learn_missing_inputs=True,checkpoint_policy='user-amended fixed 1500-step checkpoint; weights unchanged during evaluation',brain_validation_score=v['brain_validation_score'],protein_P5_mse=v['protein_P5_mse'])
 (dest/'manifest.json').write_text(json.dumps(m,indent=2))
 shutil.copy2(source.parent/'validation_latest.csv',dest/'validation_latest.csv');shutil.copy2(source.parent/'protein_validation.csv',dest/'protein_validation.csv')
 brain=evaluate_brain(model,data,'test','cuda',n=1024);pd.DataFrame(brain).to_csv(dest/'brain_test_by_route_unit.csv',index=False);data._brain_eval_cache.clear()
 print('Main brain held-out evaluation complete',flush=True)
 pd.DataFrame(evaluate_protein(model,data,'test','cuda',dest)).to_csv(dest/'protein_test_by_unit_type.csv',index=False)
 print('Main P6-P8 evaluation complete',flush=True)
 evaluate_species(model,data,'cuda',dest)
 assert m['checkpoint_sha256']==sha(source)
 m.update(status='complete',completed=datetime.datetime.now().astimezone().isoformat(),elapsed_seconds=time.monotonic()-begin)
 (dest/'manifest.json').write_text(json.dumps(m,indent=2))
 metrics=['mmd','energy','mean_mse','mean_pearson','pred_var'];rows=[];proteins=[]
 for name,path in [('repaired_main',dest),('fixed_input_control',control)]:
  d=pd.read_csv(path/'brain_test_by_route_unit.csv');d['arm']=name;rows.append(d)
  p=pd.read_csv(path/'protein_test_by_unit_type.csv');p['arm']=name;proteins.append(p)
 d=pd.concat(rows);d.to_csv(OUT/'brain_all_route_unit.csv',index=False)
 donor=d.groupby(['arm','unit','disease'])[metrics].mean().reset_index();donor.to_csv(OUT/'brain_donor_summary.csv',index=False)
 macro=donor.groupby('arm')[metrics].mean().reset_index();macro.to_csv(OUT/'brain_macro_summary.csv',index=False)
 a=donor[donor.arm=='repaired_main'].set_index(['unit','disease'])[metrics];b=donor[donor.arm=='fixed_input_control'].set_index(['unit','disease'])[metrics]
 assert a.index.equals(b.index);(a-b).reset_index().to_csv(OUT/'paired_brain_differences.csv',index=False)
 p=pd.concat(proteins);p.to_csv(OUT/'protein_all_rows.csv',index=False)
 u=p.groupby(['arm','unit','celltype']).agg(mse=('mse','mean'),median_pearson=('pearson','median')).reset_index();u.to_csv(OUT/'protein_donor_summary.csv',index=False)
 protein_macro=u.groupby(['arm','celltype'])[['mse','median_pearson']].mean().reset_index();protein_macro.to_csv(OUT/'protein_macro_summary.csv',index=False)
 original=pd.read_csv(ROOT/'results/sampling_corrected/brain_macro_summary.csv');original=original[original.arm=='joint_full']
 compare=pd.concat([original.assign(stage='original_1000'),macro[macro.arm=='repaired_main'].assign(stage='repaired_1500')]);compare.to_csv(OUT/'within_model_budget_change.csv',index=False)
 summary=dict(status='complete',completed=datetime.datetime.now().astimezone().isoformat(),main_checkpoint=str(source),main_step=1500,control_step=1500,additional_training_updates_total=1250,accounting='3 pilots x250 + main continuation250 + fixed-input control continuation250; total1250 versus original21000',matched_comparison_scope='Missing RNA input weights, same mixed sampling and 1500 total steps; does not re-establish every original component contrast',original_20_arms_preserved=True,corrected_20_evaluations_complete=True,recommendation=recommendation,brain=macro.to_dict('records'),protein_ALL=protein_macro[protein_macro.celltype=='ALL'].to_dict('records'))
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 status.update(status='complete',current=None,current_pid=None,completed_at=summary['completed'],completion_scope='20 original arms,20 corrected evaluations,3 validation pilots,2 matched1500-step models',cancelled='Broad9-10arm continuation',results=str(OUT.relative_to(ROOT)),additional_training_updates_total=1250)
 (ROOT/'runs/repair_status.json').write_text(json.dumps(status,indent=2))
 print(json.dumps(summary,indent=2),flush=True)
# The complete original evaluator above is retained as the provenance source.
# Figure exports below never call its training-status mutations or model selection.
FIG=ROOT/'analysis/figure_build_20261031';FD=FIG/'data'
def figure_metadata(data):
 counts=[];comps=[]
 for name,d in data.data.items():
  o=d['obs']
  for (assay,split,disease),z in o.groupby(['assay','split','disease']):
   counts.append(dict(dataset=name,assay=assay,split=split,disease=disease,cells=len(z),units=z.unit.nunique()))
  z=o.groupby(['unit','celltype','disease','split']).size().reset_index(name='n');z['dataset']=name;comps.append(z)
 pd.DataFrame(counts).to_csv(FD/'cohorts.csv',index=False)
 pd.concat(comps).to_csv(FD/'composition.csv',index=False)
 genes=pd.read_csv(data.root/'genes.csv');genes.to_csv(FD/'genes.csv',index=False)
 annotations=pd.read_csv(data.root/'protein_annotations.csv').fillna('')
 cite=data.data['cite'];o=cite['obs'];pairs=[]
 for donor in ['P6','P7','P8']:
  path=ROOT/'runs/uniform1500/joint_full/seed_42'/f'{donor}_protein_predictions.npz'
  pp=np.load(path);idx=pp['indices'];assert (o.iloc[idx].unit==donor).all()
  for antigen,ct in [('CD14','Mono'),('CD4','CD4 T'),('CD8a','CD8 T'),('CD19','B')]:
   hits=annotations.index[annotations.adt_feature.str.casefold()==antigen.casefold()]
   if len(hits)!=1:print('ANTIGEN_UNAVAILABLE',antigen,flush=True);continue
   j=int(hits[0]);keep=np.asarray(o.iloc[idx].celltype==ct);rows=idx[keep]
   pairs.append(pd.DataFrame(dict(donor=donor,antigen=antigen,celltype=ct,index=rows,observed=np.asarray(cite['y'][rows,j]),predicted=pp['predicted'][keep,j])))
 pd.concat(pairs).to_csv(FD/'protein_cell_pairs.csv.gz',index=False)
 lookup={}
 for i,r in genes.iterrows():
  for s in str(r.symbols).split('|'):lookup.setdefault(s,[]).append(i)
 coupling=[]
 for _,a in annotations.iterrows():
  ss=str(a.gene_symbols).split('|');ix=lookup.get(ss[0],[]) if len(ss)==1 else []
  if len(ix)!=1 or str(a.isotype_control).lower()=='true':continue
  j=int(a.protein_index)
  for ct in sorted(o.celltype.unique()):
   ii=np.flatnonzero((o.split=='train')&(o.celltype==ct))
   if len(ii)<20:continue
   xx=cite['x'][ii,ix[0]].toarray().ravel();yy=np.asarray(cite['y'][ii,j]);r=np.corrcoef(xx,yy)[0,1] if xx.std()>0 and yy.std()>0 else np.nan
   coupling.append(dict(protein_index=j,antigen=a.adt_feature,gene=ss[0],celltype=ct,n=len(ii),train_rna_adt_r=r,train_detection=float(np.mean(xx>0))))
 pd.DataFrame(coupling).to_csv(FD/'protein_train_coupling.csv',index=False)
 print('METADATA_COMPLETE',flush=True)

@torch.no_grad()
def figure_brain(data):
 from evaluation import rng_for
 from data_io import tensor
 for arm in ['joint_full','human_full']:
  ckpath=ROOT/'runs/uniform1500'/arm/'seed_42/last.pt';before=sha(ckpath)
  ck=torch.load(ckpath,map_location='cpu',weights_only=False);assert ck['step']==1500 and data.genes==ck['gene_names']
  cfg=V3Config(**ck['config']);model=DistributionModel(representation_matrix(data.esm,cfg.representation),data.pre['gene_mean'],data.pre['protein_mean'],data.pre['protein_sd'],cfg).eval();model.load_state_dict(ck['model'])
  allmeans=[];dist=[];d=data.data['human_brain']
  for route in data.routes('human'):
   label,src,tgt=route['disease'],route['source_type'],route['target_type']
   pool=data.pools.get(('human_brain','test',label,tgt),{})
   if not pool:continue
   xb,sm,_=data.sample('human_blood',label,src,1024,rng_for('source','human',label,src));c=data.condition(1024,'human',label,src,tgt,'cpu')
   noise=tensor(rng_for('noise','human',label,src,tgt).normal(size=(1024,cfg.latent_dim)).astype('f'),'cpu')
   pred=model.predict_brain(tensor(xb,'cpu'),c,noise=noise,source_mask=tensor(sm,'cpu'))[0].numpy()
   for assay,units in pool.items():
    mask=data.masks[assay];pp=pred[:,mask];pm=pp.mean(0);nodes=np.asarray(data.genes)[mask]
    for unit,ids in sorted(units.items()):
     ym=np.asarray(d['x'][ids][:,mask].mean(0)).ravel()
     allmeans.append(pd.DataFrame(dict(arm=arm,unit=unit,source_type=src,target_type=tgt,node=nodes,observed=ym,predicted=pm)))
     ids2=rng_for('target','human','test',assay,unit,tgt).choice(ids,min(1024,len(ids)),False)
     yy=d['x'][ids2][:,mask]
     for lab,vals in [('Observed',np.asarray(yy.mean(1)).ravel()),('Predicted',pp.mean(1))]:
      dist.append(pd.DataFrame(dict(arm=arm,unit=unit,source_type=src,target_type=tgt,kind=lab,value=vals)))
   print('BRAIN',arm,label,src,tgt,flush=True)
  pd.concat(allmeans).to_csv(FD/f'brain_feature_means_{arm}.csv.gz',index=False)
  pd.concat(dist).to_csv(FD/f'brain_distributions_{arm}.csv.gz',index=False)
  assert sha(ckpath)==before
  (FD/f'{arm}_export_identity.json').write_text(json.dumps(dict(checkpoint=str(ckpath),sha256=before,step=1500,predicted_cells=1024,nodes=len(data.genes),weights_unchanged=True,sampling='original evaluation rng_for; corrected DistributionModel; CPU float32'),indent=2))
  del model,ck

def figure_atlas(data):
 from sklearn.decomposition import TruncatedSVD
 import umap
 for name in ['human_blood','human_brain','mouse_blood','mouse_brain']:
  dest=FD/f'atlas_{name}.csv.gz'
  if dest.exists():continue
  x=data.data[name]['x'];o=data.data[name]['obs'];svd=TruncatedSVD(n_components=50,random_state=42);z=svd.fit_transform(x)
  # All cells are projected. The lower-dimensional representation is for display only.
  u=umap.UMAP(n_neighbors=30,min_dist=.3,metric='euclidean',random_state=42,n_jobs=1).fit_transform(z)
  out=o[['unit','celltype','disease','split','assay']].copy();out['x']=u[:,0];out['y']=u[:,1];out['dataset']=name;out.to_csv(dest,index=True)
  (FD/f'atlas_{name}.json').write_text(json.dumps(dict(cells=len(o),features=x.shape[1],svd_dimensions=50,random_state=42,n_neighbors=30,min_dist=.3,explained_variance=float(svd.explained_variance_ratio_.sum()),display_only=True),indent=2))
  print('ATLAS',name,len(o),flush=True)

def summarize_figure_exports():
 out=[]
 for arm in ['joint_full','human_full']:
  parts=[]
  for d in pd.read_csv(FD/f'brain_feature_means_{arm}.csv.gz',chunksize=500000):
   d['error']=(d.predicted-d.observed)**2;d['n']=1
   parts.append(d.groupby(['unit','node'])[['observed','predicted','error','n']].sum())
  a=pd.concat(parts).groupby(level=[0,1]).sum();a[['observed','predicted','error']]=a[['observed','predicted','error']].div(a.n,axis=0);a=a.reset_index();a['arm']=arm;out.append(a)
  print('SUMMARIZED',arm,len(a),flush=True)
 p=pd.concat(out);p.to_csv(FD/'brain_gene_donor_means.csv.gz',index=False)
 a=p[p.arm.eq('joint_full')].set_index(['unit','node']);b=p[p.arm.eq('human_full')].set_index(['unit','node']);assert a.index.equals(b.index)
 d=(b.error-a.error).rename('improvement').reset_index().groupby('node').improvement.agg(['mean','std']).reset_index().rename(columns={'mean':'improvement','std':'donor_sd'})
 genes=pd.read_csv(ROOT/'data/processed/genes.csv');d=d.merge(genes,left_on='node',right_on='node_id',validate='one_to_one')
 pre=np.load(ROOT/'data/processed/preprocessing.npz');mean=pd.Series(pre['gene_mean'],index=genes.node_id);d['training_mean']=d.node.map(mean);d['orthology_class']=np.select([d.n_human.eq(1)&d.n_mouse.eq(1),d.n_human.gt(0)&d.n_mouse.gt(0)],['One-to-one','Multiple members'],default='Species-specific');d.to_csv(FD/'transfer_feature_errors.csv.gz',index=False)
 dist=pd.read_csv(FD/'brain_distributions_joint_full.csv.gz');dist=dist[dist.source_type.eq('Mono')&dist.target_type.eq('Micro')];dist.to_csv(FD/'brain_mono_micro_distributions.csv.gz',index=False)
 # Region tracks retain every variant and use the same coordinate system as coloc.
 mr=ROOT/'analysis/mr_network_v3_repair';reg=[];sens=[]
 for gene,assay in [('BIN1','GCST90090757'),('GRN','GCST90088852')]:
  for kind,path,pval in [('Protein pQTL',mr/f'data/mr/processed/ages/regions/{assay}.tsv.gz','p_exposure'),('Bellenguez AD',mr/f'data/mr/processed/ad_ages_regions/{assay}.tsv.gz','p_outcome'),('FinnGen AD',mr/f'data/mr/processed/finngen/AGES_{assay}.tsv.gz','p_outcome')]:
   z=pd.read_csv(path,sep='\t');z['neglog10p']=-np.log10(z[pval].clip(lower=1e-300));z['track']=kind;z['gene']=gene;z['assay']=assay;reg.append(z[['gene','assay','track','rsid','chrom','pos','neglog10p']])
  for outcome in ['Bellenguez_stageI','FinnGen_R13']:
   z=json.loads((mr/f'results/mr/public_followup/coloc/{assay}_{outcome}.json').read_text())
   for q in z['sensitivity']:sens.append(dict(gene=gene,assay=assay,outcome=outcome,analysis='Prior',x=q['p12'],PP_H4=q['PP_H4']))
   (FD/f'coloc_{assay}_{outcome}.json').write_text(json.dumps(z,indent=2))
 pd.concat(reg).to_csv(FD/'regional_association_tracks.csv.gz',index=False);pd.DataFrame(sens).to_csv(FD/'coloc_prior_sensitivity.csv',index=False)
 print('SUMMARIES_COMPLETE',flush=True)
if __name__=='__main__':summarize_figure_exports()
