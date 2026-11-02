#!/usr/bin/env python3
"""Train and evaluate the full blood-to-brain world model."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import argparse,json,time,hashlib,random,csv,collections
import numpy as np,pandas as pd,torch
from world_model import WorldModel,V3Config,representation_matrix
from data_io import Dataset,tensor,ROOT
from evaluation import evaluate_brain,evaluate_protein,validation_score

def atomic_save(obj,path):
 tmp=path.with_suffix(path.suffix+'.tmp');torch.save(obj,tmp);tmp.replace(path)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--seed',type=int,required=True)
 ap.add_argument('--device',default='cuda');ap.add_argument('--run-root',default='runs/formal');ap.add_argument('--smoke',action='store_true')
 ap.add_argument('--steps',type=int);args=ap.parse_args()
 spec=json.loads(Path(args.config).read_text());contract=json.loads((ROOT/'docs/SCOPE_CONTRACT.json').read_text())
 assert spec['batch']==contract['reference_config']['batch_size']==1024
 if args.steps and not args.smoke:raise ValueError('Formal update budgets cannot be reduced from command line')
 if args.smoke:spec['steps']=args.steps or 20
 protocol=json.loads((ROOT/'configs/ablation_manifest.json').read_text())
 assert args.smoke or spec['steps']>=protocol['minimum_formal_updates']
 if not args.smoke:
  assert args.seed in protocol['seeds']
  assert spec['steps']==protocol['updates_by_arm'][spec['name']]
 run=ROOT/args.run_root/spec['name']/f'seed_{args.seed}';run.mkdir(parents=True,exist_ok=False)
 random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(4)
 if args.device=='cuda':
  assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.24,0);torch.backends.cuda.matmul.allow_tf32=False
 rng=np.random.default_rng(args.seed);data=Dataset()
 assert data.genes==contract['reference_gene_names']
 config={**contract['reference_config'],**spec.get('model',{}),'cond_dim':data.cond_dim,'n_proteins':len(data.pre['protein_mean']),
         'ct_brain_labels':data.layout['brain'],'ct_blood_labels':data.layout['blood'],'seed':args.seed,
         'model_dir':str(run),'output_dir':str(run)}
 cfg=V3Config(**config)
 esm=representation_matrix(data.esm,cfg.representation)
 model=WorldModel(esm,data.pre['gene_mean'],data.pre['protein_mean'],data.pre['protein_sd'],cfg).to(args.device)
 opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=spec['lr'])
 manifest=dict(spec=spec,seed=args.seed,status='running',scope='full_v2_plus_extensions',started=time.strftime('%FT%T%z'),
  model_config=cfg.__dict__,condition_layout=data.layout,n_nodes=len(data.genes),batch=spec['batch'],
  data_manifest_sha256=sha(data.root/'data_manifest.json'),source_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},
  gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),torch=torch.__version__,parameter_count=sum(p.numel() for p in model.parameters()),
  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),test_selection=False)
 (run/'manifest.json').write_text(json.dumps(manifest,indent=2))
 routes={s:data.routes(s) for s in ['human','mouse']};positions={s:0 for s in routes};counts=collections.Counter()
 for s in routes:rng.shuffle(routes[s])
 best=float('inf');beststep=0;start=time.monotonic();batch=spec['batch'];every=spec['eval_every']
 fields=['step','species','source_type','target_type','disease','loss','distribution','recon','kl','drift','protein','elapsed']
 with (run/'train.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for step in range(1,spec['steps']+1):
   model.train();species='human' if spec['species']=='human' or step%2 else 'mouse'
   pos=positions[species]
   if pos==len(routes[species]):rng.shuffle(routes[species]);pos=0
   route=routes[species][pos];positions[species]=pos+1
   label,src,tgt=route['disease'],route['source_type'],route['target_type']
   target_label=('AD' if label=='CTRL' else 'CTRL') if species=='mouse' and spec.get('mouse_label_swap') else label
   # If a swapped rare type has no opposite-state observations, keep it as an explicitly unsupported diagnostic stratum.
   if (species+'_brain','train',target_label,tgt) not in data.pools:
    raise RuntimeError('Label-swap control lacks opposite-state target: '+str(route))
   xb,sm,_=data.sample(species+'_blood',label,src,batch,rng)
   yb,tm,meta=data.sample(species+'_brain',target_label,tgt,batch,rng)
   xc,u,cm=data.sample_cite(batch,rng)
   c=data.condition(batch,species,label,src,tgt,args.device)
   losses=model.losses(tensor(xb,args.device),tensor(yb,args.device),c,tensor(xc,args.device),tensor(u,args.device),
      tensor(data.scales[meta['assay']],args.device),spec.get('shuffle_protein',False),
      tensor(sm,args.device),tensor(tm,args.device),tensor(cm,args.device))
   if not torch.isfinite(losses['loss']):raise FloatingPointError('Nonfinite loss at '+str(step))
   opt.zero_grad(set_to_none=True);losses['loss'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
   counts[(species,label,src,tgt)]+=1
   if step%10==0 or step==1:
    writer.writerow(dict(step=step,species=species,source_type=src,target_type=tgt,disease=label,
       **{k:float(v.detach().cpu()) for k,v in losses.items()},elapsed=time.monotonic()-start));f.flush()
   if step%every==0 or step==spec['steps']:
    rows=evaluate_brain(model,data,'val',args.device,n=32 if args.smoke else 1024)
    score=validation_score(rows);pd.DataFrame(rows).to_csv(run/'validation_latest.csv',index=False)
    ck=dict(model=model.state_dict(),config=cfg.__dict__,spec=spec,step=step,optimizer=opt.state_dict(),metric=score,
       gene_names=data.genes,protein_features=json.loads((data.root/'cite/proteins.json').read_text()),condition_layout=data.layout,
       assay_names=data.assays,assay_masks=data.masks,route_registry=data.registry.to_dict('records'),
       numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if args.device=='cuda' else None,
       data_manifest_sha256=manifest['data_manifest_sha256'])
    atomic_save(ck,run/'last.pt')
    if score<best:best=score;beststep=step;atomic_save(ck,run/'best.pt')
    progress=dict(step=step,planned_updates=spec['steps'],best_step=beststep,best_validation=best,observed_strata=len(counts),elapsed=time.monotonic()-start)
    (run/'progress.json').write_text(json.dumps(progress,indent=2));print(json.dumps(dict(arm=spec['name'],seed=args.seed,**progress)),flush=True)
    pd.DataFrame([dict(species=k[0],disease=k[1],source_type=k[2],target_type=k[3],updates=v) for k,v in counts.items()]).to_csv(run/'training_stratum_counts.csv',index=False)
 if not args.smoke:
  expected=len(routes['human'])+(len(routes['mouse']) if spec['species']=='joint' else 0)
  assert len(counts)==expected,'Every registered active stratum must receive actual training updates'
 ck=torch.load(run/'best.pt',map_location=args.device,weights_only=False);model.load_state_dict(ck['model'])
 if hasattr(data,'_brain_eval_cache'):data._brain_eval_cache.clear()
 if not args.smoke:
  pd.DataFrame(evaluate_brain(model,data,'test',args.device,n=1024)).to_csv(run/'brain_test_by_route_unit.csv',index=False)
  if hasattr(data,'_brain_eval_cache'):data._brain_eval_cache.clear()
  if cfg.protein_weight:
   pd.DataFrame(evaluate_protein(model,data,'test',args.device,run)).to_csv(run/'protein_test_by_unit_type.csv',index=False)
   pd.DataFrame(evaluate_protein(model,data,'val',args.device)).to_csv(run/'protein_validation.csv',index=False)
  from species_diagnostics import evaluate as species_evaluate
  if spec['species']=='joint':species_evaluate(model,data,args.device,run)
 manifest.update(status='smoke_complete' if args.smoke else 'complete',completed=time.strftime('%FT%T%z'),
   final_step=step,best_step=beststep,best_validation=best,elapsed_seconds=time.monotonic()-start,
   human_updates=step if spec['species']=='human' else (step+1)//2,selected_human_updates=beststep if spec['species']=='human' else (beststep+1)//2,
   observed_strata=len(counts),peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated()) if args.device=='cuda' else 0)
 (run/'manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
