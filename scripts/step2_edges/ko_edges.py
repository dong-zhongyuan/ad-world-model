#!/usr/bin/env python3
"""Generate virtual peripheral gene-clamp responses across six blood-to-brain routes."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

COMBOS = 'Micro:CD4T,Micro:CD8T,Micro:NK,Micro:B,Micro:Mono,Oligo:Mono'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def atomic_json(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def random_sources(det,candidates,n=512):
    rng=np.random.RandomState(7)
    strata=np.digitize(det[candidates],np.quantile(det[candidates],np.linspace(0,1,11))[1:-1])
    count=min(n,len(candidates));parts=[]
    for s in np.unique(strata):
        pool=candidates[strata==s];want=max(1,round(count*(strata==s).mean()))
        parts.append(rng.choice(pool,min(want,len(pool)),replace=False))
    return np.concatenate(parts)[:count]

def empirical_p(sorted_null, values):
    """Per-target upper-tail ranks, including ties, using identical |mean delta|."""
    n,t=sorted_null.shape
    lo=np.zeros(values.shape,dtype=np.int64);hi=np.full(values.shape,n,dtype=np.int64)
    cols=np.arange(t)[None,:]
    while np.any(lo<hi):
        active=lo<hi;mid=(lo+hi)//2
        lower=sorted_null[np.minimum(mid,n-1),cols]<values
        lo=np.where(active&lower,mid+1,lo);hi=np.where(active&~lower,mid,hi)
    return (1+n-lo)/(1+n)

def select_qualified(mean,p,eligible,top_k=15):
    """Filter every eligible target first, then retain strongest qualified edges."""
    valid=eligible&(p<=.05)&(np.abs(mean)>1e-7)
    indices=np.flatnonzero(valid)
    return indices[np.argsort(-np.abs(mean[indices]),kind='stable')[:top_k]],int(valid.sum())

@torch.no_grad()
def projected_delta(model,x,c,mask,genes,targets):
    """Exact input-clamp algebra; subtract one projection row instead of redoing X@ESM.
    FP32 operations throughout. Only requested decoder rows are materialized.
    """
    if model.training or model.config.mapping!='ode' or not model.config.source_expression:
        raise ValueError('Requires eval full source-dependent ODE model')
    projection=model.esm_matrix if model.raw_projection is None else model.raw_projection.weight.T
    h=((x-model.gene_mean)*mask)@projection
    mu,_=model.encoder(h)
    z=model.forward_ode(mu,'blood_to_brain',model.condition(c))
    hidden=model.decoder_z(z)
    b,n=len(genes),len(x)
    perturbation=x[:,genes].T*mask[:,genes].T
    hk=h.unsqueeze(0)-perturbation[:,:,None]*projection[genes,None,:]
    muk,_=model.encoder(hk.reshape(b*n,-1))
    zk=model.forward_ode(muk,'blood_to_brain',model.condition(c.repeat(b,1)))
    hkout=model.decoder_z(zk).reshape(b,n,-1)
    # Linear readout bias and gene_mean cancel analytically in the response.
    d=F.linear(hkout-hidden.unsqueeze(0),model.dec_out.weight[targets])
    d.masked_fill_(perturbation[:,:,None]==0,0.)
    return d

@torch.no_grad()
def response_stats(model,x,c,m,genes,targets,cell_batch):
    device=next(model.parameters()).device
    shape=(len(genes),len(targets));total=torch.zeros(shape,dtype=torch.float64,device=device)
    squares=total.clone();abs_sum=total.clone();n=len(x)
    for k in range(0,n,cell_batch):
        xt=torch.as_tensor(x[k:k+cell_batch],device=device)
        ct=torch.as_tensor(c[k:k+cell_batch],device=device)
        mt=torch.as_tensor(m[k:k+cell_batch],device=device)
        d=projected_delta(model,xt,ct,mt,genes,targets).double()
        if not torch.isfinite(d).all():raise RuntimeError('Nonfinite response')
        total+=d.sum(1);squares+=d.square().sum(1);abs_sum+=d.abs().sum(1)
    mean=total/n
    sd=((squares-total.square()/n)/max(n-1,1)).clamp_min(0).sqrt()
    z=mean.abs()/(sd/np.sqrt(n)+1e-12)
    return mean.cpu().numpy(),z.cpu().numpy(),(abs_sum/n).cpu().numpy()

@torch.no_grad()
def verify_projection(model,x,c,m,genes,targets):
    device=next(model.parameters()).device
    xt=torch.as_tensor(x[:7],device=device);ct=torch.as_tensor(c[:7],device=device);mt=torch.as_tensor(m[:7],device=device)
    base=model.predict_brain(xt,ct,source_mask=mt)[0]
    delta=projected_delta(model,xt,ct,mt,genes[:3],targets)
    errors=[]
    for j,g in enumerate(genes[:3]):
        ko=xt.clone();ko[:,g]=0
        ref=(model.predict_brain(ko,ct,source_mask=mt)[0]-base)[:,targets]
        errors.append(float((ref-delta[j]).abs().max()))
        torch.testing.assert_close(delta[j],ref,rtol=2e-3,atol=2e-5)
    return max(errors)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--columns',type=Path,required=True)
    ap.add_argument('--combos',default=COMBOS);ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--ko-batch-size',type=int,default=8);ap.add_argument('--cell-batch',type=int,default=128)
    ap.add_argument('--check-only',action='store_true');ap.add_argument('--pilot-sources',type=int,default=0);args=ap.parse_args()
    sys.path.insert(0,str(args.root/'scripts/step1_world_model'))
    from world_model import WorldModel,V3Config
    from data_io import Dataset
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    if args.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.24,torch.device(args.device))
    current=json.loads((args.root/'results/CURRENT_RESULTS.json').read_text())
    checkpoint=args.root/current['recommended_checkpoint'];ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['step']==current['recommended_checkpoint_actual_step']==1500
    assert ck['spec']['name']=='joint_full' and len(ck['gene_names'])==25790
    d=Dataset();assert d.genes==ck['gene_names'] and d.layout==ck['condition_layout']
    assert sha(d.root/'data_manifest.json')==ck['data_manifest_sha256']
    assert all(np.array_equal(d.masks[a],v) for a,v in ck['assay_masks'].items())
    s=ck['model'];model=WorldModel(s['esm_matrix'].numpy(),s['gene_mean'].numpy(),s['protein_mean'].numpy(),s['protein_sd'].numpy(),V3Config(**ck['config']))
    model.load_state_dict(s,strict=True);model.to(args.device).eval();del ck,s
    columns=pd.read_csv(args.columns).set_index('node_id');kinds=columns['kind'].to_dict()
    args.out.mkdir(parents=True,exist_ok=True)
    contract=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),step=1500,
        code_sha256=sha(__file__),model_code_sha256=sha(args.root/'scripts/step1_world_model/world_model.py'),
        data_manifest_sha256=sha(d.root/'data_manifest.json'),columns_sha256=sha(args.columns),
        observation_sha256={t:sha(d.root/t/'obs.csv') for t in ['human_blood','human_brain']},
        combos=args.combos,min_det=.05,min_target_det=.01,max_cells=2048,top_k=15,
        random_sources=512,random_seed=7,cell_seed=42,randko_cutoff=.05,
        direction='blood_to_brain',latent='posterior_mean',nodes=len(d.genes),
        repair_version='matched_signed_mean_filter_before_top15_all_members',
        statistic='abs(mean_cell_delta) for both actual and random sources',
        selection_order='shared_og_nonself_and_empirical_p<=.05_then_top15_by_abs_mean',
        pilot_sources=args.pilot_sources)
    lock=args.out/'contract.json'
    if lock.exists():assert json.loads(lock.read_text())==contract,'Run provenance changed'
    else:atomic_json(lock,contract)
    contract_sha=sha(lock);src=d.data['human_blood'];tgt=d.data['human_brain'];summary=[]
    for combo in args.combos.split(','):
        tc,sc=combo.split(':');tag='rev_'+sc+'_'+tc;dest=args.out/tag;dest.mkdir(exist_ok=True)
        status=dest/'status.json'
        if status.exists():
            prior=json.loads(status.read_text());assert prior['contract_sha256']==contract_sha
            if prior['status']=='complete':
                assert sha(dest/'edges.csv')==prior['edges_sha256'];summary.append(prior);continue
        ids=np.flatnonzero(src['obs'].celltype.to_numpy()==sc)
        if len(ids)>2048:ids=np.random.RandomState(42).choice(ids,2048,replace=False)
        if len(ids)<2:raise ValueError('Insufficient source cells '+combo)
        o=src['obs'].iloc[ids];labels=set(o.disease);registered=d.registry
        assert all(((registered.species=='human')&(registered.disease==label)&(registered.source_type==sc)&(registered.target_type==tc)&(registered.status=='active')).any() for label in labels)
        x=src['x'][ids].toarray().astype('f');m=np.stack([d.masks[a] for a in o.assay]).astype('f')
        c=np.zeros((len(x),d.cond_dim),dtype='f')
        for label in labels:
            loc=np.flatnonzero(o.disease.to_numpy()==label)
            c[loc]=d.condition(len(loc),'human',label,sc,tc,'cpu').numpy()
        det=(x>1e-6).mean(0);target_ids=np.flatnonzero(tgt['obs'].celltype.to_numpy()==tc)
        assert len(target_ids)>0
        target_det=np.asarray((tgt['x'][target_ids]>1e-6).mean(0)).ravel()
        candidates=np.flatnonzero((det>=.05)&m.any(0));targets=np.flatnonzero(target_det>=.01)
        if not len(candidates) or not len(targets):raise RuntimeError('Empty feature pool '+combo)
        error=verify_projection(model,x,c,m,candidates,targets)
        null_ids=random_sources(det,candidates);null_lookup={int(g):i for i,g in enumerate(null_ids)}
        selected_sources=candidates
        if args.pilot_sources:
            selected_sources=candidates[np.linspace(0,len(candidates)-1,min(args.pilot_sources,len(candidates)),dtype=int)]
            candidates=np.union1d(null_ids,selected_sources)
        print(json.dumps(dict(combo=combo,cells=len(x),sources=len(candidates),targets=len(targets),projection_max_abs_error=error)),flush=True)
        if args.check_only:continue
        atomic_json(status,dict(status='running',contract_sha256=contract_sha,combo=combo,total_sources=len(candidates)))
        np.save(dest/'source_cell_indices.npy',ids)
        chunks=dest/'chunks';chunks.mkdir(exist_ok=True);begin=time.monotonic()
        for k in range(0,len(candidates),args.ko_batch_size):
            gs=candidates[k:k+args.ko_batch_size];f=chunks/f'{k:06d}.npz'
            if f.exists():
                with np.load(f) as a:assert np.array_equal(a['genes'],gs)
                continue
            mean,z,ab=response_stats(model,x,c,m,gs,targets,args.cell_batch)
            null_rows=[j for j,g in enumerate(gs) if int(g) in null_lookup]
            # Full response retained so subsequent filtering never depends on a premature top-k.
            tmp=f.with_suffix('.tmp')
            with tmp.open('wb') as fh:np.savez_compressed(fh,genes=gs,targets=targets,mean=mean,z=z,null_genes=gs[null_rows],null_abs=np.abs(mean[null_rows]),legacy_null_abs=ab[null_rows])
            tmp.replace(f)
            if k%(args.ko_batch_size*25)==0:
                atomic_json(status,dict(status='running',contract_sha256=contract_sha,combo=combo,processed_sources=k+len(gs),total_sources=len(candidates),elapsed_seconds=time.monotonic()-begin))
                print(tag,k+len(gs),'/',len(candidates),round(time.monotonic()-begin,1),'s',flush=True)
        null=np.empty((len(null_ids),len(targets)),dtype=np.float64);seen=set();rows=[]
        for f in sorted(chunks.glob('*.npz')):
            with np.load(f) as a:
                for g,v in zip(a['null_genes'],a['null_abs']):null[null_lookup[int(g)]]=v;seen.add(int(g))
        assert seen==set(map(int,null_ids))
        sorted_null=np.sort(null,axis=0);diagnostics=[];selection=set(map(int,selected_sources))
        target_shared=np.array([kinds[d.genes[t]]=='shared_og' for t in targets])
        for f in sorted(chunks.glob('*.npz')):
            with np.load(f) as a:
                p=empirical_p(sorted_null,np.abs(a['mean']))
                for j,g in enumerate(a['genes']):
                    if int(g) not in selection:continue
                    eligible=target_shared&(targets!=g)&(kinds[d.genes[g]]=='shared_og')
                    positions,count=select_qualified(a['mean'][j],p[j],eligible)
                    oldmean=a['mean'][j].copy();oldmean[targets==g]=0.
                    oldtop=np.argsort(-np.abs(oldmean),kind='stable')[:15]
                    oldvalid=eligible[oldtop]&(p[j,oldtop]<=.05)&(np.abs(oldmean[oldtop])>1e-7)
                    diagnostics.append(dict(source=d.genes[g],n_qualified_before_cap=count,n_retained=len(positions),n_qualified_in_old_top15=int(oldvalid.sum())))
                    for pos in positions:
                        t=targets[pos];rows.append(dict(source=d.genes[g],target=d.genes[t],delta_mean=a['mean'][j,pos],z=a['z'][j,pos],source_det=det[g],target_det=target_det[t],source_kind=kinds[d.genes[g]],target_kind=kinds[d.genes[t]],randko_q=p[j,pos],randko_p=p[j,pos]))
        pd.DataFrame(diagnostics).to_csv(dest/'selection_diagnostics.csv',index=False)
        e=pd.DataFrame(rows,columns=['source','target','delta_mean','z','source_det','target_det','source_kind','target_kind','randko_q','randko_p'])
        e.sort_values('z',ascending=False).to_csv(dest/'edges.csv',index=False)
        info=dict(status='complete',contract_sha256=contract_sha,combo=combo,cells=len(x),sources=len(candidates),targets=len(targets),edges=len(e),random_sources=len(null_ids),edges_sha256=sha(dest/'edges.csv'),projection_max_abs_error=error,elapsed_seconds=time.monotonic()-begin)
        atomic_json(status,info);summary.append(info)
    atomic_json(args.out/'status.json',dict(status='checks_passed' if args.check_only else 'complete',contract_sha256=contract_sha,routes=summary))

if __name__=='__main__':main()
