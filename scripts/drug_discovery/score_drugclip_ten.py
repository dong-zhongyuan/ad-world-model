#!/usr/bin/env python3
"""Six-fold official DrugCLIP ranking for all ten authorized targets.

Encoder operations follow THU-ATOM/Drug-The-Whole-Genome's
unimol/tasks/drugclip.py::retrieval_multi_folds, retained under software/.
"""
import argparse
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import csv
import lmdb
import numpy as np
import torch

from paths import DRUG_WORK as ROOT
REPO=ROOT/'software/Drug-The-Whole-Genome'
sys.path.insert(0,str(REPO))
import unimol
from unimol.tasks.drugclip import DrugCLIP
from unimol.models.drugclip import BindingAffinityModel
from unicore import utils


def write(path,data):
    temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def gate_check():
    path=ROOT/'provenance/ten_candidate_lease.json'
    lease=json.loads(path.read_text())
    assert time.time()-path.stat().st_mtime < 90, 'Candidate decision lease expired'
    assert len(lease['gates'])==10 and all(g['allowed'] for g in lease['gates'].values()), 'Candidate stopped'
    return lease


def lmdb_write(path,records):
    if path.exists():
        return
    env=lmdb.open(str(path),subdir=False,map_size=128*1024*1024)
    with env.begin(write=True) as txn:
        for i,record in enumerate(records):
            txn.put(str(i).encode(),pickle.dumps(record,protocol=4))
    env.sync();env.close()


def canonical_rows(values, actual_ids, expected_ids):
    assert len(values)==len(actual_ids)==len(expected_ids)
    assert len(set(actual_ids))==len(actual_ids) and set(actual_ids)==set(expected_ids)
    positions={name:i for i,name in enumerate(actual_ids)}
    return values[np.array([positions[name] for name in expected_ids])]


@torch.inference_mode()
def encode(model,sample,kind):
    sample=utils.move_to_cuda(sample)
    data=sample['net_input']
    dist=data[kind+'_src_distance'];edge=data[kind+'_src_edge_type'];tokens=data[kind+'_src_tokens']
    encoder=getattr(model,kind+'_model')
    features=encoder.gbf(dist,edge)
    bias=encoder.gbf_proj(features).permute(0,3,1,2).contiguous()
    bias=bias.view(-1,dist.shape[-1],dist.shape[-1])
    outputs=encoder.encoder(encoder.embed_tokens(tokens),padding_mask=tokens.eq(encoder.padding_idx),attn_mask=bias)
    embedding=getattr(model,kind+'_project')(outputs[0][:,0,:])
    embedding=embedding/embedding.norm(dim=-1,keepdim=True)
    assert torch.isfinite(embedding).all()
    return embedding.float().cpu().numpy()


def main():
    gate_check()
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    source=ROOT/'inputs/ten_targets/verification/acceleration_inputs/drugclip_ten_ready.json.gz'
    with gzip.open(source,'rt') as f:data=json.load(f)
    pocket_source=ROOT/'inputs/ten_targets/pockets/all_ten_pockets.json.gz'
    with gzip.open(pocket_source,'rt') as f:data['pockets']=json.load(f)
    assert {p['target'] for p in data['pockets']}==set(gate_check()['gates'])
    work=ROOT/'inputs/ten_targets/ready_lmdb';work.mkdir(exist_ok=True)
    fingerprint=hashlib.sha256(source.read_bytes()+pocket_source.read_bytes()).hexdigest()
    identity=work/'source.sha256'
    if identity.exists():
        assert identity.read_text()==fingerprint, 'Export changed; use a new LMDB snapshot'
    else:
        identity.write_text(fingerprint)
    mol_records=[{'atoms':m['atoms'],'coordinates':np.array(m['coordinates'],dtype=np.float32),
                  'smi':m['state_id'],'label':0} for m in data['molecules']]
    pocket_records=[dict(p,pocket_coordinates=np.array(p['pocket_coordinates'],dtype=np.float32)) for p in data['pockets']]
    lmdb_write(work/'mols.lmdb',mol_records);lmdb_write(work/'pockets.lmdb',pocket_records)
    max_atoms=max(len(p['pocket_atoms']) for p in pocket_records)
    args=argparse.Namespace(data=str(REPO/'dict'),seed=42,max_seq_len=max(512,max_atoms+2),max_pocket_atoms=max(511,max_atoms),
                            pocket_encoder_layers=15)
    task=DrugCLIP.setup_task(args)
    model=BindingAffinityModel.build_model(args,task).eval().cuda().half()
    mol_data=task.load_mols_dataset(str(work/'mols.lmdb'),'atoms','coordinates')
    pocket_data=task.load_pockets_dataset(str(work/'pockets.lmdb'))
    mol_loader=torch.utils.data.DataLoader(mol_data,batch_size=32,collate_fn=mol_data.collater,num_workers=0)
    pocket_loader=torch.utils.data.DataLoader(pocket_data,batch_size=3,collate_fn=pocket_data.collater,num_workers=0)
    output=ROOT/'results/drugclip_ten';output.mkdir(exist_ok=True)
    cache=output/'embedding_cache';cache.mkdir(exist_ok=True)
    record={'status':'running','scope':data['scope'],'device':torch.cuda.get_device_name(0),
            'torch_version':torch.__version__,'precision':'float16, official inference setting',
            'molecules':len(mol_records),'pockets':len(pocket_records),'folds_planned':6,
            'pocket_heavy_atom_counts':[len(p['pocket_atoms']) for p in pocket_records],
            'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'pocket_input_sha256':hashlib.sha256(pocket_source.read_bytes()).hexdigest(),
            'target_count':10,'target_pocket_counts':{g:sum(p['target']==g for p in data['pockets']) for g in sorted(gate_check()['gates'])},
            'max_pocket_atoms':args.max_pocket_atoms,'max_seq_len':args.max_seq_len,'folds':[]}
    write(output/'progress.json',record)
    scores=[];started=time.monotonic()
    for fold in range(6):
        gate_check()
        path=ROOT/f'weights/model_weights/6_folds/fold_{fold}.pt'
        check=json.loads(path.with_suffix('.download.json').read_text());assert check['status']=='complete'
        t=time.monotonic()
        state=torch.load(path,map_location='cpu',weights_only=False)
        incompatible=model.load_state_dict(state['model'],strict=False)
        critical=('mol_model.','pocket_model.','mol_project.','pocket_project.')
        assert not any(k.startswith(critical) for k in incompatible.missing_keys),incompatible.missing_keys
        assert not any(k.startswith(critical) for k in incompatible.unexpected_keys),incompatible.unexpected_keys
        del state;gc.collect();torch.cuda.synchronize()
        load_seconds=time.monotonic()-t
        if fold==0:
            encode(model,next(iter(mol_loader)),'mol');torch.cuda.synchronize()
        t=time.monotonic();mols=[];mol_ids=[]
        cached=cache/f'fold_{fold}_molecules.npy'
        cache_id=cache/f'fold_{fold}_identity.json'
        expected_id={'molecule_input_sha256':record['input_sha256'],'model_sha256':check['sha256'],'precision':'float16','row_order':'canonical_source'}
        if cached.exists() and cache_id.exists():
            cache_info=json.loads(cache_id.read_text())
            assert {k:cache_info.get(k) for k in expected_id}==expected_id
            assert hashlib.sha256(cached.read_bytes()).hexdigest()==cache_info['embedding_sha256']
            mols=np.load(cached)
        else:
            for batch in mol_loader:
                gate_check()
                mols.append(encode(model,batch,'mol'))
                mol_ids.extend(batch['smi_name'])
            mols=np.concatenate(mols)
            mols=canonical_rows(mols,mol_ids,[m['state_id'] for m in data['molecules']])
            np.save(cached,mols)
            write(cache_id,dict(expected_id,embedding_sha256=hashlib.sha256(cached.read_bytes()).hexdigest()))
        torch.cuda.synchronize();mol_seconds=time.monotonic()-t
        t=time.monotonic();pockets=[];pocket_ids=[]
        for batch in pocket_loader:
            gate_check();pockets.append(encode(model,batch,'pocket'))
            pocket_ids.extend(batch['pocket_name'])
        torch.cuda.synchronize();pocket_seconds=time.monotonic()-t
        pockets=np.concatenate(pockets)
        pockets=canonical_rows(pockets,pocket_ids,[p['pocket'] for p in data['pockets']])
        assert mols.shape==(len(mol_records),128) and pockets.shape==(len(pocket_records),128)
        matrix=pockets@mols.T;assert np.isfinite(matrix).all();scores.append(matrix)
        np.save(cache/f'fold_{fold}_pockets.npy',pockets)
        np.save(output/f'fold_{fold}_similarity.npy',matrix)
        result={'fold':fold,'model_sha256':check['sha256'],'load_seconds':load_seconds,
                'mol_seconds':mol_seconds,'pocket_seconds':pocket_seconds,
                'molecules_per_second':len(mol_records)/mol_seconds,
                'missing_auxiliary_keys':list(incompatible.missing_keys),
                'unexpected_auxiliary_keys':list(incompatible.unexpected_keys)}
        record['folds'].append(result);write(output/'progress.json',record)
        print(json.dumps(result),flush=True)
    np.save(output/'six_fold_similarity.npy',np.stack(scores))
    mean=np.mean(scores,axis=0)
    np.save(output/'six_fold_mean_similarity.npy',mean)
    (output/'input_identity.json').write_text(json.dumps({
        'molecules':[{k:m[k] for k in ['state_id','chembl_id','smi','name','source_sha256']} for m in data['molecules']],
        'pockets':data['pockets'],'parent_statuses':data['parent_statuses']},indent=2))
    write(output/'row_identity_verification.json',{'status':'verified_direct_loader_row_order',
          'molecules':len(mol_records),'pockets':len(pocket_records),'all_six_models':True,
          'row_order':'canonical_source','method':'align every encoder output by actual batch smi_name and pocket_name; cached molecule arrays require canonical-order metadata and matching embedding SHA256'})
    actual=time.monotonic()-started
    full_states=data['ready_states_at_export']
    record.update(status='complete',total_measured_seconds=actual,ready_states_at_export=full_states,
                  ready_parents_at_export=data['ready_parents_at_export'],
                  all_ready_microstates_scored=len(mol_records)==full_states,
                  candidate_gate=gate_check(),
                  nomination='Per-target parent rankings follow in finalize_drugclip_ten.py; no global cross-target ranking',
                  timing_scope='Measured six-model loading, encoding, similarity scoring and table output; excludes environment startup, initial LMDB preparation and transfer',
                  all_six_checkpoints_loaded=True,all_embeddings_finite=True,
                  no_pocket_cropping_required=True,peak_device_memory_bytes=torch.cuda.max_memory_allocated())
    write(output/'progress.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
