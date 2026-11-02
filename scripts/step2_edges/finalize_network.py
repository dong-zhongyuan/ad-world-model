#!/usr/bin/env python3
"""Build the member-level blood-brain network and rank targets with MultiXrank."""
import argparse,json,sys,shutil
from pathlib import Path
import numpy as np,pandas as pd
from ko_edges import sha,atomic_json,empirical_p,select_qualified

def expand_members(e):
    """Preserve OG identity and distribute each OG edge's graph mass across members."""
    e=e.copy()
    e['source_members']=e.source_symbols.str.split(';').map(lambda x:sorted(set(x)))
    e['target_members']=e.target_symbols.str.split(';').map(lambda x:sorted(set(x)))
    e['n_source_members']=e.source_members.map(len);e['n_target_members']=e.target_members.map(len)
    e['member_pairs']=e.apply(lambda r:r.n_source_members*r.n_target_members-len(set(r.source_members)&set(r.target_members)),axis=1)
    e=e[e.member_pairs>0].copy()
    e['graph_weight']=e.delta_mean.abs()/e.member_pairs
    e=e.explode('source_members').explode('target_members')
    e['source']=e.pop('source_members');e['target']=e.pop('target_members')
    e['mapping_status']=np.where(e.n_source_members==1,'unique_human_member','ambiguous_orthogroup_member')
    return e[e.source!=e.target].copy()

def select_mapped_edges(path,d,sc,tc,kinds,groups):
    """Apply member-level nonself eligibility before the top15 cap, using cached full responses."""
    resume_file=path.parent.parent/'mapped_resume_cache.json'
    if resume_file.exists():
        manifest=json.loads(resume_file.read_text())
        assert manifest['contract_sha256']==sha(path.parent/'contract.json')
        cached=manifest['routes'].get(path.name)
        if cached:
            assert sha(path/'mapped_edges.csv')==cached['mapped_edges_sha256']
            assert sha(path/'mapped_selection_diagnostics.csv')==cached['diagnostics_sha256']
            assert sha(path/'status.json')==cached['route_status_sha256']
            e=pd.read_csv(path/'mapped_edges.csv',float_precision='round_trip')
            assert len(e)==cached['n_edges']
            print('reused_completed_mapping',path.name,len(e),flush=True)
            return e
    print('mapping_route_start',path.name,flush=True)
    chunks=sorted((path/'chunks').glob('*.npz'));null=[]
    for f in chunks:
        with np.load(f) as a:
            null.extend(a['null_abs']);targets=a['targets']
    null=np.sort(np.asarray(null),axis=0)
    src=d.data['human_blood'];tgt=d.data['human_brain'];ids=np.load(path/'source_cell_indices.npy')
    det=np.asarray((src['x'][ids]>1e-6).mean(0)).ravel()
    td=np.asarray((tgt['x'][tgt['obs'].celltype.to_numpy()==tc]>1e-6).mean(0)).ravel()
    members={g:set(s.split(';')) for g,s in groups.human_symbols.dropna().items()}
    target_members=[members.get(d.genes[t],set()) for t in targets]
    target_single=np.array([next(iter(x)) if len(x)==1 else '' for x in target_members])
    base=np.array([kinds[d.genes[t]]=='shared_og' for t in targets]);rows=[];diagnostics=[]
    for f in chunks:
        with np.load(f) as a:
            assert np.array_equal(targets,a['targets'])
            mean=a['mean'];z=a['z'];genes=a['genes']
            p=empirical_p(null,np.abs(mean))
            for j,g in enumerate(genes):
                sm=members.get(d.genes[g],set())
                valid_pairs=(target_single!=next(iter(sm))) if len(sm)==1 else np.full(len(targets),bool(sm))
                eligible=base&(targets!=g)&valid_pairs&(kinds[d.genes[g]]=='shared_og')
                positions,n=select_qualified(mean[j],p[j],eligible)
                diagnostics.append(dict(source=d.genes[g],n_qualified_before_cap=n,n_retained=len(positions)))
                for pos in positions:
                    t=targets[pos];rows.append(dict(source=d.genes[g],target=d.genes[t],delta_mean=mean[j,pos],z=z[j,pos],source_det=det[g],target_det=td[t],source_kind=kinds[d.genes[g]],target_kind=kinds[d.genes[t]],randko_q=p[j,pos],randko_p=p[j,pos]))
    pd.DataFrame(diagnostics).to_csv(path/'mapped_selection_diagnostics.csv',index=False)
    e=pd.DataFrame(rows,columns=['source','target','delta_mean','z','source_det','target_det','source_kind','target_kind','randko_q','randko_p'])
    e.to_csv(path/'mapped_edges.csv',index=False)
    print('mapping_route_complete',path.name,len(e),flush=True)
    return e

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    r=a.root;out=r/'results/network_v3_repair';edge_dir=out/'edges'
    status=json.loads((edge_dir/'status.json').read_text());assert status['status']=='complete'
    contract=json.loads((edge_dir/'contract.json').read_text());assert status['contract_sha256']==sha(edge_dir/'contract.json')
    assert sha(contract['checkpoint'])==contract['checkpoint_sha256']
    groups=pd.read_csv(r/'data/mappings/orthogroups_member_level.csv',dtype=str).set_index('og_id')
    sys.path.insert(0,str(r/'scripts/step1_world_model'))
    from data_io import Dataset
    d=Dataset();kinds=pd.read_csv(r/'data/member_level/columns.csv').set_index('node_id').kind.to_dict()
    assert sha(d.root/'data_manifest.json')==contract['data_manifest_sha256']
    assert all(sha(d.root/t/'obs.csv')==h for t,h in contract['observation_sha256'].items())
    rows=[]
    for combo in contract['combos'].split(','):
        target,source=combo.split(':');path=edge_dir/('rev_'+source+'_'+target)
        m=json.loads((path/'status.json').read_text());assert m['status']=='complete' and m['contract_sha256']==status['contract_sha256']
        assert sha(path/'edges.csv')==m['edges_sha256']
        e=select_mapped_edges(path,d,source,target,kinds,groups)
        e['source_node']=e.source;e['target_node']=e.target
        e['source_symbols']=e.source.map(groups.human_symbols);e['target_symbols']=e.target.map(groups.human_symbols)
        assert e[['source_symbols','target_symbols']].notna().all().all()
        e=e[e.source_node!=e.target_node];e['direction']='reverse';e['combo']=source+'_'+target;rows.append(e)
    og_edges=pd.concat(rows,ignore_index=True).drop_duplicates(['source_node','target_node','combo'])
    og_edges.to_csv(out/'qualified_orthogroup_edges.csv',index=False)
    edges=expand_members(og_edges)
    total=edges.groupby(['source_node','target_node','combo']).graph_weight.sum()
    expected=og_edges.set_index(['source_node','target_node','combo']).delta_mean.abs().reindex(total.index)
    assert np.allclose(total,expected,rtol=1e-12,atol=1e-14)
    if edges.empty:raise RuntimeError('No qualified v3 edges; no legacy fallback')
    edges.to_csv(out/'validated_edges_reverse.csv',index=False)
    candidates=pd.DataFrame({'gene':sorted(edges.source.unique())})
    candidates['n_edges']=candidates.gene.map(edges.source.value_counts())
    mappings=[]
    for gene,e in edges.groupby('source'):
        nodes=sorted(set(e.source_node));symbols=sorted({s for a in e.source_symbols for s in a.split(';')})
        status='multiple_orthogroups' if len(nodes)>1 else ('ambiguous_orthogroup_member' if len(symbols)>1 else 'unique_human_member')
        mappings.append(dict(gene=gene,source_node=';'.join(nodes),source_symbols=';'.join(symbols),n_model_nodes=len(nodes),n_source_members=len(symbols),mapping_status=status))
    candidates=candidates.merge(pd.DataFrame(mappings),on='gene',validate='one_to_one')
    edges[['source','source_node','source_symbols','n_source_members']].drop_duplicates().to_csv(out/'gene_orthogroup_memberships.csv',index=False)
    candidates['candidate_origin']='v3_joint_full1500_repaired_all_members'
    candidates.to_csv(out/'candidate_coverage.csv',index=False)
    cache=r/'configs/multixrank';dest=out/'hub';dest.mkdir(exist_ok=True)
    shutil.copy2(cache/'config.yml',dest/'config.yml');nodes=set()
    for tissue in ['brain','blood']:
        name=Path('multiplex')/tissue/(tissue+'_layer.tsv');(dest/name).parent.mkdir(parents=True,exist_ok=True)
        if not (dest/name).exists():(dest/name).symlink_to((cache/name).resolve())
        net=pd.read_csv(dest/name,sep='\t',header=None);nodes.update(net[0]);nodes.update(net[1])
    # Reverse-route legacy layer names: brain_* is the biological blood layer.
    # Multiple cell routes retain their records; graph uses strongest route per member pair.
    graph=edges.groupby(['source','target'],as_index=False).graph_weight.max()
    cross=pd.DataFrame({'source':'brain_'+graph.source,'target':'blood_'+graph.target,'strength':graph.graph_weight})
    (dest/'bipartite').mkdir(exist_ok=True)
    cross.to_csv(dest/'bipartite/brain_to_blood.tsv',sep='\t',header=False,index=False)
    nodes.update(cross.source);nodes.update(cross.target)
    (dest/'seeds.txt').write_text('\n'.join(sorted(nodes))+'\n')
    from multixrank import Multixrank
    model=Multixrank(config=str(dest/'config.yml'),wdir=str(dest))
    scores=model.random_walk_rank();assert np.isfinite(scores.score).all() and (scores.score>=0).all()
    assert np.isclose(scores.score.sum(),1.,rtol=1e-6)
    from multixrank.TransitionMatrix import TransitionMatrix
    from scipy.sparse import eye
    from scipy.sparse.linalg import gmres
    matrix=TransitionMatrix(multiplex_all=model.multiplexall_obj,bipartite_matrix=model.bipartiteall_obj.bipartite_matrix,lamb=model.lamb).transition_matrixcoo.tocsr()
    assert np.isfinite(matrix.data).all() and (matrix.data>=0).all()
    assert np.allclose(np.asarray(matrix.sum(axis=0)).ravel(),1.,atol=1e-12)
    prior,_=model.seed_obj.get_seed_scores(transition=matrix);prior=np.asarray(prior).ravel();prior/=prior.sum()
    independent,info=gmres(eye(len(prior),format='csr')-(1-model.r)*matrix,model.r*prior,rtol=1e-12,atol=1e-14)
    assert info==0 and np.allclose(independent,scores.score.to_numpy(),rtol=1e-6,atol=1e-10)
    atomic_json(dest/'verification.json',dict(status='passed',transition_columns_sum_to_one=True,independent_linear_system_matches=True,orthogroup_edge_mass_conserved=True,all_human_members_included=True,graph_route_aggregation='maximum_weight_per_member_pair',legacy_layer_semantics={'brain':'blood','blood':'brain'},max_abs_score_difference=float(np.max(np.abs(independent-scores.score.to_numpy())))))
    scores.to_csv(dest/'all_layer_scores.csv',index=False)
    blood=scores[scores.multiplex=='brain'].copy();blood['gene']=blood.node.str.replace('^brain_','',regex=True)
    assert blood.gene.is_unique and set(candidates.gene)<=set(blood.gene)
    blood['rank']=blood.score.rank(ascending=False,method='min').astype(int)
    blood['top_percent']=100*(blood['rank']-1)/max(len(blood)-1,1)
    cutoff=float(np.percentile(blood.score,50));blood['hub_top50']=blood.score>=cutoff
    blood.sort_values('rank').to_csv(dest/'blood_hub_ranks.csv',index=False)
    ranked=candidates.merge(blood[['gene','score','rank','top_percent','hub_top50']],on='gene',validate='one_to_one')
    ranked.to_csv(dest/'candidate_hub_ranks.csv',index=False)
    selected=ranked[ranked.hub_top50];selected.to_csv(out/'hub50_candidates.csv',index=False)
    result=dict(status='network_complete',repair_version=contract['repair_version'],checkpoint=contract['checkpoint'],checkpoint_sha256=contract['checkpoint_sha256'],n_edges=len(edges),n_orthogroup_edges=len(og_edges),n_graph_edges=len(graph),n_candidates=len(candidates),n_ambiguous_member_candidates=int((candidates.n_source_members>1).sum()),n_blood_nodes=len(blood),hub_cutoff_score=cutoff,n_hub50_candidates=len(selected),hashes={str(f.relative_to(out)):sha(f) for f in [out/'validated_edges_reverse.csv',out/'candidate_coverage.csv',dest/'candidate_hub_ranks.csv',dest/'blood_hub_ranks.csv']})
    result.update(n_ambiguous_member_candidates=int((candidates.mapping_status!='unique_human_member').sum()),mapping_source_sha256=sha(r/'data/mappings/orthogroups_member_level.csv'),finalization_code_sha256=sha(__file__),member_mapping='All human members and all model OG memberships retained; nonself member pairs share each OG edge mass equally',selection='matched abs(mean_cell_delta) empirical rank<=.05 and valid shared-OG member pairs before top15',n_genes_with_multiple_model_nodes=int((candidates.n_model_nodes>1).sum()))
    atomic_json(out/'network_summary.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
