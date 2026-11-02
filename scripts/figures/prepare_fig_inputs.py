from pathlib import Path
import json,numpy as np,pandas as pd
from paths import FIGURE_WORK as W
D=W/'data'
T=['BIN1','CR1','PILRB','GRN','BLNK','CD55','ACP5','ST13','GNAS','BST2'];routes=['B_Micro','CD4T_Micro','CD8T_Micro','Mono_Micro','NK_Micro','Mono_Oligo']
e=pd.read_csv(D/'validated_edges_reverse.csv');og=e.drop_duplicates(['combo','source_node','target_node']).copy();og['weight']=og.delta_mean.abs()
route=og.groupby('combo').agg(count=('weight','sum'),og_edges=('weight','size')).reset_index();route['source']=route.combo.str.split('_').str[0];route['target']=route.combo.str.split('_').str[1];route['group']='Blood to brain';route.to_csv(D/'route_weights.csv',index=False)
sel=e[e.source.isin(T)].drop_duplicates(['source','combo','source_node','target_node']).copy();sel.to_csv(D/'target_edges.csv',index=False)
# One shared display set, ranked by unsigned response mass over all ten targets and all routes.
order=sel.groupby('target_node').delta_mean.apply(lambda x:x.abs().sum()).sort_values(ascending=False).head(28).index.tolist()
labels=sel.drop_duplicates('target_node').set_index('target_node').target_symbols.to_dict()
rows=[]
for r in routes:
 for t in T:
  for node in order:
   z=sel[(sel.combo==r)&(sel.source==t)&(sel.target_node==node)]
   rows.append(dict(combo=r,gene=t,node=node,brain_gene=labels[node],response=z.delta_mean.mean() if len(z) else np.nan))
pd.DataFrame(rows).to_csv(D/'route_fingerprints.csv',index=False)
sets={t:set(sel[sel.source.eq(t)].target_node) for t in T}
pd.DataFrame([[len(sets[a]&sets[b])/len(sets[a]|sets[b]) if sets[a]|sets[b] else np.nan for b in T]for a in T],index=T,columns=T).to_csv(D/'response_jaccard.csv')
# Local network panels: strongest ten unique brain nodes per target; all source data remain in target_edges.csv.
local=[]
for t in ['BIN1','GRN','BLNK']:
 z=sel[sel.source.eq(t)].groupby('target_node').agg(delta=('delta_mean','mean'),magnitude=('delta_mean',lambda x:x.abs().max()),label=('target_symbols','first')).sort_values('magnitude',ascending=False).head(10)
 for i,(node,r) in enumerate(z.iterrows()):
  a=2*np.pi*i/len(z);local.append(dict(target=t,path=t,gene=r.label,angle=a,x=np.cos(a),y=np.sin(a),cx=0,cy=0,delta=r.delta,magnitude=r.magnitude))
pd.DataFrame(local).to_csv(D/'target_local_networks.csv',index=False)
g=pd.read_csv(D/'genes.csv');g['source']=np.where(g.n_mouse.eq(1),'1 mouse gene',np.where(g.n_mouse.gt(1),'>1 mouse genes','No mouse gene'));g['target']=np.where(g.n_human.eq(1),'1 human gene',np.where(g.n_human.gt(1),'>1 human genes','No human gene'));q=g.groupby(['source','target']).size().reset_index(name='count');q['group']='Orthology features';q.to_csv(D/'orthology_counts.csv',index=False)
# Exact assay-level ring matrix. Gene-level q is an annotation, never substituted for assay q.
u=pd.read_csv(D/'mr_gene_union_with_hub_gate.csv');u=u[u.union_significant].sort_values(['eligible_union_target','q_union'],ascending=[False,True]);m=pd.read_csv(D/'mr_mr_all_with_sensitivity.csv');m=m[m.gene.isin(u.gene)&m.status.eq('estimated')];c=pd.read_csv(D/'mr_coloc_all.csv');rows=[]
for (gene,assay,src),a in m.groupby(['gene','assay','exposure_source'],sort=False):
 z=u.set_index('gene').loc[gene];r=dict(protein=gene,assay=assay,source=src,network=100-z.hub_top_percent,hub50=z.hub_top50,mapping=z.mapping_status,identity='NESP55' if gene=='GNAS' else str(a.uniprot.iloc[0]),q_union=z.q_union,q_both=z.q_both_outcomes)
 for key,out in [('bell','Bellenguez_stageI'),('finn','FinnGen_R13')]:
  b=a[a.outcome.eq(out)];r[key+'_q']=float(b.q_joint_sources.iloc[0]) if len(b) else np.nan;r[key+'_beta']=float(b.beta_MR.iloc[0])if len(b)else np.nan
  cc=c[c.gene.eq(gene)&c.assay.eq(assay)&c.exposure_source.eq(src)&c.outcome.eq(out)&c.status.eq('complete')];assert len(cc)<=1;r[key+'_h4']=float(cc.PP_H4.iloc[0])if len(cc)else np.nan
 r['direction']=('Both positive' if r['bell_beta']>0 else 'Both negative')if np.isfinite(r['bell_beta'])and np.isfinite(r['finn_beta'])and np.sign(r['bell_beta'])==np.sign(r['finn_beta'])else 'Opposite / unavailable'
 rows.append(r)
r=pd.DataFrame(rows);r['protein']=pd.Categorical(r.protein,u.gene);r=r.sort_values(['protein','source','assay']);r.to_csv(D/'evidence_wheel.csv',index=False)
# All top-20 parents across ten targets, one row per chemical identity.
a=pd.concat([pd.read_csv(D/f'drugclip_{t}_all_parents.tsv',sep='\t') for t in T]);ids=set(a[a['rank']<=20].chembl_id);b=a[a.chembl_id.isin(ids)].copy();b['rank_percentile']=100*(1-(b['rank']-1)/(2297-1));b.to_csv(D/'chemical_top20_union.csv',index=False)
# Representative molecules are frozen from approved figure design, independent of docking energies.
choice={'BIN1':'FOMEPIZOLE','GRN':'FOSCARNET','BLNK':'PAMIDRONIC ACID','CR1':'FOMEPIZOLE','PILRB':'AMIFAMPRIDINE','CD55':'FLUOROURACIL','ACP5':'DALFAMPRIDINE','BST2':'HALOTHANE','GNAS':'TOLCAPONE','ST13':'CHLORAL HYDRATE'}
rep=[]
for t,name in choice.items():
 z=a[a.target.eq(t)&a.name.eq(name)];assert len(z)==1,(t,name);rep.append(z.iloc[0].to_dict())
pd.DataFrame(rep).to_csv(D/'representative_molecules.csv',index=False)
# Store definition instead of adapting selection to sign or significance.
(W/'derived_input_definitions.json').write_text(json.dumps(dict(target_order=T,route_order=routes,network_fingerprint_rows=order,display_brain_nodes=28,display_rule='largest total absolute OG response across ten targets and six routes; NA means no retained edge',local_network_display='ten strongest brain OGs per target by max absolute response',chemical_display_rows=len(ids),wheel_columns=len(r),wheel_genes=r.protein.nunique()),indent=2))
print('Prepared network, MR and chemical matrices',len(r),len(ids))
