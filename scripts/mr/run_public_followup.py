"""Predeclared public-resource follow-up: AGES, FinnGen and regional colocalization."""
from pathlib import Path
import argparse,csv,json,math
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from mr_core import harmonize,wald,bh,coloc_abf,estimate_sdy
from prepare_exposures import write_csv,sha
def key(x):
    return str(x['chrom']),int(x['pos']),tuple(sorted([str(x['effect_allele']).upper(),str(x['other_allele']).upper()]))
def valid_number(x):
    try:return math.isfinite(float(x))
    except (ValueError,TypeError):return False
def read_region(path):
    return pd.read_csv(path,sep='\t',dtype={'chrom':str}).to_dict('records')
def unique_map(rows,fields):
    result={};ambiguous=set()
    for x in rows:
        try:k=key(x)
        except (ValueError,TypeError):continue
        if k in result and any(str(result[k].get(f))!=str(x.get(f)) for f in fields):ambiguous.add(k)
        else:result[k]=x
    return {k:v for k,v in result.items() if k not in ambiguous},len(ambiguous)
def mr_row(e,ys,source,outcome):
    row=dict(e,exposure_source=source,outcome=outcome,status='not_estimated',harmonization='exact_variant_not_found',
             beta_outcome='',se_outcome='',sign='',beta_MR='',se_MR='',se_second_order='',p_MR='',p_second_order='',
             CI_low='',CI_high='',OR='',OR_low='',OR_high='',q_source='',q_joint_sources='',q_secondary='',outcome_rsid='')
    ym,amb=unique_map(ys,['beta_outcome','se_outcome']);y=ym.get(key(e))
    if y:
        sg,reason=harmonize(e,y,allow_coordinate_identity=True);row['harmonization']=reason
        if sg is not None:
            try:values=wald(e['beta_exposure'],e['se_exposure'],sg*float(y['beta_outcome']),y['se_outcome'])
            except (ValueError,TypeError):row['harmonization']='invalid_effect_or_SE'
            else:row.update(values,status='estimated',beta_outcome=y['beta_outcome'],se_outcome=y['se_outcome'],sign=sg,outcome_rsid=y['rsid'])
    return row
def coloc(e_rows,y_rows,n):
    xm,xd=unique_map(e_rows,['beta_exposure','se_exposure']);ym,yd=unique_map(y_rows,['beta_outcome','se_outcome'])
    aligned=[];excluded={}
    for k in sorted(set(xm)&set(ym)):
        e,y=xm[k],ym[k];sg,reason=harmonize(e,y,allow_coordinate_identity=True)
        if sg is None:excluded[reason]=excluded.get(reason,0)+1;continue
        vals=[e['beta_exposure'],e['se_exposure'],y['beta_outcome'],y['se_outcome'],e['eaf_exposure']]
        if not all(valid_number(x) for x in vals) or float(vals[1])<=0 or float(vals[3])<=0 or not 0<float(vals[4])<1:
            excluded['invalid_regional_statistics']=excluded.get('invalid_regional_statistics',0)+1;continue
        bx,sx,by,sy,f=map(float,vals);aligned.append((e['rsid'],k[1],bx,sx,sg*by,sy,f))
    result=dict(n_exposure=len(e_rows),n_outcome=len(y_rows),n_shared=len(set(xm)&set(ym)),n_aligned=len(aligned),
                ambiguous_exposure_keys=xd,ambiguous_outcome_keys=yd,exclusions=excluded)
    if len(aligned)<100:return dict(result,status='insufficient_shared_variants')
    ids,pos,bx,sx,by,sy,f=zip(*aligned);sdy=estimate_sdy(sx,f,n)
    sensitivities=[]
    for p12 in [1e-6,1e-5,1e-4]:
        sensitivities.append(dict(p12=p12,primary=p12==1e-5,**coloc_abf(bx,sx,by,sy,prior_sd_x=.15*sdy,p12=p12)))
    primary=next(x for x in sensitivities if x['primary'])
    def labf(b,s,sd):
        b,s=np.asarray(b),np.asarray(s);v=s*s;w=sd*sd;return .5*(np.log(v/(v+w))+b*b*w/(v*(v+w)))
    l=labf(bx,sx,.15*sdy)+labf(by,sy,.2);prob=np.exp(l-logsumexp(l));idx=int(np.argmax(prob))
    scale_sensitivity=[dict(sdY_multiplier=m,**coloc_abf(bx,sx,by,sy,prior_sd_x=.15*sdy*m)) for m in [.5,2.0]]
    return dict(result,status='complete',sdY_estimated=sdy,sdY_method='coloc::sdY.est(SE^2,EAF,N)',
                PP_H3=primary['PP_H3'],PP_H4=primary['PP_H4'],top_shared_variant=ids[idx],top_shared_position=pos[idx],
                top_variant_conditional_PP_H4=float(prob[idx]),sensitivity=sensitivities,scale_sensitivity=scale_sensitivity)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);r=ap.parse_args().root
    out=r/'results/mr/public_followup';(out/'coloc').mkdir(parents=True,exist_ok=True)
    sun=list(csv.DictReader((r/'data/mr/processed/primary_instruments.csv').open()))
    ages=list(csv.DictReader((r/'data/mr/processed/ages/all_assay_instruments.csv').open()))
    rows=[];colocs=[]
    for source,exposures in [('UKB',sun),('AGES',ages)]:
        for e in exposures:
            for outcome in ['Bellenguez_stageI','FinnGen_R13']:
                name=source+'_'+(e['assay'] if source=='AGES' else e['gene'])
                path=(r/'data/mr/processed/finngen'/(name+'.tsv.gz')) if outcome=='FinnGen_R13' else ((r/'data/mr/processed/ad_ages_regions'/(e['assay']+'.tsv.gz')) if source=='AGES' else (r/'data/mr/processed/ad_regions'/(e['gene']+'.tsv.gz')))
                assert path.exists(),path;ys=read_region(path);rows.append(mr_row(e,ys,source,outcome))
                if source=='AGES':
                    xp=r/'data/mr/processed/ages/regions'/(e['assay']+'.tsv.gz');res=coloc(read_region(xp),ys,int(e['n_exposure']))
                    res.update(gene=e['gene'],assay=e['assay'],exposure_source=source,outcome=outcome,primary=e['primary'],
                               exposure_sha256=sha(xp),outcome_sha256=sha(path),method='single-causal-variant ABF; unfiltered full regional statistics',
                               interpretation='Shared-signal evidence, not tissue mediation or efficacy')
                    (out/'coloc'/(e['assay']+'_'+outcome+'.json')).write_text(json.dumps(res,indent=2));colocs.append(res)
    primary=lambda x:str(x['primary']).lower()=='true'
    for outcome in ['Bellenguez_stageI','FinnGen_R13']:
        for source in ['UKB','AGES']:
            subset=[x for x in rows if x['outcome']==outcome and x['exposure_source']==source and primary(x) and x['status']=='estimated']
            for x,q in zip(subset,bh([x['p_MR'] for x in subset])):x['q_source']=q
        subset=[x for x in rows if x['outcome']==outcome and primary(x) and x['status']=='estimated']
        for x,q in zip(subset,bh([x['p_MR'] for x in subset])):x['q_joint_sources']=q
        secondary=[x for x in rows if x['outcome']==outcome and not primary(x) and x['status']=='estimated']
        for x,q in zip(secondary,bh([x['p_MR'] for x in secondary])):x['q_secondary']=q
    fields=list(dict.fromkeys(k for x in rows for k in x));write_csv(out/'mr_all.csv',rows,fields)
    compact=[{k:x.get(k,'') for k in ['gene','assay','exposure_source','outcome','primary','status','n_aligned','sdY_estimated','PP_H3','PP_H4','top_shared_variant','top_variant_conditional_PP_H4']} for x in colocs]
    write_csv(out/'coloc_all.csv',compact)
    for old in csv.DictReader((r/'results/mr/primary_mr.csv').open()):
        new=next(x for x in rows if x['exposure_source']=='UKB' and x['outcome']=='Bellenguez_stageI' and x['gene']==old['gene'])
        assert new['status']==old['status'],(old['gene'],old['status'],new['status'])
        if old['status']=='estimated':
            for before,after in [('beta_MR','beta_MR'),('p_MR','p_MR'),('q_BH','q_source')]:assert math.isclose(float(old[before]),float(new[after]),rel_tol=1e-10,abs_tol=1e-14)
    significant=[x for x in rows if primary(x) and x['status']=='estimated' and x['q_source']<.05]
    write_csv(out/'significant_estimates.csv',significant,fields)
    summary=dict(status='public_followup_complete',n_mr_rows=len(rows),n_estimated=sum(x['status']=='estimated' for x in rows),
                 n_significant_estimates=len(significant),significant_genes=sorted({x['gene'] for x in significant}),
                 coloc_completed=sum(x['status']=='complete' for x in colocs),UKB_coloc='Unavailable: Synapse download authorization required; user elected public data only',
                 primary_UKB_results_preserved=True,primary_regions='500 kb each side of exposure-selected lead; same for all assays',
                 selection='AGES leads frozen by exposure F before AD follow-up; no selection based on outcome results',
                 methods='Single-instrument MR; source-wise BH and joint-source BH sensitivity; coloc ABF with sdY estimated and prior sensitivity',
                 hashes={p.name:sha(p) for p in out.glob('*.csv')})
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
