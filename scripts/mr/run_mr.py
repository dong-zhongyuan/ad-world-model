"""Run all locked primary protein-to-AD Wald estimates, including disposition of every candidate."""
from pathlib import Path
import argparse,csv,json,math
from mr_core import harmonize,wald,bh,canonical_rsid
from prepare_exposures import write_csv,sha
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);r=ap.parse_args().root
    exp_path=r/'data/mr/processed/primary_instruments.csv';ys=r/'data/mr/processed/ad_sentinel_statistics.csv'
    lock=json.loads((r/'provenance/mr/exposure_lock.json').read_text())
    ext=json.loads((r/'provenance/mr/outcome_extraction.json').read_text())
    assert ext['status']=='complete'
    assert sha(exp_path)==lock['instruments_sha256']==ext['instruments_sha256']
    assert sha(ys)==ext['sentinels_sha256']
    assert sha(r/'configs/mr/protocol.json')==ext['protocol_sha256']
    alias_path=r/'provenance/mr/dbsnp_aliases.json';aliases={}
    if ext.get('aliases_sha256'):
        assert sha(alias_path)==ext['aliases_sha256']
        alias_info=json.loads(alias_path.read_text());assert alias_info['instruments_sha256']==sha(exp_path)
        aliases=alias_info['aliases']
    e=list(csv.DictReader(exp_path.open()));y=list(csv.DictReader(ys.open()));rows=[]
    for x in e:
        matches=[z for z in y if canonical_rsid(z['rsid'],aliases)==canonical_rsid(x['rsid'],aliases)];valid=[];reasons=[]
        for z in matches:
            sign,reason=harmonize(x,z,aliases=aliases)
            reasons.append(reason)
            if sign is not None:
                try:res=wald(x['beta_exposure'],x['se_exposure'],sign*float(z['beta_outcome']),z['se_outcome'])
                except ValueError:reasons.append('invalid_outcome_statistics');continue
                valid.append((z,sign,reason,res))
        unique={(z['chrom'],z['pos'],z['effect_allele'],z['other_allele'],z['beta_outcome'],z['se_outcome']):(z,s,h,v) for z,s,h,v in valid}
        base=dict(x,outcome='Bellenguez2022_GCST90027158_AD_including_proxy',status='',harmonization='',
            sign='',outcome_rsid='',identifier_match='',beta_outcome='',se_outcome='',eaf_outcome='',n_cases='',n_controls='',
            beta_MR='',se_MR='',se_second_order='',p_MR='',p_second_order='',CI_low='',CI_high='',
            OR='',OR_low='',OR_high='',q_BH='',q_second_order='',q_no_PAV='',
            sample_overlap='Possible UKB overlap; not independent cohort replication',
            colocalization_status='pending_full_regional_pQTL',evidence='not_evaluated')
        if len(unique)==1:
            z,sign,reason,res=next(iter(unique.values()))
            base.update(status='estimated',harmonization=reason,sign=sign,outcome_rsid=z['rsid'],
                identifier_match='exact_rsid' if x['rsid']==z['rsid'] else 'official_dbsnp_merge',
                beta_outcome=z['beta_outcome'],se_outcome=z['se_outcome'],eaf_outcome=z['eaf_outcome'],
                n_cases=z['n_cases'],n_controls=z['n_controls'],evidence='MR_association_not_colocalization_confirmed',**res)
        elif len(unique)>1:base.update(status='ambiguous_duplicate_outcome',harmonization='multiple_valid_records')
        else:base.update(status='not_estimated',harmonization=';'.join(sorted(set(reasons))) if reasons else 'exact_variant_not_found')
        rows.append(base)
    estimated=[x for x in rows if x['status']=='estimated']
    for x,q,q2 in zip(estimated,bh([x['p_MR'] for x in estimated]),bh([x['p_second_order'] for x in estimated])):
        x['q_BH']=q;x['q_second_order']=q2
    no_pav=[x for x in estimated if str(x['PAV_linked']).lower()=='false']
    for x,q in zip(no_pav,bh([x['p_MR'] for x in no_pav])):x['q_no_PAV']=q
    out=r/'results/mr';out.mkdir(exist_ok=True)
    write_csv(out/'primary_mr.csv',rows)
    write_csv(out/'sensitivity_no_PAV.csv',no_pav,list(rows[0]))
    coverage=list(csv.DictReader((r/'data/mr/processed/candidate_coverage.csv').open()))
    by_gene={x['gene']:x for x in rows}
    for x in coverage:
        z=by_gene.get(x['gene'],{})
        x.update(mr_status=z.get('status','not_assessable_from_this_resource'),harmonization=z.get('harmonization',''),
            beta_MR=z.get('beta_MR',''),OR=z.get('OR',''),p_MR=z.get('p_MR',''),q_BH=z.get('q_BH',''),
            PAV_linked=z.get('PAV_linked',''),colocalization_status=z.get('colocalization_status','not_applicable_no_instrument'))
    write_csv(out/'all_candidates.csv',coverage)
    summary=dict(pipeline_status='MR_stage_complete',candidate_origin=lock['candidate_version'],candidates=len(coverage),
        primary_instruments=len(e),estimated=len(estimated),unestimated=len(e)-len(estimated),
        nominal_p05=sum(x['p_MR']<.05 for x in estimated),FDR05=sum(x['q_BH']<.05 for x in estimated),
        FDR05_no_PAV=sum(x['q_no_PAV']<.05 for x in no_pav),colocalization_completed=0,
        exposure_resource='Sun2023 complete supplement',outcome='GCST90027158; European; AD and proxy cases; catalog N487511',
        n_policy='Per-variant cases/controls from original GWAS; not full publication N788989',
        overlap='UKB exposure and AD meta-analysis may overlap; independent-source extension remains needed',
        interpretation='Association estimates for genetically proxied protein on AD log odds; no brain-gene mediation or efficacy claim',
        selected_without_outcome=True,files_sha256={p.name:sha(p) for p in out.glob('*.csv')})
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
