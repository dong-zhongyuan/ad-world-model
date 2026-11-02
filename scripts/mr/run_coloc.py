"""Optional regional coloc ABF. Requires full, unfiltered pQTL regional statistics."""
from pathlib import Path
import argparse,csv,json
import pandas as pd
from mr_core import harmonize,coloc_abf
from prepare_exposures import sha
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--gene',required=True);ap.add_argument('--pqtl-region',type=Path,required=True)
    ap.add_argument('--metadata',type=Path,required=True);a=ap.parse_args();r=a.root
    protocol=json.loads((r/'configs/mr/protocol.json').read_text())
    request=next(x for x in json.loads((r/'data/mr/processed/coloc_region_requests.json').read_text()) if x['gene']==a.gene)
    meta=json.loads(a.metadata.read_text())
    assert meta['assay']==request['assay'] and meta['build']=='GRCh38'
    assert meta['region_is_complete'] is True and meta['filtered_by_p_value'] is False,'Significant-only tables cannot be used for coloc'
    assert str(meta['chrom'])==request['chrom'] and meta['start']<=request['start'] and meta['end']>=request['end']
    assert meta['data_sha256']==sha(a.pqtl_region)
    assert meta['exposure_sdY']==1,'Default quantitative prior requires standardized exposure; supply validated scaling first'
    e=pd.read_csv(a.pqtl_region,sep='\t',dtype={'chrom':str})
    y=pd.read_csv(r/'data/mr/processed/ad_regions'/(a.gene+'.tsv.gz'),sep='\t',dtype={'chrom':str})
    keys=['rsid','chrom','pos']
    assert not e.duplicated(keys).any() and not y.duplicated(keys).any(),'Resolve multiallelic/duplicate coordinates explicitly'
    joined=e.merge(y,on=keys,suffixes=('_x','_y'),validate='one_to_one')
    aligned=[];excluded={}
    for row in joined.to_dict('records'):
        ex={k:row[k+'_x'] for k in ['build','effect_allele','other_allele']}
        yy={k:row[k+'_y'] for k in ['build','effect_allele','other_allele']}
        ex.update({k:row[k] for k in keys});yy.update({k:row[k] for k in keys})
        ex['eaf_exposure']=row.get('eaf_exposure');yy['eaf_outcome']=row.get('eaf_outcome')
        sign,reason=harmonize(ex,yy)
        if sign is None:excluded[reason]=excluded.get(reason,0)+1;continue
        bx,sx,by,sy=[float(row[k]) for k in ['beta_exposure','se_exposure','beta_outcome','se_outcome']]
        if not (sx>0 and sy>0):excluded['invalid_SE']=excluded.get('invalid_SE',0)+1;continue
        aligned.append((bx,sx,sign*by,sy))
    if len(aligned)<protocol['coloc']['minimum_common_variants']:raise ValueError('Too few shared variants for regional coloc')
    bx,sx,by,sy=zip(*aligned)
    results=[]
    for p12 in protocol['coloc']['p12_sensitivity']:
        result=coloc_abf(bx,sx,by,sy,p12=p12)
        results.append(dict(p12=p12,primary=p12==protocol['coloc']['p12'],**result))
    out=r/'results/mr/coloc';out.mkdir(parents=True,exist_ok=True)
    result=dict(gene=a.gene,assay=meta['assay'],status='coloc_ABF_complete',method='single_causal_variant_ABF',
        n_pqtl=len(e),n_AD=len(y),n_shared=len(joined),n_aligned=len(aligned),exclusions=excluded,
        source_metadata=meta,pqtl_sha256=sha(a.pqtl_region),results=results,
        sample_overlap=meta.get('sample_overlap','unknown; UKB exposure and AD meta-analysis may overlap'),
        dependence_note='Standard ABF assumes independent study estimates; overlapping cohorts require cautious interpretation or independent exposure replication',
        interpretation='Regional shared-signal evidence under single-causal-variant assumption; not mediation or treatment efficacy',
        multicausal_status='Not assessed; conditional/SuSiE coloc requires valid LD and full summary statistics')
    (out/(a.gene+'.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
