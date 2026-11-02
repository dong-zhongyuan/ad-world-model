"""Complete regional extraction from checksum-verified EBI GRCh38 harmonised AGES files."""
from pathlib import Path
import argparse,csv,json,concurrent.futures
import pandas as pd
from prepare_exposures import write_csv,sha
from download_sources import checksum
COHORT_CONTEXT='AGES exposure is separate from UKB/FinnGen. The analyzed Bellenguez file is stage I, whose published consortium list excludes CHARGE and FinnGen (stage II); participant-level overlap was not independently audited.'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--assay');a=ap.parse_args();r=a.root
    sources=json.loads((r/'provenance/mr/ages_download_plan.json').read_text())
    if a.assay:sources=[x for x in sources if x['accession']==a.assay];assert len(sources)==1
    for x in sources:x['path']=str(r/'data/mr/raw/ages'/x['filename'])
    out=r/'data/mr/processed/ages';(out/'regions').mkdir(parents=True,exist_ok=True)
    def extract(x):
        cached=out/'regions'/(x['accession']+'.instrument.json');mp=out/'regions'/(x['accession']+'.json')
        if cached.exists() and mp.exists():
            meta=json.loads(mp.read_text());assert meta['source_md5']==x['expected_md5']
            assert meta['data_sha256']==sha(out/'regions'/(x['accession']+'.tsv.gz'))
            if meta.get('sample_overlap')!=COHORT_CONTEXT:
                meta['sample_overlap']=COHORT_CONTEXT;meta['cohort_source']='https://pmc.ncbi.nlm.nih.gov/articles/PMC9005347/'
                mp.write_text(json.dumps(meta,indent=2))
            return json.loads(cached.read_text()),meta
        p=Path(x['path']);assert checksum(p)==x['expected_md5']
        keep=[];leads=[];count=0
        use=['hm_rsid','hm_chrom','hm_pos','hm_other_allele','hm_effect_allele','hm_beta','hm_effect_allele_frequency','hm_code',
             'variant_id','chromosome','base_pair_location','effect_allele','other_allele','beta','standard_error','p_value']
        for d in pd.read_csv(p,sep='\t',compression='gzip',usecols=use,chunksize=200000,dtype={'hm_chrom':str,'chromosome':str},low_memory=False):
            count+=len(d);leads.extend(d[d.variant_id.eq(x['rsid'])].to_dict('records'))
            keep.append(d[d.hm_chrom.eq(str(x['chrom']))])
        # EBI legacy .h files have already lifted both ordinary and hm coordinate
        # columns to GRCh38. The publication's GRCh37 position is retained separately.
        exact=[z for z in leads if str(z['chromosome'])==str(x['chrom'])]
        assert len(exact)==1,(x['accession'],'original sentinel not unique',len(exact))
        lead=exact[0]
        assert lead['effect_allele']==x['effect_allele']
        assert abs(float(lead['beta'])-float(x['beta_exposure']))<1e-5
        assert abs(float(lead['standard_error'])-float(x['se_exposure']))<1e-6
        assert str(lead['hm_chrom'])==str(x['chrom']) and pd.notna(lead['hm_pos'])
        build_map=r/'provenance/mr/ages_variant_builds.json'
        if build_map.exists():
            v=json.loads(build_map.read_text())[x['rsid']]
            assert int(v['pos37'])==int(x['pos37']) and int(v['pos38'])==int(lead['hm_pos'])
        pos=int(lead['hm_pos']);start=max(1,pos-500000);end=pos+500000
        allchr=pd.concat(keep,ignore_index=True);part=allchr[allchr.hm_pos.between(start,end)].copy()
        renamed=part.rename(columns={'hm_rsid':'rsid','hm_chrom':'chrom','hm_pos':'pos','hm_effect_allele':'effect_allele_x','hm_other_allele':'other_allele_x',
            'hm_beta':'beta_exposure','standard_error':'se_exposure','hm_effect_allele_frequency':'eaf_exposure','p_value':'p_exposure'})
        selected=renamed[['rsid','chrom','pos','effect_allele_x','other_allele_x','beta_exposure','se_exposure','eaf_exposure','p_exposure','hm_code']].rename(columns={'effect_allele_x':'effect_allele','other_allele_x':'other_allele'})
        selected['build']='GRCh38';selected['pos']=selected['pos'].astype(int)
        dest=out/'regions'/(x['accession']+'.tsv.gz');selected.to_csv(dest,sep='\t',index=False,compression='gzip')
        instrument=dict(gene=x['gene'],assay=x['accession'],SOMAmer=x['SOMAmer'],chrom=str(x['chrom']),pos=pos,pos37=x['pos37'],build='GRCh38',
            rsid=lead['hm_rsid'],published_rsid=x['rsid'],effect_allele=lead['hm_effect_allele'],other_allele=lead['hm_other_allele'],
            beta_exposure=lead['hm_beta'],se_exposure=lead['standard_error'],eaf_exposure=lead['hm_effect_allele_frequency'],
            F_exposure=(float(lead['hm_beta'])/float(lead['standard_error']))**2,p_exposure=lead['p_value'],n_exposure=x['sample_size'],
            source='Gudjonsson2022_AGES_serum',specimen='serum',primary=x['primary_within_gene'],PAV_linked='not_yet_annotated')
        assert instrument['F_exposure']>=10
        meta=dict(gene=x['gene'],assay=x['accession'],chrom=str(x['chrom']),start=start,end=end,build='GRCh38',n=x['sample_size'],
            source_url=x['url'],source_md5=x['expected_md5'],data_sha256=sha(dest),source_rows=count,region_rows=len(selected),
            region_is_complete=True,filtered_by_p_value=False,scale_method='Estimate sdY from SE, EAF and sample N using coloc::sdY.est',
            sample_overlap=COHORT_CONTEXT,cohort_source='https://pmc.ncbi.nlm.nih.gov/articles/PMC9005347/')
        (out/'regions'/(x['accession']+'.json')).write_text(json.dumps(meta,indent=2))
        cached.write_text(json.dumps(instrument,indent=2))
        print('EXTRACTED',x['gene'],x['accession'],count,len(selected),flush=True)
        return instrument,meta
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:done=list(pool.map(extract,sources))
    if a.assay:return
    rows=[x[0] for x in done];write_csv(out/'all_assay_instruments.csv',rows)
    primary=[x for x in rows if str(x['primary']).lower()=='true'];assert len(primary)==9
    write_csv(out/'primary_instruments.csv',primary)
    (out/'requests.json').write_text(json.dumps([x[1] for x in done],indent=2))
    (r/'provenance/mr/ages_extraction.json').write_text(json.dumps(dict(status='complete',n_assays=len(rows),n_primary=len(primary),
        primary_sha256=sha(out/'primary_instruments.csv'),all_assay_sha256=sha(out/'all_assay_instruments.csv'),requests_sha256=sha(out/'requests.json')),indent=2))
if __name__=='__main__':main()
