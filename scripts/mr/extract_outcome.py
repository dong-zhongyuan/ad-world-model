"""Extract fixed sentinel variants and complete local regions from the original AD GWAS."""
from pathlib import Path
import argparse,csv,gzip,hashlib,json
import pandas as pd
from prepare_exposures import sha
from mr_core import canonical_rsid
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);r=ap.parse_args().root
    lock=json.loads((r/'provenance/mr/exposure_lock.json').read_text())
    exp=r/'data/mr/processed/primary_instruments.csv'
    assert sha(exp)==lock['instruments_sha256']
    protocol=json.loads((r/'configs/mr/protocol.json').read_text())
    verified=json.loads((r/'provenance/mr/ad_download_verified.json').read_text())
    assert verified['md5']==protocol['outcome']['md5'] and verified['status']=='complete'
    raw=Path(verified['path']); assert raw.stat().st_size==verified['bytes']
    digest=hashlib.md5()
    with raw.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    assert digest.hexdigest()==verified['md5'],'AD source changed after download verification'
    source=list(csv.DictReader(exp.open()))
    processed=r/'data/mr/processed';region_dir=processed/'ad_regions';region_dir.mkdir(exist_ok=True)
    alias_path=r/'provenance/mr/dbsnp_aliases.json'
    alias_info=json.loads(alias_path.read_text()) if alias_path.exists() else {}
    if alias_info:assert alias_info['instruments_sha256']==sha(exp)
    aliases=alias_info.get('aliases',{})
    targets={x['rsid'] for x in source};targets|={canonical_rsid(x,aliases) for x in targets}
    found=[];count=0;requests=[]
    by_chr={}
    for x in source:
        pos=int(x['pos']);start=max(1,pos-protocol['coloc']['window_bp']);end=pos+protocol['coloc']['window_bp']
        spec=dict(gene=x['gene'],assay=x['assay'],chrom=str(x['chrom']),start=start,end=end,build='GRCh38',sentinel=x['rsid'])
        requests.append(spec);by_chr.setdefault(str(x['chrom']),[]).append(spec)
    columns=['rsid','chrom','pos','build','effect_allele','other_allele','beta_outcome','se_outcome','eaf_outcome','p_outcome','n_cases','n_controls']
    handles={};writers={};regional_counts={}
    for x in source:
        h=gzip.open(region_dir/(x['gene']+'.tsv.gz'),'wt',newline='');handles[x['gene']]=h
        w=csv.DictWriter(h,fieldnames=columns,delimiter='\t');w.writeheader();writers[x['gene']]=w;regional_counts[x['gene']]=0
    def canonical(d):
        return dict(rsid=d['variant_id'],chrom=str(d['chromosome']),pos=int(d['base_pair_location']),build='GRCh38',
            effect_allele=d['effect_allele'],other_allele=d['other_allele'],beta_outcome=d['beta'],
            se_outcome=d['standard_error'],eaf_outcome=d['effect_allele_frequency'],p_outcome=d['p_value'],
            n_cases=d['n_cases'],n_controls=d['n_controls'])
    try:
        for chunk in pd.read_csv(raw,sep='\t',compression='gzip',chunksize=150000,dtype={'chromosome':str,'variant_id':str},low_memory=False):
            count+=len(chunk)
            for d in chunk[chunk.variant_id.isin(targets)].to_dict('records'):found.append(canonical(d))
            for chrom,specs in by_chr.items():
                same=chunk[chunk.chromosome==chrom]
                for x in specs:
                    subset=same[same.base_pair_location.between(x['start'],x['end'])]
                    for d in subset.to_dict('records'):writers[x['gene']].writerow(canonical(d))
                    regional_counts[x['gene']]+=len(subset)
            if count%1500000==0:print('AD rows read',count,flush=True)
    finally:
        for h in handles.values():h.close()
    dest=processed/'ad_sentinel_statistics.csv'
    with dest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(found)
    (processed/'coloc_region_requests.json').write_text(json.dumps(requests,indent=2))
    manifest=dict(status='complete',gwas_rows_read=count,requested_variants=len(source),requested_identifiers=len(targets),matched_records=len(found),
        matched_rsids=len({x['rsid'] for x in found}),source_md5=verified['md5'],instruments_sha256=sha(exp),
        sentinels_sha256=sha(dest),region_counts=regional_counts,region_sha256={p.name:sha(p) for p in region_dir.glob('*.gz')},
        regional_source='Unfiltered original genome-wide AD GWAS; no outcome p-value threshold',
        protocol_sha256=sha(r/'configs/mr/protocol.json'),aliases_sha256=sha(alias_path) if alias_path.exists() else None)
    (r/'provenance/mr/outcome_extraction.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)
if __name__=='__main__':main()
