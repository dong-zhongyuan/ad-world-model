"""Extend complete AD regional extraction to the independently locked AGES assays."""
from pathlib import Path
import argparse,csv,gzip,json
import pandas as pd
from prepare_exposures import sha
from download_sources import checksum
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);r=ap.parse_args().root
    requests=json.loads((r/'data/mr/processed/ages/requests.json').read_text())
    source=json.loads((r/'provenance/mr/ad_download_verified.json').read_text());p=Path(source['path']);assert checksum(p)==source['md5']
    out=r/'data/mr/processed/ad_ages_regions';out.mkdir(parents=True,exist_ok=True)
    cols=['rsid','chrom','pos','build','effect_allele','other_allele','beta_outcome','se_outcome','eaf_outcome','p_outcome','n_cases','n_controls']
    handles={x['assay']:gzip.open(out/(x['assay']+'.tsv.gz'),'wt',newline='') for x in requests}
    writers={k:csv.DictWriter(v,fieldnames=cols,delimiter='\t') for k,v in handles.items()}
    for w in writers.values():w.writeheader()
    counts={k:0 for k in handles};n=0
    try:
        for d in pd.read_csv(p,sep='\t',compression='gzip',chunksize=200000,dtype={'chromosome':str},low_memory=False):
            n+=len(d)
            for x in requests:
                rows=d[d.chromosome.eq(str(x['chrom'])) & d.base_pair_location.between(x['start'],x['end'])]
                for v in rows.to_dict('records'):
                    writers[x['assay']].writerow(dict(rsid=v['variant_id'],chrom=v['chromosome'],pos=v['base_pair_location'],build='GRCh38',
                        effect_allele=v['effect_allele'],other_allele=v['other_allele'],beta_outcome=v['beta'],se_outcome=v['standard_error'],
                        eaf_outcome=v['effect_allele_frequency'],p_outcome=v['p_value'],n_cases=v['n_cases'],n_controls=v['n_controls']))
                counts[x['assay']]+=len(rows)
            if n%2000000==0:print('AD_EXTENSION_ROWS',n,flush=True)
    finally:
        for h in handles.values():h.close()
    (r/'provenance/mr/ad_ages_extraction.json').write_text(json.dumps(dict(status='complete',source_md5=source['md5'],source_rows=n,
        counts=counts,hashes={p.name:sha(p) for p in out.glob('*.tsv.gz')},filtered_by_p_value=False),indent=2))
if __name__=='__main__':main()
