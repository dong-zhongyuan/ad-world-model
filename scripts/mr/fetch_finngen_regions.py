"""All tested variants in locked FinnGen R13 AD regions, via public BGZF/tabix ranges."""
from pathlib import Path
import argparse,csv,gzip,json,hashlib,concurrent.futures
from range_tabix import RangeTabix
URL='https://storage.googleapis.com/finngen-public-data-r13/summary_stats/finngen_R13_G6_ALZHEIMER.gz'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--requests',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--prefix',required=True);a=ap.parse_args()
    reqs=json.loads(a.requests.read_text());a.out.mkdir(parents=True,exist_ok=True)
    tab=RangeTabix(URL,a.out/'cache');assert tab.header[:4]==['chrom','pos','ref','alt']
    def fetch(x):
        name=a.prefix+'_'+(x['assay'] if a.prefix=='AGES' else x['gene']);p=a.out/(name+'.tsv.gz');meta=p.with_suffix('.json')
        if p.exists() and meta.exists():return json.loads(meta.read_text())
        rows=tab.fetch(x['chrom'],int(x['start']),int(x['end']));converted=[]
        for d in rows:
            converted.append(dict(rsid=d['rsids'],chrom=d['chrom'],pos=d['pos'],build='GRCh38',effect_allele=d['alt'],other_allele=d['ref'],
                beta_outcome=d['beta'],se_outcome=d['sebeta'],eaf_outcome=d['af_alt'],p_outcome=d['pval'],n_cases=16652,n_controls=483534))
        assert converted,(name,'empty region')
        with gzip.open(p,'wt',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(converted[0]),delimiter='\t');w.writeheader();w.writerows(converted)
        m=dict(x,id=name,source_url=URL,outcome='FinnGen_R13_G6_ALZHEIMER',rows=len(rows),filtered_by_p_value=False,region_is_complete=True,
            path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),integrity='Checked HTTP Content-Range and BGZF CRC; exact tabix region, not full-file download',
            n_cases=16652,n_controls=483534,allele_definition='Effect and frequency refer to GRCh38 ALT',source_overlap='FinnGen independent of UKB/AGES exposure cohorts; overlap with Bellenguez outcome assessed separately')
        meta.write_text(json.dumps(m,indent=2));print('FINNGEN',name,len(rows),flush=True);return m
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:done=list(pool.map(fetch,reqs))
    (a.out/(a.prefix+'_manifest.json')).write_text(json.dumps(done,indent=2))
if __name__=='__main__':main()
