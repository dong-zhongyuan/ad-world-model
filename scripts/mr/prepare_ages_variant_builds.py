"""Independent dbSNP build check for the preselected AGES sentinel variants."""
from pathlib import Path
import argparse,csv,json,urllib.request,concurrent.futures,hashlib
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--leads',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    leads=list(csv.DictReader(a.leads.open()))
    def f(x):
        rs=x['rsid'];p=a.out/(rs+'.json');url='https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/'+rs[2:]
        if not p.exists():p.write_bytes(urllib.request.urlopen(url,timeout=40).read())
        d=json.loads(p.read_text());positions={}
        for v in d['primary_snapshot_data']['placements_with_allele']:
            if not v['seq_id'].startswith('NC_'):continue
            for asm in v['placement_annot'].get('seq_id_traits_by_assembly',[]):
                for build in ['GRCh37','GRCh38']:
                    if asm['assembly_name'].startswith(build):
                        spdi=v['alleles'][0]['allele']['spdi'];positions[build]=spdi['position']+1
        assert positions['GRCh37']==int(x['pos37']),(rs,positions,x['pos37'])
        return rs,dict(pos37=positions['GRCh37'],pos38=positions['GRCh38'],chrom=x['chrom'],source_url=url,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:values=dict(pool.map(f,leads))
    (a.out/'ages_variant_builds.json').write_text(json.dumps(values,indent=2));print(json.dumps(values,indent=2))
if __name__=='__main__':main()
