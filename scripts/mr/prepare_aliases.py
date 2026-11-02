"""Resolve only official dbSNP merged identifiers; never replace a genetic instrument."""
from pathlib import Path
import argparse,csv,json,concurrent.futures,urllib.request,re
from datetime import datetime,timezone
from prepare_exposures import sha
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);r=ap.parse_args().root
    ep=r/'data/mr/processed/primary_instruments.csv';dest=r/'provenance/mr/dbsnp_aliases.json'
    if dest.exists():
        cached=json.loads(dest.read_text())
        assert cached['instruments_sha256']==sha(ep),'Alias source instruments changed'
        if cached['status']=='complete':print('Using verified cached dbSNP identifier records');return
    ids=sorted({x['rsid'] for x in csv.DictReader(ep.open()) if re.fullmatch(r'rs\d+',x['rsid'])})
    raw=r/'data/mr/raw/dbsnp';raw.mkdir(parents=True,exist_ok=True)
    def fetch(rs):
        chain=[];current=rs
        try:
            for _ in range(8):
                url='https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/'+current[2:]
                f=raw/(current+'.json')
                if not f.exists():
                    data=urllib.request.urlopen(url,timeout=40).read();json.loads(data);f.write_bytes(data)
                d=json.loads(f.read_text());assert str(d['refsnp_id'])==current[2:]
                merged=d.get('merged_snapshot_data',{}).get('merged_into',[])
                chain.append(dict(rsid=current,url=url,sha256=sha(f)))
                if not merged:return dict(requested=rs,canonical=current,chain=chain,status='resolved')
                if len(merged)!=1:raise ValueError('Non-unique merge')
                current='rs'+str(merged[0])
            raise ValueError('Merge chain too long')
        except Exception as e:return dict(requested=rs,chain=chain,status='unresolved',error=str(e))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(fetch,ids))
    aliases={}
    for x in records:
        if x['status']=='resolved':
            for step in x['chain']:
                if step['rsid']!=x['canonical']:aliases[step['rsid']]=x['canonical']
    out=dict(status='complete' if all(x['status']=='resolved' for x in records) else 'partial',
        instruments_sha256=sha(ep),retrieved_utc=datetime.now(timezone.utc).isoformat(),aliases=aliases,records=records,
        rule='Official merged identifiers only; same GRCh38 coordinate and sequence alleles still required; no LD proxy')
    dest.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='records'},indent=2))
if __name__=='__main__':main()
