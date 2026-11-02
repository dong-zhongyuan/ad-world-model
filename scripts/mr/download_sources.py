"""Adapted from download_ad.py: parallel resumable public downloads, final official MD5."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,time
import requests
def checksum(p):
    h=hashlib.md5()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--ranges-only',action='store_true');a=ap.parse_args()
    jobs=json.loads(a.plan.read_text());a.out.mkdir(parents=True,exist_ok=True);pending=[];done=[]
    for x in jobs:
        p=a.out/x['filename']
        if p.exists():
            assert p.stat().st_size==x['expected_bytes'] and checksum(p)==x['expected_md5']
            done.append(dict(x,path=str(p),status='verified'));continue
        pending.append(x)
    tasks=[]
    for x in pending:
        parts=a.out/(x['accession']+'_parts');parts.mkdir(exist_ok=True)
        for s in range(0,x['expected_bytes'],4*1024*1024):tasks.append((x,s,min(s+4*1024*1024-1,x['expected_bytes']-1),parts))
    def fetch(t):
        x,s,e,parts=t;p=parts/f'{s}-{e}.part'
        if p.exists() and p.stat().st_size==e-s+1:return
        for attempt in range(4):
            try:
                z=requests.get(x['url'],headers={'Range':f'bytes={s}-{e}','Accept-Encoding':'identity'},timeout=(20,90));z.raise_for_status()
                assert z.status_code==206 and z.headers.get('Content-Range')==f'bytes {s}-{e}/{x["expected_bytes"]}'
                assert len(z.content)==e-s+1
                tmp=p.with_suffix('.tmp');tmp.write_bytes(z.content);tmp.replace(p);return
            except Exception:
                if attempt==3:raise
                time.sleep(2+attempt)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(fetch,t) for t in tasks]
        for n,f in enumerate(concurrent.futures.as_completed(futures),1):
            f.result()
            if n%25==0:print('CHUNKS',n,'/',len(tasks),flush=True)
    if a.ranges_only:
        print('ALL_RANGES_READY; processing worker performs complete file MD5 verification',flush=True);return
    for x in pending:
        dest=a.out/x['filename'];temp=dest.with_suffix('.assembling');h=hashlib.md5()
        with temp.open('wb') as w:
            for _,s,e,parts in [t for t in tasks if t[0]['accession']==x['accession']]:
                b=(parts/f'{s}-{e}.part').read_bytes();h.update(b);w.write(b)
        assert temp.stat().st_size==x['expected_bytes'] and h.hexdigest()==x['expected_md5'],x['accession']
        temp.replace(dest);done.append(dict(x,path=str(dest),status='verified'));print('VERIFIED',x['gene'],x['accession'],flush=True)
    (a.out/'verified_sources.json').write_text(json.dumps(done,indent=2))
if __name__=='__main__':main()
