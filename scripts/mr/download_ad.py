"""Resumable public-file acquisition with exact byte-range and final MD5 checks."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,os,time
import requests
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    r=a.root;job=json.loads((r/'provenance/mr/download_job.json').read_text())
    dest=r/'data/mr/raw/GCST90027158_buildGRCh38.tsv.gz';partial=Path(job['path'])
    if dest.exists():
        digest=hashlib.md5()
        with dest.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):digest.update(b)
        assert dest.stat().st_size==job['expected_bytes'] and digest.hexdigest()==job['expected_md5']
        (r/'provenance/mr/ad_download_verified.json').write_text(json.dumps({'status':'complete','path':str(dest),'bytes':dest.stat().st_size,'md5':digest.hexdigest(),'source_url':job['url']},indent=2))
        print('Already downloaded and verified',flush=True);return
    parts=r/'data/mr/raw/ad_download_parts';parts.mkdir(exist_ok=True)
    meta=parts/'manifest.json'
    if not meta.exists():meta.write_text(json.dumps({'prefix_bytes':partial.stat().st_size if partial.exists() else 0,'total':job['expected_bytes'],'url':job['url']}))
    m=json.loads(meta.read_text());prefix=m['prefix_bytes'];assert partial.stat().st_size==prefix
    chunk=4*1024*1024
    ranges=[(s,min(s+chunk-1,m['total']-1)) for s in range(prefix,m['total'],chunk)]
    def fetch(pair):
        s,e=pair;p=parts/f'{s}-{e}.part'
        if p.exists() and p.stat().st_size==e-s+1:return
        for attempt in range(4):
            try:
                z=requests.get(m['url'],headers={'Range':f'bytes={s}-{e}','Accept-Encoding':'identity'},timeout=(20,90))
                z.raise_for_status()
                assert z.status_code==206 and z.headers.get('Content-Range')==f'bytes {s}-{e}/{m["total"]}'
                assert len(z.content)==e-s+1
                tmp=p.with_suffix('.tmp');tmp.write_bytes(z.content);tmp.replace(p);return
            except Exception:
                if attempt==3:raise
                time.sleep(2+attempt)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        done=0
        for _ in pool.map(fetch,ranges):
            done+=1
            if done%10==0:print('chunks',done,'/',len(ranges),flush=True)
    assembled=dest.with_suffix('.assembling')
    digest=hashlib.md5()
    with assembled.open('wb') as f:
        for p in [partial]+[parts/f'{s}-{e}.part' for s,e in ranges]:
            with p.open('rb') as src:
                for b in iter(lambda:src.read(1024*1024),b''):digest.update(b);f.write(b)
    assert assembled.stat().st_size==job['expected_bytes']
    assert digest.hexdigest()==job['expected_md5'],digest.hexdigest()
    assembled.replace(dest)
    (r/'provenance/mr/ad_download_verified.json').write_text(json.dumps({'status':'complete','path':str(dest),'bytes':dest.stat().st_size,'md5':digest.hexdigest(),'source_url':job['url']},indent=2))
    print('DOWNLOAD_VERIFIED',str(dest),flush=True)
if __name__=='__main__':main()
