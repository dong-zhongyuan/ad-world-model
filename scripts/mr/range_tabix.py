"""Read public BGZF/tabix point records with checked HTTP byte ranges and CRCs."""
import urllib.request,struct,gzip,zlib,time,hashlib,json,threading
from pathlib import Path

def get(url,start=None,end=None):
 for attempt in range(4):
  try:
   headers={} if start is None else {'Range':f'bytes={start}-{end}'}
   with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=45) as r:
    if start is not None:
     assert r.status==206 and r.headers['Content-Range'].startswith(f'bytes {start}-'),(r.status,r.headers)
    b=r.read()
    if start is not None:
     actual_end=int(r.headers['Content-Range'].split('-')[1].split('/')[0]);assert len(b)==actual_end-start+1
    return b
  except Exception:
   if attempt==3:raise
   time.sleep(1+attempt)

def blocks(b,base=0):
 i=0
 while i+18<=len(b):
  assert b[i:i+3]==b'\x1f\x8b\x08',(base+i,'invalid BGZF magic')
  size=struct.unpack_from('<H',b,i+16)[0]+1
  if i+size>len(b):break
  yield base+i,zlib.decompress(b[i:i+size],31)
  i+=size
class RangeTabix:
 def __init__(self,url,cache):
  self.url=url;self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True)
  idx=self.cache/(url.split('/')[-1]+'.tbi')
  if not idx.exists():idx.write_bytes(get(url+'.tbi'))
  b=gzip.decompress(idx.read_bytes());assert b[:4]==b'TBI\x01'
  self.b=b;self.i=4
  def read(fmt):
   x=struct.unpack_from(fmt,b,self.i);self.i+=struct.calcsize(fmt);return x
  nref,fmt,colseq,colbeg,colend,meta,skip,ln=read('<8i');self.columns=(colseq,colbeg)
  names=b[self.i:self.i+ln].decode().rstrip('\0').split('\0');self.i+=ln;self.refs={}
  for name in names:
   bins={}
   for _ in range(read('<i')[0]):
    bin_id,n=read('<Ii');bins[bin_id]=[read('<QQ') for _ in range(n)]
   ni=read('<i')[0];linear=read('<'+'Q'*ni) if ni else []
   self.refs[name]=(bins,linear)
  head=b''.join(x for _,x in blocks(get(url,0,65535)))
  self.header=head.decode().splitlines()[0].lstrip('#').split('\t')
 def fetch(self,chrom,pos,end_pos=None):
  stop=pos if end_pos is None else end_pos
  assert 1<=pos<=stop
  beg=pos-1;end=stop-1
  bins,linear=self.refs[str(chrom)];ids=[0]+list(range(1+(beg>>26),2+(end>>26)))+list(range(9+(beg>>23),10+(end>>23)))+list(range(73+(beg>>20),74+(end>>20)))+list(range(585+(beg>>17),586+(end>>17)))+list(range(4681+(beg>>14),4682+(end>>14)))
  minoff=linear[beg>>14] if beg>>14<len(linear) else 0
  chunks=sorted((max(a,minoff),z) for k in ids for a,z in bins.get(k,[]) if z>minoff and a<z)
  merged=[]
  for a,z in chunks:
   if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(z,merged[-1][1]))
   else:merged.append((a,z))
  rows=[]
  for a,z in merged:
   start=a>>16;end=z>>16;key=hashlib.sha256(f'{self.url}:{start}:{end}'.encode()).hexdigest();p=self.cache/(key+'.bgzf')
   if p.exists():raw=p.read_bytes()
   else:
    raw=get(self.url,start,end+65535);tmp=p.with_suffix('.tmp'+str(threading.get_ident()));tmp.write_bytes(raw);tmp.replace(p)
   parts=[]
   for offset,dec in blocks(raw,start):
    if offset>end:break
    lo=(a&65535) if offset==start else 0
    hi=(z&65535) if offset==end else len(dec)
    parts.append(dec[lo:hi])
   for line in b''.join(parts).decode().splitlines():
    v=line.split('\t')
    if len(v)==len(self.header) and v[self.columns[0]-1]==str(chrom) and pos<=int(v[self.columns[1]-1])<=stop:rows.append(dict(zip(self.header,v)))
  return list({tuple(r.items()):r for r in rows}.values())
