#!/usr/bin/env python3
"""Copied from export_drugclip_ready.py; extend verified pocket export to ten targets.
Use the whole annotated mature human protein, sequence-checked against UniProt.
Keep every default-fpocket cavity, its complete 6-A residue neighbourhood, and confidence.
Existing experimental BIN1 pockets are retained unchanged as additional inputs.
"""
import gzip, hashlib, json, os, re, subprocess, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from paths import DRUG_WORK as ROOT
INP=ROOT/'inputs/ten_targets'
OUT=INP/'pockets'
OUT.mkdir(parents=True,exist_ok=True)
FPOCKET=ROOT/'software/fpocket'
FPOCKET.chmod(0o755)
PARSER=PDBParser(QUIET=True)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)

def prepare(row):
 gene=row['gene']; folder=INP/'structures'/gene
 manifest=json.loads((folder/'manifest.json').read_text())
 assert manifest['accession']==row['uniprot']
 u=next(x for x in json.loads((INP/'uniprot'/f'{gene}.json').read_text())['results'] if x['primaryAccession']==row['uniprot'])
 sequence=u['sequence']['value']
 # Select the canonical model by accession and exact full sequence, independently of scores.
 eligible=[a for a in manifest['alphafold_models'] if a['sequence']==sequence and a['entry']=='AF-'+row['uniprot']+'-F1']
 assert len(eligible)==1,(gene,'canonical AlphaFold model unavailable')
 af=eligible[0];source=Path(af['path']);assert sha(source)==af['sha256']
 removed=set()
 for feature in u.get('features',[]):
  if feature['type'] in ['Signal','Propeptide']:
   a=feature['location']['start'];b=feature['location']['end']
   if a.get('modifier')=='EXACT' and b.get('modifier')=='EXACT':removed.update(range(a['value'],b['value']+1))
 model=PDBParser(QUIET=True).get_structure(gene,source)[0]
 for residue in model.get_residues():
  assert seq1(residue.resname)==sequence[residue.id[1]-1],(gene,residue.id,residue.resname)
 mature=folder/(gene+'_mature.pdb')
 lines=[line for line in source.read_text().splitlines() if line.startswith('ATOM  ') and int(line[22:26]) not in removed]
 mature.write_text('\n'.join(lines)+'\nTER\nEND\n')
 fpout=folder/(mature.stem+'_out')
 log=folder/'fpocket.log'
 marker=folder/'fpocket_complete.json'
 identity={'input_sha256':sha(mature),'fpocket_binary_sha256':sha(FPOCKET)}
 if not marker.exists():
  with log.open('w') as f:
   result=subprocess.run([str(FPOCKET),'-f',str(mature)],stdout=f,stderr=subprocess.STDOUT,timeout=600,env=dict(os.environ,OMP_NUM_THREADS='1'))
  assert result.returncode==0,(gene,result.returncode,log.read_text()[-2000:])
  write(marker,identity)
 else:assert json.loads(marker.read_text())==identity
 structure=PDBParser(QUIET=True).get_structure(gene,mature)[0]
 residues=[]
 for residue in structure.get_residues():
  heavy=[a for a in residue if a.element!='H']
  if heavy:residues.append((residue,heavy,np.asarray([a.coord for a in heavy])))
 pockets=[]
 for pqr in sorted((fpout/'pockets').glob('pocket*_vert.pqr'),key=lambda p:int(re.search(r'pocket(\d+)',p.name)[1])):
  centers=np.array([[float(line[30:38]),float(line[38:46]),float(line[46:54])] for line in pqr.read_text().splitlines() if line.startswith(('ATOM  ','HETATM'))])
  assert len(centers)>0,(gene,pqr)
  selected=[];resids=[];confidence=[]
  for residue,heavy,xyz in residues:
   if np.linalg.norm(xyz[:,None,:]-centers[None,:,:],axis=-1).min()<=6.0:
    selected.extend(heavy);resids.append(residue.id[1]);confidence.append(float(residue['CA'].bfactor))
  assert selected,(gene,pqr)
  pid=gene+'_'+af['entry']+'_'+pqr.stem.replace('_vert','')
  pocket={'pocket':pid,'target':gene,'screening_name':row['screening_name'],'uniprot':row['uniprot'],
          'pocket_atoms':[a.name for a in selected], 'pocket_coordinates':np.array([a.coord for a in selected]).tolist(),
          'residues_uniprot':resids,'definition':'whole mature-protein residues within 6 A of every alpha sphere in one default-fpocket cavity',
          'source_kind':'AlphaFold predicted structure; fpocket predicted cavity','structure_url':af['url'],
          'receptor_sha256':sha(mature),'pqr_sha256':sha(pqr),'mean_residue_pLDDT':float(np.mean(confidence)),
          'fraction_residues_pLDDT_ge70':float(np.mean(np.array(confidence)>=70)), 'complete_pocket_atoms':len(selected),
          'fpocket_binary_sha256':identity['fpocket_binary_sha256']}
  pockets.append(pocket)
 result={'target':gene,'status':'pockets_ready' if pockets else 'no_cavity_detected','uniprot':row['uniprot'],
         'sequence_verified':True,'removed_signal_propeptide_residues':sorted(removed),'pocket_count':len(pockets),
         'heavy_atom_counts':[len(p['pocket_atoms']) for p in pockets],'pockets':pockets}
 write(OUT/(gene+'.json'),result)
 print(json.dumps({k:v for k,v in result.items() if k not in ['pockets','removed_signal_propeptide_residues']}),flush=True)
 return result

def main():
 registry=json.loads((INP/'verification/acceleration_inputs/drugclip_ten_target_identity.json').read_text())
 write(OUT/'progress.json',{'status':'running','planned_targets':10,'completed_targets':[]})
 results=[]
 with ThreadPoolExecutor(max_workers=5) as pool:
  futures={pool.submit(prepare,r):r['gene'] for r in registry['targets']}
  for future in as_completed(futures):
   results.append(future.result())
   write(OUT/'progress.json',{'status':'running','planned_targets':10,'completed_targets':[r['target'] for r in results]})
 pockets=[p for r in sorted(results,key=lambda r:r['target']) for p in r['pockets']]
 with gzip.open(INP/'verification/acceleration_inputs/drugclip_ten_ready.json.gz','rt') as f: old=json.load(f)
 for pocket in old['pockets']:
  pockets.append(dict(pocket,target='BIN1',screening_name='BIN1',uniprot='O00499',source_kind='experimental NMR c-Myc interface',complete_pocket_atoms=len(pocket['pocket_atoms'])))
 assert len({p['pocket'] for p in pockets})==len(pockets)
 with gzip.open(OUT/'all_ten_pockets.json.gz','wt') as f:json.dump(pockets,f)
 progress={'status':'complete','planned_targets':10,'completed_targets':sorted(r['target'] for r in results),
           'pocket_counts':{g:sum(p['target']==g for p in pockets) for g in registry['gates']},
           'all_targets_have_pockets':all(any(p['target']==g for p in pockets) for g in registry['gates']),
           'pockets':len(pockets),'max_pocket_atoms':max(len(p['pocket_atoms']) for p in pockets),
           'no_pocket_score_filtering':True,'sequence_identity_verified_all':True}
 write(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)

if __name__=='__main__':main()
