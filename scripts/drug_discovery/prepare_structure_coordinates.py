from pathlib import Path
import json,numpy as np,hashlib,csv,os
from scipy.spatial import cKDTree
p=Path(os.environ.get('YB_STRUCTURE_DATA',str(Path(__file__).resolve().parents[1]/'data/drd5_structure')));d=json.loads((p/'coordinate_analysis.json').read_text());r=d['receptor'];lig=d['ligand'];orig=[]
for ln in Path(d.get('receptor_path',p/'DRD5.src.pdb')).read_text().splitlines():
 if ln.startswith('ATOM') and ln[76:78].strip()!='H':orig.append(dict(atom=ln[12:16].strip(),res=ln[17:20].strip(),resi=int(ln[22:26]),chain=ln[21],xyz=[float(ln[30:38]),float(ln[38:46]),float(ln[46:54])],plddt=float(ln[60:66])))
dist,idx=cKDTree(np.array([a['xyz']for a in orig])).query(np.array([a['xyz']for a in r]));assert max(dist)<.002
for i,a in enumerate(r):
 b=orig[int(idx[i])]
 a.update(id=i,source_res=a['res'],res=b['res'],source_atom=a['atom'],atom=b['atom'],source_chain=a['chain'],source_resi=a['resi'],chain=b['chain'],resi=b['resi'],plddt=b['plddt'])
byid={a['serial']:a for a in lig}
if 'ligand_bonds' in d:bonds=d['ligand_bonds']
else:
 smimap={}
 for ln in (p/'DRD5__EJ_SQFNCEKTEKNPCN.pdbqt').read_text().splitlines():
  if ln.startswith('REMARK SMILES IDX'):
   nums=list(map(int,ln.split()[3:]));smimap.update(zip(nums[::2],nums[1::2]))
 chem=json.loads((p/'ligand_chemistry.json').read_text())
 assert all(byid[smimap[i+1]]['element']==e for i,e in enumerate(chem['atoms']))
 bonds=[dict(a=smimap[b['a']],b=smimap[b['b']],order=b['order'])for b in chem['bonds']]
x=np.array([a['xyz']for a in r]);y=np.array([a['xyz']for a in lig]);ds=np.linalg.norm(x[:,None,:]-y[None,:,:],axis=2);groups={}
for i,a in enumerate(r):
 k=(a['chain'],a['resi']);j=int(ds[i].argmin());v=float(ds[i,j])
 if k not in groups or v<groups[k]['distance']:groups[k]=dict(res=a['res'],resi=a['resi'],chain=a['chain'],r_id=i,l_serial=lig[j]['serial'],distance=v)
contacts=sorted([v for v in groups.values()if v['distance']<4.5],key=lambda v:v['distance'])
polar=[]
for i,a in enumerate(r):
 for j,b in enumerate(lig):
  if a['element']in ['N','O'] and b['element']in ['N','O'] and ds[i,j]<3.6:polar.append(dict(res=a['res'],resi=a['resi'],r_id=i,l_serial=b['serial'],distance=float(ds[i,j])))
# receptor bonds inferred only within the same original residue, or consecutive C-N peptide connections
radii={'C':.76,'N':.71,'O':.66,'S':1.05};rb=[]
for i,j in sorted(cKDTree(x).query_pairs(2.0)):
 a,b=r[i],r[j];v=float(np.linalg.norm(x[i]-x[j]));same=a['resi']==b['resi'];peptide=(b['resi']==a['resi']+1 and a['atom']=='C' and b['atom']=='N') or (a['resi']==b['resi']+1 and b['atom']=='C' and a['atom']=='N')
 if a['chain']==b['chain'] and (same or peptide) and .5<v<radii.get(a['element'],.75)+radii.get(b['element'],.75)+.42:rb.append([i,j])
result=dict(receptor=r,ligand=lig,full_receptor=orig,ligand_bonds=bonds,receptor_bonds=rb,contacts=contacts,polar_proximities=polar,coordinate_map_max_error=float(max(dist)))
result.update({k:d[k] for k in ['target','candidate_id','cid','pose_seed','selected_residues','model_source','source_hashes'] if k in d})
(p/'structure_scene.json').write_text(json.dumps(result,indent=2))
print('Mapped',len(r),'receptor heavy atoms; max coordinate error',max(dist));print('Contacts',contacts);print('Polar proximities',polar)
