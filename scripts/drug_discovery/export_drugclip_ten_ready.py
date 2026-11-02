#!/usr/bin/env python3
"""Export the same complete ready-parent snapshot for all ten authorized targets."""
import gzip
import hashlib
import json
import csv
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
from candidate_gate import require_allowed

from paths import DRUG_WORK as ROOT
OUT = ROOT/'verification/acceleration_inputs'
OUT.mkdir(parents=True, exist_ok=True)
registry = json.loads((ROOT/'execution/drugclip_ten_targets.json').read_text())
genes = [r['gene'] for r in registry['targets']]
assert len(genes) == len(set(genes)) == 10
gates = {g: require_allowed(g) for g in genes}
project = ROOT.parents[1]
authority = json.loads((project/'results/CURRENT_RESULTS.json').read_text())
mr = list(csv.DictReader((project/authority['mr_results']/'mr_all.csv').open()))
identities=[]
verified_uniprot={'BIN1':'O00499','CR1':'P17927','PILRB':'Q9UKJ0','GRN':'P28799','BLNK':'Q8WV28',
                  'CD55':'P08174','ACP5':'P13686','ST13':'P50502','GNAS':'O95467','BST2':'Q10589'}
for row in registry['targets']:
    assays=[x for x in mr if x['gene']==row['gene'] and x['primary']=='True']
    accessions=sorted({x['uniprot'] for x in assays if x['uniprot']})
    accession=verified_uniprot[row['gene']]
    assert not accessions or accessions==[accession], (row['gene'], accessions)
    identities.append(dict(row, uniprot=accession, mr_accessions_recorded=accessions,
                           accession_source='MR assay and current reviewed human UniProt entry' if accessions else 'current reviewed human UniProt entry; MR assay accession field empty',
                           assays=sorted({x['assay'] for x in assays}),
                           screening_name='NESP55' if row['gene']=='GNAS' else row['gene']))
(OUT/'drugclip_ten_target_identity.json').write_text(json.dumps({'targets':identities,'groups':registry['groups'],'gates':gates},indent=2))
parents = [json.loads(p.read_text()) for p in sorted((ROOT/'prepared/ligands').glob('CHEMBL*/preparation.json'))]
ready = [p for p in parents if p['status'] == 'ready']
ready.sort(key=lambda p: (Chem.MolFromSmiles(p['parent_smiles']).GetNumHeavyAtoms(), p['chembl_id']))
molecules=[]
for parent in ready:
  for state in parent['states']:
    path=ROOT/state['sdf']
    mol=Chem.RemoveHs(Chem.SDMolSupplier(str(path), removeHs=False)[0])
    molecules.append({'state_id':state['state_id'],'chembl_id':parent['chembl_id'],
                      'name':parent['name'],
                      'smi':state['smiles'],'atoms':[a.GetSymbol() for a in mol.GetAtoms()],
                      'coordinates':mol.GetConformer().GetPositions().tolist(),
                      'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
parser=PDBParser(QUIET=True)
reference=parser.get_structure('reference',ROOT/'inputs/raw/BIN1_1MV0.pdb')
pockets=[]
for path in sorted((ROOT/'prepared/receptors/BIN1').glob('*canonical.pdb')):
    model=int(path.stem.split('_')[2])-1
    peptide=np.array([a.coord for a in reference[model]['A'].get_atoms() if a.element!='H'])
    chain=next(parser.get_structure(path.stem,path)[0].get_chains())
    selected=[];residues=[]
    for residue in chain:
        heavy=[a for a in residue if a.element!='H']
        xyz=np.array([a.coord for a in heavy])
        if np.linalg.norm(xyz[:,None,:]-peptide[None,:,:],axis=-1).min()<=6.0:
            selected.extend(heavy);residues.append(residue.id[1])
    pockets.append({'pocket':path.stem,'pocket_atoms':[a.name for a in selected],
                    'pocket_coordinates':np.array([a.coord for a in selected]).tolist(),
                    'residues_source_numbering':residues,'definition':'whole residues within 6 A of experimental c-Myc partner',
                    'receptor_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
data={'scope':'All states of all fully prepared parents at export; common library for ten-target DrugCLIP screening',
      'candidate_gates':gates,
      'parent_statuses':{p['chembl_id']:p['status'] for p in parents},
      'ready_parents_at_export':len(ready),'ready_states_at_export':sum(len(p['states']) for p in ready),
      'molecules':molecules,'pockets':pockets}
assert len(molecules)==data['ready_states_at_export']
assert len({m['state_id'] for m in molecules})==len(molecules)
with gzip.open(OUT/'drugclip_ten_ready.json.gz','wt') as f:json.dump(data,f)
print(json.dumps({'exported_states':len(molecules),'ready_parents':len(ready),'pocket_heavy_atoms':[len(p['pocket_atoms']) for p in pockets],
                  'ligand_heavy_atom_range':[min(len(m['atoms']) for m in molecules),max(len(m['atoms']) for m in molecules)]}))
