#!/usr/bin/env python3
"""Prepare all supported unique library parents; retain every exclusion/failure."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from paths import DRUG_WORK as ROOT
OUT = ROOT / 'prepared/ligands'


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def prepare_one(row):
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
    from dimorphite_dl import protonate_smiles
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    cid = row['chembl_id']
    folder = OUT / cid
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / 'preparation.json'
    if output.exists():
        return json.loads(output.read_text())
    result = {'chembl_id': cid, 'name': row['name'], 'parent_smiles': row['parent_smiles'],
              'source_inchikey': row['inchikey'], 'states': [], 'errors': [], 'status': 'pending'}
    try:
        mol = Chem.MolFromSmiles(row['parent_smiles'])
        if mol is None:
            raise ValueError('Server RDKit cannot parse parent structure')
        if mol.GetNumHeavyAtoms() < 6:
            raise ValueError('Outside organic ligand scope: fewer than six heavy atoms')
        elements = {a.GetAtomicNum() for a in mol.GetAtoms()}
        if not elements <= {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}:
            raise ValueError('Requires separate chemistry/metal parameterization: ' + str(sorted(elements)))
        if rdMolDescriptors.CalcNumRotatableBonds(mol) > 32:
            raise ValueError('Exceeds standard Vina torsion scope of 32 rotatable bonds')
        undefined = sum(label == '?' for _, label in Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        if undefined > 4:
            raise ValueError('More than four unspecified stereocentres; identity review required before assigning configurations')
        stereo = list(EnumerateStereoisomers(mol, options=StereoEnumerationOptions(onlyUnassigned=True, maxIsomers=0, unique=True)))
        microstates = set()
        for iso in stereo:
            states = protonate_smiles(Chem.MolToSmiles(iso, isomericSmiles=True),
                                      ph_min=6.9, ph_max=7.9, max_variants=128)
            if len(states) >= 128:
                raise ValueError('Protonation enumeration reached cap; no truncated ensemble accepted')
            microstates.update(states)
        if not microstates:
            raise ValueError('No valid protonation state')
        for i, smiles in enumerate(sorted(microstates)):
            state_id = f'{cid}_s{i:03d}'
            state = {'state_id': state_id, 'smiles': smiles, 'status': 'pending'}
            try:
                m = Chem.AddHs(Chem.MolFromSmiles(smiles))
                seed = int(hashlib.sha256(state_id.encode()).hexdigest()[:7], 16)
                params = AllChem.ETKDGv3()
                params.randomSeed = seed
                params.numThreads = 1
                confs = AllChem.EmbedMultipleConfs(m, numConfs=10, params=params)
                if not confs:
                    params.useRandomCoords = True
                    confs = AllChem.EmbedMultipleConfs(m, numConfs=10, params=params)
                if not confs:
                    raise ValueError('3D embedding failed')
                if not AllChem.MMFFHasAllMoleculeParams(m):
                    raise ValueError('MMFF94s preparation parameters incomplete')
                optimized = AllChem.MMFFOptimizeMoleculeConfs(m, numThreads=1, maxIters=1500, mmffVariant='MMFF94s')
                converged = [j for j, (flag, energy) in enumerate(optimized) if flag == 0]
                if not converged:
                    raise ValueError('No converged MMFF94s conformer')
                best = min(converged, key=lambda j: optimized[j][1])
                chosen = Chem.Mol(m)
                conformer = Chem.Conformer(m.GetConformer(confs[best]))
                chosen.RemoveAllConformers()
                chosen.AddConformer(conformer, assignId=True)
                chosen.SetProp('_Name', state_id)
                chosen.SetProp('CHEMBL_ID', cid)
                chosen.SetProp('MICROSTATE_SMILES', smiles)
                sdf = folder / (state_id + '.sdf')
                writer = Chem.SDWriter(str(sdf))
                writer.write(chosen)
                writer.close()
                setups = MoleculePreparation().prepare(chosen)
                if len(setups) != 1:
                    raise ValueError('Multiple macrocycle setup variants require explicit handling')
                text, ok, err = PDBQTWriterLegacy.write_string(setups[0])
                if not ok:
                    raise ValueError(err)
                pdbqt = folder / (state_id + '.pdbqt')
                pdbqt.write_text(text)
                state.update(status='ready', sdf=str(sdf.relative_to(ROOT)), pdbqt=str(pdbqt.relative_to(ROOT)),
                             seed=seed, converged_conformers=len(converged),
                             preparation_energy_kcal_mol=float(optimized[best][1]),
                             formal_charge=Chem.GetFormalCharge(chosen),
                             stereochemistry_enumerated=bool(undefined))
            except Exception as exc:
                state.update(status='preparation_failed', error=str(exc))
            result['states'].append(state)
        n_ready = sum(s['status'] == 'ready' for s in result['states'])
        result['status'] = 'ready' if n_ready == len(result['states']) else ('partial' if n_ready else 'preparation_failed')
    except Exception as exc:
        result.update(status='requires_review', errors=[str(exc)])
    atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / '.preparation.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (ROOT / 'inputs/library_identity.tsv').open() as f:
        rows = [r for r in csv.DictReader(f, delimiter='\t') if r['check_status'] == 'structure_valid' and not r['duplicate_parent_of']]
    start = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(prepare_one, r) for r in rows]
        for future in as_completed(futures):
            results.append(future.result())
            progress = {'status': 'running', 'completed': len(results), 'total_parents': len(rows),
                        'ready_parents': sum(r['status'] == 'ready' for r in results),
                        'ready_microstates': sum(s['status'] == 'ready' for r in results for s in r['states']),
                        'elapsed_seconds': time.time() - start}
            atomic_json(OUT / 'progress.json', progress)
            if len(results) % 25 == 0:
                print(json.dumps(progress), flush=True)
    results.sort(key=lambda r: r['chembl_id'])
    atomic_json(OUT / 'preparation_manifest.json', results)
    progress['status'] = 'complete'
    atomic_json(OUT / 'progress.json', progress)
    print(json.dumps(progress), flush=True)


if __name__ == '__main__':
    main()
