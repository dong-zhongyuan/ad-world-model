#!/usr/bin/env python3
"""Vina ensemble screening with preserved states, poses, seeds and failures."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from candidate_gate import status, require_allowed

from paths import DRUG_WORK as ROOT


def atomic_json(path, data):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def receptors(target):
    if target == 'BIN1':
        from Bio.PDB import PDBParser
        source = PDBParser(QUIET=True).get_structure('BIN1', ROOT / 'inputs/raw/BIN1_1MV0.pdb')
        rows = []
        for file in sorted((ROOT / 'prepared/receptors/BIN1').glob('BIN1_model_*_canonical.pdbqt')):
            model = int(file.stem.split('_')[2]) - 1
            xyz = np.array([a.coord for a in source[model]['A'].get_atoms() if a.element != 'H'])
            rows.append({'name': file.stem, 'pdbqt': str(file.relative_to(ROOT)),
                         'pdb': str(file.with_suffix('.pdb').relative_to(ROOT)),
                         'box_center_A': ((xyz.min(0)+xyz.max(0))/2).tolist(),
                         'box_size_A': (xyz.max(0)-xyz.min(0)+16.0).tolist()})
        return rows
    path = ROOT / 'prepared/receptors/GRN/ensemble_manifest.json'
    return json.loads(path.read_text()) if path.exists() else []


def one(job):
    from vina import Vina
    from meeko import PDBQTMolecule, RDKitMolCreate
    from rdkit import Chem
    target, receptor, ligand, seed, folder = job
    require_allowed(target)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    name = receptor['name'] + '__' + ligand['state_id'] + f'__seed{seed}'
    output = folder / (name + '.json')
    ligand_path = ROOT / ligand['pdbqt']
    receptor_path = ROOT / receptor['pdbqt']
    fingerprint = hashlib.sha256((ligand_path.read_text()+receptor_path.read_text()+json.dumps(receptor, sort_keys=True)+str(seed)+'exhaustiveness32').encode()).hexdigest()
    if output.exists():
        cached = json.loads(output.read_text())
        if cached['input_fingerprint'] != fingerprint:
            raise ValueError('Changed inputs for an existing result: ' + str(output))
        return cached
    started = time.monotonic()
    result = {'target': target, 'receptor': receptor['name'], 'state_id': ligand['state_id'],
              'chembl_id': ligand['chembl_id'], 'name': ligand['name'], 'seed': seed,
              'input_fingerprint': fingerprint, 'exhaustiveness': 32, 'status': 'running'}
    try:
        vina = Vina(sf_name='vina', cpu=1, seed=seed, verbosity=0)
        vina.set_receptor(str(receptor_path))
        vina.set_ligand_from_file(str(ligand_path))
        vina.compute_vina_maps(center=receptor['box_center_A'], box_size=receptor['box_size_A'])
        vina.dock(exhaustiveness=32, n_poses=20)
        energies = vina.energies(n_poses=10)
        if not len(energies) or not np.isfinite(energies).all():
            raise ValueError('No finite docking energies')
        pose_file = folder / (name + '.pdbqt')
        vina.write_poses(str(pose_file), n_poses=10, overwrite=False)
        parsed = PDBQTMolecule.from_file(str(pose_file), skip_typing=True)
        molecules = RDKitMolCreate.from_pdbqt_mol(parsed)
        if len(molecules) != 1 or molecules[0] is None:
            raise ValueError('Pose chemical reconstruction failed')
        original = Chem.SDMolSupplier(str(ROOT / ligand['sdf']), removeHs=False)[0]
        recovered = molecules[0]
        if Chem.MolToInchiKey(Chem.RemoveHs(original)) != Chem.MolToInchiKey(Chem.RemoveHs(recovered)):
            raise ValueError('Exported pose identity differs from the submitted microstate')
        sdf = folder / (name + '.sdf')
        writer = Chem.SDWriter(str(sdf))
        recovered.SetProp('CHEMBL_ID', ligand['chembl_id'])
        recovered.SetProp('STATE_ID', ligand['state_id'])
        for conf in recovered.GetConformers():
            writer.write(recovered, confId=conf.GetId())
        writer.close()
        result.update(status='complete', top_score_kcal_mol=float(energies[0,0]),
                      energies_kcal_mol=energies.tolist(), poses=str(pose_file.relative_to(ROOT)),
                      sdf=str(sdf.relative_to(ROOT)), pose_identity_verified=True,
                      box_center_A=receptor['box_center_A'], box_size_A=receptor['box_size_A'])
    except Exception as exc:
        result.update(status='failed', error=str(exc))
    result['seconds'] = time.monotonic() - started
    atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True, choices=['BIN1','GRN'])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--benchmark-parents', type=int, default=0)
    args = parser.parse_args()
    require_allowed(args.target)
    mode = 'benchmark' if args.benchmark_parents else 'screen'
    folder = ROOT / 'docking' / args.target / mode
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / '.dock.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    receptor_rows = receptors(args.target)
    if not receptor_rows:
        raise ValueError('No prepared receptor')
    records = [json.loads(p.read_text()) for p in sorted((ROOT / 'prepared/ligands').glob('CHEMBL*/preparation.json'))]
    ready = [r for r in records if r['status'] == 'ready']
    if args.benchmark_parents:
        import csv
        with (ROOT / 'inputs/library_identity.tsv').open() as f:
            identity = {r['chembl_id']:r for r in csv.DictReader(f, delimiter='\t')}
        ready.sort(key=lambda r:(float(identity[r['chembl_id']]['rotatable_bonds']),r['chembl_id']))
        selected = np.linspace(0, max(0,len(ready)-1), min(args.benchmark_parents,len(ready)), dtype=int)
        ready = [ready[i] for i in selected]
        receptor_rows = receptor_rows[:1]
    else:
        prep = json.loads((ROOT / 'prepared/ligands/progress.json').read_text())
        if prep['status'] != 'complete' or len(receptor_rows) != 3:
            raise ValueError('Full screening requires completed preparation and three ensemble receptors')
    jobs = []
    for parent in ready:
        states = parent['states'][:1] if args.benchmark_parents else parent['states']
        for ligand in states:
            ligand = dict(ligand, chembl_id=parent['chembl_id'], name=parent['name'])
            for receptor in receptor_rows:
                jobs.append((args.target,receptor,ligand,42,str(folder)))
    atomic_json(folder / 'registered_jobs.json', {'mode': mode, 'target':args.target,
                'parents':len(ready),'jobs':len(jobs),'receptors':receptor_rows,
                'exhaustiveness':32,'seed':42,'benchmark_excluded_from_primary_ranking':bool(args.benchmark_parents),
                'job_identity':[{'receptor':j[1]['name'],'state_id':j[2]['state_id'],'seed':j[3]} for j in jobs]})
    if not jobs:
        raise ValueError('No chemically prepared ligand jobs')
    pool = ProcessPoolExecutor(max_workers=args.workers)
    pending = {}
    todo = iter(jobs)
    rows = []
    started = time.monotonic()
    exhausted = False
    while pending or not exhausted:
        gate = status(args.target)
        if not gate['allowed']:
            for process in pool._processes.values():
                process.terminate()
            pool.shutdown(wait=True, cancel_futures=True)
            atomic_json(folder/'progress.json',{'status':'stopped_by_updated_candidates','candidate_gate':gate,
                        'completed':len(rows),'total_jobs':len(jobs)})
            return
        while len(pending) < args.workers*2 and not exhausted:
            try:
                job = next(todo)
            except StopIteration:
                exhausted = True
                break
            pending[pool.submit(one,job)] = job
        done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
        for future in done:
            job = pending.pop(future)
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append({'target':job[0], 'receptor':job[1]['name'], 'state_id':job[2]['state_id'],
                             'status':'failed', 'error':str(exc)})
        if done:
            progress = {'status':'running','mode':mode,'target':args.target,'completed':len(rows),
                        'total_jobs':len(jobs),'successful':sum(r['status']=='complete' for r in rows),
                        'failed':sum(r['status']=='failed' for r in rows),'elapsed_seconds':time.monotonic()-started}
            atomic_json(folder/'progress.json',progress)
            print(json.dumps(progress),flush=True)
    pool.shutdown()
    progress['status'] = 'complete' if progress['failed'] == 0 else 'finished_with_failures'
    atomic_json(folder/'progress.json',progress)
    atomic_json(folder/'results.json',rows)


def shortlist_main():
    from paths import FIGURE_WORK
    fig=FIGURE_WORK;folder=ROOT/'docking/figure_shortlist';folder.mkdir(exist_ok=True)
    lock=(folder/'.dock.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=[]
    for r in json.loads((fig/'data/shortlist_receptors.json').read_text()):
        a=r['representative'];parent=json.loads((ROOT/'prepared/ligands'/a['chembl_id']/'preparation.json').read_text())
        ligand=next(x for x in parent['states'] if x['state_id']==a['best_state_id']);ligand=dict(ligand,chembl_id=parent['chembl_id'],name=parent['name'])
        jobs.append((r['target'],r,ligand,42,str(folder)))
    assert len(jobs)==10
    atomic_json(folder/'registered_jobs.json',dict(purpose='poses for the ten preselected DrugCLIP figure representatives',exhaustiveness=32,seed=42,targets=[x[0] for x in jobs],jobs=[dict(target=x[0],receptor=x[1]['name'],state_id=x[2]['state_id'])for x in jobs]))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for r in pool.map(one,jobs):
            rows.append(r);atomic_json(folder/'results.json',rows);print(r['target'],r['status'],r.get('top_score_kcal_mol',r.get('error')),flush=True)
    atomic_json(folder/'progress.json',dict(status='complete' if all(r['status']=='complete'for r in rows)else 'finished_with_failures',completed=len(rows),successful=sum(r['status']=='complete'for r in rows),total_jobs=10))

if __name__ == '__main__':
    shortlist_main()
