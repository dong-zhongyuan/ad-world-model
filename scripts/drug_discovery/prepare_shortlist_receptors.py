#!/usr/bin/env python3
"""Prepare exact-state receptor inputs for shortlisted target-compound pairs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
if os.getenv('AD_SHORTLIST','1')!='1':
    from Bio.PDB import PDBParser
    import openmm as mm
    from openmm import app, unit
    from pdbfixer import PDBFixer
from candidate_gate import require_allowed, status

from paths import DRUG_WORK as ROOT
OUT = ROOT / 'prepared/receptors'
CPU = mm.Platform.getPlatformByName('CPU') if os.getenv('AD_SHORTLIST','1')!='1' else None


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def fit_rmsd(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    correction = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
    rotation = u @ correction @ vt
    return float(np.sqrt(np.mean(np.sum((a @ rotation - b)**2, axis=1))))


def medoids():
    parser = PDBParser(QUIET=True)
    paths = sorted((ROOT / 'inputs/structure_templates').glob('BIN1_1MV0_B_model_*_source_numbering.pdb'))
    assert len(paths) == 20
    arrays = [np.array([r['CA'].coord for r in parser.get_structure(p.stem, p)[0]['B'] if r.id[1] >= 409]) for p in paths]
    distances = np.array([[fit_rmsd(a, b) for b in arrays] for a in arrays])
    selected = [int(np.argmin(distances.sum(axis=1)))]
    while len(selected) < 3:
        selected.append(int(np.argmax(distances[:, selected].min(axis=1))))
    for _ in range(30):
        assignment = distances[:, selected].argmin(axis=1)
        updated = []
        for k, old in enumerate(selected):
            members = np.where(assignment == k)[0]
            updated.append(int(members[np.argmin(distances[np.ix_(members, members)].sum(axis=1))]))
        if updated == selected:
            break
        selected = updated
    return paths, selected, distances


def prepare(target, source, name, mutate=False):
    require_allowed(target)
    folder = OUT / target
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / (name + '.pdb')
    record_path = folder / (name + '.json')
    if record_path.exists() and output.exists():
        return json.loads(record_path.read_text())
    fixer = PDBFixer(filename=str(source), platform=CPU)
    if mutate:
        fixer.applyMutations(['LYS-465-GLU'], 'B')
    fixer.findMissingResidues()
    if fixer.missingResidues:
        raise ValueError('Unexpected missing residues; do not silently omit or model loops')
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=42)
    # Use OpenMM's public API: installed PDBFixer expects a newer private
    # hydrogen-definition helper than this preparation environment provides.
    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(app.ForceField('amber14-all.xml'), pH=7.4, platform=CPU)
    fixer.topology, fixer.positions = modeller.topology, modeller.positions
    pairs = sorted(tuple(sorted((int(a.residue.id), int(b.residue.id))))
                   for a,b in fixer.topology.bonds() if a.name == b.name == 'SG')
    if target == 'GRN':
        assert len(pairs) == 44 and len({x for p in pairs for x in p}) == 88
        uniprot = json.loads((ROOT / 'inputs/raw/GRN_P28799.json').read_text())
        annotated = {(f['location']['start']['value'], f['location']['end']['value']) for f in uniprot['features'] if f['type'] == 'Disulfide bond'}
        assert annotated <= set(pairs)
    ff = app.ForceField('amber14-all.xml', 'implicit/gbn2.xml')
    system = ff.createSystem(fixer.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    restraint = mm.CustomExternalForce('0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    restraint.addGlobalParameter('k', 1000.0)
    for n in ['x0','y0','z0']:
        restraint.addPerParticleParameter(n)
    original = np.array(fixer.positions.value_in_unit(unit.nanometer))
    heavy = []
    for atom in fixer.topology.atoms():
        if atom.element.symbol != 'H':
            heavy.append(atom.index)
            restraint.addParticle(atom.index, original[atom.index])
    system.addForce(restraint)
    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    simulation = app.Simulation(fixer.topology, system, integrator, CPU, {'Threads': '2'})
    simulation.context.setPositions(fixer.positions)
    simulation.minimizeEnergy(maxIterations=1000)
    state = simulation.context.getState(getPositions=True, getEnergy=True)
    energy = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    if not np.isfinite(energy):
        raise ValueError('Nonfinite minimized energy')
    pos = state.getPositions()
    displacement = (np.array(pos.value_in_unit(unit.nanometer)) - original) * 10
    rmsd = float(np.sqrt(np.mean(np.sum(displacement[heavy] ** 2, axis=1))))
    if rmsd > 2.0:
        raise ValueError('Preparation displaced heavy atoms more than 2 Angstrom')
    with output.open('w') as f:
        app.PDBFile.writeFile(fixer.topology, pos, f, keepIds=True)
    record = {'target': target, 'name': name, 'source': str(source.relative_to(ROOT)),
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'pdb': str(output.relative_to(ROOT)), 'canonical_E576_rebuilt': mutate,
              'protein_residues': len(list(fixer.topology.residues())), 'disulfide_pairs_source_numbering': pairs,
              'glycosylation_model': 'unglycosylated protein',
              'heavy_atom_preparation_rmsd_A': rmsd, 'preparation_energy_kj_mol': energy,
              'minimization': 'Amber14 + GBN2; heavy-atom restraint 1000 kJ/mol/nm^2; CPU',
              'candidate_gate': status(target)}
    write_json(record_path, record)
    print('PREPARED', target, name, 'atoms', fixer.topology.getNumAtoms(), 'rmsd_A', rmsd, flush=True)
    return record


def meeko(pdb, output):
    command = [sys.executable, str(Path(sys.executable).with_name('mk_prepare_receptor.py')),
               '--read_pdb', str(pdb), '-p', str(output)]
    with output.with_suffix('.meeko.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=log, check=True)
    if not output.exists() or not any(x.startswith('ATOM') for x in output.read_text().splitlines()):
        raise ValueError('Receptor PDBQT missing')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    if status('BIN1')['allowed']:
        paths, selected, distances = medoids()
        write_json(OUT / 'BIN1_ensemble_selection.json', {'all_models': len(paths), 'selected_models': [i+1 for i in selected],
                    'selection': 'three deterministic k-medoids of SH3 CA RMSD over all 20 deposited NMR models',
                    'pairwise_rmsd_A': distances.tolist()})
        checks = json.loads((ROOT / 'inputs/structure_checks.json').read_text())['BIN1']['templates']
        source = PDBParser(QUIET=True).get_structure('complex', ROOT / 'inputs/raw/BIN1_1MV0.pdb')
        for i in selected:
            name = f'BIN1_model_{i+1:02d}_canonical'
            record = prepare('BIN1', paths[i], name, mutate=True)
            peptide = np.array([a.coord for a in source[i]['A'].get_atoms() if a.element != 'H'])
            record['box_center_A'] = ((peptide.min(0) + peptide.max(0))/2).tolist()
            record['box_size_A'] = (peptide.max(0)-peptide.min(0)+16.0).tolist()
            record['box_basis'] = 'complete c-Myc partner envelope plus 8 A on every side'
            pdbqt = OUT / 'BIN1' / (name + '.pdbqt')
            meeko(ROOT / record['pdb'], pdbqt)
            record['pdbqt'] = str(pdbqt.relative_to(ROOT))
            records.append(record)
    if status('GRN')['allowed']:
        record = prepare('GRN', ROOT / 'inputs/structure_templates/GRN_P28799_mature_18_593_template.pdb', 'GRN_mature_relaxed')
        record['docking_ready'] = False
        record['next_stage'] = 'explicit-solvent apo MD ensemble of full mature protein; sample flexible C-terminal region'
        records.append(record)
    write_json(OUT / 'receptor_manifest.json', records)
    print('RECEPTOR PREPARATION COMPLETE', flush=True)


def shortlist_receptors():
    import pandas as pd
    from paths import FIGURE_WORK
    fig=FIGURE_WORK
    rows=pd.read_csv(fig/'data/representative_molecules.csv')
    pockets={p['pocket']:p for p in json.loads((ROOT/'results/drugclip_ten/input_identity.json').read_text())['pockets']}
    output=ROOT/'prepared/figure_shortlist';output.mkdir(exist_ok=True)
    records=[]
    for row in rows.to_dict('records'):
        target=row['target'];require_allowed(target);pocket=pockets[row['best_pocket']]
        source=ROOT/'inputs/ten_targets/structures'/target/(target+'_mature.pdb')
        assert hashlib.sha256(source.read_bytes()).hexdigest()==pocket['receptor_sha256']
        pdbqt=output/(target+'.pdbqt')
        if not pdbqt.exists():meeko(source,pdbqt)
        # Preserve every source heavy atom and its exact geometry; Meeko supplies typing and H.
        src={(l[21],l[22:27],l[12:16].strip()):np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])]) for l in source.read_text().splitlines() if l.startswith('ATOM') and l[76:78].strip()!='H'}
        dst={(l[21],l[22:27],l[12:16].strip()):np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])]) for l in pdbqt.read_text().splitlines() if l.startswith('ATOM') and l[12:16].strip()[0]!='H'}
        assert src.keys()==dst.keys(),(target,'Atom identities changed',len(src),len(dst))
        renames=[]
        for k,xyz0 in src.items():
            if np.linalg.norm(xyz0-dst[k])<.002:continue
            equivalent=[kk for kk,xyz1 in dst.items() if kk[:2]==k[:2] and kk[2][0]==k[2][0] and np.linalg.norm(xyz0-xyz1)<.002]
            assert len(equivalent)==1,(target,k,'Changed heavy atom geometry')
            kk=equivalent[0]
            allowed=[{'NH1','NH2'},{'OE1','OE2'},{'OD1','OD2'},{'O','OXT'},{'CD1','CD2'},{'CE1','CE2'}]
            assert any({k[2],kk[2]}==pair for pair in allowed),(target,k,kk)
            renames.append(dict(source=list(k),prepared=list(kk)))
        xyz=np.array(pocket['pocket_coordinates']);center=(xyz.min(0)+xyz.max(0))/2;size=xyz.max(0)-xyz.min(0)+12.
        record=dict(target=target,name=row['best_pocket'],pdbqt=str(pdbqt.relative_to(ROOT)),pdb=str(source.relative_to(ROOT)),box_center_A=center.tolist(),box_size_A=size.tolist(),source_sha256=pocket['receptor_sha256'],heavy_atom_geometry_preserved=True,equivalent_atom_renames=renames,pocket=pocket,representative=row)
        records.append(record);write_json(fig/'data/shortlist_receptors.json',records);print('READY',target,len(src),flush=True)

if __name__ == '__main__':
    shortlist_receptors()
