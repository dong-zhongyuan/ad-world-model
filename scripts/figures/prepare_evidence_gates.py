"""Join existing decisions to exact primary assays; never refit or reselect MR."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

W = Path(__file__).resolve().parents[1]
D = W / 'data'
genes = pd.read_csv(D / 'mr_gene_union_with_hub_gate.csv')
genes = genes.sort_values('q_union').head(60).sort_values(['eligible_union_target', 'q_union'], ascending=[False, True])
mr = pd.read_csv(D / 'mr_mr_all_with_sensitivity.csv')
mr = mr[mr.primary.eq(True) & mr.gene.isin(genes.gene)]
coloc = pd.read_csv(D / 'mr_coloc_all.csv')
representatives = pd.read_csv(D / 'representative_molecules.csv').set_index('target')
docking = {x['target']: x for x in json.loads((D / 'docking_pose_results.json').read_text())}
layers = ['network', 'mr_bell', 'mr_finn', 'direction_outcome', 'direction_platform', 'coloc_bell', 'coloc_finn', 'drugclip', 'docking_qc']
rules = {
    'network': 'Existing complete blood-network rank in the top 50%, including ties.',
    'mr_bell': 'Exact primary assay Bellenguez q_joint_sources < 0.05; unchanged outcome-specific correction family.',
    'mr_finn': 'Exact primary assay FinnGen q_joint_sources < 0.05; unchanged outcome-specific correction family.',
    'direction_outcome': 'Both estimated MR coefficients for this exact assay have the same nonzero sign.',
    'direction_platform': 'For this protein, the primary AGES and UKB MR effects agree in sign in both outcomes. Missing comparable source/outcome is unresolved.',
    'coloc_bell': 'Exact primary assay Bellenguez completed colocalization PP_H4 >= 0.8; inherited from mr_union595/scripts/mr/finalize_union.py.',
    'coloc_finn': 'Exact primary assay FinnGen completed colocalization PP_H4 >= 0.8; same existing cutoff.',
    'drugclip': 'The molecule fixed for the structure figure is in the existing DrugCLIP top-20 parent shortlist for this target.',
    'docking_qc': 'The fixed molecule and microstate match the intended receptor; the docking job completed, has finite energies, and its pose chemical identity passed the existing check.',
}

def decide(value, predicate):
    return 'unresolved' if not np.isfinite(value) else ('pass' if predicate(value) else 'not_met')

rows = []
for gene in genes.itertuples():
    gm = mr[mr.gene.eq(gene.gene)]
    platform_signs = []
    complete_platforms = True
    for outcome in ['Bellenguez_stageI', 'FinnGen_R13']:
        z = gm[gm.status.eq('estimated') & gm.outcome.eq(outcome)]
        if not {'AGES', 'UKB'} <= set(z.exposure_source):
            complete_platforms = False
        else:
            signs = np.sign(z.beta_MR.to_numpy(float))
            platform_signs.append(bool(np.isfinite(signs).all() and np.all(signs == signs[0]) and signs[0] != 0))
    platform = 'pass' if complete_platforms and all(platform_signs) else ('not_met' if platform_signs and not all(platform_signs) else 'unresolved')
    for (source, assay), a in gm.groupby(['exposure_source', 'assay'], sort=True):
        r = dict(protein=gene.gene, assay=assay, source=source, primary=True, hub50=gene.hub_top50,
                 hub_top_percent=gene.hub_top_percent, q_union=gene.q_union,
                 network='pass' if gene.hub_top50 else 'not_met', direction_platform=platform)
        for key, outcome in [('bell', 'Bellenguez_stageI'), ('finn', 'FinnGen_R13')]:
            b = a[a.outcome.eq(outcome) & a.status.eq('estimated')]
            assert len(b) <= 1
            r[key + '_q'] = float(b.q_joint_sources.iloc[0]) if len(b) else np.nan
            r[key + '_beta'] = float(b.beta_MR.iloc[0]) if len(b) else np.nan
            r['mr_' + key] = decide(r[key + '_q'], lambda x: x < .05)
            c = coloc[coloc.gene.eq(gene.gene) & coloc.assay.eq(assay) & coloc.exposure_source.eq(source) & coloc.outcome.eq(outcome) & coloc.status.eq('complete') & coloc.primary.eq(True)]
            assert len(c) <= 1
            r[key + '_h4'] = float(c.PP_H4.iloc[0]) if len(c) else np.nan
            r['coloc_' + key] = decide(r[key + '_h4'], lambda x: x >= .8)
        r['direction_outcome'] = decide(r['bell_beta'] * r['finn_beta'], lambda x: x > 0)
        r.update(drugclip='unresolved', docking_qc='unresolved', molecule='', molecule_rank=np.nan, vina_score=np.nan)
        if gene.gene in representatives.index:
            rep = representatives.loc[gene.gene]
            r.update(molecule=rep.chembl_id, molecule_rank=int(rep['rank']), drugclip='pass' if rep['rank'] <= 20 else 'not_met')
            raw_rank = pd.read_csv(D / f'drugclip_{gene.gene}_all_parents.tsv', sep='\t')
            match = raw_rank[raw_rank.chembl_id.eq(rep.chembl_id)]
            assert len(match) == 1 and int(match['rank'].iloc[0]) == int(rep['rank'])
            dock = docking.get(gene.gene)
            if dock:
                identity = dock['chembl_id'] == rep.chembl_id and dock['state_id'] == rep.best_state_id and dock['receptor'] == rep.best_pocket
                passed = identity and dock['status'] == 'complete' and dock['pose_identity_verified'] and np.isfinite(dock['top_score_kcal_mol'])
                r.update(docking_qc='pass' if passed else 'not_met', vina_score=dock['top_score_kcal_mol'])
        r['passed_count'] = sum(r[key] == 'pass' for key in layers)
        r['unresolved_count'] = sum(r[key] == 'unresolved' for key in layers)
        rows.append(r)
matrix = pd.DataFrame(rows)
assert matrix.protein.nunique() == 60 and len(matrix) == 79
assert set(matrix[layers].stack()) <= {'pass', 'not_met', 'unresolved'}
assert (matrix.passed_count == matrix[layers].eq('pass').sum(axis=1)).all()
assert {'ACP5', 'PILRA'} <= set(matrix.loc[matrix.direction_platform.eq('not_met'), 'protein'])
matrix.to_csv(D / 'evidence_gates.csv', index=False)
(D / 'evidence_gate_definitions.json').write_text(json.dumps(dict(layers=layers, rules=rules,
    display_unit='Exact primary assay within protein sector; 60 genes ranked by existing q_union for display, 79 primary assays; discovery definitions unchanged.',
    outer_bars='Number of passed criteria, from 0 to 9; not a weighted evidence score.',
    docking_qc='Pose identity/computation quality control, not an affinity or binding confirmation.',
    source_family='391 instrument-ready genes; no change to nomination or multiple testing.'), indent=2))
print(matrix[['protein','source','passed_count','unresolved_count','direction_platform']].to_string(index=False))
