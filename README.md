# AD blood–brain world model

Code, analysis configurations, metadata and figure-source data for **A blood–brain world model nominates peripheral interventions for Alzheimer's disease**.

The repository covers the current main analysis only:

- a shared RNA–protein encoder and conditional neural ODE for blood-to-brain prediction;
- donor-held-out brain RNA and CITE-seq/ADT evaluation, with all registered model comparisons;
- virtual peripheral gene-clamp responses across six routes;
- member-level response networks and MultiXrank target prioritization;
- cis-pQTL Mendelian randomization and regional colocalization;
- ADNI blood-expression associations with longitudinal MMSE decline;
- six-model DrugCLIP screening and exact-state AutoDock Vina poses;
- source data and rendering code for the five main figures.

Historical training schedules, repair queues, debugging scripts, old side projects, raw expression matrices and model checkpoints are excluded. Public-data locations and file hashes are retained in `configs/` and `data/processed/`.

## Repository layout

```text
configs/                  model, comparison, MR and MultiXrank configurations
data/processed/           gene, protein, assay, donor-split and cell-count metadata
data/mappings/            orthogroup-to-human-member mapping
data/member_level/        molecular-coordinate classes used by the response network
docs/SCOPE_CONTRACT.json  model dimensions, coordinate order and scope checks
scripts/data/             public-data preparation and scope validation
scripts/step1_world_model/ model, training, inference and evaluation
scripts/step2_edges/      virtual perturbation and MultiXrank network construction
scripts/mr/               MR, allele harmonization and colocalization workflows
scripts/clinical/         ADNI endpoint construction and Cox/KM analysis
scripts/drug_discovery/   DrugCLIP, chemical-state preparation and exact-state docking
scripts/figures/          figure-input derivation and workflow graphics
figure_work/              main-figure source data, layout and R renderer
results/CURRENT_RESULTS.json  release checkpoint and analysis entry points
tests/                    model and MR test suites
```

## Python environment

Python 3.10 or newer is recommended. The main analysis used PyTorch 2.6.0.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

DrugCLIP additionally requires its Uni-Mol/Uni-Core implementation and the six official model checkpoints. Place the implementation at `work/drug_discovery/software/Drug-The-Whole-Genome/` and checkpoints under `work/drug_discovery/weights/model_weights/6_folds/`, or set `AD_DRUG_WORK` to the corresponding work directory. Pocket detection requires an fpocket executable at `work/drug_discovery/software/fpocket` unless the equivalent path is configured in the preparation script. AutoDock Vina and Meeko are covered by `requirements.txt`.

## Public data

The metadata retain the following resources:

- human blood: GSE181279;
- human brain: GSE138852 and GSE157827;
- mouse blood and brain: GSE329430, including BM1/2, BM3/4 and Brain1;
- paired PBMC RNA–ADT: GSE164378;
- longitudinal clinical outcomes and blood microarray expression: ADNI;
- protein instruments: UKB plasma proteomics and AGES–Reykjavik serum;
- AD outcomes: Bellenguez stage I (GCST90027158) and FinnGen R13 G6_ALZHEIMER;
- structures and compounds: AlphaFold, UniProt, ChEMBL and DrugCLIP resources.

`configs/download_manifest.json` records raw GSE157827 file URLs, sizes and hashes. `configs/data_manifest.json` records processed-matrix hashes. The expression matrices and `.pt` checkpoints are not stored in GitHub. After preparing the data, place the selected full-model checkpoint at `checkpoints/joint_full.pt` before running the registered perturbation workflow.

## Tests

```bash
pytest -q
```

The tests construct and backpropagate every registered comparison model, check protein-gradient routing, deterministic perturbation, coordinate masks, checkpoint round trips, MR allele harmonization, Wald estimates, colocalization and tabix region extraction.

## Main workflow

Prepare and validate the matrices:

```bash
python scripts/data/prepare_raw_intermediates.py --stage all
python scripts/data/prepare_full_data.py
python scripts/data/validate_full_scope.py
```

Train the full model:

```bash
python scripts/step1_world_model/train.py \
  --config configs/ablations/joint_full.json \
  --seed 42 --device cuda
```

Generate response edges and run MultiXrank:

```bash
python scripts/step2_edges/ko_edges.py \
  --root "$PWD" \
  --out results/network_v3_repair/edges \
  --columns data/member_level/columns.csv
python scripts/step2_edges/finalize_network.py --root "$PWD"
```

Run the MR workflow after downloading the corresponding public instruments and outcomes:

```bash
python scripts/mr/run_pipeline.py --root "$PWD" --stage all
python scripts/mr/run_public_followup.py --root "$PWD"
```

Run the ADNI analysis using locally authorized downloads. The scripts do not contain server-specific paths:

```bash
python scripts/clinical/run_adni_target_survival.py \
  --expression data/raw/adni/ADNI_Gene_Expression_Profile.zip \
  --mmse data/raw/adni/MMSE.csv \
  --diagnosis data/raw/adni/DXSUM.csv \
  --evidence figure_work/data/evidence_wheel.csv \
  --out results/adni_targets
python scripts/clinical/run_wheel_survival.py \
  --expression data/raw/adni/ADNI_Gene_Expression_Profile.zip \
  --mmse data/raw/adni/MMSE.csv \
  --diagnosis data/raw/adni/DXSUM.csv \
  --gates figure_work/data/evidence_gates.csv \
  --definitions figure_work/data/evidence_gate_definitions.json \
  --out results/adni_wheel
```

The primary analysis uses baseline blood samples (`bl`/`v03`) with visit-matched clinical records and subsequent MMSE measurements. The event is the first post-sampling MMSE decline of at least three points from the baseline reference score. Cox models adjust for baseline MMSE and diagnosis. The figure-ready `bst2_km_curve.csv` contains aggregate survival estimates and 95% intervals; participant-level ADNI records are not stored in the repository.

The drug-discovery scripts are ordered as follows:

1. `prepare_ligands.py`
2. `prepare_drugclip_ten_pockets.py`
3. `score_drugclip_ten.py`
4. `finalize_drugclip_ten.py`
5. `derive_fold_ranks.py`
6. `prepare_shortlist_receptors.py`
7. `dock_shortlist.py`

They use `work/drug_discovery/` by default. Derived rankings and pose summaries used in the article are already present in `figure_work/data/`, so the figure workflow does not require rerunning the structural screen.

## Main figures

The R renderer uses the included source data and `figure_work/page_layout.json`:

```bash
python scripts/figures/workflows_current.py
python scripts/figures/evidence_wheel.py
Rscript figure_work/render_current.R
```

The evidence wheel has ten rings, including the ADNI MMSE-association layer. To regenerate the aggregate KM source data after the clinical analysis, run `Rscript scripts/clinical/export_bst2_km.R --input results/adni_targets/analysis_cohort.csv --out figure_work/data`.

Required R packages are `ggplot2`, `cowplot`, `jsonlite`, `ggrepel`, `ggalluvial`, `ComplexHeatmap`, `circlize` and `survival`, together with their standard dependencies. `AD_WORLD_MODEL_ROOT` and `AD_FIGURE_WORK` can override the repository and figure-work paths. Rendering writes to `figure_work/build/`.

## Citation

If you use this repository, cite the associated manuscript and the original public-data and software resources listed above.
