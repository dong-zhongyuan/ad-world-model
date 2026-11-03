# AD blood–brain world model

Code, analysis configurations, metadata and figure-source data for **A blood–brain world model translates peripheral interventions into brain-cell programs in Alzheimer's disease**.

The repository covers the current main analysis only:

- a shared RNA–protein encoder and conditional neural ODE for blood-to-brain prediction;
- donor-held-out brain RNA and CITE-seq/ADT evaluation, with all registered model comparisons;
- virtual peripheral gene-clamp responses and measured CD4/myeloid Perturb-seq actions;
- same-animal peripheral-to-microglial dynamic adaptation with processing-held-out tests;
- member-level response networks and MultiXrank target prioritization;
- cis-pQTL Mendelian randomization, regional colocalization and independent GRN/BIN1 brain responses;
- six-model DrugCLIP screening and exact-state AutoDock Vina poses;
- source data and rendering code for the six main figures.

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
scripts/real_action/      CD4/myeloid guide assignment, action matrices and propagation
scripts/dynamic_response/ animal-level LPS adaptation, GRN/BIN1 brain analyses and source tables
scripts/mr/               MR, allele harmonization and colocalization workflows
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
- primary CD4 Perturb-seq: GSE314342;
- primary myeloid Perturb-seq: GSE327644;
- peripheral LPS brain response: GSE245309;
- aged Grn-knockout microglia: GSE272377;
- microglial Bin1 conditional knockout: Mendeley Data DOI 10.17632/whgkcy72yv.1;
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

The drug-discovery scripts are ordered as follows:

1. `prepare_ligands.py`
2. `prepare_drugclip_ten_pockets.py`
3. `score_drugclip_ten.py`
4. `finalize_drugclip_ten.py`
5. `derive_fold_ranks.py`
6. `prepare_shortlist_receptors.py`
7. `dock_shortlist.py`

They use `work/drug_discovery/` by default. Derived rankings and pose summaries used in the article are already present in `figure_work/data/`, so the figure workflow does not require rerunning the structural screen.

## Real-action and dynamic-response extension

`scripts/real_action/` contains guide assignment, orthogroup count aggregation, measured-action propagation and candidate response tables. `scripts/dynamic_response/` contains the fixed peripheral-plus-world ridge adapter, reciprocal animal holdout evaluation, headline-metric calculation, GRN/BIN1 brain analyses and figure-source export. These reference implementations retain the release analysis-root layout; place them in the directory containing the corresponding `data/` and `results/` resources before running.

The fixed combined model uses a ridge penalty of 1.0 and is evaluated in both processing-transfer directions. Its reported 53.28% MSE reduction, 0.740 response correlation and 61.25% top-100 recovery are generated from the held-out-animal result tables rather than hardcoded figure labels.

## Main figures

The current six-figure extension uses `make_real_action_figure.py`, `make_dynamic_figure.py`, `make_genetics_brain_panel.py` and `assemble_six_figures.py` in `scripts/figures/`. The original R renderer uses the included source data and `figure_work/page_layout.json`:

```bash
Rscript figure_work/render_current.R
```

Required R packages are `ggplot2`, `cowplot`, `jsonlite`, `ggrepel`, `ggalluvial`, `ComplexHeatmap` and `circlize`, together with their standard dependencies. `AD_WORLD_MODEL_ROOT` and `AD_FIGURE_WORK` can override the repository and figure-work paths. Rendering writes to `figure_work/build/`.

## Citation

If you use this repository, cite the associated manuscript and the original public-data and software resources listed above.
