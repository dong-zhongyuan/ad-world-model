# AD blood–brain world model

Code, analysis configurations, metadata and figure-source data for **A blood–brain world model nominates peripheral interventions for Alzheimer's disease**.

- **69.05% lower microglial response MSE.** Across four held-out LPS animals, a ridge readout combining brain-isolated phagocyte expression and frozen world-model predictions reduced mean squared error relative to a zero-response predictor. [Held-out results](results/dynamic_response/headline/headline_by_method.csv)
- **2.08-fold enrichment for human AD genetic evidence.** After accounting for linkage disequilibrium, 15 of 130 model-nominated genes (11.5%) and 7 of 126 blood differential-expression candidates (5.6%) ranked in the top decile of Bellenguez protein–AD genetic evidence. [Gene-level scores and list membership](figure_work/data/gene_scores_LD_component_representatives_Bellenguez_stageI.csv)

## Run the published test dataset

The [eight GSE245309 test inputs](results/dynamic_response/benchmark_inputs) ship with the checkout. The following commands run the frozen model on CPU and evaluate both processing-held-out folds. Start in a new directory with Python 3.11.

```bash
git clone https://github.com/dong-zhongyuan/ad-world-model.git
cd ad-world-model
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-benchmark.txt
python scripts/dynamic_response/fetch_model.py
AD_OMICS_RESULTS=work/benchmark python scripts/dynamic_response/benchmark.py
```

**Expected output:** `OFFICIAL_V3_DYNAMIC_BENCHMARK_VERIFIED`. In `work/benchmark/headline/headline_metrics.json`, the four LPS animals give a **69.05% reduction in MSE** against zero response, a **0.835 mean response correlation** and **61.5% mean top-100 recovery**. The 95% animal-resampling interval for the MSE reduction is 60.01–76.27%. Per-animal metrics and gene-level predictions are written alongside the summary. [Input schema, raw-count reconstruction and outputs](docs/prediction_benchmark.md).

## Analysis scope

- a shared RNA–protein encoder and conditional neural ODE for blood-to-brain prediction;
- donor-held-out brain RNA and CITE-seq/ADT evaluation, with all registered model comparisons;
- virtual peripheral gene-clamp responses and measured CD4/myeloid Perturb-seq actions;
- brain-isolated phagocyte-to-microglia dynamic adaptation with processing-held-out tests;
- member-level response networks and MultiXrank target prioritization;
- cis-pQTL Mendelian randomization, regional colocalization and independent GRN/BIN1 brain responses;
- six-model DrugCLIP screening and exact-state AutoDock Vina poses;
- source data and rendering code for the six main figures.

The [versioned release](https://github.com/dong-zhongyuan/ad-world-model/releases/tag/v3.0.1) includes the frozen inference model and training references. Public test matrices, source-data tables and file checksums are available in this checkout.

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

The [dataset index](docs/data_sources.md) links the training atlases, perturbation measurements, independent brain cohorts, genetic outcomes and structural resources. It explains outcome definitions, sample annotations and data versions. [Source Data 1 and 2](results/source_data/README.md) include the complete human and mouse comparison tables and biological-unit uncertainty. The [training-input guide](docs/training_inputs.md) gives the raw-count reconstruction commands and reference bundle.

## Tests

```bash
pytest -q
```

The tests construct and backpropagate every registered comparison model, check protein-gradient routing, deterministic perturbation, coordinate masks, checkpoint round trips, MR allele harmonization, Wald estimates, colocalization and tabix region extraction.

## Main workflow

After downloading the data and reference bundle described in the [training-input guide](docs/training_inputs.md), prepare and validate the matrices:

```bash
python scripts/data/build_reference_atlases.py
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

## Main figures

[Figure sources](figure_work/main_grid/source) contain the fixed input tables, workflow artwork and vector masters used to compose the six figures. From the repository root:

```bash
cd figure_work/main_grid
Rscript design_figures.R .
python normalize_pdfs.py
python reflow_figures.py
python space_panel_a.py output output
```

The final PDFs are written to `figure_work/main_grid/output/`. Figures 3 and 4 use R with ggplot2, gtable, dplyr, tidyr, jsonlite, png and ragg. The PDF composition uses pypdf, PyMuPDF, NumPy and Arial Bold. The version of R and its packages is recorded in [R_sessionInfo.txt](figure_work/main_grid/R_sessionInfo.txt).

## Citation

If you use this repository, cite the associated manuscript and the original public data and software resources listed above.
