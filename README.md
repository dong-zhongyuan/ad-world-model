# AD blood–brain world model

Code, analysis configurations, metadata and figure-source data for **A blood–brain world model nominates peripheral interventions for Alzheimer's disease**.

- **69.05% lower microglial response MSE.** Across four held-out LPS animals, a ridge readout combining brain-isolated phagocyte expression and frozen world-model predictions reduced mean squared error relative to a zero-response predictor. [Held-out results](results/dynamic_response/headline/headline_by_method.csv)
- **2.08-fold enrichment for human AD genetic evidence.** After accounting for linkage disequilibrium, 15 of 130 model-nominated genes (11.5%) and 7 of 126 blood differential-expression candidates (5.6%) ranked in the top decile of Bellenguez protein–AD genetic evidence. [Gene-level scores and list membership](figure_work/data/gene_scores_LD_component_representatives_Bellenguez_stageI.csv)

## Analysis scope

- a shared RNA–protein encoder and conditional neural ODE for blood-to-brain prediction;
- donor-held-out brain RNA and CITE-seq/ADT evaluation, with all registered model comparisons;
- virtual peripheral gene-clamp responses and measured CD4/myeloid Perturb-seq actions;
- brain-isolated phagocyte-to-microglia dynamic adaptation with processing-held-out tests;
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
- brain CD45-positive cells after systemic saline or LPS: GSE245309;
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

## Dynamic response benchmark

The frozen 1,500-step model and gene-wise ridge readouts (penalty 1.0) were evaluated across reciprocal tissue-processing groups. The readouts use the source phagocyte state and the frozen model prediction. The four held-out LPS animals gave a mean response correlation of 0.835 and 61.50% top-100 gene recovery. [Per-animal results](results/dynamic_response/headline/combined_by_test_animal.csv)

Recompute the headline from the published per-animal comparison table:

```bash
python scripts/dynamic_response/summarize_dynamic_headline.py
```

For full inference, place the separately provisioned model bundle, benchmark inputs and GO gene-set file at the locations in [the benchmark configuration](configs/dynamic_response.json), then run `python scripts/dynamic_response/benchmark.py`. It verifies the frozen inputs, refits the two training-fold readouts and regenerates the response vectors and enrichment tables.

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
