# Predicting microglial responses from phagocyte expression

The test data are eight mouse brain CD45+ samples from [GSE245309](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE245309). Each tissue-processing group contains two LPS and two saline animals. Each group is held out in turn. Its ridge readout is fitted only on the other four animals. The frozen mouse Mono-to-Micro route uses the CTRL disease condition throughout.

## Inputs and outputs

- [Selected source cells](../results/dynamic_response/benchmark_inputs/selected_world_inputs.npz) contain the ordered 25,790 coordinates and one source-expression matrix per GEO sample. The seed-42 subset contains up to 128 phagocytes per sample.
- [Population means](../results/dynamic_response/benchmark_inputs/state_means.npz) contain phagocyte means, microglial means, microglial detection frequencies and archived frozen-model outputs. The benchmark recomputes model outputs from the selected cells. Test microglial means supply the observed outcome for scoring.
- The [release](https://github.com/dong-zhongyuan/ad-world-model/releases/tag/v3.0.1) supplies the 1,500-step inference checkpoint, its model definition and the two training-fold ridge heads. The downloader checks SHA-256 before installation. The benchmark verifies the four input files, checks coordinate order, recomputes the frozen backbone, refits the heads and verifies all animal-level metrics against the frozen reference.
- The reported reduction is one minus the ratio of mean MSEs across four LPS animals. Predicted and observed changes each use the two saline references in the same held-out processing group. Correlation and top-100 recovery are averaged over animals. The shared evaluation set contains 11,330 orthogroups.
- The output directory contains individual predictions and observations in `scene/dynamic_response_vectors.npz`, per-animal scores in `adaptation/metrics_by_test_animal.csv` and aggregate results in `headline/headline_metrics.json`. Small CPU floating-point differences are tolerated during recomputation. Counts, input hashes, sample identities and coordinate order are checked separately.

## Starting from the deposited counts

Download the [GSE245309 archive](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE245nnn/GSE245309/suppl/GSE245309_RAW.tar) and extract its feature, matrix and barcode files into one directory. From a checkout with the benchmark requirements and released model installed, run:

```bash
python scripts/dynamic_response/prepare_benchmark_inputs.py --raw /path/to/extracted/GSE245309 --out work/reconstructed_inputs
```

The program uses RNA counts with at least 200 detected genes, positive library size and at most 20% mitochondrial counts per cell. A two-component mixture on pooled microglial marker scores assigns microglia. A second mixture on monocyte marker scores among the remaining cells is fitted separately in each sample. Counts are summed on the model coordinates and log1p-transformed after normalization to 10,000 counts. The pooled detection mask, fixed source-cell subset and frozen model roles match the benchmark. The program checks both generated arrays against the released inputs, including every sample, coordinate and observed target. It writes sample counts to `sample_cell_counts.csv`.
