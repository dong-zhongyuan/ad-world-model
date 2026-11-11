# Rebuilding the training inputs

Use the full analysis requirements in the repository root. From the [data-source index](data_sources.md), download the original files for GSE138852, GSE157827, GSE164378, GSE181279 and GSE329430 into `data/raw/<accession>/`. Extract the 10x triples for GSE181279 and GSE157827 there, and keep the deposited sample-prefixed filenames. For GSE164378, retain the original TAR and the 3P metadata CSV. For GSE329430 use the source count matrices, BM1/BM2 assignment files and the four Brain1 sample H5 files. [Filenames and checksums](../data/reference/input_checksums.csv) identify the exact inputs.

Download the [training reference bundle](https://github.com/dong-zhongyuan/ad-world-model/releases/download/v3.0.1/training_reference_inputs.tar.gz). It supplies the original ESM-2 node embeddings and repairs, gene and multiplex mappings, BM3/BM4 assignments, and the mouse cell annotations used in training. Verify the digest before extracting at the repository root:

```bash
python -c "import hashlib, pathlib; assert hashlib.sha256(pathlib.Path('training_reference_inputs.tar.gz').read_bytes()).hexdigest() == '3dbdf8b7783dbc0d50c6820b9b53370be3ddc63467277ad6beb2bb694fececdb'"
tar -xzf training_reference_inputs.tar.gz
python scripts/data/build_reference_atlases.py
python scripts/data/prepare_raw_intermediates.py --stage all
python scripts/data/prepare_full_data.py
python scripts/data/validate_full_scope.py
python scripts/data/verify_training_snapshot.py
```

The first command after extraction reconstructs five expression atlases from counts on the released coordinate list. It checks the blood row IDs, donors and marker labels before joining the archived Leiden annotations. [Reference annotation programs](../scripts/data/reference_builders) document the original cell labels and HTO demultiplexing. The second stage prepares the expanded mouse-brain atlas, cortex and same-cell RNA/ADT data, including marker assignments, cell QC, normalization and donor splits. The final transformation constructs the CSR matrices, donor metadata, assay masks, centered ADT targets, ESM-2 features and training-only standardization. The [release manifest](../configs/data_manifest.json) records the original matrix shapes and hashes.

## Transformations and verification environment

The five reference matrices start from the deposited counts and fixed molecular coordinates. The original mouse BM labels are refined on the mapped CD4, CD8A and CD8B expression values. The two-stage implementation and the frozen human blood labels preserve the training annotations. For the expanded Brain1, GSE157827 and CITE RNA atlases, repeated upper-case symbols are summed. Cells must have at least 500 counts and 200 detected genes and a mitochondrial fraction below 0.10. Counts are scaled to 10,000 per cell and log1p transformed. GSE157827 and expanded Brain1 marker assignments require at least two detected markers and a winning score margin of 0.10. The CITE cohort uses its deposited cell annotations, with RNA and ADT measurements joined by barcode after doublet exclusion.

Assay masks record the measured coordinates. Human training donors contribute equally within assay to gene means and squared moments on available coordinates. Protein means and variances use the four training CITE donors; the variance floor is 0.0001. ESM-2 features use the archived per-node vectors and corrections, zero vectors for uncovered nodes, and column-wise L2 scaling with a floor of one. The processed-matrix manifest covers CSR data and indices, observations, coordinate labels, protein targets, embeddings, assay masks and preprocessing parameters.

The raw-count rebuild used Linux x86-64, Python 3.10.20, NumPy 1.26.4, pandas 2.2.3, SciPy 1.15.3, scikit-learn 1.7.2, Scanpy 1.11.5, AnnData 0.10.8, h5py 3.14.0 and PyTorch 2.4.0. The lightweight inference example uses the separate environment in `requirements-benchmark.txt`.

The rebuilt 36 files matched all SHA-256 values in the frozen manifest. The scope validator retained all 25,790 nodes and all 348 active training strata.
