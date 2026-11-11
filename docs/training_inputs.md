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
```

The first command after extraction reconstructs five expression atlases from counts on the released coordinate list. It checks the blood row IDs, donors and marker labels before joining the archived Leiden annotations. [Reference annotation programs](../scripts/data/reference_builders) document the original cell labels and HTO demultiplexing. The second stage prepares the expanded mouse-brain atlas, cortex and same-cell RNA/ADT data, including marker assignments, cell QC, normalization and donor splits. The final transformation constructs the CSR matrices, donor metadata, assay masks, centered ADT targets, ESM-2 features and training-only standardization. The [release manifest](../configs/data_manifest.json) records the original matrix shapes and hashes.
