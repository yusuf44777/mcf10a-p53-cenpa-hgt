# Reproducibility Notes

## Environment

Core pipeline:

```bash
python3 -m pip install -r requirements.txt
python3 -m src.run_pipeline --config configs/default.json
```

Optional HGT:

```bash
python3 -m pip install torch-geometric
python3 -m src.train_hgt --config configs/default.json
python3 -m src.train_homogeneous_gnn --config configs/default.json
python3 -m src.post_hgt_analysis --config configs/default.json
```

Optional external annotation:

```bash
python3 -m src.external_annotation --config configs/default.json
python3 -m src.enrichment_figures --config configs/default.json
```

This command sends the candidate Ensembl IDs to Ensembl REST and g:Profiler. Run it only when external data export is acceptable for the project.

## Data

The pipeline uses the supplied dense H5AD file:

`dataset/EMTAB9861_Human_MCF10-2a_CENPAandTP53perturb_Y2021_9201Cells_FromAuthors_Counts_Jeffery_Almouzni.h5ad`

Expected integrity checks:

- 9,201 cells
- 40,295 genes
- 6 samples
- 2 p53 states
- 3 CENPA states

## Outputs

- `results/dataset_summary.json`
- `results/preprocessed.npz`
- `results/baseline_metrics.csv`
- `results/hetero_graph.npz`
- `results/gene_modules.csv`
- `results/candidate_gene_scores.csv`
- `results/candidate_gene_scores_annotated.csv` if external annotation is run
- `results/candidate_enrichment_gprofiler.csv` if external enrichment is run
- `figures/*.png`
- `paper/main.tex`

## LaTeX

No local TeX compiler was detected during planning. To compile:

```bash
cd paper
latexmk -pdf main.tex
```

Tectonic is also supported:

```bash
cd paper
TECTONIC_CACHE_DIR=../.tectonic-cache tectonic main.tex
```
