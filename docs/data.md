# Data Files

The GitHub repository intentionally excludes the raw local dataset because the
files are large enough to exceed normal GitHub limits.

Expected local files:

- `dataset/EMTAB9861_Human_MCF10-2a_CENPAandTP53perturb_Y2021_9201Cells_FromAuthors_Counts_Jeffery_Almouzni.h5ad`
- `dataset/cell_annotation.csv`
- `dataset/raw_counts_MCF10-2A_p53onoff_CENPAoverexpression_EMTAB9861.csv`

Place these files under `dataset/` before running:

```bash
python3 -m src.run_pipeline --config configs/default.json
```

Generated lightweight CSV/JSON summaries are tracked so the manuscript figures
and reported metrics remain inspectable without committing the raw matrix.
