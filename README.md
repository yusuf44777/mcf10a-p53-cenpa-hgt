# HGT Modeling of p53 and CENP-A Regulation in MCF10A Cells

This repository turns the available MCF10A p53/CENPA single-cell expression dataset into a reproducible manuscript and analysis pipeline for:

- dataset integrity checks,
- scRNA-seq preprocessing without requiring Scanpy,
- k-means and supervised baselines,
- heterogeneous graph construction with cell, gene, and module nodes,
- optional PyTorch Geometric HGT training, and
- LaTeX manuscript generation.

Raw local data files are intentionally not committed because they are large. See
`docs/data.md` for the expected dataset filenames.

## Quick Start

```bash
python3 -m src.run_pipeline --config configs/default.json
python3 -m pytest
```

The local machine currently has PyTorch but not `torch_geometric`; the pipeline still produces baseline metrics, graph artifacts, figures, and LaTeX sources. Install the optional dependency to train the full HGT model:

```bash
python3 -m pip install torch-geometric
python3 -m src.train_hgt --config configs/default.json
python3 -m src.train_homogeneous_gnn --config configs/default.json
python3 -m src.post_hgt_analysis --config configs/default.json
```

Optional external annotation is implemented but intentionally not run automatically because it sends candidate gene IDs to third-party services:

```bash
python3 -m src.external_annotation --config configs/default.json
python3 -m src.enrichment_figures --config configs/default.json
```

LaTeX sources are under `paper/`. If no TeX distribution is installed, compile on a machine with TinyTeX, MacTeX, or TeX Live:

```bash
cd paper
latexmk -pdf main.tex
```

This workspace also supports Tectonic:

```bash
cd paper
TECTONIC_CACHE_DIR=../.tectonic-cache tectonic main.tex
```
