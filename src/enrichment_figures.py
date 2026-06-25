from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import load_config


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_annotated_candidates(results_dir: Path, figures_dir: Path, top_n: int = 20) -> None:
    path = results_dir / "candidate_gene_scores_annotated.csv"
    if not path.exists():
        return
    data = pd.read_csv(path)
    data["gene_label"] = data["gene_symbol"].fillna(data["gene_id"]).astype(str)
    top = data.sort_values("candidate_score", ascending=False).head(top_n).copy()
    top["direction"] = np.where(
        top["acute_vs_control_log_delta"].abs() >= top["p53DN_vs_p53WT_log_delta"].abs(),
        "CENP-A acute/control",
        "p53 DN/WT",
    )
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    sns.barplot(
        data=top,
        y="gene_label",
        x="candidate_score",
        hue="direction",
        palette={"CENP-A acute/control": "#C75D4D", "p53 DN/WT": "#4169A8"},
        dodge=False,
        ax=ax,
    )
    ax.set_title("Annotated candidate regulatory genes")
    ax.set_xlabel("|acute-control| + |p53DN-p53WT| log-expression delta")
    ax.set_ylabel("Gene symbol")
    ax.legend(title="Dominant contrast", loc="lower right", frameon=False)
    _save(fig, figures_dir / "candidate_gene_scores_annotated.png")


def plot_enrichment_terms(results_dir: Path, figures_dir: Path, top_n: int = 15) -> None:
    path = results_dir / "candidate_enrichment_gprofiler.csv"
    if not path.exists():
        return
    data = pd.read_csv(path)
    data = data[data["significant"].astype(bool)].copy()
    if data.empty:
        return
    data["minus_log10_p"] = -np.log10(data["p_value"].clip(lower=np.finfo(float).tiny))
    top = data.sort_values("p_value", ascending=True).head(top_n).copy()
    top["term_label"] = [
        "\n".join(textwrap.wrap(f"{name} ({source})", width=46))
        for name, source in zip(top["name"].astype(str), top["source"].astype(str))
    ]
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    sns.scatterplot(
        data=top,
        x="minus_log10_p",
        y="term_label",
        size="intersection_size",
        hue="source",
        palette={"GO:BP": "#3C7D6B", "REAC": "#9A5BA5"},
        sizes=(70, 360),
        edgecolor="#263238",
        linewidth=0.5,
        ax=ax,
    )
    ax.set_title("g:Profiler enrichment of top candidate genes")
    ax.set_xlabel("-log10 adjusted p-value")
    ax.set_ylabel("")
    ax.legend(title="Source / overlap", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    _save(fig, figures_dir / "candidate_enrichment_terms.png")


def generate_enrichment_figures(config_path: str) -> None:
    config = load_config(config_path)
    results_dir = Path(config["paths"]["results_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    plot_annotated_candidates(results_dir, figures_dir)
    plot_enrichment_terms(results_dir, figures_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    generate_enrichment_figures(args.config)


if __name__ == "__main__":
    main()
