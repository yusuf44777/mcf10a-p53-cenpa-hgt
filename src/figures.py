from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .graph_build import HeteroGraphArtifact
from .preprocess import PreprocessedData


PALETTE = {
    "p53-WT|control": "#3C7D6B",
    "p53-WT|acute_oe": "#79A73A",
    "p53-WT|chronic_oe": "#B89B27",
    "p53-DN|control": "#4169A8",
    "p53-DN|acute_oe": "#9A5BA5",
    "p53-DN|chronic_oe": "#C75D4D",
}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_workflow(figures_dir: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.axis("off")
    steps = [
        ("Raw H5AD", "9,201 cells\n40,295 genes"),
        ("Preprocess", "library normalize\nlog1p, HVG, PCA"),
        ("Hetero graph", "cell, gene,\nmodule nodes"),
        ("Models", "k-means baselines\nmultitask HGT"),
        ("Interpret", "embeddings,\nmodules, candidates"),
    ]
    xs = np.linspace(0.08, 0.92, len(steps))
    colors = ["#F0F3F6", "#E2F0EA", "#E8EDF8", "#F7EBD9", "#F7E4E0"]
    for i, ((title, body), x, color) in enumerate(zip(steps, xs, colors)):
        ax.add_patch(
            plt.Rectangle((x - 0.075, 0.38), 0.15, 0.34, facecolor=color, edgecolor="#263238", lw=1.1)
        )
        ax.text(x, 0.625, title, ha="center", va="center", fontsize=10.5, fontweight="bold")
        ax.text(x, 0.495, body, ha="center", va="center", fontsize=8.8)
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.09, 0.55),
                xytext=(x + 0.09, 0.55),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#263238"),
            )
    ax.text(0.5, 0.18, "Reproducible workflow from expression matrix to graph-based regulatory interpretation", ha="center", fontsize=11)
    _save(fig, Path(figures_dir) / "workflow.png")


def plot_condition_landscape(data: PreprocessedData, figures_dir: str | Path) -> None:
    obs = data.obs.copy()
    obs["PC1"] = data.x_pca[:, 0]
    obs["PC2"] = data.x_pca[:, 1]
    obs["condition"] = obs["p53_status"].astype(str) + "|" + obs["CENPA_status"].astype(str)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    sns.scatterplot(
        data=obs,
        x="PC1",
        y="PC2",
        hue="condition",
        palette=PALETTE,
        s=12,
        alpha=0.72,
        linewidth=0,
        ax=ax,
    )
    ax.set_title("Condition structure in PCA space")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    _save(fig, Path(figures_dir) / "condition_landscape.png")


def plot_graph_schema(figures_dir: str | Path) -> None:
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(["cell", "gene", "module"])
    directed_edges = [
        ("cell", "gene", "expresses"),
        ("gene", "cell", "expressed by"),
        ("gene", "module", "in module"),
        ("module", "gene", "contains"),
    ]
    for src, dst, label in directed_edges:
        graph.add_edge(src, dst, label=label)
    pos = {"cell": (0, 0), "gene": (1.7, 0), "module": (0.85, 1.05)}
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=["#DDEFE7", "#E5EAF7", "#F5E6D1"],
        edgecolors="#263238",
        node_size=2600,
        linewidths=1.2,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, pos, font_size=12, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(graph, pos, arrowstyle="-|>", arrowsize=18, width=1.1, connectionstyle="arc3,rad=0.12", ax=ax)
    label_pos = {
        ("cell", "gene"): (0.85, -0.13),
        ("gene", "cell"): (0.85, 0.18),
        ("gene", "module"): (1.30, 0.62),
        ("module", "gene"): (1.06, 0.84),
    }
    for src, dst, label in directed_edges:
        x, y = label_pos[(src, dst)]
        ax.text(x, y, label, fontsize=8.7, ha="center", va="center", color="#37474F")
    badges = [
        (-0.08, -0.44, "cell-cell: near"),
        (1.78, -0.44, "gene-gene: coexpressed"),
    ]
    for x, y, label in badges:
        ax.text(
            x,
            y,
            label,
            fontsize=9,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#F0F3F6", edgecolor="#90A4AE", linewidth=0.8),
        )
    ax.set_title("Heterogeneous graph schema")
    ax.set_xlim(-0.45, 2.15)
    ax.set_ylim(-0.62, 1.38)
    ax.axis("off")
    _save(fig, Path(figures_dir) / "hetero_graph_schema.png")


def plot_baseline_comparison(metrics: pd.DataFrame, figures_dir: str | Path) -> None:
    plot_df = metrics[metrics["target"] == "condition"].copy()
    plot_df = plot_df.sort_values("macro_f1", ascending=False)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    sns.barplot(data=plot_df, y="model", x="macro_f1", color="#4169A8", ax=ax)
    ax.set_xlim(0, max(0.1, min(1.0, plot_df["macro_f1"].max() * 1.15)))
    ax.set_title("Condition classification baseline comparison")
    ax.set_xlabel("Macro-F1")
    ax.set_ylabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)
    _save(fig, Path(figures_dir) / "baseline_comparison.png")


def plot_module_summary(graph: HeteroGraphArtifact, figures_dir: str | Path) -> None:
    table = graph.module_table.sort_values("n_genes", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    sns.scatterplot(
        data=table,
        x="mean_detected_fraction",
        y="mean_var",
        size="n_genes",
        hue="n_genes",
        palette="viridis",
        sizes=(45, 420),
        edgecolor="#263238",
        linewidth=0.4,
        ax=ax,
    )
    ax.set_title("Expression-derived gene modules")
    ax.set_xlabel("Mean detected fraction")
    ax.set_ylabel("Mean log-expression variance")
    ax.legend(title="Genes", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    _save(fig, Path(figures_dir) / "gene_module_summary.png")


def plot_umap(data: PreprocessedData, figures_dir: str | Path) -> None:
    obs = data.obs.copy()
    obs["UMAP1"] = data.x_umap[:, 0]
    obs["UMAP2"] = data.x_umap[:, 1]
    obs["condition"] = obs["p53_status"].astype(str) + "|" + obs["CENPA_status"].astype(str)
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    sns.scatterplot(
        data=obs,
        x="UMAP1",
        y="UMAP2",
        hue="condition",
        palette=PALETTE,
        s=10,
        alpha=0.70,
        linewidth=0,
        ax=ax,
    )
    ax.set_title("UMAP of MCF10A cells colored by perturbation condition")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    _save(fig, Path(figures_dir) / "umap_condition.png")


def plot_confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: dict[str, int],
    model_name: str,
    figures_dir: str | Path,
) -> None:
    labels_order = sorted(label_map, key=label_map.get)
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels_order,
        yticklabels=labels_order,
        ax=ax,
        linewidths=0.4,
        linecolor="#e0e0e0",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Fraction of true class"},
    )
    ax.set_title(f"Confusion matrix — {model_name} (6-condition, normalized)")
    ax.set_xlabel("Predicted condition")
    ax.set_ylabel("True condition")
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    safe_name = model_name.replace(" ", "_").lower()
    _save(fig, Path(figures_dir) / f"confusion_matrix_{safe_name}.png")


def plot_label_efficiency(efficiency_df: pd.DataFrame, figures_dir: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for model_name, grp in efficiency_df.groupby("model"):
        ax.plot(
            grp["label_fraction"] * 100,
            grp["macro_f1"],
            marker="o",
            label=model_name,
            linewidth=1.8,
        )
    ceiling = float(efficiency_df["macro_f1"].max())
    ax.axhline(y=ceiling, color="gray", linestyle="--", linewidth=0.8, label=f"100% ceiling ({ceiling:.3f})")
    ax.set_xlabel("Labeled fraction (%)")
    ax.set_ylabel("Macro-F1 (6-condition, held-out test set)")
    ax.set_title("Label efficiency: condition classification vs. labeled data fraction")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    _save(fig, Path(figures_dir) / "label_efficiency_curve.png")


def plot_embedding_quality(quality_df: pd.DataFrame, figures_dir: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))
    for ax, metric, title in zip(
        axes,
        ["silhouette", "calinski_harabasz"],
        ["Silhouette score (higher = better)", "Calinski-Harabasz index (higher = better)"],
    ):
        sns.barplot(data=quality_df, y="embedding", x=metric, color="#4169A8", ax=ax)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(metric.replace("_", "-").capitalize())
        ax.set_ylabel("")
        fmt = "%.4f" if metric == "silhouette" else "%.0f"
        for container in ax.containers:
            ax.bar_label(container, fmt=fmt, padding=2, fontsize=8)
    fig.suptitle("Embedding quality comparison across representation methods", fontsize=11)
    plt.tight_layout()
    _save(fig, Path(figures_dir) / "embedding_quality_comparison.png")


def plot_candidate_genes(data: PreprocessedData, figures_dir: str | Path) -> pd.DataFrame:
    obs = data.obs
    conditions = obs["p53_status"].astype(str) + "|" + obs["CENPA_status"].astype(str)
    acute = conditions.str.contains("acute_oe").to_numpy()
    control = conditions.str.contains("control").to_numpy()
    wt = obs["p53_status"].astype(str).eq("p53-WT").to_numpy()
    dn = obs["p53_status"].astype(str).eq("p53-DN").to_numpy()
    eps = 1e-6
    score_acute = data.x_log_hvg[acute].mean(axis=0) - data.x_log_hvg[control].mean(axis=0)
    score_p53 = data.x_log_hvg[dn].mean(axis=0) - data.x_log_hvg[wt].mean(axis=0)
    score = np.abs(score_acute) + np.abs(score_p53)
    table = data.gene_stats.copy()
    table["acute_vs_control_log_delta"] = score_acute
    table["p53DN_vs_p53WT_log_delta"] = score_p53
    table["candidate_score"] = score + eps
    top = table.sort_values("candidate_score", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    sns.barplot(
        data=top,
        y="gene_id",
        x="candidate_score",
        color="#C75D4D",
        ax=ax,
    )
    ax.set_title("Top expression-derived candidate regulatory genes")
    ax.set_xlabel("|acute-control| + |p53DN-p53WT| log-expression delta")
    ax.set_ylabel("Ensembl gene ID")
    _save(fig, Path(figures_dir) / "candidate_gene_scores.png")
    return table.sort_values("candidate_score", ascending=False)


def generate_all_figures(
    data: PreprocessedData,
    graph: HeteroGraphArtifact,
    metrics: pd.DataFrame,
    config: dict[str, Any],
    efficiency_df: pd.DataFrame | None = None,
    quality_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    figures_dir = config["paths"]["figures_dir"]
    plot_workflow(figures_dir)
    plot_condition_landscape(data, figures_dir)
    plot_umap(data, figures_dir)
    plot_graph_schema(figures_dir)
    plot_baseline_comparison(metrics, figures_dir)
    plot_module_summary(graph, figures_dir)
    if efficiency_df is not None:
        plot_label_efficiency(efficiency_df, figures_dir)
    if quality_df is not None:
        plot_embedding_quality(quality_df, figures_dir)
    return plot_candidate_genes(data, figures_dir)
