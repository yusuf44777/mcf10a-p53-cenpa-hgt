from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from .config import load_config
from .figures import PALETTE


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _load_metric_tables(results_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(results_dir / "baseline_metrics.csv")]
    optional = [
        results_dir / "homogeneous_gnn_metrics.csv",
        results_dir / "hgt_metrics.csv",
    ]
    for path in optional:
        if path.exists():
            frames.append(pd.read_csv(path))
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(results_dir / "model_metrics_combined.csv", index=False)
    return combined


def plot_combined_model_comparison(metrics: pd.DataFrame, figures_dir: Path) -> None:
    plot_df = metrics[metrics["target"] == "condition"].copy()
    label_map = {
        "kmeans_hvg": "k-means HVG",
        "pca_kmeans": "PCA k-means",
        "logistic_pca": "Logistic PCA",
        "random_forest_pca": "Random forest PCA",
        "mlp_pca": "MLP PCA",
        "homogeneous_gcn_cell_knn": "Homogeneous GCN",
        "hgt_multitask": "Multitask HGT",
    }
    plot_df["display_model"] = plot_df["model"].map(label_map).fillna(plot_df["model"])
    plot_df = plot_df.sort_values("macro_f1", ascending=False)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    colors = ["#C75D4D" if model == "Multitask HGT" else "#4169A8" for model in plot_df["display_model"]]
    sns.barplot(data=plot_df, y="display_model", x="macro_f1", palette=colors, hue="display_model", legend=False, ax=ax)
    ax.set_xlim(0, 1.0)
    ax.set_title("Condition classification with graph-learning ablations")
    ax.set_xlabel("Macro-F1")
    ax.set_ylabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)
    _save(fig, figures_dir / "model_comparison_with_hgt.png")


def plot_hgt_embedding(results_dir: Path, figures_dir: Path) -> None:
    embedding_path = results_dir / "hgt_cell_embedding.npy"
    if not embedding_path.exists():
        return
    embedding = np.load(embedding_path)
    coords = PCA(n_components=2, random_state=42).fit_transform(embedding)
    metadata = pd.read_csv(results_dir / "cell_metadata.csv")
    metadata["HGT1"] = coords[:, 0]
    metadata["HGT2"] = coords[:, 1]
    metadata["condition"] = metadata["p53_status"].astype(str) + "|" + metadata["CENPA_status"].astype(str)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    sns.scatterplot(
        data=metadata,
        x="HGT1",
        y="HGT2",
        hue="condition",
        palette=PALETTE,
        s=12,
        alpha=0.75,
        linewidth=0,
        ax=ax,
    )
    ax.set_title("HGT cell embedding landscape")
    ax.set_xlabel("HGT embedding PC1")
    ax.set_ylabel("HGT embedding PC2")
    ax.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    _save(fig, figures_dir / "hgt_embedding_landscape.png")


def plot_hgt_training(results_dir: Path, figures_dir: Path) -> None:
    history_path = results_dir / "hgt_training_history.csv"
    if not history_path.exists():
        return
    history = pd.read_csv(history_path)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    sns.lineplot(data=history, x="epoch", y="loss", marker="o", color="#C75D4D", ax=ax)
    ax.set_title("HGT multitask training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted multitask loss")
    _save(fig, figures_dir / "hgt_training_curve.png")


def run_mcnemar_hgt_vs_logistic(results_dir: Path) -> None:
    hgt_pred_path = results_dir / "hgt_condition_predictions.npy"
    logistic_pred_path = results_dir / "logistic_condition_predictions.npy"
    y_true_path = results_dir / "test_condition_labels.npy"
    if not (hgt_pred_path.exists() and logistic_pred_path.exists() and y_true_path.exists()):
        return
    from .baselines import mcnemar_test
    y_true = np.load(str(y_true_path))
    y_hgt = np.load(str(hgt_pred_path))
    y_logistic = np.load(str(logistic_pred_path))
    result = mcnemar_test(y_true, y_logistic, y_hgt, "logistic_pca", "hgt_multitask")
    pd.DataFrame([result]).to_csv(results_dir / "mcnemar_hgt_vs_logistic.csv", index=False)


def extend_embedding_quality_with_hgt(results_dir: Path, figures_dir: Path) -> None:
    existing_path = results_dir / "embedding_quality.csv"
    hgt_path = results_dir / "hgt_cell_embedding.npy"
    metadata_path = results_dir / "cell_metadata.csv"
    label_maps_path = results_dir / "label_maps.json"
    if not (existing_path.exists() and hgt_path.exists()):
        return
    import json as _json
    from .baselines import compute_embedding_quality
    from .figures import plot_embedding_quality
    existing = pd.read_csv(existing_path)
    hgt_emb = np.load(str(hgt_path))
    metadata = pd.read_csv(metadata_path)
    with label_maps_path.open(encoding="utf-8") as fh:
        label_maps = _json.load(fh)
    condition_map = label_maps["condition"]
    condition_str = metadata["p53_status"].astype(str) + "|" + metadata["CENPA_status"].astype(str)
    y_condition = np.array([condition_map[c] for c in condition_str], dtype=int)
    hgt_quality = compute_embedding_quality({"HGT embedding (64D)": hgt_emb}, y_condition)
    combined = pd.concat([existing, hgt_quality], ignore_index=True)
    combined.to_csv(existing_path, index=False)
    plot_embedding_quality(combined, figures_dir)


def run_post_hgt_analysis(config_path: str) -> None:
    config = load_config(config_path)
    results_dir = Path(config["paths"]["results_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    metrics = _load_metric_tables(results_dir)
    plot_combined_model_comparison(metrics, figures_dir)
    plot_hgt_embedding(results_dir, figures_dir)
    plot_hgt_training(results_dir, figures_dir)
    run_mcnemar_hgt_vs_logistic(results_dir)
    extend_embedding_quality_with_hgt(results_dir, figures_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    run_post_hgt_analysis(args.config)


if __name__ == "__main__":
    main()
