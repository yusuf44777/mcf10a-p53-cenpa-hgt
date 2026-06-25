from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import (
    compute_embedding_quality,
    mcnemar_test,
    run_baselines,
    run_baselines_cv,
    run_label_efficiency_experiment,
)
from .config import ensure_dirs, load_config
from .data_io import dataset_summary, load_h5ad_dense, write_json
from .figures import generate_all_figures, plot_confusion_matrix_figure
from .graph_build import build_hetero_graph, save_graph_artifact
from .preprocess import preprocess_dataset, save_preprocessed


def run_pipeline(config_path: str) -> None:
    config = load_config(config_path)
    ensure_dirs(config)
    results_dir = Path(config["paths"]["results_dir"])
    dataset = load_h5ad_dense(config["paths"]["h5ad"])
    summary = dataset_summary(dataset)
    write_json(results_dir / "dataset_summary.json", summary)

    preprocessed = preprocess_dataset(dataset, config)
    save_preprocessed(str(results_dir / "preprocessed.npz"), preprocessed)
    preprocessed.gene_stats.to_csv(results_dir / "selected_gene_stats.csv", index=False)
    preprocessed.obs.to_csv(results_dir / "cell_metadata.csv", index=False)
    with (results_dir / "label_maps.json").open("w", encoding="utf-8") as handle:
        json.dump(preprocessed.label_maps, handle, indent=2, sort_keys=True)

    baselines = run_baselines(preprocessed, config)
    baselines.metrics.to_csv(results_dir / "baseline_metrics.csv", index=False)
    if not baselines.feature_importance.empty:
        baselines.feature_importance.to_csv(results_dir / "baseline_feature_importance.csv", index=False)

    # 5-fold cross-validation
    n_folds = int(config.get("cv", {}).get("n_folds", 5))
    cv_df = run_baselines_cv(preprocessed, config, n_folds=n_folds)
    cv_df.to_csv(results_dir / "baseline_metrics_cv.csv", index=False)
    cv_summary = (
        cv_df.groupby(["model", "target"])[["accuracy", "balanced_accuracy", "macro_f1"]]
        .agg(["mean", "std"])
        .round(4)
    )
    cv_summary.to_csv(results_dir / "baseline_metrics_cv_summary.csv")

    # Label efficiency experiment
    efficiency_df = run_label_efficiency_experiment(preprocessed, config)
    efficiency_df.to_csv(results_dir / "label_efficiency.csv", index=False)

    # Embedding quality: PCA + UMAP (HGT added by post_hgt_analysis)
    quality_df = compute_embedding_quality(
        {"PCA (40 PCs)": preprocessed.x_pca, "UMAP (2D)": preprocessed.x_umap},
        preprocessed.y["condition"],
    )
    quality_df.to_csv(results_dir / "embedding_quality.csv", index=False)

    # Confusion matrix and predictions for McNemar
    y_test_condition = preprocessed.y["condition"][preprocessed.test_index]
    if "logistic_pca" in baselines.predictions:
        plot_confusion_matrix_figure(
            y_test_condition,
            baselines.predictions["logistic_pca"],
            preprocessed.label_maps["condition"],
            "Logistic PCA",
            config["paths"]["figures_dir"],
        )
        np.save(str(results_dir / "logistic_condition_predictions.npy"),
                baselines.predictions["logistic_pca"])
        np.save(str(results_dir / "test_condition_labels.npy"), y_test_condition)

    # McNemar: logistic vs MLP on fixed test split
    if "logistic_pca" in baselines.predictions and "mlp_pca" in baselines.predictions:
        mcn = mcnemar_test(
            y_test_condition,
            baselines.predictions["logistic_pca"],
            baselines.predictions["mlp_pca"],
            "logistic_pca",
            "mlp_pca",
        )
        pd.DataFrame([mcn]).to_csv(results_dir / "mcnemar_tests.csv", index=False)

    graph = build_hetero_graph(preprocessed, config)
    save_graph_artifact(results_dir / "hetero_graph.npz", graph)
    graph.module_table.to_csv(results_dir / "gene_modules.csv", index=False)
    write_json(results_dir / "hetero_graph_metadata.json", graph.metadata)

    candidates = generate_all_figures(
        preprocessed, graph, baselines.metrics, config,
        efficiency_df=efficiency_df,
        quality_df=quality_df,
    )
    candidates.to_csv(results_dir / "candidate_gene_scores.csv", index=False)

    status = {
        "pipeline": "complete",
        "hgt_training": "not_run",
        "hgt_note": "Run python3 -m src.train_hgt --config configs/default.json after installing torch-geometric.",
    }
    write_json(results_dir / "pipeline_status.json", status)

    manuscript_table = baselines.metrics.copy()
    manuscript_table["macro_f1"] = manuscript_table["macro_f1"].round(3)
    manuscript_table["balanced_accuracy"] = manuscript_table["balanced_accuracy"].round(3)
    manuscript_table["accuracy"] = manuscript_table["accuracy"].round(3)
    manuscript_table.to_csv(results_dir / "table_model_metrics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
