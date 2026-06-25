from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .data_io import ExpressionDataset


@dataclass
class PreprocessedData:
    x_log_hvg: np.ndarray
    x_pca: np.ndarray
    x_umap: np.ndarray
    selected_gene_indices: np.ndarray
    selected_genes: pd.DataFrame
    gene_stats: pd.DataFrame
    obs: pd.DataFrame
    label_maps: dict[str, dict[str, int]]
    y: dict[str, np.ndarray]
    train_index: np.ndarray
    test_index: np.ndarray
    pca_explained_variance: np.ndarray


def normalize_log1p(expression: np.ndarray, target_sum: float) -> np.ndarray:
    library = expression.sum(axis=1, keepdims=True).astype(np.float32)
    library[library == 0] = 1.0
    normalized = expression / library * np.float32(target_sum)
    return np.log1p(normalized, dtype=np.float32)


def _gene_stats(x_log: np.ndarray, genes: pd.DataFrame) -> pd.DataFrame:
    detected = x_log > 0
    stats = genes.copy()
    stats["mean_log"] = x_log.mean(axis=0)
    stats["var_log"] = x_log.var(axis=0)
    stats["detected_cells"] = detected.sum(axis=0).astype(int)
    stats["detected_fraction"] = detected.mean(axis=0)
    return stats


def select_hvg(
    x_log: np.ndarray,
    genes: pd.DataFrame,
    n_hvg: int,
    min_cells: int,
    always_keep: dict[str, str],
) -> tuple[np.ndarray, pd.DataFrame]:
    stats = _gene_stats(x_log, genes)
    eligible = stats["detected_cells"].to_numpy() >= min_cells
    ranking = np.argsort(-stats["var_log"].to_numpy())
    selected: list[int] = []
    for idx in ranking:
        if eligible[idx]:
            selected.append(int(idx))
        if len(selected) >= n_hvg:
            break

    gene_ids = stats["gene_id"].to_numpy()
    for gene_id in always_keep.values():
        matches = np.flatnonzero(gene_ids == gene_id)
        if len(matches) and int(matches[0]) not in selected:
            selected.append(int(matches[0]))

    selected_indices = np.array(sorted(set(selected)), dtype=int)
    selected_stats = stats.iloc[selected_indices].reset_index(drop=True)
    selected_stats["original_index"] = selected_indices
    return selected_indices, selected_stats


def compute_umap(
    x_pca: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError("umap-learn is required. Install with `pip install umap-learn`.") from exc
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric="euclidean",
    )
    return reducer.fit_transform(x_pca).astype(np.float32)


def encode_labels(obs: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, dict[str, int]]]:
    labels = {
        "p53": obs["p53_status"].astype(str).to_numpy(),
        "cenpa": obs["CENPA_status"].astype(str).to_numpy(),
        "condition": (
            obs["p53_status"].astype(str) + "|" + obs["CENPA_status"].astype(str)
        ).to_numpy(),
    }
    encoded: dict[str, np.ndarray] = {}
    maps: dict[str, dict[str, int]] = {}
    for key, values in labels.items():
        encoder = LabelEncoder()
        encoded[key] = encoder.fit_transform(values)
        maps[key] = {label: int(code) for code, label in enumerate(encoder.classes_)}
    return encoded, maps


def preprocess_dataset(dataset: ExpressionDataset, config: dict[str, Any]) -> PreprocessedData:
    pp = config["preprocess"]
    x_log = normalize_log1p(dataset.expression, target_sum=float(pp["target_sum"]))
    selected_idx, selected_genes = select_hvg(
        x_log,
        dataset.genes,
        n_hvg=int(pp["n_hvg"]),
        min_cells=int(pp["min_cells_per_gene"]),
        always_keep=dict(pp["always_keep_genes"]),
    )
    x_hvg = x_log[:, selected_idx].astype(np.float32, copy=False)
    scaler = StandardScaler(with_mean=True, with_std=True)
    x_scaled = scaler.fit_transform(x_hvg)
    pca = PCA(n_components=int(pp["n_pca"]), random_state=int(config["seed"]))
    x_pca = pca.fit_transform(x_scaled).astype(np.float32)
    x_umap = compute_umap(
        x_pca,
        n_neighbors=int(pp.get("umap_n_neighbors", 15)),
        min_dist=float(pp.get("umap_min_dist", 0.1)),
        random_state=int(config["seed"]),
    )
    y, label_maps = encode_labels(dataset.obs)
    train_index, test_index = train_test_split(
        np.arange(x_pca.shape[0]),
        test_size=float(config["baselines"]["test_size"]),
        random_state=int(config["seed"]),
        stratify=y["condition"],
    )
    gene_stats = selected_genes.copy()
    return PreprocessedData(
        x_log_hvg=x_hvg,
        x_pca=x_pca,
        x_umap=x_umap,
        selected_gene_indices=selected_idx,
        selected_genes=selected_genes,
        gene_stats=gene_stats,
        obs=dataset.obs.copy(),
        label_maps=label_maps,
        y=y,
        train_index=np.array(sorted(train_index), dtype=int),
        test_index=np.array(sorted(test_index), dtype=int),
        pca_explained_variance=pca.explained_variance_ratio_,
    )


def save_preprocessed(path: str, data: PreprocessedData) -> None:
    np.savez_compressed(
        path,
        x_log_hvg=data.x_log_hvg,
        x_pca=data.x_pca,
        x_umap=data.x_umap,
        selected_gene_indices=data.selected_gene_indices,
        train_index=data.train_index,
        test_index=data.test_index,
        y_p53=data.y["p53"],
        y_cenpa=data.y["cenpa"],
        y_condition=data.y["condition"],
        pca_explained_variance=data.pca_explained_variance,
    )
