from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .preprocess import PreprocessedData


@dataclass
class HeteroGraphArtifact:
    node_features: dict[str, np.ndarray]
    edge_index: dict[str, np.ndarray]
    edge_weight: dict[str, np.ndarray]
    module_assignments: np.ndarray
    module_table: pd.DataFrame
    metadata: dict[str, Any]


def _top_expression_edges(x_hvg: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    n_cells, n_genes = x_hvg.shape
    k = min(top_k, n_genes)
    gene_idx = np.argpartition(x_hvg, kth=n_genes - k, axis=1)[:, -k:]
    cell_idx = np.repeat(np.arange(n_cells), k)
    gene_flat = gene_idx.reshape(-1)
    weights = x_hvg[cell_idx, gene_flat]
    order = weights > 0
    edge = np.vstack([cell_idx[order], gene_flat[order]]).astype(np.int64)
    return edge, weights[order].astype(np.float32)


def _knn_edges(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(k + 1, x.shape[0])
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(x)
    distances, indices = nn.kneighbors(x)
    src = np.repeat(np.arange(x.shape[0]), k - 1)
    dst = indices[:, 1:].reshape(-1)
    dist = distances[:, 1:].reshape(-1)
    weights = 1.0 / (1.0 + dist)
    return np.vstack([src, dst]).astype(np.int64), weights.astype(np.float32)


def _gene_coexpression_edges(x_hvg: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    z = StandardScaler().fit_transform(x_hvg).astype(np.float32)
    corr = cosine_similarity(z.T)
    np.fill_diagonal(corr, -np.inf)
    k = min(k, corr.shape[0] - 1)
    dst = np.argpartition(corr, kth=corr.shape[1] - k, axis=1)[:, -k:]
    src = np.repeat(np.arange(corr.shape[0]), k)
    dst_flat = dst.reshape(-1)
    weights = corr[src, dst_flat]
    keep = weights > 0
    return np.vstack([src[keep], dst_flat[keep]]).astype(np.int64), weights[keep].astype(np.float32)


def _module_nodes(x_hvg: np.ndarray, n_modules: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    gene_profiles = StandardScaler().fit_transform(x_hvg).T
    n_modules = min(n_modules, gene_profiles.shape[0])
    kmeans = KMeans(n_clusters=n_modules, random_state=seed, n_init=20)
    assignments = kmeans.fit_predict(gene_profiles)
    module_features = []
    for module_id in range(n_modules):
        members = gene_profiles[assignments == module_id]
        if len(members) == 0:
            module_features.append(np.zeros(8, dtype=np.float32))
            continue
        module_features.append(
            np.array(
                [
                    float(len(members)),
                    float(members.mean()),
                    float(members.std()),
                    float(np.median(members)),
                    float(np.quantile(members, 0.1)),
                    float(np.quantile(members, 0.9)),
                    float(np.mean(members > 0)),
                    float(np.mean(members < 0)),
                ],
                dtype=np.float32,
            )
        )
    return assignments.astype(np.int64), np.vstack(module_features).astype(np.float32)


def build_hetero_graph(data: PreprocessedData, config: dict[str, Any]) -> HeteroGraphArtifact:
    graph_cfg = config["graph"]
    seed = int(config["seed"])
    gene_features = data.gene_stats[
        ["mean_log", "var_log", "detected_cells", "detected_fraction"]
    ].to_numpy(dtype=np.float32)
    gene_features[:, 2] = gene_features[:, 2] / max(float(data.x_log_hvg.shape[0]), 1.0)
    module_assignments, module_features = _module_nodes(
        data.x_log_hvg,
        n_modules=int(graph_cfg["n_gene_modules"]),
        seed=seed,
    )
    expr_edge, expr_weight = _top_expression_edges(
        data.x_log_hvg,
        top_k=int(graph_cfg["top_expression_edges_per_cell"]),
    )
    cell_edge, cell_weight = _knn_edges(data.x_pca, k=int(graph_cfg["cell_knn"]))
    gene_edge, gene_weight = _gene_coexpression_edges(
        data.x_log_hvg,
        k=int(graph_cfg["gene_knn"]),
    )
    gene_module_edge = np.vstack(
        [np.arange(len(module_assignments)), module_assignments]
    ).astype(np.int64)
    gene_module_weight = np.ones(gene_module_edge.shape[1], dtype=np.float32)

    module_table = (
        data.gene_stats.assign(module_id=module_assignments)
        .groupby("module_id")
        .agg(
            n_genes=("gene_id", "size"),
            mean_log=("mean_log", "mean"),
            mean_var=("var_log", "mean"),
            mean_detected_fraction=("detected_fraction", "mean"),
        )
        .reset_index()
    )

    edges = {
        "cell__expresses__gene": expr_edge,
        "gene__expressed_by__cell": expr_edge[::-1],
        "cell__near__cell": cell_edge,
        "gene__coexpressed__gene": gene_edge,
        "gene__in_module__module": gene_module_edge,
        "module__contains__gene": gene_module_edge[::-1],
    }
    weights = {
        "cell__expresses__gene": expr_weight,
        "gene__expressed_by__cell": expr_weight,
        "cell__near__cell": cell_weight,
        "gene__coexpressed__gene": gene_weight,
        "gene__in_module__module": gene_module_weight,
        "module__contains__gene": gene_module_weight,
    }
    metadata = {
        "node_types": ["cell", "gene", "module"],
        "edge_types": list(edges.keys()),
        "n_cells": int(data.x_pca.shape[0]),
        "n_genes": int(data.x_log_hvg.shape[1]),
        "n_modules": int(module_features.shape[0]),
        "notes": "Reciprocal edges are included so HGT message passing can update all node types.",
    }
    return HeteroGraphArtifact(
        node_features={
            "cell": data.x_pca.astype(np.float32),
            "gene": gene_features.astype(np.float32),
            "module": module_features.astype(np.float32),
        },
        edge_index=edges,
        edge_weight=weights,
        module_assignments=module_assignments,
        module_table=module_table,
        metadata=metadata,
    )


def save_graph_artifact(path: str | Path, graph: HeteroGraphArtifact) -> None:
    path = Path(path)
    payload: dict[str, np.ndarray] = {}
    for node_type, features in graph.node_features.items():
        payload[f"node_{node_type}"] = features
    for edge_name, edge in graph.edge_index.items():
        payload[f"edge_{edge_name}"] = edge
        payload[f"weight_{edge_name}"] = graph.edge_weight[edge_name]
    payload["module_assignments"] = graph.module_assignments
    np.savez_compressed(path, **payload)
