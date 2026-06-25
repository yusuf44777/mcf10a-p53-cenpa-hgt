from __future__ import annotations

import sklearn.base
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .preprocess import PreprocessedData


@dataclass
class BaselineResults:
    metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    predictions: dict[str, np.ndarray] = field(default_factory=dict)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def _kmeans_map(train_labels: np.ndarray, train_clusters: np.ndarray, test_clusters: np.ndarray) -> np.ndarray:
    n_clusters = int(max(train_clusters.max(), test_clusters.max()) + 1)
    n_classes = int(train_labels.max() + 1)
    contingency = np.zeros((n_clusters, n_classes), dtype=int)
    for cluster, label in zip(train_clusters, train_labels):
        contingency[int(cluster), int(label)] += 1
    row, col = linear_sum_assignment(-contingency)
    mapping = {int(r): int(c) for r, c in zip(row, col)}
    fallback = int(np.bincount(train_labels).argmax())
    return np.array([mapping.get(int(cluster), fallback) for cluster in test_clusters])


def _run_kmeans(
    name: str,
    x: np.ndarray,
    y_condition: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> list[dict[str, Any]]:
    kmeans = KMeans(n_clusters=len(np.unique(y_condition)), random_state=seed, n_init=25)
    train_clusters = kmeans.fit_predict(x[train_idx])
    test_clusters = kmeans.predict(x[test_idx])
    mapped = _kmeans_map(y_condition[train_idx], train_clusters, test_clusters)
    metrics = _classification_metrics(y_condition[test_idx], mapped)
    metrics.update(
        {
            "model": name,
            "target": "condition",
            "ari": float(adjusted_rand_score(y_condition[test_idx], test_clusters)),
            "nmi": float(normalized_mutual_info_score(y_condition[test_idx], test_clusters)),
        }
    )
    return [metrics]


def _fit_supervised_models(
    data: PreprocessedData, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, np.ndarray]]:
    seed = int(config["seed"])
    baseline_cfg = config["baselines"]
    x_train = data.x_pca[data.train_index]
    x_test = data.x_pca[data.test_index]
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    models = {
        "logistic_pca": LogisticRegression(
            max_iter=int(baseline_cfg["max_iter"]),
            random_state=seed,
            class_weight="balanced",
            n_jobs=None,
        ),
        "random_forest_pca": RandomForestClassifier(
            n_estimators=int(baseline_cfg["random_forest_trees"]),
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
        "mlp_pca": MLPClassifier(
            hidden_layer_sizes=tuple(baseline_cfg["mlp_hidden_layer_sizes"]),
            max_iter=int(baseline_cfg["max_iter"]),
            random_state=seed,
            early_stopping=True,
        ),
    }
    rows: list[dict[str, Any]] = []
    importances: list[pd.DataFrame] = []
    predictions: dict[str, np.ndarray] = {}
    for target, labels in data.y.items():
        y_train = labels[data.train_index]
        y_test = labels[data.test_index]
        for name, model in models.items():
            xtr = x_train_scaled if name != "random_forest_pca" else x_train
            xte = x_test_scaled if name != "random_forest_pca" else x_test
            model.fit(xtr, y_train)
            pred = model.predict(xte)
            metrics = _classification_metrics(y_test, pred)
            metrics.update({"model": name, "target": target, "ari": np.nan, "nmi": np.nan})
            rows.append(metrics)
            if target == "condition":
                predictions[name] = pred
            if target == "condition" and name == "random_forest_pca":
                importances.append(
                    pd.DataFrame(
                        {
                            "feature": [f"PC{i + 1}" for i in range(len(model.feature_importances_))],
                            "importance": model.feature_importances_,
                            "model": name,
                        }
                    )
                )
    importance_df = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
    return rows, importance_df, predictions


def run_baselines(data: PreprocessedData, config: dict[str, Any]) -> BaselineResults:
    rows: list[dict[str, Any]] = []
    seed = int(config["seed"])
    rows.extend(
        _run_kmeans("kmeans_hvg", data.x_log_hvg, data.y["condition"],
                    data.train_index, data.test_index, seed)
    )
    rows.extend(
        _run_kmeans("pca_kmeans", data.x_pca, data.y["condition"],
                    data.train_index, data.test_index, seed)
    )
    supervised_rows, importance, predictions = _fit_supervised_models(data, config)
    rows.extend(supervised_rows)
    return BaselineResults(
        metrics=pd.DataFrame(rows),
        feature_importance=importance,
        predictions=predictions,
    )


def run_baselines_cv(
    data: PreprocessedData,
    config: dict[str, Any],
    n_folds: int = 5,
) -> pd.DataFrame:
    seed = int(config["seed"])
    baseline_cfg = config["baselines"]

    # Reconstruct full dataset in original cell order
    all_indices = np.concatenate([data.train_index, data.test_index])
    sort_order = np.argsort(all_indices)
    x_full = np.vstack([data.x_pca[data.train_index], data.x_pca[data.test_index]])[sort_order]

    rows: list[dict[str, Any]] = []

    for target, labels in data.y.items():
        y_full = labels  # original cell order

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        model_templates = {
            "logistic_pca": LogisticRegression(
                max_iter=int(baseline_cfg["max_iter"]),
                random_state=seed,
                class_weight="balanced",
            ),
            "random_forest_pca": RandomForestClassifier(
                n_estimators=int(baseline_cfg["random_forest_trees"]),
                random_state=seed,
                class_weight="balanced_subsample",
                n_jobs=-1,
            ),
            "mlp_pca": MLPClassifier(
                hidden_layer_sizes=tuple(baseline_cfg["mlp_hidden_layer_sizes"]),
                max_iter=int(baseline_cfg["max_iter"]),
                random_state=seed,
                early_stopping=True,
            ),
        }

        for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(x_full, y_full)):
            x_tr, x_te = x_full[tr_idx], x_full[te_idx]
            y_tr, y_te = y_full[tr_idx], y_full[te_idx]

            scaler = StandardScaler()
            x_tr_scaled = scaler.fit_transform(x_tr)
            x_te_scaled = scaler.transform(x_te)

            for name, template in model_templates.items():
                m = sklearn.base.clone(template)
                xtr = x_tr_scaled if name != "random_forest_pca" else x_tr
                xte = x_te_scaled if name != "random_forest_pca" else x_te
                m.fit(xtr, y_tr)
                pred = m.predict(xte)
                metrics = _classification_metrics(y_te, pred)
                metrics.update({"model": name, "target": target, "fold": fold_idx})
                rows.append(metrics)

    return pd.DataFrame(rows)


def mcnemar_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    model1_name: str = "model1",
    model2_name: str = "model2",
) -> dict[str, Any]:
    from statsmodels.stats.contingency_tables import mcnemar as _mcnemar

    correct1 = y_pred1 == y_true
    correct2 = y_pred2 == y_true
    b = int(np.sum(correct1 & ~correct2))
    c = int(np.sum(~correct1 & correct2))
    table = np.array([[0, b], [c, 0]])
    exact = (b + c) < 25
    result = _mcnemar(table, exact=exact, correction=True)
    return {
        "model1": model1_name,
        "model2": model2_name,
        "b": b,
        "c": c,
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "exact": exact,
    }


def compute_embedding_quality(
    embeddings: dict[str, np.ndarray],
    y_condition: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for name, X in embeddings.items():
        n = X.shape[0]
        if n > 3000:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, 3000, replace=False)
            X_sub, y_sub = X[idx], y_condition[idx]
        else:
            X_sub, y_sub = X, y_condition
        sil = float(silhouette_score(X_sub, y_sub, metric="euclidean", random_state=42))
        ch = float(calinski_harabasz_score(X, y_condition))
        rows.append({"embedding": name, "silhouette": round(sil, 4), "calinski_harabasz": round(ch, 1)})
    return pd.DataFrame(rows)


def run_label_efficiency_experiment(
    data: PreprocessedData,
    config: dict[str, Any],
    label_fractions: list[float] | None = None,
) -> pd.DataFrame:
    if label_fractions is None:
        label_fractions = list(config.get("cv", {}).get(
            "label_fractions", [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
        ))

    seed = int(config["seed"])
    baseline_cfg = config["baselines"]
    x_train = data.x_pca[data.train_index]
    x_test = data.x_pca[data.test_index]
    y_condition = data.y["condition"]
    y_train = y_condition[data.train_index]
    y_test = y_condition[data.test_index]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    rows: list[dict[str, Any]] = []
    for frac in label_fractions:
        if frac >= 1.0:
            x_tr_sub = x_train_scaled
            y_tr_sub = y_train
        else:
            _, x_tr_sub, _, y_tr_sub = train_test_split(
                x_train_scaled, y_train,
                test_size=frac, stratify=y_train, random_state=seed,
            )
        lr = LogisticRegression(
            max_iter=int(baseline_cfg["max_iter"]),
            random_state=seed,
            class_weight="balanced",
        )
        lr.fit(x_tr_sub, y_tr_sub)
        pred = lr.predict(x_test_scaled)
        m = _classification_metrics(y_test, pred)
        m["model"] = "logistic_pca"
        m["label_fraction"] = frac
        rows.append(m)

    return pd.DataFrame(rows)
