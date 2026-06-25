from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExpressionDataset:
    expression: np.ndarray
    obs: pd.DataFrame
    genes: pd.DataFrame

    @property
    def condition_labels(self) -> np.ndarray:
        return (
            self.obs["p53_status"].astype(str)
            + "|"
            + self.obs["CENPA_status"].astype(str)
        ).to_numpy()


def _decode_array(values: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def _read_obs_column(group: h5py.Group, key: str) -> list[str]:
    obj = group[key]
    if isinstance(obj, h5py.Dataset):
        return _decode_array(obj[:])

    categories = _decode_array(obj["categories"][:])
    codes = obj["codes"][:]
    return [categories[int(code)] for code in codes]


def load_h5ad_dense(path: str | Path) -> ExpressionDataset:
    """Load the specific dense AnnData/H5AD layout used by the supplied dataset."""
    path = Path(path)
    with h5py.File(path, "r") as handle:
        expression = handle["X"][:].astype(np.float32, copy=False)
        obs_group = handle["obs"]
        obs = pd.DataFrame(
            {
                "_index": _read_obs_column(obs_group, "_index"),
                "sample_name": _read_obs_column(obs_group, "sample_name"),
                "p53_status": _read_obs_column(obs_group, "p53_status"),
                "CENPA_status": _read_obs_column(obs_group, "CENPA_status"),
            }
        )
        genes = pd.DataFrame(
            {
                "gene_id": _decode_array(handle["var"]["_index"][:]),
                "gene_full_id": _decode_array(handle["var"]["Full"][:]),
            }
        )
    return ExpressionDataset(expression=expression, obs=obs, genes=genes)


def dataset_summary(dataset: ExpressionDataset) -> dict[str, Any]:
    expression = dataset.expression
    obs = dataset.obs
    detected = expression > 0
    summary: dict[str, Any] = {
        "n_cells": int(expression.shape[0]),
        "n_genes": int(expression.shape[1]),
        "samples": obs["sample_name"].value_counts().sort_index().to_dict(),
        "p53_status": obs["p53_status"].value_counts().sort_index().to_dict(),
        "CENPA_status": obs["CENPA_status"].value_counts().sort_index().to_dict(),
        "conditions": dataset.condition_labels.tolist(),
        "condition_counts": pd.Series(dataset.condition_labels)
        .value_counts()
        .sort_index()
        .to_dict(),
        "cell_library_sum_quantiles": np.quantile(expression.sum(axis=1), [0, 0.25, 0.5, 0.75, 1]).round(3).tolist(),
        "cell_detected_gene_quantiles": np.quantile(detected.sum(axis=1), [0, 0.25, 0.5, 0.75, 1]).round(3).tolist(),
        "gene_detected_cell_quantiles": np.quantile(detected.sum(axis=0), [0, 0.25, 0.5, 0.75, 1]).round(3).tolist(),
    }
    summary["conditions"] = sorted(summary["condition_counts"].keys())
    return summary


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
