from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .config import load_config
from .hgt_model import HGTMultiTask, build_pyg_data


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def train_hgt(config_path: str) -> None:
    config = load_config(config_path)
    results_dir = Path(config["paths"]["results_dir"])
    graph_path = results_dir / "hetero_graph.npz"
    preprocessed_path = results_dir / "preprocessed.npz"
    if not graph_path.exists() or not preprocessed_path.exists():
        raise FileNotFoundError(
            "Run `python3 -m src.run_pipeline --config configs/default.json` before HGT training."
        )

    torch.manual_seed(int(config["seed"]))
    batch = build_pyg_data(str(graph_path), str(preprocessed_path))
    data = batch.data
    metadata = data.metadata()
    in_dims = {node_type: int(data[node_type].x.shape[1]) for node_type in data.node_types}
    n_classes = {
        "p53": int(data["cell"].y_p53.max().item() + 1),
        "cenpa": int(data["cell"].y_cenpa.max().item() + 1),
        "condition": int(data["cell"].y_condition.max().item() + 1),
    }
    hgt_cfg = config["hgt"]
    model = HGTMultiTask(
        metadata=metadata,
        in_dims=in_dims,
        n_p53=n_classes["p53"],
        n_cenpa=n_classes["cenpa"],
        n_condition=n_classes["condition"],
        hidden_dim=int(hgt_cfg["hidden_dim"]),
        heads=int(hgt_cfg["heads"]),
        layers=int(hgt_cfg["layers"]),
        dropout=float(hgt_cfg["dropout"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(hgt_cfg["learning_rate"]),
        weight_decay=float(hgt_cfg["weight_decay"]),
    )
    weights = hgt_cfg["loss_weights"]
    history: list[dict[str, float]] = []
    for epoch in range(1, int(hgt_cfg["epochs"]) + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        train_idx = batch.train_index
        loss = (
            float(weights["p53"]) * F.cross_entropy(out["p53"][train_idx], data["cell"].y_p53[train_idx])
            + float(weights["cenpa"]) * F.cross_entropy(out["cenpa"][train_idx], data["cell"].y_cenpa[train_idx])
            + float(weights["condition"]) * F.cross_entropy(out["condition"][train_idx], data["cell"].y_condition[train_idx])
        )
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0:
            history.append({"epoch": float(epoch), "loss": float(loss.detach().cpu())})

    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
    test_idx = batch.test_index
    rows = []
    for target, y_attr in [
        ("p53", "y_p53"),
        ("cenpa", "y_cenpa"),
        ("condition", "y_condition"),
    ]:
        y_true = getattr(data["cell"], y_attr)[test_idx].cpu().numpy()
        y_pred = out[target][test_idx].argmax(dim=1).cpu().numpy()
        row = _metrics(y_true, y_pred)
        row.update({"model": "hgt_multitask", "target": target})
        rows.append(row)
    pd.DataFrame(rows).to_csv(results_dir / "hgt_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(results_dir / "hgt_training_history.csv", index=False)
    torch.save(model.state_dict(), results_dir / "hgt_model.pt")
    np.save(results_dir / "hgt_cell_embedding.npy", out["cell_embedding"].cpu().numpy())
    # Save condition predictions for McNemar test against logistic regression
    y_pred_condition = out["condition"][test_idx].argmax(dim=1).cpu().numpy()
    np.save(results_dir / "hgt_condition_predictions.npy", y_pred_condition)
    with (results_dir / "hgt_status.json").open("w", encoding="utf-8") as handle:
        json.dump({"status": "trained", "epochs": int(hgt_cfg["epochs"])}, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    train_hgt(args.config)


if __name__ == "__main__":
    main()
