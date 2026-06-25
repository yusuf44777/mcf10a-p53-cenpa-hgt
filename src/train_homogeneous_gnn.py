from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .config import load_config


class CellGCN(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int, dropout: float) -> None:
        super().__init__()
        try:
            from torch_geometric.nn import GCNConv
        except ImportError as exc:
            raise ImportError(
                "torch_geometric is required for the homogeneous GNN ablation."
            ) from exc
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, n_classes)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        return self.classifier(x)


def train_homogeneous_gnn(config_path: str) -> None:
    try:
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError(
            "torch_geometric is required. Install with `python3 -m pip install torch-geometric`."
        ) from exc

    config = load_config(config_path)
    results_dir = Path(config["paths"]["results_dir"])
    preprocessed = np.load(results_dir / "preprocessed.npz")
    graph = np.load(results_dir / "hetero_graph.npz")
    x = torch.tensor(preprocessed["x_pca"], dtype=torch.float32)
    edge_index = torch.tensor(graph["edge_cell__near__cell"], dtype=torch.long)
    y = torch.tensor(preprocessed["y_condition"], dtype=torch.long)
    train_index = torch.tensor(preprocessed["train_index"], dtype=torch.long)
    test_index = torch.tensor(preprocessed["test_index"], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)

    torch.manual_seed(int(config["seed"]))
    model = CellGCN(
        in_dim=int(data.x.shape[1]),
        hidden_dim=int(config["hgt"]["hidden_dim"]),
        n_classes=int(data.y.max().item() + 1),
        dropout=float(config["hgt"]["dropout"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["hgt"]["learning_rate"]),
        weight_decay=float(config["hgt"]["weight_decay"]),
    )
    for _ in range(int(config["hgt"]["epochs"])):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = torch.nn.functional.cross_entropy(logits[train_index], data.y[train_index])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index)[test_index].argmax(dim=1).cpu().numpy()
    y_true = data.y[test_index].cpu().numpy()
    metrics = {
        "model": "homogeneous_gcn_cell_knn",
        "target": "condition",
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
    }
    pd.DataFrame([metrics]).to_csv(results_dir / "homogeneous_gnn_metrics.csv", index=False)
    torch.save(model.state_dict(), results_dir / "homogeneous_gnn_model.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    train_homogeneous_gnn(args.config)


if __name__ == "__main__":
    main()
