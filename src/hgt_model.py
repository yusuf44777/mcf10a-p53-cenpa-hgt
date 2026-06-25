from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


EDGE_NAME_TO_TUPLE = {
    "cell__expresses__gene": ("cell", "expresses", "gene"),
    "gene__expressed_by__cell": ("gene", "expressed_by", "cell"),
    "cell__near__cell": ("cell", "near", "cell"),
    "gene__coexpressed__gene": ("gene", "coexpressed", "gene"),
    "gene__in_module__module": ("gene", "in_module", "module"),
    "module__contains__gene": ("module", "contains", "gene"),
}


@dataclass
class HGTBatch:
    data: Any
    train_index: torch.Tensor
    test_index: torch.Tensor


class HGTMultiTask(nn.Module):
    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        in_dims: dict[str, int],
        n_p53: int,
        n_cenpa: int,
        n_condition: int,
        hidden_dim: int = 64,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        try:
            from torch_geometric.nn import HGTConv
        except ImportError as exc:
            raise ImportError(
                "torch_geometric is required for HGTMultiTask. Install with "
                "`python3 -m pip install torch-geometric`."
            ) from exc

        self.proj = nn.ModuleDict(
            {node_type: nn.Linear(dim, hidden_dim) for node_type, dim in in_dims.items()}
        )
        self.convs = nn.ModuleList(
            [
                HGTConv(
                    in_channels={node_type: hidden_dim for node_type in in_dims},
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=heads,
                )
                for _ in range(layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.p53_head = nn.Linear(hidden_dim, n_p53)
        self.cenpa_head = nn.Linear(hidden_dim, n_cenpa)
        self.condition_head = nn.Linear(hidden_dim, n_condition)

    def forward(self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict[Any, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = {key: F.relu(layer(x_dict[key])) for key, layer in self.proj.items()}
        for conv in self.convs:
            out = conv(x, edge_index_dict)
            x = {
                key: self.dropout(F.relu(out[key])) if out.get(key) is not None else x[key]
                for key in x
            }
        cell = x["cell"]
        return {
            "p53": self.p53_head(cell),
            "cenpa": self.cenpa_head(cell),
            "condition": self.condition_head(cell),
            "cell_embedding": cell,
        }


def build_pyg_data(
    graph_npz: str,
    labels_npz: str,
) -> HGTBatch:
    try:
        from torch_geometric.data import HeteroData
    except ImportError as exc:
        raise ImportError(
            "torch_geometric is required to materialize HeteroData."
        ) from exc

    import numpy as np

    graph = np.load(graph_npz)
    labels = np.load(labels_npz)
    data = HeteroData()
    for node_type in ("cell", "gene", "module"):
        data[node_type].x = torch.tensor(graph[f"node_{node_type}"], dtype=torch.float32)
    for edge_name, edge_type in EDGE_NAME_TO_TUPLE.items():
        data[edge_type].edge_index = torch.tensor(graph[f"edge_{edge_name}"], dtype=torch.long)
    data["cell"].y_p53 = torch.tensor(labels["y_p53"], dtype=torch.long)
    data["cell"].y_cenpa = torch.tensor(labels["y_cenpa"], dtype=torch.long)
    data["cell"].y_condition = torch.tensor(labels["y_condition"], dtype=torch.long)
    return HGTBatch(
        data=data,
        train_index=torch.tensor(labels["train_index"], dtype=torch.long),
        test_index=torch.tensor(labels["test_index"], dtype=torch.long),
    )
