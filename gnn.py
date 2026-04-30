"""
gnn.py — GraphSAGE-based GNN refiner for scene-graph hallucination filtering.

GraphBuilder  : assembles a PyG Data object from raw feature arrays.
GNNRefiner    : GraphSAGE encoder + MLP edge scorer.
                Input  → node [N, NODE_DIM] and edge [E, EDGE_DIM] features.
                Output → per-edge logit; sigmoid(logit) = P(edge is real).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

NODE_DIM = 1032   
EDGE_DIM = 3096   


class GraphBuilder:
    """
    Assembles a PyG Data object from per-image feature arrays.

    Node deduplication (overlapping boxes) is handled upstream by NMS
    before this class is called.
    """

    def build(self, node_feats: Tensor, edge_feats: Tensor,
              edge_index: Tensor,
              edge_labels: Optional[Tensor] = None) -> Data:
        """
        Args:
            node_feats  : [N, NODE_DIM]
            edge_feats  : [E, EDGE_DIM]
            edge_index  : [2, E] COO adjacency (source, dest)
            edge_labels : [E] float32 binary labels — provided during training only.

        Returns PyG Data with .x, .edge_index, .edge_attr, and optionally .edge_label.
        """
        data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_feats)
        if edge_labels is not None:
            data.edge_label = edge_labels.float()
        return data


class GNNRefiner(nn.Module):
    """
    GraphSAGE encoder + MLP edge scorer for hallucination filtering.

    Architecture:
        node_proj   : Linear(NODE_DIM → hidden) + LayerNorm + GELU
        edge_proj   : Linear(EDGE_DIM → hidden) + LayerNorm + GELU
        SAGEConv ×N : message passing — aggregates neighbour means into node reprs
        edge_scorer : MLP( [node_i ‖ node_j ‖ edge_h] ) → scalar logit

    Use BCEWithLogitsLoss during training (logits) and sigmoid() for inference.

    Args:
        node_in_dim    : Input node feature dim (= NODE_DIM).
        edge_in_dim    : Input edge feature dim (= EDGE_DIM).
        hidden_dim     : Hidden size used throughout.
        num_sage_layers: Number of GraphSAGE message-passing rounds.
        dropout        : Dropout rate inside the edge-scorer MLP.
    """

    def __init__(self, node_in_dim=NODE_DIM, edge_in_dim=EDGE_DIM,
                 hidden_dim=256, num_sage_layers=2, dropout=0.1):
        super().__init__()

        
        self.node_proj = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )

        
        self.sage_layers = nn.ModuleList(
            [SAGEConv(hidden_dim, hidden_dim) for _ in range(num_sage_layers)]
        )
        self.sage_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_sage_layers)]
        )

        
        self.edge_scorer = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data: Data) -> Tensor:
        """
        Args:
            data : PyG Data with .x [N, NODE_DIM], .edge_index [2, E],
                   .edge_attr [E, EDGE_DIM].

        Returns:
            logits Tensor[E] — raw (pre-sigmoid) edge existence scores.
            DEBUG: if logits collapse to a constant, verify grad_fn is not None.
        """
        x      = self.node_proj(data.x)         
        edge_h = self.edge_proj(data.edge_attr)  

        
        for sage, norm in zip(self.sage_layers, self.sage_norms):
            x = norm(F.gelu(sage(x, data.edge_index)))

        src, dst = data.edge_index
        
        edge_input = torch.cat([x[src], x[dst], edge_h], dim=-1)  
        return self.edge_scorer(edge_input).squeeze(-1)             

    def predict_proba(self, data: Data) -> Tensor:
        """Returns sigmoid probabilities [E] — use for inference, not training."""
        return torch.sigmoid(self.forward(data))