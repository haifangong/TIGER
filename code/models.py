"""Multimodal peptide encoders and pair / single prediction heads."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import (
    GATConv,
    GATv2Conv,
    GINConv,
    GraphConv,
    GraphNorm,
    SAGEConv,
    GlobalAttention,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from .utils.config import Config
from .utils.constants import CFU_LEVELS, NUM_AA_TOKENS


class GNN(nn.Module):
    def __init__(self, num_layer: int, input_dim: int, emb_dim: int, gnn_type: str):
        super().__init__()
        self.layers = nn.ModuleList()
        for layer in range(num_layer):
            in_dim = input_dim if layer == 0 else emb_dim
            if gnn_type == "gin":
                self.layers.append(
                    GINConv(nn.Sequential(nn.Linear(in_dim, emb_dim), GraphNorm(emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim)))
                )
            elif gnn_type == "gcn":
                self.layers.append(GraphConv(in_dim, emb_dim))
            elif gnn_type == "gat":
                self.layers.append(GATConv(in_dim, emb_dim))
            elif gnn_type == "gatv2":
                self.layers.append(GATv2Conv(in_dim, emb_dim))
            elif gnn_type == "graphsage":
                self.layers.append(SAGEConv(in_dim, emb_dim))
            else:
                raise ValueError(f"Unsupported gnn_type={gnn_type}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = F.relu(layer(h, edge_index))
        return h


class SimpleSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.query = nn.Linear(embedding_dim, embedding_dim * num_heads)
        self.key = nn.Linear(embedding_dim, embedding_dim * num_heads)
        self.value = nn.Linear(embedding_dim, embedding_dim * num_heads)
        self.proj = nn.Linear(embedding_dim * num_heads, embedding_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        batch_size = x1.shape[0]
        x = torch.stack((x1, x2, x3), dim=1)
        q = self.query(x).view(batch_size, 3, self.num_heads, self.embedding_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, 3, self.num_heads, self.embedding_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, 3, self.num_heads, self.embedding_dim).transpose(1, 2)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embedding_dim), dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, 3, self.num_heads * self.embedding_dim)
        return self.proj(out).sum(dim=1)


def node_input_dim(cfg: Config) -> int:
    """AA(20) + energy(20) [+ coords(3) if enabled]."""
    return 43 if bool(getattr(cfg, "include_node_coords", False)) else 40


class PeptideEncoder(nn.Module):
    def __init__(self, cfg: Config, input_dim: int | None = None):
        super().__init__()
        self.cfg = cfg
        input_dim = int(input_dim if input_dim is not None else node_input_dim(cfg))
        self.input_dim = input_dim
        self.gnn = GNN(cfg.num_layer, input_dim, cfg.emb_dim, cfg.gnn_type)
        if cfg.graph_pooling == "sum":
            self.pool = global_add_pool
        elif cfg.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif cfg.graph_pooling == "max":
            self.pool = global_max_pool
        elif cfg.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=nn.Linear(cfg.emb_dim, 1))
        else:
            raise ValueError(f"Unsupported pooling={cfg.graph_pooling}")

        encoding = str(cfg.seq_encoding).lower()
        if encoding not in {"integer", "embedding", "onehot"}:
            raise ValueError(f"Unsupported seq_encoding={cfg.seq_encoding}")
        self.seq_encoding = encoding
        self.max_len = cfg.max_len
        if encoding == "embedding":
            # Categorical AA ids (1..20) + learnable positional encoding; masked mean pool.
            self.aa_embedding = nn.Embedding(NUM_AA_TOKENS, cfg.emb_dim, padding_idx=0)
            self.pos_embedding = nn.Embedding(cfg.max_len, cfg.emb_dim)
            self.aa_proj = None
            self.seq_encoder = nn.Sequential(
                nn.Linear(cfg.emb_dim, cfg.emb_dim), nn.LeakyReLU(0.1), nn.Dropout(cfg.dropout_ratio)
            )
        elif encoding == "onehot":
            self.aa_embedding = None
            self.pos_embedding = nn.Embedding(cfg.max_len, cfg.emb_dim)
            self.aa_proj = nn.Linear(NUM_AA_TOKENS, cfg.emb_dim)
            self.seq_encoder = nn.Sequential(
                nn.Linear(cfg.emb_dim, cfg.emb_dim), nn.LeakyReLU(0.1), nn.Dropout(cfg.dropout_ratio)
            )
        else:
            # Strongest empirical recipe: position-slot Linear over AA codes 1..20 (0=pad).
            self.aa_embedding = None
            self.pos_embedding = None
            self.aa_proj = None
            self.seq_encoder = nn.Sequential(
                nn.Linear(cfg.max_len, cfg.emb_dim), nn.LeakyReLU(0.1), nn.Dropout(cfg.dropout_ratio)
            )
        self.global_encoder = nn.Sequential(
            nn.Linear(10, cfg.emb_dim), nn.LeakyReLU(0.1), nn.Dropout(cfg.dropout_ratio)
        )
        self.cfu_encoder = nn.Embedding(len(CFU_LEVELS), cfg.emb_dim)
        self.att = SimpleSelfAttention(cfg.emb_dim)
        self.concat_proj = nn.Sequential(
            nn.Linear(cfg.emb_dim * 3, cfg.emb_dim), nn.LeakyReLU(0.1), nn.Dropout(cfg.dropout_ratio)
        )

    def _masked_node_features(self, x: torch.Tensor) -> torch.Tensor:
        keep = self.cfg.structure_features.lower()
        # "sep"/"se"/"all" keep every available channel (coords may be absent).
        if keep in {"sep", "se", "all"} and x.size(1) <= 40:
            return x
        if keep in {"sep", "all"} and x.size(1) >= 43:
            return x
        mask = torch.zeros(x.size(1), dtype=x.dtype, device=x.device)
        if "s" in keep:
            mask[:20] = 1
        if "e" in keep and x.size(1) >= 40:
            mask[20:40] = 1
        if "p" in keep and x.size(1) >= 43:
            mask[40:43] = 1
        return x * mask

    def _encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        if self.seq_encoding == "integer":
            return self.seq_encoder(seq.float().view(seq.size(0), -1))
        idx = seq.long().view(seq.size(0), -1).clamp(0, NUM_AA_TOKENS - 1)
        mask = (idx > 0).to(dtype=torch.float32)
        positions = torch.arange(idx.size(1), device=idx.device).unsqueeze(0).expand_as(idx)
        pos = self.pos_embedding(positions)
        if self.seq_encoding == "embedding":
            emb = self.aa_embedding(idx) + pos
        else:
            emb = self.aa_proj(F.one_hot(idx, num_classes=NUM_AA_TOKENS).to(dtype=torch.float32)) + pos
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (emb * mask.unsqueeze(-1)).sum(dim=1) / lengths
        return self.seq_encoder(pooled)

    def forward(self, data: Batch) -> torch.Tensor:
        modalities = self.cfg.feature_modalities.lower()
        batch_size = data.seq.size(0)
        node_rep = self.gnn(self._masked_node_features(data.x_s), data.edge_index_s)
        graph_rep = self.pool(node_rep, data.x_s_batch)
        seq_rep = self._encode_sequence(data.seq)
        global_rep = self.global_encoder(data.global_f.float())
        zero = torch.zeros((batch_size, self.cfg.emb_dim), dtype=graph_rep.dtype, device=graph_rep.device)
        graph_rep = graph_rep if "h" in modalities else zero
        seq_rep = seq_rep if "s" in modalities else zero
        global_rep = global_rep if "g" in modalities else zero
        if self.cfg.fusion == "concat":
            fused = self.concat_proj(torch.cat([graph_rep, seq_rep, global_rep], dim=1))
        else:
            fused = self.att(graph_rep, seq_rep, global_rep)
        if self.cfg.use_cfu_feature:
            fused = fused + self.cfu_encoder(data.cfu_id.view(-1).long())
        return fused


class PairModel(nn.Module):
    """Legacy-style pair head: encode both peptides, concat or diff, then MLP."""

    def __init__(self, cfg: Config, interaction: str | None = None):
        super().__init__()
        self.encoder = PeptideEncoder(cfg)
        self.interaction = interaction or cfg.pair_interaction
        width = cfg.emb_dim * 2 if self.interaction == "concat" else cfg.emb_dim
        self.head = nn.Sequential(
            nn.Linear(width, cfg.emb_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(cfg.dropout_ratio),
            nn.Linear(cfg.emb_dim, 1),
        )

    def forward(self, a, b, teacher_delta=None, task_id=None):
        ea, eb = self.encoder(a), self.encoder(b)
        pair = torch.cat([ea, eb], dim=1) if self.interaction == "concat" else ea - eb
        return self.head(pair).squeeze(-1)


class SingleDeltaModel(nn.Module):
    """Score each peptide; delta = score(a) - score(b)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = PeptideEncoder(cfg)
        self.head = nn.Sequential(
            nn.Linear(cfg.emb_dim, cfg.emb_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(cfg.dropout_ratio),
            nn.Linear(cfg.emb_dim, 1),
        )

    def score(self, batch):
        return self.head(self.encoder(batch)).squeeze(-1)

    def forward(self, a, b, teacher_delta=None, task_id=None):
        return self.score(a) - self.score(b)


def build_model(cfg: Config) -> nn.Module:
    if cfg.model_kind == "single":
        return SingleDeltaModel(cfg)
    return PairModel(cfg, cfg.pair_interaction)


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, epochs: int):
    if str(cfg.lr_scheduler).lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(epochs)))
    return None
