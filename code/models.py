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
    """Multi-head attention over modality tokens with configurable Q/K/V roles.

    Unified modality letters: g=graph (structure GNN), s=sequence, p=physicochemical
    (handcrafted global properties).

    Modes
    -----
    self_gsp / self_sgp / self_pgs
        Self-attention over three stacked modalities in the named order.
    cross_qg / cross_qs / cross_qp
        Cross-attention: Q from one modality, K/V from the other two.
    """

    _SELF_ORDERS = {
        "self_gsp": ("g", "s", "p"),
        "self_sgp": ("s", "g", "p"),
        "self_pgs": ("p", "g", "s"),
    }
    _CROSS_Q = {
        "cross_qg": "g",
        "cross_qs": "s",
        "cross_qp": "p",
    }

    def __init__(self, embedding_dim: int, num_heads: int = 4, mode: str = "self_gsp"):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mode = str(mode or "self_gsp").lower()
        if self.mode not in self._SELF_ORDERS and self.mode not in self._CROSS_Q:
            raise ValueError(
                f"Unsupported fusion_attn_mode={mode!r}; "
                f"expected one of {sorted(self._SELF_ORDERS) + sorted(self._CROSS_Q)}"
            )
        self.query = nn.Linear(embedding_dim, embedding_dim * num_heads)
        self.key = nn.Linear(embedding_dim, embedding_dim * num_heads)
        self.value = nn.Linear(embedding_dim, embedding_dim * num_heads)
        self.proj = nn.Linear(embedding_dim * num_heads, embedding_dim)

    def _pack(self, graph: torch.Tensor, seq: torch.Tensor, p: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"g": graph, "s": seq, "p": p}

    def _attend(self, q_tok: torch.Tensor, kv_tok: torch.Tensor) -> torch.Tensor:
        """q_tok: [B, Q, D], kv_tok: [B, K, D] -> pooled [B, D]."""
        bsz, n_q, _ = q_tok.shape
        n_k = kv_tok.size(1)
        q = self.query(q_tok).view(bsz, n_q, self.num_heads, self.embedding_dim).transpose(1, 2)
        k = self.key(kv_tok).view(bsz, n_k, self.num_heads, self.embedding_dim).transpose(1, 2)
        v = self.value(kv_tok).view(bsz, n_k, self.num_heads, self.embedding_dim).transpose(1, 2)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embedding_dim), dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, n_q, self.num_heads * self.embedding_dim)
        return self.proj(out).sum(dim=1)

    def forward(self, graph: torch.Tensor, seq: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        mods = self._pack(graph, seq, p)
        if self.mode in self._SELF_ORDERS:
            order = self._SELF_ORDERS[self.mode]
            stacked = torch.stack([mods[k] for k in order], dim=1)
            return self._attend(stacked, stacked)
        q_key = self._CROSS_Q[self.mode]
        kv_keys = [k for k in ("g", "s", "p") if k != q_key]
        q_tok = mods[q_key].unsqueeze(1)
        kv_tok = torch.stack([mods[k] for k in kv_keys], dim=1)
        return self._attend(q_tok, kv_tok)


def node_input_dim(cfg: Config) -> int:
    """GNN input width by node_features_mode: prior = AA(20)+energy(20)[+coords];
    prior+emb = that base plus a learnable per-AA embedding; emb = embedding only."""
    base = 43 if bool(getattr(cfg, "include_node_coords", False)) else 40
    mode = str(getattr(cfg, "node_features_mode", "prior")).lower()
    dim = int(getattr(cfg, "node_emb_dim", 90))
    if mode == "emb":
        return dim
    if mode == "prior+emb":
        return base + dim
    return base


class PeptideEncoder(nn.Module):
    def __init__(self, cfg: Config, input_dim: int | None = None):
        super().__init__()
        self.cfg = cfg
        input_dim = int(input_dim if input_dim is not None else node_input_dim(cfg))
        self.input_dim = input_dim
        self.gnn = GNN(cfg.num_layer, input_dim, cfg.emb_dim, cfg.gnn_type)
        # Node amino-acid representation mode (see Config.node_features_mode):
        # prior = frozen descriptor channels only (legacy); prior+emb = descriptor
        # channels concat a learnable per-AA embedding; emb = embedding only.
        self.node_features_mode = str(getattr(cfg, "node_features_mode", "prior")).lower()
        if self.node_features_mode not in ("prior", "prior+emb", "emb"):
            raise ValueError(f"Unsupported node_features_mode={self.node_features_mode}")
        if self.node_features_mode in ("prior+emb", "emb"):
            self.node_aa_embedding = nn.Embedding(
                NUM_AA_TOKENS, int(getattr(cfg, "node_emb_dim", 90)), padding_idx=0
            )
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
        if encoding not in {"integer", "embedding", "onehot", "onehot_prior"}:
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
            # Prior coupling: fixed orthonormal projection of the 20-dim handcrafted
            # AA descriptor. Used as the init (seq_emb_prior_init) and/or the anchor
            # target of the L2 regularizer (seq_emb_prior_reg, train.py).
            if bool(getattr(cfg, "seq_emb_prior_init", False)) or float(getattr(cfg, "seq_emb_prior_reg", 0) or 0) > 0:
                from .utils.features import aa_prior_projected

                self.register_buffer("aa_prior90", aa_prior_projected(cfg.feature_path, cfg.emb_dim, seed=0))
                if bool(getattr(cfg, "seq_emb_prior_init", False)):
                    with torch.no_grad():
                        self.aa_embedding.weight.copy_(self.aa_prior90)
        elif encoding in ("onehot", "onehot_prior"):
            # onehot_prior: per-residue input = [one-hot identity (21) || frozen
            # 20-dim handcrafted prior] -> Linear; i.e. input-level concatenation of
            # a learnable identity basis with the fixed descriptor (prior branch).
            self.aa_embedding = None
            self.pos_embedding = nn.Embedding(cfg.max_len, cfg.emb_dim)
            in_w = NUM_AA_TOKENS + (20 if encoding == "onehot_prior" else 0)
            self.aa_proj = nn.Linear(in_w, cfg.emb_dim)
            if encoding == "onehot_prior":
                from .utils.features import aa_prior_table

                self.register_buffer("aa_prior20", aa_prior_table(cfg.feature_path))
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
        # p_f = 10 base physicochemical props (+2 hydrophobic moments when
        # cfg.p_feature_hmom, see utils/features.py).
        p_dim = 12 if bool(getattr(cfg, "p_feature_hmom", False)) else 10
        self.p_encoder = nn.Sequential(
            nn.Linear(p_dim, cfg.emb_dim), nn.LeakyReLU(0.1), nn.Dropout(cfg.dropout_ratio)
        )
        self.cfu_encoder = nn.Embedding(len(CFU_LEVELS), cfg.emb_dim)
        self.att = SimpleSelfAttention(
            cfg.emb_dim, mode=str(getattr(cfg, "fusion_attn_mode", "self_gsp") or "self_gsp")
        )
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
        elif self.seq_encoding == "onehot_prior":
            oneh = F.one_hot(idx, num_classes=NUM_AA_TOKENS).to(dtype=torch.float32)
            prior = self.aa_prior20[idx]
            emb = self.aa_proj(torch.cat([oneh, prior], dim=-1)) + pos
        else:
            emb = self.aa_proj(F.one_hot(idx, num_classes=NUM_AA_TOKENS).to(dtype=torch.float32)) + pos
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (emb * mask.unsqueeze(-1)).sum(dim=1) / lengths
        return self.seq_encoder(pooled)

    def forward(self, data: Batch) -> torch.Tensor:
        modalities = self.cfg.feature_modalities.lower()
        batch_size = data.seq.size(0)
        if self.node_features_mode == "emb":
            node_x = self.node_aa_embedding(data.node_aa)
        else:
            node_x = self._masked_node_features(data.x_s)
            if self.node_features_mode == "prior+emb":
                node_x = torch.cat([node_x, self.node_aa_embedding(data.node_aa)], dim=-1)
        node_rep = self.gnn(node_x, data.edge_index_s)
        graph_rep = self.pool(node_rep, data.x_s_batch)
        seq_rep = self._encode_sequence(data.seq)
        p_rep = self.p_encoder(data.p_f.float())
        zero = torch.zeros((batch_size, self.cfg.emb_dim), dtype=graph_rep.dtype, device=graph_rep.device)
        graph_rep = graph_rep if "g" in modalities else zero
        seq_rep = seq_rep if "s" in modalities else zero
        p_rep = p_rep if "p" in modalities else zero
        if self.cfg.fusion == "concat":
            fused = self.concat_proj(torch.cat([graph_rep, seq_rep, p_rep], dim=1))
        else:
            fused = self.att(graph_rep, seq_rep, p_rep)
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
        # Optional signed per-pair hydrophobic-moment deltas appended to the head
        # input (Config.pair_hmom_delta): same scale/orientation as the target.
        self.pair_hmom_delta = bool(getattr(cfg, "pair_hmom_delta", False))
        if self.pair_hmom_delta:
            width += 2
        self.head = nn.Sequential(
            nn.Linear(width, cfg.emb_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(cfg.dropout_ratio),
            nn.Linear(cfg.emb_dim, 1),
        )
        # Anchor multitask (Config.anchor_multitask > 0): shared-encoder absolute
        # score head trained jointly with the pair-delta objective (see train.py);
        # mirrors SingleDeltaModel's head shape.
        self.anchor_multitask = float(getattr(cfg, "anchor_multitask", 0) or 0)
        self.anchor_head = (
            nn.Sequential(
                nn.Linear(cfg.emb_dim, cfg.emb_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(cfg.dropout_ratio),
                nn.Linear(cfg.emb_dim, 1),
            )
            if self.anchor_multitask > 0 else None
        )

    def score(self, batch):
        """Absolute per-peptide score (anchor head); requires anchor_multitask."""
        return self.anchor_head(self.encoder(batch)).squeeze(-1)

    def forward(self, a, b, teacher_delta=None, task_id=None):
        ea, eb = self.encoder(a), self.encoder(b)
        pair = torch.cat([ea, eb], dim=1) if self.interaction == "concat" else ea - eb
        if self.pair_hmom_delta:
            pair = torch.cat([pair, a.hmom - b.hmom], dim=1)
        return self.head(pair).squeeze(-1)


class SingleDeltaModel(nn.Module):
    """Internal ablation: one peptide → log2(MIC); no siamese pair head.

    Train / checkpoint selection use ``score(peptide)``. Pair-Δ evaluation
    still calls ``forward(template, mutated) = score(template) - score(mutated)``.
    """

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
