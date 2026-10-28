"""Deep models for toxicity: FusionPeptide-style, TIGER seq+global, metric learning."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class FusionSeqGlob(nn.Module):
    """POAP/TIGER toxicity FusionPeptide mode=101 (sequence GRU + global MLP)."""

    def __init__(self, q_encoder: str = "gru", glob_dim: int = 10, classes: int = 1):
        super().__init__()
        if q_encoder == "lstm":
            self.q_encoder = nn.LSTM(
                input_size=21, hidden_size=256, num_layers=2,
                dropout=0.1, batch_first=True, bidirectional=True,
            )
        elif q_encoder == "gru":
            self.q_encoder = nn.GRU(
                input_size=21, hidden_size=256, num_layers=2,
                dropout=0.1, batch_first=True, bidirectional=True,
            )
        elif q_encoder == "rnn":
            self.q_encoder = nn.RNN(
                input_size=21, hidden_size=256, num_layers=2,
                dropout=0.1, batch_first=True, bidirectional=True,
            )
        else:
            raise ValueError(q_encoder)
        self.g_encoder = MLP(glob_dim, 128, 128, 3, 0.3)
        self.fc = nn.Sequential(
            nn.Linear(512 + 128, 128), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(64, classes),
        )

    def encode(self, seq, globf):
        q = self.q_encoder(seq)[0][:, -1, :]
        g = self.g_encoder(globf)
        return torch.cat([q, g], dim=-1)

    def forward(self, seq, globf):
        return self.fc(self.encode(seq, globf))


class TIGERSeqGlob(nn.Module):
    """TIGER-style seq+global fusion without structure/GNN."""

    def __init__(self, max_len: int = 50, emb_dim: int = 128, glob_dim: int = 10, dropout: float = 0.2):
        super().__init__()
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.seq_encoder = nn.Sequential(
            nn.Linear(max_len, emb_dim), nn.LeakyReLU(0.1), nn.Dropout(dropout)
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(glob_dim, emb_dim), nn.LeakyReLU(0.1), nn.Dropout(dropout)
        )
        self.query = nn.Linear(emb_dim, emb_dim)
        self.key = nn.Linear(emb_dim, emb_dim)
        self.value = nn.Linear(emb_dim, emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 1),
        )

    def encode(self, seq, globf):
        # seq: [B, L] integer ids
        s = self.seq_encoder(seq.float().view(seq.size(0), -1))
        g = self.global_encoder(globf.float())
        tokens = torch.stack([s, g], dim=1)  # [B, 2, D]
        q = self.query(tokens)
        k = self.key(tokens)
        v = self.value(tokens)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.emb_dim), dim=-1)
        return torch.matmul(attn, v).sum(dim=1)

    def forward(self, seq, globf):
        return self.head(self.encode(seq, globf))


class MetricLearningNet(nn.Module):
    """Embedding network + classification head for supervised metric learning."""

    def __init__(self, max_len: int = 50, emb_dim: int = 128, glob_dim: int = 10, proj_dim: int = 64):
        super().__init__()
        self.backbone = TIGERSeqGlob(max_len=max_len, emb_dim=emb_dim, glob_dim=glob_dim)
        # reuse encoders, replace head with projection + classifier
        self.encoder = self.backbone
        self.projector = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(emb_dim, proj_dim),
        )
        self.classifier = nn.Linear(emb_dim, 1)

    def embed(self, seq, globf):
        return self.encoder.encode(seq, globf)

    def project(self, seq, globf):
        z = F.normalize(self.projector(self.embed(seq, globf)), dim=-1)
        return z

    def forward(self, seq, globf):
        h = self.embed(seq, globf)
        logits = self.classifier(h)
        return logits, F.normalize(self.projector(h), dim=-1)


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss on L2-normalized embeddings."""
    device = features.device
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    logits = torch.matmul(features, features.T) / temperature
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=device)
    mask = mask * logits_mask
    # numerical stability
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
    # mean of positive pairs
    pos_per_sample = mask.sum(dim=1)
    # samples without positives contribute 0
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / pos_per_sample.clamp_min(1.0)
    loss = -mean_log_prob_pos
    valid = pos_per_sample > 0
    if valid.any():
        return loss[valid].mean()
    return features.new_tensor(0.0)
