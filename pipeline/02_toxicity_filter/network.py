#!/usr/bin/env python3
"""Toxicity model used by Step 2 (sequence + physicochemical fusion, mode=101)."""

from __future__ import annotations

import torch
import torch.nn as nn


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


class FusionPeptide(nn.Module):
    """
    Lightweight FusionPeptide for hemolytic toxicity inference.

    Default checkpoint uses:
      mode='101' (sequence + globf), q_encoder='gru', g_encoder='mlp'
    """

    def __init__(
        self,
        v_encoder: str = "resnet34",
        q_encoder: str = "gru",
        g_encoder: str = "mlp",
        mode: str = "101",
        classes: int = 1,
        channels: int = 32,
    ):
        super().__init__()
        del v_encoder, channels  # unused in mode=101; kept for checkpoint API compatibility
        if mode == "000":
            raise KeyError("None of the modules are activated")

        self.classes = classes
        self.mode = [False, False, False]
        final_dim = 0

        if mode[0] == "1":
            final_dim += 512
            self.mode[0] = True
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
                raise NotImplementedError(
                    f"q_encoder='{q_encoder}' is not included in this slim package. "
                    "Use 'gru' (default), 'lstm', or 'rnn'."
                )

        if mode[1] == "1":
            raise NotImplementedError(
                "Voxel encoder (mode[1]=='1') is excluded from this slim inference package."
            )

        if mode[2] == "1":
            final_dim += 128
            self.mode[2] = True
            if g_encoder != "mlp":
                raise NotImplementedError(g_encoder)
            self.g_encoder = MLP(10, 128, 128, 3, 0.3)

        self.fc = nn.Sequential(
            nn.Linear(final_dim, 128), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(64, self.classes),
        )

    def forward(self, x):
        _, seq, globf = x
        fusion = []
        if self.mode[0]:
            fusion.append(self.q_encoder(seq)[0][:, -1, :])
        if self.mode[2]:
            fusion.append(self.g_encoder(globf))
        return self.fc(torch.cat(fusion, dim=-1))
