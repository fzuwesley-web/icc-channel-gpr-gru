"""
gru_model.py — Joint multi-frequency GRU for channel magnitude prediction.

Input at each timestep: all 100 frequency bins (magnitude in dB, normalized).
Predicts the next timestep across all frequencies.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ChannelGRU(nn.Module):
    """Joint multi-frequency GRU: all 100 frequency bins processed together."""

    def __init__(self, n_freq: int = 100, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_freq,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, n_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_freq) → (batch, n_freq)"""
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

    def predict_sequence(self, seed: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        seed: (1, window, n_freq) → returns (n_steps, n_freq)
        """
        self.eval()
        preds = []
        buf = seed.clone()
        with torch.no_grad():
            for _ in range(n_steps):
                y = self(buf)
                preds.append(y[0])
                buf = torch.cat([buf[:, 1:, :], y.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=0)


class ChannelGRU1D(nn.Module):
    """Per-frequency 1D GRU: each frequency channel modelled independently."""

    def __init__(self, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, 1) → (batch, 1)"""
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

    def predict_sequence(self, seed: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        seed: (1, window, 1) → returns (n_steps,)
        """
        self.eval()
        preds = []
        buf = seed.clone()
        with torch.no_grad():
            for _ in range(n_steps):
                y = self(buf)          # (1, 1)
                preds.append(y[0, 0])
                buf = torch.cat([buf[:, 1:, :], y.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=0)


class ChannelGRUFreqSlice(nn.Module):
    """Per-time-slice GRU: processes the 100 frequency bins as a 1D sequence,
    predicts the next time step's full frequency profile.
    Analogue of GPR's per-time 1D baseline."""

    def __init__(self, n_freq: int = 100, hidden_size: int = 64,
                 num_layers: int = 1):
        super().__init__()
        self.n_freq = n_freq
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, n_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_freq, 1) → (batch, n_freq)"""
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

    def predict_sequence(self, seed: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        seed: (1, n_freq, 1) — normalized freq profile at last training step
        returns: (n_steps, n_freq)
        """
        self.eval()
        preds = []
        current = seed.clone()
        with torch.no_grad():
            for _ in range(n_steps):
                y = self(current)              # (1, n_freq)
                preds.append(y[0])
                current = y.unsqueeze(-1)      # (1, n_freq, 1)
        return torch.stack(preds, dim=0)       # (n_steps, n_freq)
