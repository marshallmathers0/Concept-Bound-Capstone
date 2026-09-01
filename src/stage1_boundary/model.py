"""
Stage 1 — Pedagogical Boundary Detector.

This is the core trainable contribution of Stage 1: a lightweight temporal
encoder that maps the fused per-time-step feature stream
    x_t = [ocr_change_t, topic_drift_t, visual_change_t, clip_embed_t]
to a sequence of embeddings z_t such that:
    - z_t and z_{t+1} are similar when t, t+1 belong to the same
      concept-coherent segment, and
    - z_t and z_{t+1} are dissimilar when a true concept boundary lies
      between t and t+1.

Two interchangeable encoder backbones are provided:
  - "cnn":          a small stack of dilated 1D convolutions (causal-ish,
                     captures local rhythm patterns cheaply).
  - "transformer":  a small Transformer encoder with sinusoidal positional
                     encoding (captures longer-range dependencies).

The boundary *score* at time t is derived post-hoc as
    b_t = 1 - cosine_similarity(z_t, z_{t+1})
which is high at concept boundaries. See `segment.py` for converting this
score sequence into discrete (start_time, end_time) segments.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class CNNBoundaryEncoder(nn.Module):
    """1D dilated-CNN encoder over the time-series feature stream."""

    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(nn.Conv1d(in_ch, hidden_dim, kernel_size=3,
                                     padding=dilation, dilation=dilation))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_ch = hidden_dim
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, D, T) for Conv1d
        h = self.conv(x.transpose(1, 2))
        h = h.transpose(1, 2)  # (B, T, hidden)
        return self.proj(h)


class TransformerBoundaryEncoder(nn.Module):
    """Small Transformer encoder over the time-series feature stream."""

    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int,
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)
        return self.proj(h)


class BiLSTMBoundaryEncoder(nn.Module):
    """Bi-LSTM encoder over the time-series feature stream.

    Uses bidirectional LSTM to capture both past and future context at each
    time-step. This is particularly effective for long (1-hour) lectures
    where the Transformer encoder may lose context due to quadratic attention
    scaling, while the LSTM's hidden state preserves information over the
    entire sequence.
    """

    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        # hidden_dim // 2 per direction -> hidden_dim total when concatenated
        self.lstm = nn.LSTM(
            input_dim, hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, T, D) -> (B, T, hidden_dim)
        if key_padding_mask is not None:
            # Pack padded sequences for efficient LSTM processing
            lengths = (~key_padding_mask).sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False)
            output, _ = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True, total_length=x.size(1))
        else:
            output, _ = self.lstm(x)
        return self.proj(output)


class PedagogicalBoundaryDetector(nn.Module):
    """Wraps a CNN, Transformer, or Bi-LSTM backbone and L2-normalizes the
    output embeddings so that cosine similarity reduces to a dot product."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, embed_dim: int = 64,
                 encoder_type: str = "bilstm", num_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == "cnn":
            self.backbone = CNNBoundaryEncoder(input_dim, hidden_dim, embed_dim,
                                                num_layers=num_layers, dropout=dropout)
        elif encoder_type == "transformer":
            self.backbone = TransformerBoundaryEncoder(input_dim, hidden_dim, embed_dim,
                                                         num_layers=num_layers,
                                                         num_heads=num_heads, dropout=dropout)
        elif encoder_type == "bilstm":
            self.backbone = BiLSTMBoundaryEncoder(input_dim, hidden_dim, embed_dim,
                                                    num_layers=num_layers, dropout=dropout)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """x: (B, T, D) -> z: (B, T, embed_dim), L2-normalized along last dim."""
        if self.encoder_type in ("transformer", "bilstm"):
            z = self.backbone(x, key_padding_mask=key_padding_mask)
        else:
            z = self.backbone(x)
        z = nn.functional.normalize(z, dim=-1)
        return z

    @staticmethod
    def boundary_scores(z: torch.Tensor) -> torch.Tensor:
        """Compute boundary score b_t = 1 - cos_sim(z_t, z_{t+1}) for t=0..T-2.

        z: (B, T, E) -> scores: (B, T-1)
        """
        sim = (z[:, :-1, :] * z[:, 1:, :]).sum(dim=-1)
        return 1.0 - sim
