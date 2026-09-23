from typing import Tuple

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class AttentionTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_classes: int = 4,
        max_seq_len: int = 128,
        per_frame: bool = False,
    ):
        super().__init__()
        self.per_frame = per_frame
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos = PositionalEncoding(hidden_dim, max_len=max_seq_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """``key_padding_mask``: [B, T] bool, True at padded positions.

        Returns [B, T, C] when ``per_frame`` else [B, C] (last valid timestep).
        """
        z = self.input_proj(x)
        z = self.pos(z)
        z = self.encoder(z, src_key_padding_mask=key_padding_mask)
        if self.per_frame:
            return self.head(z)
        if key_padding_mask is not None:
            lengths = (~key_padding_mask).long().sum(dim=1).clamp(min=1)
            idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, z.size(-1))
            last = z.gather(1, idx).squeeze(1)
            return self.head(last)
        return self.head(z[:, -1, :])


def logits_to_pred(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits, dim=-1)
    conf, pred = torch.max(probs, dim=-1)
    return pred, conf

