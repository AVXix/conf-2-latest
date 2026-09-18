import math

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 64) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] > self.encoding.shape[1]:
            raise ValueError("Sequence is longer than the configured positional encoding")
        return x + self.encoding[:, : x.shape[1]]


class MusicTransformer(nn.Module):
    def __init__(self, d_model: int = 128, heads: int = 8, layers: int = 4, ff_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.position = SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff_dim, dropout, activation="gelu", batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d_model))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != 128:
            raise ValueError("Expected Transformer input with shape [B, windows, 128]")
        return self.encoder(self.position(x)).mean(dim=1)