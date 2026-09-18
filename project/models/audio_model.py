from typing import Optional

import torch
from torch import Tensor, nn

from ..config import ModelConfig
from .musicnn_encoder import MusicNNEncoder
from .transformer import MusicTransformer
from .vae import VAE


class AudioRepresentationModel(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None, feature_extractor: Optional[nn.Module] = None) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.config = config
        self.musicnn = MusicNNEncoder(config.musicnn_feature_dim, config.transformer_d_model, config.freeze_musicnn, feature_extractor)
        self.transformer = MusicTransformer(config.transformer_d_model, config.transformer_heads, config.transformer_layers, config.transformer_ff_dim, config.transformer_dropout)
        self.vae = VAE(config.vae_input_dim, config.vae_hidden_dim, config.vae_latent_dim, config.vae_dropout)

    def forward(self, windows_or_features: Tensor) -> dict[str, Tensor]:
        tokens = self.musicnn(windows_or_features)
        transformer_embedding = self.transformer(tokens)
        reconstruction, mu, log_var, z = self.vae(transformer_embedding)
        return {"transformer_embedding": transformer_embedding, "reconstruction": reconstruction, "mu": mu, "log_var": log_var, "z": z}