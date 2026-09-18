import torch

from project.config import ModelConfig
from project.models.transformer import MusicTransformer
from project.models.vae import VAE


def test_transformer_shape():
    assert MusicTransformer()(torch.randn(8, 4, 128)).shape == (8, 128)


def test_vae_shapes():
    reconstruction, mu, log_var, z = VAE()(torch.randn(8, 128))
    assert reconstruction.shape == (8, 128)
    assert mu.shape == log_var.shape == z.shape == (8, ModelConfig().vae_latent_dim)