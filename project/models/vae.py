import torch
from torch import Tensor, nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 128, latent_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim))

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        hidden = self.encoder(x)
        mu, log_var = self.mu(hidden), self.log_var(hidden)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var, z


def vae_loss(reconstruction: Tensor, target: Tensor, mu: Tensor, log_var: Tensor, beta: float = 1.0) -> tuple[Tensor, Tensor, Tensor]:
    reconstruction_loss = F.mse_loss(reconstruction, target)
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return reconstruction_loss + beta * kl_loss, reconstruction_loss, kl_loss