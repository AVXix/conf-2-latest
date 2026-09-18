from typing import Optional

import torch
from torch import Tensor, nn


class MusicNNEncoder(nn.Module):
    """Project official musicnn penultimate embeddings into Transformer tokens.

    The local official MTT_musicnn implementation returns ``penultimate`` with
    200 dimensions (the ``num_units_backend=200`` dense layer). It is not the
    50-tag sigmoid output. ``feature_extractor`` may be supplied to turn raw
    audio windows into these embeddings; tensor inputs are useful for training
    from cached musicnn features and for tests.
    """

    def __init__(
        self,
        feature_dim: int = 200,
        output_dim: int = 128,
        freeze_musicnn: bool = True,
        feature_extractor: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or output_dim <= 0:
            raise ValueError("feature_dim and output_dim must be positive")
        self.feature_dim = feature_dim
        self.feature_extractor = feature_extractor
        self.projection = nn.Linear(feature_dim, output_dim)
        self.freeze_musicnn = freeze_musicnn
        if self.feature_extractor is not None:
            for parameter in self.feature_extractor.parameters():
                parameter.requires_grad = not freeze_musicnn

    def forward(self, windows_or_features: Tensor) -> Tensor:
        if windows_or_features.ndim != 3:
            raise ValueError("Expected [batch, windows, feature_dim] or [batch, windows, samples]")
        features = windows_or_features
        if features.shape[-1] != self.feature_dim:
            if self.feature_extractor is None:
                raise ValueError(
                    "Raw audio windows require a feature_extractor that returns "
                    f"[B, W, {self.feature_dim}]"
                )
            if not isinstance(self.feature_extractor, nn.Module):
                raise TypeError("feature_extractor must be a torch.nn.Module")
            features = self.feature_extractor(features)
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"feature_extractor must return [B, W, {self.feature_dim}]")
        if self.freeze_musicnn:
            features = features.detach()
        return self.projection(features)