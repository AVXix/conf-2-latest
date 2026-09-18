import torch

from project.models.audio_model import AudioRepresentationModel


def test_complete_forward_and_backward():
    model = AudioRepresentationModel()
    outputs = model(torch.randn(8, 4, 200))
    assert {key: value.shape for key, value in outputs.items()} == {
        "transformer_embedding": (8, 128), "reconstruction": (8, 128),
        "mu": (8, 64), "log_var": (8, 64), "z": (8, 64),
    }
    sum(value.square().mean() for value in outputs.values()).backward()
    assert any(parameter.grad is not None for parameter in model.transformer.parameters())