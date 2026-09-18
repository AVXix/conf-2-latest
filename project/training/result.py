from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def summarize_model_outputs(outputs: dict[str, Tensor]) -> dict[str, tuple[int, ...]]:
    """Return a compact shape summary of the model outputs for reporting."""
    summary: dict[str, tuple[int, ...]] = {}
    for name, value in outputs.items():
        if isinstance(value, torch.Tensor):
            summary[name] = tuple(value.shape)
    return summary


def print_result_summary(outputs: dict[str, Tensor], *, prefix: str = "output") -> None:
    """Print tensor shapes in a compact, research-friendly format."""
    for name, value in outputs.items():
        if isinstance(value, torch.Tensor):
            shape = tuple(value.shape)
            print(f"{prefix}.{name}: {shape}")


if __name__ == "__main__":
    sample = {
        "transformer_embedding": torch.randn(8, 128),
        "reconstruction": torch.randn(8, 128),
        "mu": torch.randn(8, 64),
        "log_var": torch.randn(8, 64),
        "z": torch.randn(8, 64),
    }
    print_result_summary(sample)
