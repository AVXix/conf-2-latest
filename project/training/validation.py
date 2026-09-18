import torch
from torch.utils.data import DataLoader

from .losses import audio_vae_loss


@torch.no_grad()
def validate(model, loader: DataLoader, device: torch.device, beta: float = 1.0) -> float:
    model.eval()
    losses = []
    for batch in loader:
        outputs = model(batch["audio"].to(device))
        losses.append(audio_vae_loss(outputs, beta)[0].item())
    return sum(losses) / max(1, len(losses))