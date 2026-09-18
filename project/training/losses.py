from torch import Tensor

from ..models.vae import vae_loss


def audio_vae_loss(outputs: dict[str, Tensor], beta: float = 1.0) -> tuple[Tensor, Tensor, Tensor]:
    return vae_loss(outputs["reconstruction"], outputs["transformer_embedding"].detach(), outputs["mu"], outputs["log_var"], beta)