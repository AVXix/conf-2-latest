import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import ModelConfig
from ..data import PreprocessedAudioDataset
from ..models.audio_model import AudioRepresentationModel
from ..models.musicnn_tensorflow import TensorFlowMusicNNExtractor
from .losses import audio_vae_loss
from .validation import validate


def train(model, train_loader: DataLoader, validation_loader: DataLoader, epochs: int, learning_rate: float, beta: float, device: torch.device, checkpoint: str | Path = "best_model.pt") -> None:
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    best = float("inf")
    model.to(device)
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["audio"].to(device))
            total, _, _ = audio_vae_loss(outputs, beta)
            total.backward()
            optimizer.step()
        validation_loss = validate(model, validation_loader, device, beta)
        if validation_loss < best:
            best = validation_loss
            torch.save(model.state_dict(), checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Transformer/VAE audio representation branch")
    parser.add_argument("--data-dir", type=Path, default=Path("preprocessed"))
    parser.add_argument("--musicnn-root", type=Path, default=Path("musicnn"))
    parser.add_argument("--checkpoint", type=Path, default=Path("audio_representation.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=ModelConfig().learning_rate)
    parser.add_argument("--beta", type=float, default=ModelConfig().vae_beta)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")

    config = ModelConfig(learning_rate=args.learning_rate, vae_beta=args.beta)
    train_dataset = PreprocessedAudioDataset(args.data_dir / "train.jsonl")
    validation_dataset = PreprocessedAudioDataset(args.data_dir / "validation.jsonl")
    test_dataset = PreprocessedAudioDataset(args.data_dir / "test.jsonl")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    device = torch.device(args.device)
    extractor = TensorFlowMusicNNExtractor(args.musicnn_root, config.musicnn_model)
    model = AudioRepresentationModel(config, feature_extractor=extractor)
    try:
        train(model, train_loader, validation_loader, args.epochs, config.learning_rate, config.vae_beta, device, args.checkpoint)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        test_loss = validate(model, test_loader, device, config.vae_beta)
        print(f"test_loss={test_loss:.6f}")
        print(f"saved={args.checkpoint}")
    finally:
        extractor.close()


if __name__ == "__main__":
    main()