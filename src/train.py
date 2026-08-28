"""Training entry point for the CIFAR-10 image classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from dataset import get_dataloaders
from model import get_model


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the YAML training configuration."""

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Training configuration must contain a YAML mapping.")

    return config


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Train the model for one epoch and return loss and accuracy."""

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)

        total_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)

        total += targets.size(0)
        correct += predictions.eq(targets).sum().item()

    if total == 0:
        return 0.0, 0.0

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model and return validation loss and accuracy."""

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        batch_size = inputs.size(0)

        total_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)

        total += targets.size(0)
        correct += predictions.eq(targets).sum().item()

    if total == 0:
        return 0.0, 0.0

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def find_config_path() -> Path:
    """Find the training configuration in common local/container paths."""

    candidates = [
        Path("/app/configs/training_config.yaml"),
        Path("configs/training_config.yaml"),
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find training_config.yaml in /app/configs or configs/."
    )


def main() -> None:
    """Run the complete training pipeline."""

    config_path = find_config_path()
    config = load_config(config_path)

    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]
    output_config = config["output"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(
        json.dumps(
            {
                "event": "training_started",
                "device": str(device),
                "architecture": model_config["architecture"],
                "dataset": data_config["dataset"],
            }
        ),
        flush=True,
    )

    model = get_model(
        architecture=model_config["architecture"],
        num_classes=model_config["num_classes"],
    ).to(device)

    train_loader, val_loader = get_dataloaders(
        data_dir=data_config["data_dir"],
        batch_size=training_config["batch_size"],
        num_workers=data_config.get("num_workers", 0),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
    )

    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = Path(output_config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / output_config["model_name"]

    best_val_loss = float("inf")
    patience_counter = 0
    patience = training_config["early_stopping_patience"]

    epochs = training_config["epochs"]

    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_loss, val_accuracy = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_accuracy, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_accuracy, 4),
        }

        # Required structured JSON-line logging.
        print(json.dumps(log_entry), flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "architecture": model_config["architecture"],
                    "num_classes": model_config["num_classes"],
                },
                checkpoint_path,
            )

            print(
                json.dumps(
                    {
                        "event": "checkpoint_saved",
                        "path": str(checkpoint_path),
                        "epoch": epoch + 1,
                    }
                ),
                flush=True,
            )

        else:
            patience_counter += 1

            if patience_counter >= patience:
                print(
                    json.dumps(
                        {
                            "event": "early_stopping",
                            "epoch": epoch + 1,
                            "patience": patience,
                        }
                    ),
                    flush=True,
                )
                break

    print(
        json.dumps(
            {
                "event": "training_complete",
                "best_val_loss": round(best_val_loss, 4),
                "checkpoint": str(checkpoint_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()