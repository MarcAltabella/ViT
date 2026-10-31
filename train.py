import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def train_step(
    epochs: int,
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    resume: bool,
):

    model_path = Path(
        "/Users/marcaltabella/Desktop/Projects/Code/PersonalProjects/ViT/weights"
    )
    checkpoint_path = model_path / "checkpoint.pt"

    results = {"epoch": [], "train_loss": []}

    model = model.to(device)
    start_epoch = 0

    if resume and checkpoint_path.exists():
        checkpoint = torch.load(f=checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0

        for X_train, y_train in tqdm(dataloader):
            X_train = X_train.to(device)
            y_train = y_train.to(device)

            logits = model(X_train)

            loss = loss_fn(logits, y_train)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss_e = total_loss / len(dataloader)

        results["epoch"].append(epoch)
        results["train_loss"].append(avg_loss_e)

        print(f"Epoch: {epoch} | Avg Loss: {avg_loss_e}")

        if epoch % 5 == 0:
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "avg_loss": avg_loss_e,
            }
            torch.save(obj=checkpoint, f=checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    weights_path = model_path / "weights.pt"
    torch.save(obj=model.state_dict(), f=weights_path)
    print(f"Saved weights: {weights_path}")

    results_path = model_path / "results.json"
    with open(file=results_path, mode="w") as file:
        json.dump(obj=results, fp=file, indent=4)
