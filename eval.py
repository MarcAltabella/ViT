import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def eval_step(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
):

    model_path = Path(
        "/Users/marcaltabella/Desktop/Projects/Code/PersonalProjects/ViT/weights"
    )
    weights_path = model_path / "weights.pt"

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_test, y_test in tqdm(dataloader):
            X_test = X_test.to(device)
            y_test = y_test.to(device)

            logits = model(X_test)
            loss = loss_fn(logits, y_test)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == y_test).sum().item()
            total += y_test.shape[0]

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    results = {"eval_loss": avg_loss, "accuracy": accuracy}
    print(f"Eval Loss: {avg_loss} | Accuracy: {accuracy}")

    results_path = model_path / "eval.json"
    with open(file=results_path, mode="w") as file:
        json.dump(obj=results, fp=file, indent=4)
