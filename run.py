# %%
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from data import IMG_SIZE, N_CHANNELS, N_CLASSES, get_dataloader
from eval import eval_step
from model.vision_transformer import ViTModel
from train import train_step


def main():
    dataloader = get_dataloader(32, True)
    test_dataloader = get_dataloader(32, False, train=False)
    model = ViTModel(
        img_size=IMG_SIZE,
        n_channels=N_CHANNELS,
        patch_size=8,
        n_classes=N_CLASSES,
        d_model=64,
        n_heads=4,
        mlp_dim=128,
    )
    loss_fn = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)

    train_step(
        epochs=10,
        dataloader=dataloader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=torch.device("cpu"),
        resume=False,
    )
    eval_step(
        dataloader=test_dataloader,
        model=model,
        loss_fn=loss_fn,
        device=torch.device("cpu"),
    )


if __name__ == "__main__":
    main()
