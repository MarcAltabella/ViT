"""
1. Load images into Tensors:
    - Project images into tokens with a learnable parameter [B, C, H, W] -> [B, D, P_H, P_W]
    - Create patches P_H*P_W = P
    - Transpose from [B, D, P] -> [B, P, D]
"""

import torch.nn as nn

class PatchEmbeddings(nn.Module):
    def __init__(self, d_model, img_size, patch_size, n_channels):
        super().__init__()

        self.d_model = d_model
        self.patch_size = patch_size
        self.image_size = img_size
        self.patch_size = patch_size
        self.n_channels = n_channels

        # From n channels to dimension d of the transformer
        self.linear_project = nn.Conv2d(
            self.n_channels,
            self.d_model,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):
        x = self.linear_project(x)
        x = x.flatten(2)  # [B, D, P_H, P_W] -> [B, D, P]
        x = x.transpose(1, 2)  # [B, D, P] -> [B, P, D]

        return x
