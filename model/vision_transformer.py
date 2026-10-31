import torch.nn as nn
from model.patch_embedding import PatchEmbeddings
from model.pos_encoding import PositionalEncoding
from model.transformer_encoder import TransformerEncoder


class ViTModel(nn.Module):
    def __init__(
        self,
        img_size: int,
        n_channels: int,
        patch_size: int,
        n_classes: int,
        d_model: int,
        n_heads: int,
        mlp_dim: int,
    ):
        super().__init__()

        self.patchEmbeddings = PatchEmbeddings(
            d_model, img_size, patch_size, n_channels
        )  # create the patch embeddings
        self.posEncoding = PositionalEncoding(
            (img_size // patch_size) ** 2, d_model
        )  # add positional encoding
        self.transformer = TransformerEncoder(
            n_heads, d_model, mlp_dim
        )  # depth transformer encoder blocks
        self.mlp = nn.Linear(d_model, n_classes)

    def forward(self, x):

        x = self.patchEmbeddings(x)  # [B, P, D]
        x = self.posEncoding(x)  # [B, P+1, D]

        out = self.transformer(x)  # concat [B, P+1, D//n_heads] -> [B, P+1, D]

        cls = out[:, 0, :]  # [B, D]
        return self.mlp(cls)  # [B, n_classes]
