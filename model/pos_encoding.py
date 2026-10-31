'''
Initializing the CLS token and the Pos Embedding and adding them to the input tensor [B, P, D]
Add the CLS token and the positional embedding to the current tensor. [B, P, D] -> [B, P+1, D]
'''
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, d_model)
        )

        self.pos_embed = nn.Parameter(
            torch.rand(1, seq_len + 1, d_model)
        )

    def forward(self, x):
        B = x.shape[0]

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed

        return x
