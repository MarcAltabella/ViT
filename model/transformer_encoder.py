"""

1. LayerNorm
2. Transformer Block
3. Res Layer
4. LayerNorm
5. MLP
6. Res Layer

"""

import torch
import torch.nn as nn


class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()

        self.head_size = head_size

        # Q, K, V
        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)

    def forward(self, x):
        query_vector = self.query(x)
        key_vector = self.key(x)
        value_vector = self.value(x)

        att_scores = query_vector @ key_vector.transpose(
            -2, -1
        )  # dot product transforming the key_vector
        att_scores = att_scores / (
            self.head_size**0.5
        )  # dividing by the squareroot of the size of the output to scale
        att_weights = torch.softmax(att_scores, dim=-1)  # apply softmax

        o_vector = (
            att_weights @ value_vector
        )  # attention weights @ value vector to obtain the output vector

        return o_vector


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.head_size = d_model // n_heads  # calculate the size of each head

        self.heads = nn.ModuleList(
            [AttentionHead(d_model, self.head_size) for _ in range(n_heads)]
        )

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        For each head concat the output vector on the last result and apply a linear transformation
        [B, T+1, D] -> N_Heads * [B, T+1, D//Head_size]
        Concatenate all the heads outputs [B, T+1, D//Head_size] -> [B, T+1, D]
        Apply a linear transformation
        """
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.W_o(out)

        return out


class TransformerEncoder(nn.Module):
    def __init__(self, n_heads, d_model, mlp_h):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        # LayerNorm
        self.LayerNorm1 = nn.LayerNorm(d_model)

        # OutputVector from the MultiHead Attention
        self.mha = MultiHeadAttention(d_model, n_heads)

        # LayerNorm 2
        self.LayerNorm2 = nn.LayerNorm(d_model)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_h), nn.GELU(), nn.Linear(mlp_h, d_model)
        )

    def forward(self, x):

        # Residual Layer 1
        out = x + self.mha(self.LayerNorm1(x))

        # Residual Layer 2
        out = out + self.mlp(self.LayerNorm2(out))

        return out
