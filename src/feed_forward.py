import torch
from torch import nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation: nn.Module = nn.ReLU, p_dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = activation()
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor):
        out = self.linear1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)

        return out
