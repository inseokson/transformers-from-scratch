import torch
from torch import nn
from torch.nn.parameter import Parameter


class LayerNormalization(nn.Module):
    def __init__(self, d_model, epsilon=1e-5):
        super().__init__()
        self.gamma = Parameter(torch.ones(d_model))
        self.beta = Parameter(torch.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)

        out = (x - mean) / (var + self.epsilon) ** 0.5
        out = self.gamma * out + self.beta

        return out
