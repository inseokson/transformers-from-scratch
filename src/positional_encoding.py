import torch
from torch import nn


class SinusoidalPositionalEncoder(nn.Module):
    @torch.no_grad()
    def __init__(self, d_model: int, max_length: int):
        super().__init__()

        if (d_model <= 0) or (max_length <= 0):
            raise ValueError("d_model and max_length must be positive integer.")

        self.d_model = d_model
        self.max_length = max_length

        positions = torch.arange(0, max_length).unsqueeze(dim=1)
        dividors = 10000 ** (torch.arange(0, d_model, step=2) / d_model).unsqueeze(dim=0)
        angles = positions / dividors  # (max_length, d_model)

        even_dim_encodings = torch.sin(angles)
        odd_dim_encoding = torch.cos(angles)
        if d_model & 1 == 1:
            odd_dim_encoding = odd_dim_encoding[:, :-1]

        self.encoding = torch.empty((1, max_length, d_model))
        self.encoding[:, :, ::2] = even_dim_encodings
        self.encoding[:, :, 1::2] = odd_dim_encoding

    def forward(self, x):
        # x: (sequence_length, d_model) or (n_batch, sequence_length, d_model)
        if (x.dim() == 1) or (x.dim() >= 4):
            raise ValueError("Passed input must be 2 or 3-dimension.")
        if x.dim() == 2:
            x = x.unsqueeze(0)

        if x.shape[1] > self.max_length:
            raise ValueError("Length of the input exceeds allowable length of the encoder")
        if x.shape[-1] != self.d_model:
            raise ValueError("Embedding dimension of the input is different from the dimension specified in encoder.")

        return x + self.encoding[:, : x.shape[1]]
