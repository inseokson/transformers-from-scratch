import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    @torch.no_grad()
    def __init__(self, dim_attention):
        super().__init__()
        self.normalize_factor = dim_attention**0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        # q, k, v: (n_batch, seqeunce_length, dim_attention) or \
        #  (n_batch, n_head, sequence_length, dim_attention)
        k_t = k.transpose(-2, -1)
        score = q @ k_t / self.normalize_factor
        if mask is not None:
            score = torch.masked_fill(score, mask)

        attention_prob = self.softmax(score)
        attention_value = attention_prob @ v

        return attention_value
