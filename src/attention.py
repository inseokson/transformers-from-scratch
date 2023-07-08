from typing import Type

import torch
from torch import nn


class Attention(nn.Module):
    ...


class ScaledDotProductAttention(Attention):
    @torch.no_grad()
    def __init__(self, d_attention, *args, **kwargs):
        super().__init__()
        self.normalize_factor = d_attention**0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        # q, k, v: (n_batch, seqeunce_length, d_attention) or \
        #  (n_batch, n_head, sequence_length, d_attention)
        k_t = k.transpose(-2, -1)
        score = q @ k_t / self.normalize_factor
        if mask is not None:
            score = torch.masked_fill(score, mask, -torch.inf)

        attention_prob = self.softmax(score)
        attention_value = attention_prob @ v

        return attention_value


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_attention, attention: Type[Attention] = ScaledDotProductAttention):
        super().__init__()
        self.n_head = n_head
        self.d_attention = d_attention
        self.attention = attention(n_head=n_head, d_model=d_model, d_attention=d_attention)

        self.w_q = nn.Linear(d_model, n_head * d_attention, bias=False)
        self.w_k = nn.Linear(d_model, n_head * d_attention, bias=False)
        self.w_v = nn.Linear(d_model, n_head * d_attention, bias=False)
        self.w_o = nn.Linear(n_head * d_attention, d_model, bias=False)

    def forward(self, x, mask=None):
        q, k, v = self.split(self.w_q(x)), self.split(self.w_k(x)), self.split(self.w_v(x))
        out = self.attention(q, k, v, mask)
        out = self.concat(out)
        out = self.w_o(out)

        return out

    def split(self, x):
        n_batch, seq_length, _ = x.shape
        x = x.reshape(n_batch, seq_length, self.n_head, self.d_attention).transpose(1, 2)

        return x

    def concat(self, x):
        n_batch, _, seq_length, _ = x.shape
        x = x.transpose(1, 2).reshape(n_batch, seq_length, self.n_head * self.d_attention)

        return x
