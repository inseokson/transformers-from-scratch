import numpy as np
import pytest
import torch

from src.attention import MultiHeadAttention, ScaledDotProductAttention

# Notes: I've written tests separately for Attention-related class and for the MultiHeadAttention class.
# MultiHeadAttention class simply uses Attention class passed in the constructor.
# So, for MultiHeadAttention, tests
# 1. that the split and concat methods work correctly,
# 2. that the tensor returned by the forward method has the desired shape.


@pytest.fixture
def multi_head_attention_base():
    return MultiHeadAttention(3, 2, 4, ScaledDotProductAttention)


def test_split_method(multi_head_attention_base):
    x = torch.tensor(
        [
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]],  # 1st batch  # n_head * d_attention,
            [[0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7]],
        ]
    )  # (2 batches, 1 sequence, 3 heads * 4 d_attention)

    actual = multi_head_attention_base.split(x)
    expected = torch.tensor(
        [
            [  # 1st batch
                [[1, 2, 3, 4]],  # 1st head - 1st sequence - d_attention
                [[5, 6, 7, 8]],
                [[9, 10, 11, 12]],
            ],
            [
                [[0, 0, 0, 0]],
                [[0, 1, 2, 3]],
                [[4, 5, 6, 7]],
            ],
        ]
    )  # (2 batches, 3 heads, 1 sequence, 4 d_attention)

    np.testing.assert_array_equal(actual, expected)


def test_concat_method(multi_head_attention_base):
    x = torch.tensor(
        [
            [  # 1st batch
                [  # 1st head
                    [1, 2, 3, 4],  # 1st sequence
                    [5, 6, 7, 8],  # 2nd sequence
                ],
                [  # 2nd head
                    [9, 10, 11, 12],  # 1st sequence
                    [13, 14, 15, 16],  # 2nd sequence
                ],
                [  # 3rd head
                    [-1, -2, -3, -4],  # 1st sequence
                    [1, 2, 3, 4],  # 2nd sequence
                ],
            ],
            [
                [
                    [0, 0, 0, 0],
                    [0, 1, 2, 3],
                ],
                [
                    [4, 5, 6, 7],
                    [8, 9, 10, 11],
                ],
                [
                    [19, 2, 1, 3],
                    [9, 10, 3, 1],
                ],
            ],
        ]
    )  # (2 batches, 3 heads, 2 sequences, 4 d_attention)

    actual = multi_head_attention_base.concat(x)
    expected = torch.tensor(
        [
            [
                [1, 2, 3, 4, 9, 10, 11, 12, -1, -2, -3, -4],  # 1st batch - concat 1st sequence's values
                [5, 6, 7, 8, 13, 14, 15, 16, 1, 2, 3, 4],
            ],
            [
                [0, 0, 0, 0, 4, 5, 6, 7, 19, 2, 1, 3],
                [0, 1, 2, 3, 8, 9, 10, 11, 9, 10, 3, 1],
            ],
        ]
    )  # (2 batches, 2 sequences, 3 heads * 4 d_attention)

    np.testing.assert_array_equal(actual, expected)


def test_forward(multi_head_attention_base):
    x = torch.rand((10, 13, 2))
    output = multi_head_attention_base.forward(x)

    np.testing.assert_array_equal(x.shape, output.shape)
