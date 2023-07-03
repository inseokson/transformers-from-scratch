import numpy as np
import torch

from src.attention import ScaledDotProductAttention


def test_scaled_dot_product_single_head_single_batch():
    attention = ScaledDotProductAttention(3)

    q = torch.tensor(
        [
            [
                [0.1, 0.1, 0.7],
                [0.3, -0.4, 0.6],
            ]
        ]
    )
    k = v = torch.tensor(
        [
            [
                [0.2, 0.5, -0.9],
                [-0.4, 0.1, 1.0],
            ]
        ]
    )
    actual_without_head = attention.forward(q, k, v)
    expected_without_head = torch.tensor(
        [
            [
                [-0.202259367, 0.231827089, 0.373821327],
                [-0.193750695, 0.237499537, 0.346877202],
            ]
        ]
    )

    actual_with_head = attention.forward(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
    expected_with_head = expected_without_head.unsqueeze(0)

    np.testing.assert_array_almost_equal(actual_without_head, expected_without_head)
    np.testing.assert_array_almost_equal(actual_with_head, expected_with_head)


def test_scaled_dot_product_single_head_multi_batch():
    attention = ScaledDotProductAttention(3)

    q = torch.tensor(
        [
            [
                [0.1, 0.1, 0.7],
                [0.3, -0.4, 0.6],
            ],
            [
                [0.1, 1.3, 1.7],
                [-0.4, -0.4, 2.0],
            ],
        ]
    )
    k = v = torch.tensor(
        [
            [
                [0.2, 0.5, -0.9],
                [-0.4, 0.1, 1.0],
            ],
            [
                [-0.3, -0.5, 1.0],
                [-0.4, -0.1, 1.3],
            ],
        ]
    )
    actual = attention.forward(q, k, v)
    expected = torch.tensor(
        [
            [
                [-0.202259367, 0.231827089, 0.373821327],
                [-0.193750695, 0.237499537, 0.346877202],
            ],
            [
                [-0.364311209, -0.242755164, 1.192933627],
                [-0.356884201, -0.272463197, 1.170652602],
            ],
        ]
    )

    np.testing.assert_array_almost_equal(actual, expected)


def test_scaled_dot_product_multi_head_multi_batch():
    attention = ScaledDotProductAttention(3)

    q = torch.tensor(
        [
            [
                [
                    [0.1, 0.1, 0.7],
                    [0.3, -0.4, 0.6],
                ],
                [
                    [0.1, 1.3, 1.7],
                    [-0.4, -0.4, 2.0],
                ],
            ],
            [
                [
                    [-0.7, 2.3, 3.4],
                    [-1.0, -0.1, 1.2],
                ],
                [
                    [1.3, 0.1, 1.7],
                    [-2.0, -0.1, 1.2],
                ],
            ],
        ]
    )
    k = v = torch.tensor(
        [
            [
                [
                    [0.2, 0.5, -0.9],
                    [-0.4, 0.1, 1.0],
                ],
                [
                    [-0.3, -0.5, 1.0],
                    [-0.4, -0.1, 1.3],
                ],
            ],
            [
                [
                    [-0.1, -0.1, 1.0],
                    [4.0, -0.2, 1.3],
                ],
                [
                    [-0.1, 0.4, 1.0],
                    [0.3, 0.4, 0.7],
                ],
            ],
        ]
    )
    actual = attention.forward(q, k, v)
    expected = torch.tensor(
        [
            [
                [
                    [-0.202259367, 0.231827089, 0.373821327],
                    [-0.193750695, 0.237499537, 0.346877202],
                ],
                [
                    [-0.364311209, -0.242755164, 1.192933627],
                    [-0.356884201, -0.272463197, 1.170652602],
                ],
            ],
            [
                [
                    [0.848400456, -0.123131718, 1.069395155],
                    [0.326412612, -0.110400308, 1.031200923],
                ],
                [
                    [0.100577349, 0.4, 0.849566989],
                    [0.035423251, 0.4, 0.898432562],
                ],
            ],
        ]
    )

    np.testing.assert_array_almost_equal(actual, expected)
