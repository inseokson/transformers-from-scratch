import numpy as np
import torch

from src.normalization import LayerNormalization


def test_layer_normalization_base():
    layer_norm = LayerNormalization(3)
    x = torch.tensor(
        [
            [  # 1st batch
                [0.1, 0.2, 0.8],  # 1st sequence
                [0.3, 0.4, 1.2],
                [0.5, 0.6, -0.5],
            ],
            [
                [-0.1, -0.4, 0.1],
                [0.3, 0.5, 0.7],
                [0.7, 1.2, 2.5],
            ],
            [
                [-1.2, 0.4, 0.1],
                [0.1, 0.8, 0.3],
                [0.3, 0.9, -1.5],
            ],
        ]
    )

    with torch.no_grad():
        actual = layer_norm.forward(x)

    expect = torch.tensor(
        [
            [
                [-0.86261705, -0.539135656, 1.401752706],
                [-0.827580381, -0.579306267, 1.406886648],
                [0.604028206, 0.805370941, -1.409399147],
            ],
            [
                [0.162202214, -1.297617713, 1.135415499],
                [-1.224515296, 0.0, 1.224515296],
                [-1.010553204, -0.351496767, 1.362049971],
            ],
            [
                [-1.392030113, 0.912019729, 0.480010384],
                [-1.018990545, 1.358654059, -0.339663515],
                [0.392230385, 0.980575961, -1.372806346],
            ],
        ]
    )

    np.testing.assert_array_almost_equal(actual, expect)
