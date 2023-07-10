import numpy as np
import torch

from src.feed_forward import PositionWiseFeedForward


def test_position_wise_feed_forward_base():
    position_wise_feed_forward = PositionWiseFeedForward(10, 20)
    x = torch.rand((5, 128, 10))
    out = position_wise_feed_forward.forward(x)

    np.testing.assert_array_equal(x.shape, out.shape)
