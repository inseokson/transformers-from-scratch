import numpy as np
import pytest
import torch

from src.positional_encoding import SinusoidalPositionalEncoder


@pytest.fixture
def encoder():
    return SinusoidalPositionalEncoder(4, 8)


@pytest.fixture
def expected_encoding():
    return torch.tensor(
        [
            [
                [0, 1, 0, 1],
                [0.841470985, 0.540302306, 0.009999833, 0.99995],
                [0.909297427, -0.416146837, 0.019998667, 0.999800007],
                [0.141120008, -0.989992497, 0.0299955, 0.999550034],
                [-0.756802495, -0.653643621, 0.039989334, 0.999200107],
                [-0.958924275, 0.283662185, 0.049979169, 0.99875026],
                [-0.279415498, 0.960170287, 0.059964006, 0.99820054],
                [0.656986599, 0.753902254, 0.069942847, 0.997551],
            ]
        ]
    )


def test_sinusoidal_basic(encoder, expected_encoding):
    np.testing.assert_array_almost_equal(encoder.encoding, expected_encoding)


def test_sinusoidal_odd_d_model():
    encoder = SinusoidalPositionalEncoder(3, 8)

    expected_encoding = torch.tensor(
        [
            [
                [0, 1, 0],
                [0.841470985, 0.540302306, 0.002154433],
                [0.909297427, -0.416146837, 0.004308856],
                [0.141120008, -0.989992497, 0.006463259],
                [-0.756802495, -0.653643621, 0.008617632],
                [-0.958924275, 0.283662185, 0.010771965],
                [-0.279415498, 0.960170287, 0.012926248],
                [0.656986599, 0.753902254, 0.015080471],
            ]
        ]
    )
    np.testing.assert_array_almost_equal(encoder.encoding, expected_encoding)


def test_sinusoidal_single_d_model():
    encoder = SinusoidalPositionalEncoder(1, 8)
    expected_encoding = torch.tensor(
        [
            [
                [0],
                [0.841470985],
                [0.909297427],
                [0.141120008],
                [-0.756802495],
                [-0.958924275],
                [-0.279415498],
                [0.656986599],
            ]
        ]
    )
    np.testing.assert_array_almost_equal(encoder.encoding, expected_encoding)


def test_sinusoidal_single_length():
    encoder = SinusoidalPositionalEncoder(4, 1)
    np.testing.assert_array_equal(encoder.encoding, torch.tensor([[[0, 1, 0, 1]]]))


def test_sinusoidal_contructor_exception():
    with pytest.raises(ValueError) as e:
        SinusoidalPositionalEncoder(0, 9)
    assert str(e.value) == "d_model and max_length must be positive integer."

    with pytest.raises(ValueError) as e:
        SinusoidalPositionalEncoder(-41, 0)
    assert str(e.value) == "d_model and max_length must be positive integer."


def test_sinusoidal_output(encoder, expected_encoding):
    x = torch.zeros((3, 6, 4))
    output = encoder.forward(x)

    np.testing.assert_almost_equal((output - expected_encoding[:, :6]).abs().max(), 0)
    np.testing.assert_array_equal(output.shape, (3, 6, 4))


def test_sinusoidal_two_dimension_input(encoder, expected_encoding):
    x = torch.zeros((6, 4))
    output = encoder.forward(x)

    np.testing.assert_array_almost_equal(output, expected_encoding[:, :6])
    np.testing.assert_array_equal(output.shape, (1, 6, 4))


def test_sinusoidal_input_dimension_exception(encoder):
    with pytest.raises(ValueError) as e:
        encoder.forward(torch.zeros(6))
    assert str(e.value) == "Passed input must be 2 or 3-dimension."

    with pytest.raises(ValueError) as e:
        encoder.forward(torch.zeros((6, 3, 1, 2, 5)))
    assert str(e.value) == "Passed input must be 2 or 3-dimension."

    with pytest.raises(ValueError) as e:
        encoder.forward(torch.zeros((3, 8, 2)))
    assert str(e.value) == "Embedding dimension of the input is different from the dimension specified in encoder."


def test_sinusoidal_input_longer_than_max_length_exception(encoder):
    with pytest.raises(ValueError) as e:
        encoder.forward(torch.zeros(2, 12, 4))
    assert str(e.value) == "Length of the input exceeds allowable length of the encoder"


def test_sinusoidal_requires_grad(encoder):
    assert ~encoder.encoding.requires_grad
