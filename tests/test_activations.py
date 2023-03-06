import numpy as np
import pytest
from datacheese.activations import (
    sigmoid,
    sigmoid_derivative,
    tanh,
    tanh_derivative,
    relu,
    relu_derivative,
    leaky_relu,
    leaky_relu_derivative,
)


def test_sigmoid():
    rng = np.random.default_rng()

    f = rng.uniform(low=0, high=1, size=(3, 1, 4))
    f_prime = f * (1 - f)
    x = np.log(f / (1 - f))

    assert np.allclose(sigmoid(x), f)
    assert np.allclose(sigmoid_derivative(x=x), f_prime)
    assert np.allclose(sigmoid_derivative(f=f), f_prime)

    with pytest.raises(ValueError):
        sigmoid_derivative()


def test_tanh():
    rng = np.random.default_rng()

    f = rng.uniform(low=0, high=1, size=(3, 1, 4))
    f_prime = 1 - (f**2)
    x = 0.5 * np.log((1 + f) / (1 - f))

    assert np.allclose(tanh(x), f)
    assert np.allclose(tanh_derivative(x=x), f_prime)
    assert np.allclose(tanh_derivative(f=f), f_prime)

    with pytest.raises(ValueError):
        tanh_derivative()


def test_relu():
    rng = np.random.default_rng()

    x = rng.uniform(low=-1, high=1, size=(3, 1, 4))
    f = np.maximum(0, x)
    f_prime = (x > 0).astype(np.float64)

    assert np.allclose(relu(x), f)
    assert np.allclose(relu_derivative(x=x), f_prime)
    assert np.allclose(relu_derivative(f=f), f_prime)

    with pytest.raises(ValueError):
        relu_derivative()


def test_leaky_relu():
    rng = np.random.default_rng()

    alpha = 0.05
    f = rng.uniform(low=-1, high=1, size=(3, 1, 4))
    f_prime = np.where(f < 0, alpha, 1)
    x = np.where(f < 0, f / alpha, f)

    assert np.allclose(leaky_relu(x, alpha=alpha), f)
    assert np.allclose(leaky_relu_derivative(x=x, alpha=alpha), f_prime)
    assert np.allclose(leaky_relu_derivative(f=f, alpha=alpha), f_prime)

    with pytest.raises(ValueError):
        leaky_relu_derivative(alpha=alpha)

    with pytest.raises(ValueError):
        leaky_relu(x, alpha=3.14)

    with pytest.raises(ValueError):
        leaky_relu_derivative(x=x, f=f, alpha=3.14)
