import numpy as np
import pytest
from datacheese.neural_networks import (
    BaseLayer,
    LinearLayer,
    SigmoidLayer,
    TanhLayer,
    ReLULayer,
    LeakyReLULayer,
    MultiLayerPerceptron,
)


def test_MultiLayerPerceptron():
    rng = np.random.default_rng()

    X = rng.normal(loc=0, scale=1, size=(4, 3))
    Y = rng.integers(low=0, high=2, size=(4, 2)).astype(np.float64)

    model = MultiLayerPerceptron(lr=0.5)
    model.add_layer(LinearLayer(3, 2))
    model.fit(X, Y, epochs=3, verbose=1)

    assert model.predict(X).shape == Y.shape

    model = MultiLayerPerceptron(lr=0.5)
    model.add_layer(TanhLayer(3, 4))
    model.add_layer(SigmoidLayer(4, 2))
    model.fit(X, Y, epochs=3, verbose=1)

    assert model.predict(X).shape == Y.shape

    model = MultiLayerPerceptron(lr=0.5)
    model.add_layer(LeakyReLULayer(3, 4))
    model.add_layer(ReLULayer(4, 2))
    model.fit(X, Y, epochs=3, verbose=1)

    assert model.predict(X).shape == Y.shape


def test_BaseLayer_exceptions():
    layer = BaseLayer(3, 4)
    with pytest.raises(NotImplementedError):
        layer.activation(0)

    with pytest.raises(NotImplementedError):
        layer.activation_derivative(0)


def test_LinearLayer_feed_forward_back_propagate():
    rng = np.random.default_rng()

    layer = LinearLayer(2, 3)
    x = rng.normal(loc=0, scale=1, size=2)
    x_padded = np.concatenate((x, [1]))
    out = layer.feed_forward(x)

    assert np.allclose(out, layer.w @ x_padded)

    init_w = layer.w.copy()
    e = rng.normal(loc=0, scale=1, size=3)
    lr = 0.5
    layer.back_propagate(e, lr=lr)

    assert np.allclose(e, layer.delta)

    update_w = np.zeros(layer.w.shape, dtype=np.float64)
    for i in range(update_w.shape[0]):
        for j in range(update_w.shape[1]):
            update_w[i, j] = lr * x_padded[j] * e[i]

    assert np.allclose(update_w, layer.w - init_w)
