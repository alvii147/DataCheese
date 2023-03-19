import numpy as np
from datacheese.regression import LinearRegression


def test_linear_regression():
    rng = np.random.default_rng()

    n = 10
    d = 4

    X = rng.normal(loc=0, scale=1, size=(n, d))
    w = rng.normal(loc=0, scale=1, size=d)
    b = rng.normal(loc=0, scale=1, size=1)[0]
    y = (X @ w) + b

    model = LinearRegression()
    model.fit(X, y)
    assert np.allclose(model.w[:-1], w)
    assert np.allclose(model.w[-1], b)

    y_pred = model.predict(X)
    assert np.allclose(y_pred, y)

    r_squared = model.score(X, y)
    assert r_squared == 1
