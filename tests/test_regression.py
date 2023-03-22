import numpy as np
from datacheese.regression import LinearRegression, LogisticRegression


def test_LinearRegression():
    rng = np.random.default_rng()

    n = 10
    d = 4
    t = 2

    X = rng.normal(loc=0, scale=1, size=(n, d))
    w = rng.normal(loc=0, scale=1, size=(d, t))
    b = rng.normal(loc=0, scale=1, size=1)[0]
    y = (X @ w) + b

    model = LinearRegression()
    model.fit(X, y, Lambda=0.0)
    assert np.allclose(model.w[1:], w)
    assert np.allclose(model.w[0], b)

    y_pred = model.predict(X)
    assert np.allclose(y_pred, y)

    r_squared = model.score(X, y)
    assert np.allclose(r_squared, 1)


def test_LogisticRegression_gradient():
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.float64,
    )
    y = np.any(X, axis=1).astype(int)

    model = LogisticRegression()
    model.fit(X, y, method='gradient')

    y_pred = model.predict(X)
    assert np.allclose(y_pred, y)

    y_prob = model.predict_prob(X)
    assert np.all(np.abs(y - y_prob) < 0.5)

    accuracy = model.score(X, y, metric='accuracy')
    assert accuracy == 1

    log_loss = model.score(X, y, metric='log_loss')
    assert log_loss >= 0


def test_LogisticRegression_newton():
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.float64,
    )
    y = np.any(X, axis=1).astype(int)

    model = LogisticRegression()
    model.fit(X, y, method='newton')

    y_pred = model.predict(X)
    assert np.allclose(y_pred, y)

    y_prob = model.predict_prob(X)
    assert np.all(np.abs(y - y_prob) < 0.5)

    accuracy = model.score(X, y, metric='accuracy')
    assert accuracy == 1

    log_loss = model.score(X, y, metric='log_loss')
    assert log_loss >= 0
