import numpy as np
from datacheese.regression import LinearRegression, LogisticRegression


def test_LinearRegression():
    rng = np.random.default_rng()

    n = 10
    d = 4
    t = 2

    X = rng.normal(loc=0, scale=1, size=(n, d))
    W = rng.normal(loc=0, scale=1, size=(d, t))
    b = rng.normal(loc=0, scale=1, size=t)
    Y = (X @ W) + b

    model = LinearRegression()
    model.fit(X, Y, Lambda=0.0)
    assert np.allclose(model.W[1:], W)
    assert np.allclose(model.W[0], b)

    y_pred = model.predict(X)
    assert np.allclose(y_pred, Y)

    r_squared = model.score(X, Y)
    assert np.all(r_squared == 1)


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
    Y = np.column_stack((np.any(X, axis=1), np.all(X, axis=1))).astype(int)

    model = LogisticRegression()
    model.fit(X, Y, method='gradient')

    Y_pred = model.predict(X)
    assert np.all(Y_pred == Y)

    Y_prob = model.predict_prob(X)
    assert np.all(np.abs(Y - Y_prob) < 0.5)

    accuracy = model.score(X, Y, metric='accuracy')
    assert np.all(accuracy == 1)

    log_loss = model.score(X, Y, metric='log_loss')
    assert np.all(log_loss >= 0)


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
    Y = np.column_stack((np.any(X, axis=1), np.all(X, axis=1))).astype(int)

    model = LogisticRegression()
    model.fit(X, Y, method='newton')

    Y_pred = model.predict(X)
    assert np.all(Y_pred == Y)

    Y_prob = model.predict_prob(X)
    assert np.all(np.abs(Y - Y_prob) < 0.5)

    accuracy = model.score(X, Y, metric='accuracy')
    assert np.all(accuracy == 1)

    log_loss = model.score(X, Y, metric='log_loss')
    assert np.all(log_loss >= 0)
