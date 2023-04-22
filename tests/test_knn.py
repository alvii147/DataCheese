import numpy as np
import pytest
from datacheese.knn import KNN


knn_parameters = (
    (
        np.array(
            [
                [0],
                [1],
                [2],
                [3],
            ],
            dtype=np.float64,
        ),
        np.array(
            [0, 0, 1, 1],
            dtype=int,
        ),
        3,
        np.array(
            [
                [0.1],
                [1.1],
            ],
            dtype=np.float64,
        ),
        np.array(
            [0, 0],
            dtype=int,
        ),
        np.array(
            [0, 0],
            dtype=int,
        ),
        1.0,
    ),
    (
        np.array(
            [
                [8, 8],
                [2, 1],
                [8, 7],
                [1, 2],
                [2, 3],
                [3, 2],
                [7, 8],
                [3, 3],
                [6, 7],
                [7, 6],
            ],
            dtype=np.float64,
        ),
        np.array(
            [2, 1, 2, 1, 1, 1, 2, 1, 2, 2],
            dtype=int,
        ),
        3,
        np.array(
            [
                [7, 7],
                [4, 4],
            ],
            dtype=np.float64,
        ),
        np.array(
            [2, 2],
            dtype=int,
        ),
        np.array(
            [2, 1],
            dtype=int,
        ),
        0.5,
    ),
    (
        np.array(
            [
                [5, 4, 6],
                [5, 6, 4],
                [8, 9, 7],
                [6, 4, 5],
                [6, 5, 4],
                [3, 1, 2],
                [9, 7, 8],
                [1, 2, 3],
                [2, 3, 1],
                [7, 8, 9],
                [9, 8, 7],
                [7, 9, 8],
                [8, 7, 9],
                [3, 2, 1],
                [4, 6, 5],
                [1, 3, 2],
                [4, 5, 6],
                [2, 1, 3],
            ],
            dtype=np.float64,
        ),
        np.array(
            [2, 2, 3, 2, 2, 1, 3, 1, 1, 3, 3, 3, 3, 1, 2, 1, 2, 1],
            dtype=int,
        ),
        5,
        np.array(
            [
                [2, 2, 2],
            ],
            dtype=np.float64,
        ),
        np.array(
            [0],
            dtype=int,
        ),
        np.array(
            [1],
            dtype=int,
        ),
        0.0,
    ),
)


@pytest.mark.parametrize(
    'X_train, y_train, k, X_test, y_test, y_pred_expected, accuracy_expected',
    knn_parameters,
)
def test_knn(
    X_train,
    y_train,
    k,
    X_test,
    y_test,
    y_pred_expected,
    accuracy_expected,
):
    model = KNN()
    model.fit(X_train, y_train)
    y_pred_computed = model.predict(X_test, k=k)

    assert np.array_equal(y_pred_computed, y_pred_expected)

    accuracy_computed = model.score(X_test, y_test, k=k)

    assert np.allclose(accuracy_computed, accuracy_expected)
