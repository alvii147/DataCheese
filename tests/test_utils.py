import numpy as np
import pytest
from datacheese.utils import assert_ndarray_shape, assert_fitted, pad_array
from datacheese.exceptions import ArrayShapeError, NotFittedError


def test_assert_array_shape():
    x = np.zeros(8)
    assert_ndarray_shape(x, shape=8)
    assert_ndarray_shape(x, shape=None)

    x = np.zeros((3, 1, 4))

    assert_ndarray_shape(x, shape=(3, 1, 4))
    assert_ndarray_shape(x, shape=(3, None, 4))
    assert_ndarray_shape(x, shape=(None, None, 4))
    assert_ndarray_shape(x, shape=(None, None, None))

    x = np.zeros((6, 9))

    with pytest.raises(ArrayShapeError):
        assert_ndarray_shape(x, shape=(3, 4))

    with pytest.raises(ArrayShapeError):
        assert_ndarray_shape(x, shape=(3, None))

    with pytest.raises(ArrayShapeError):
        assert_ndarray_shape(x, shape=(None, None, None))


def test_assert_fitted():
    assert_fitted(True)
    assert_fitted(True, class_name='myclass')

    with pytest.raises(NotFittedError):
        assert_fitted(False)

    with pytest.raises(NotFittedError):
        assert_fitted(False, class_name='myclass')


def test_pad_array():
    rng = np.random.default_rng()
    A = rng.uniform(low=-1, high=1, size=(4, 4))

    Ap = pad_array(A, 'top', 3)
    assert Ap.shape == (5, 4)
    assert np.all(Ap[0] == 3)
    assert np.allclose(Ap[1:], A)

    Ap = pad_array(A, 'bottom', 1)
    assert Ap.shape == (5, 4)
    assert np.all(Ap[-1] == 1)
    assert np.allclose(Ap[:-1], A)

    Ap = pad_array(A, 'left', 4)
    assert Ap.shape == (4, 5)
    assert np.all(Ap[:, 0] == 4)
    assert np.allclose(Ap[:, 1:], A)

    Ap = pad_array(A, 'right', 1)
    assert Ap.shape == (4, 5)
    assert np.all(Ap[:, -1] == 1)
    assert np.allclose(Ap[:, :-1], A)
