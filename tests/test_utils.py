import numpy as np
import pytest
from datacheese.utils import assert_ndarray_shape, assert_fitted
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
