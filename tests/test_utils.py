import numpy as np
import pytest
from datacheese.utils import assert_ndarray_shape, ArrayShapeError


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
