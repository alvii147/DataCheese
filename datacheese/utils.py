import numpy as np
from numpy.typing import NDArray
from typing import Any
from .exceptions import ArrayShapeError, NotFittedError


def assert_ndarray_shape(
    A: NDArray[Any],
    shape: tuple[int | None, ...],
    name: str = 'ndarray',
):
    """
    Assert array is of given shape.

    Parameters
    ----------
    A : numpy.ndarray
        Array to check shape of.

    shape : tuple
        Shape to expect, represented by a tuple of dimensions. Dimensions with
        ``None`` are ignored.

    name : str, default ``ndarray``
        Display name of given array, used to construct error message.

    Examples
    --------
    >>> import numpy as np
    >>> from datacheese.utils import assert_ndarray_shape
    >>> A = np.zeros((3, 4))
    >>> assert_ndarray_shape(A, shape=(3, 4), name='A')

    ``None`` may be used to ignore a dimensions.

    >>> assert_ndarray_shape(A, shape=(3, None), name='A')

    ``ArrayShapeError`` is raised when shapes don't match.

    >>> assert_ndarray_shape(A, shape=(None, 7), name='A')
    Traceback (most recent call last):
        raise ArrayShapeError(
    ArrayShapeError: Invalid shape for A, expected shape (None, 7), got shape
    (3, 4)

    ``ArrayShapeError`` is raised when number of dimensions don't match.

    >>> assert_ndarray_shape(A, shape=(3, 4, 9), name='A')
    Traceback (most recent call last):
        raise ArrayShapeError(
    ArrayShapeError: Invalid number of dimensions for A, expected 3 dimensions,
    got 2 dimensions
    """
    A_shape = np.shape(A)
    # convert scalar shape to tuple
    try:
        iter(shape)
    except TypeError:
        shape = (shape,)

    # raise exception if number of dimensions are not equal
    if len(A_shape) != len(shape):
        raise ArrayShapeError(
            f'Invalid number of dimensions for {name}, '
            f'expected {len(shape)} dimensions, got {len(A_shape)} dimensions'
        )

    # raise exception if shape invalid in any dimension
    for a, s in zip(A_shape, shape):
        if s is not None and a != s:
            raise ArrayShapeError(
                f'Invalid shape for {name}, '
                f'expected shape {shape}, got shape {A_shape}'
            )


def assert_fitted(fitted, class_name='class'):
    """
    Assert that an estimator has been fitted.

    Parameters
    ----------
    fitted : bool
        Whether or not the given estimator has been fitted.

    class_name : str, default ``class``
        Display name of class instance, used to construct error message.

    Examples
    --------
    >>> from datacheese.utils import assert_fitted
    >>> assert_fitted(True)
    >>> assert_fitted(False)
    Traceback (most recent call last):
        raise NotFittedError(
    datacheese.exceptions.NotFittedError: This class instance has not been
    fitted yet. Call 'fit' method before using this estimator.
    >>> assert_fitted(False, class_name='myclass')
    Traceback (most recent call last):
        raise NotFittedError(
    datacheese.exceptions.NotFittedError: This myclass instance has not been
    fitted yet. Call 'fit' method before using this estimator.
    """
    # raise error if estimator class not fitted
    if not fitted:
        raise NotFittedError(
            f'This {class_name} instance has not been fitted yet. '
            'Call \'fit\' method before using this estimator.'
        )


def pad_array(A: NDArray[np.float64], edge: str, c: float):
    """
    Add constant padding to 2D array on one side.

    Parameters
    ----------
    A : numpy.ndarray
        Array to be padded.

    edge : str
        Edge on which padding is to be added. Must be one of ``top``,
        ``bottom``, ``left``, or ``right``.

    c : float
        Constant to be padded.

    Returns
    -------
    Ap : numpy.ndarray
        Padded array.

    Examples
    --------
    >>> import numpy as np
    >>> from datacheese.utils import pad_array
    >>> A = np.zeros((3, 4), dtype=np.float64)
    >>> A
    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])
    >>> pad_array(A, 'right', 2)
    array([[0., 0., 0., 0., 2.],
           [0., 0., 0., 0., 2.],
           [0., 0., 0., 0., 2.]])
    >>> pad_array(A, 'bottom', -1)
    array([[ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.],
           [-1., -1., -1., -1.]])
    """
    pad_width_map = {
        'top': ((1, 0), (0, 0)),
        'bottom': ((0, 1), (0, 0)),
        'left': ((0, 0), (1, 0)),
        'right': ((0, 0), (0, 1)),
    }
    pad_width = pad_width_map[edge]
    Ap = np.pad(A, pad_width, mode='constant', constant_values=c)

    return Ap
