import numpy as np
from numpy.typing import NDArray
from typing import Any
from .exceptions import ArrayShapeError, NotFittedError


def assert_ndarray_shape(
    A: NDArray[Any],
    shape: tuple[int | None, ...],
    array_name: str = 'ndarray',
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

    array_name : str, default ``ndarray``
        Display name of given array, used to construct error message.

    Examples
    --------
    >>> import numpy as np
    >>> from datacheese.utils import assert_ndarray_shape
    >>> A = np.zeros((3, 4))
    >>> assert_ndarray_shape(A, shape=(3, 4), array_name='A')

    ``None`` may be used to ignore a dimensions.

    >>> assert_ndarray_shape(A, shape=(3, None), array_name='A')

    ``ArrayShapeError`` is raised when shapes don't match.

    >>> assert_ndarray_shape(A, shape=(None, 7), array_name='A')
    Traceback (most recent call last):
        raise ArrayShapeError(
    ArrayShapeError: Invalid shape for A, expected shape (None, 7), got shape
    (3, 4)

    ``ArrayShapeError`` is raised when number of dimensions don't match.

    >>> assert_ndarray_shape(A, shape=(3, 4, 9), array_name='A')
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
            f'Invalid number of dimensions for {array_name}, '
            f'expected {len(shape)} dimensions, got {len(A_shape)} dimensions'
        )

    # raise exception if shape invalid in any dimension
    for a, s in zip(A_shape, shape):
        if s is not None and a != s:
            raise ArrayShapeError(
                f'Invalid shape for {array_name}, '
                f'expected shape {shape}, got shape {A_shape}'
            )


def assert_fitted(fitted: bool, class_name: str = 'class'):
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
    >>> assert_fitted(True, class_name='myclass')
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


def assert_str_choice(
    str_val: str,
    choices: list[str],
    str_name: str = 'string',
    case_insensitive: bool = False,
):
    """
    Assert that a string value belongs to given list of allowed choices.

    Parameters
    ----------
    str_val : str
        String value.

    choices : list
        List of allowed choices.

    str_name : str, default ``string``
        Display name of string variable, used to construct error message.

    case_insensitive : bool
        Case sensitivity when checking if string value is in given list.

    Examples
    --------
    >>> from datacheese.utils import assert_str_choice
    >>> eu_country = 'Germany'
    >>> choices = ['Germany', 'Italy', 'Spain']
    >>> assert_str_choice(eu_country, choices, str_name='EU country')
    >>> eu_country = 'Britain'
    >>> assert_str_choice(eu_country, choices, str_name='EU country')
    Traceback (most recent call last):
        raise ValueError(
    ValueError: Invalid value 'Britain' for 'EU country', must be one of
    'Germany', 'Italy', 'Spain'.

    Set ``case_insensitive`` to ``True`` to ignore case:

    >>> eu_country = 'germany'
    >>> assert_str_choice(
    ...     eu_country,
    ...     choices,
    ...     str_name='EU country',
    ...     case_insensitive=True,
    ... )
    """
    # convert everything to lowercase if case insensitive
    if case_insensitive:
        str_val_to_check = str_val.lower()
        choices_to_check = [c.lower() for c in choices]
    else:
        str_val_to_check = str_val
        choices_to_check = choices

    # raise error if string value does not match one of given choices
    if str_val_to_check not in choices_to_check:
        choices_str = ', '.join([f'\'{c}\'' for c in choices])
        raise ValueError(
            f'Invalid value \'{str_val}\' for \'{str_name}\', '
            f'must be one of {choices_str}.'
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
    # get padding width based on edge
    pad_width_map = {
        'top': ((1, 0), (0, 0)),
        'bottom': ((0, 1), (0, 0)),
        'left': ((0, 0), (1, 0)),
        'right': ((0, 0), (0, 1)),
    }

    assert_str_choice(
        edge,
        pad_width_map.keys(),
        str_name='edge',
        case_insensitive=True,
    )

    pad_width = pad_width_map[edge]
    Ap = np.pad(A, pad_width, mode='constant', constant_values=c)

    return Ap
