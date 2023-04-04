import numpy as np
from numpy.typing import NDArray
from .utils import assert_ndarray_shape


def conv2d(
    img: NDArray[np.float64],
    kernel: NDArray[np.float64],
    stride: tuple[int, int] = (1, 1),
    padding: int = 0,
    fill: float = 0,
) -> NDArray[np.float64]:
    """
    Perform 2D convolution operation over image array using kernel.

    Parameters
    ----------
    img : numpy.ndarray
        2D image array to perform convolution on.

    kernel : numpy.ndarray
        2D kernel array.

    stride : tuple
        2-value tuple representing stride in each dimension.

    padding : int
        Number of layers of padding to add around image.

    fill : float
        Fill value to use when padding.

    Returns
    -------
    out : numpy.nadarray
        2D output array.

    Examples
    --------
    >>> import numpy as np
    >>> from datacheese.convolution import conv2d

    Define image and kernel:

    >>> img = np.array(
    ...     [
    ...         [5, -2, 8, 1],
    ...         [3, -1, 0, -2],
    ...         [-9, 2, 8, -3],
    ...         [4, 7, -3, 4],
    ...     ]
    ... )
    >>> kernel = np.array(
    ...     [
    ...         [-2, 3],
    ...         [0, 1],
    ...     ]
    ... )

    Perform convolution of kernel over image using strides of 2 in both
    dimensions:

    >>> conv2d(img, kernel, stride=(2, 2))
    array([[-17, -15],
           [ 31, -21]])
    """
    assert_ndarray_shape(img, (None, None), array_name='img')
    assert_ndarray_shape(kernel, (None, None), array_name='kernel')
    assert_ndarray_shape(stride, 2, array_name='stride')

    # add padding to image on all sides
    padded_img = np.pad(
        img,
        ((padding, padding), (padding, padding)),
        mode='constant',
        constant_values=fill,
    )

    img_shape = padded_img.shape
    kernel_shape = kernel.shape

    # compute output array shape
    out_shape = (
        ((img_shape[0] - kernel_shape[0]) // stride[0]) + 1,
        ((img_shape[1] - kernel_shape[1]) // stride[1]) + 1,
    )

    # construct array of row indices
    i0 = np.arange(kernel_shape[0])[None, :] + (
        stride[0] * np.arange(out_shape[0])[:, None]
    )
    i = np.repeat(
        np.repeat(i0, kernel_shape[1], axis=1),
        out_shape[1],
        axis=0,
    )

    # construct array of column indices
    j0 = np.arange(kernel_shape[1])[None, :] + (
        stride[1] * np.arange(out_shape[1])[:, None]
    )
    j = np.tile(j0, (out_shape[0], kernel_shape[0]))

    # compute dot product of kernel over image
    out = np.dot(padded_img[i, j], kernel.reshape(-1)).reshape(out_shape)

    return out
