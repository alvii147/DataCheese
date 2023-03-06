import numpy as np
from numpy.typing import NDArray


def sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Sigmoid function, :math:`\\sigma(x)`.

    .. math::
        \\sigma(x) = \\frac{1}{1 + e^{-x}}

    Parameters
    ----------
    x : numpy.ndarray
        Input values.

    Returns
    -------
    f : numpy.ndarray
        Output values.
    """
    f = 1 / (1 + np.exp(-x))

    return f


def sigmoid_derivative(
    x: NDArray[np.float64] | None = None,
    f: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """
    Sigmoid derivative function, :math:`\\sigma(x)'`.

    If ``x`` is provided, the output is computed directly:

    .. math::
        \\sigma'(x) = \\frac{e^{-x}}{(1 + e^{-x})^2}

    If the output of antiderivative ``f`` is provided, the output is computed
    using the antiderivative:

    .. math::
        \\sigma'(x) = \\sigma(x) \\bigg(1 - \\sigma(x)\\bigg)

    Parameters
    ----------
    x : numpy.ndarray, default ``None``
        Input values.

    f : numpy.ndarray, default ``None``
        Output values of antiderivative function.

    Returns
    -------
    f_prime : numpy.ndarray
        Output values.
    """
    if x is not None:
        ex = np.exp(-x)
        f_prime = ex / ((1 + ex) ** 2)

        return f_prime

    if f is not None:
        f_prime = f * (1 - f)

        return f_prime

    raise ValueError('One of parameters x or f must be passed')


def tanh(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Hyperbolic tangent function, :math:`\\tanh(x)`.

    .. math::
        \\tanh(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}

    Parameters
    ----------
    x : numpy.ndarray
        Input values.

    Returns
    -------
    f : numpy.ndarray
        Output values.
    """
    f = np.tanh(x)

    return f


def tanh_derivative(
    x: NDArray[np.float64] | None = None,
    f: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """
    Hyperbolic tangent derivative function, :math:`\\tanh'(x)`.

    If ``x`` is provided, the output is computed directly:

    .. math::
        \\tanh'(x) = 1 - \\bigg(\\frac{e^x - e^{-x}}{e^x + e^{-x}}\\bigg)^2

    If the output of antiderivative ``f`` is provided, the output is computed
    using the antiderivative:

    .. math::
        \\tanh'(x) = 1 - \\tanh^2(x)

    Parameters
    ----------
    x : numpy.ndarray, default ``None``
        Input values.

    f : numpy.ndarray, default ``None``
        Output values of antiderivative function.

    Returns
    -------
    f_prime : numpy.ndarray
        Output values.
    """
    if x is not None:
        f_prime = 1 - (np.tanh(x) ** 2)

        return f_prime

    if f is not None:
        f_prime = 1 - (f**2)

        return f_prime

    raise ValueError('One of parameters x or f must be passed')


def relu(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    ReLU (rectified linear unit) function.

    .. math::
        \\text{relu}(x) = \\max(0, x)

    Parameters
    ----------
    x : numpy.ndarray
        Input values.

    Returns
    -------
    f : numpy.ndarray
        Output values.
    """
    f = np.maximum(0, x)

    return f


def relu_derivative(
    x: NDArray[np.float64] | None = None,
    f: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """
    ReLU (rectified linear unit) derivative function.

    If ``x`` is provided, the output is computed directly:

    .. math::
        \\text{relu}'(x) = \\begin{cases}
            0 & \\text{ if } x \\le 0 \\\\
            1 & \\text{ if } x > 0
        \\end{cases}

    If the output of antiderivative ``f`` is provided, the output is computed
    using the antiderivative:

    .. math::
        \\text{relu}'(x) = \\begin{cases}
            0 & \\text{ if } \\text{relu}(x) = 0 \\\\
            1 & \\text{ if } \\text{relu}(x) \\ne 0
        \\end{cases}

    Parameters
    ----------
    x : numpy.ndarray, default ``None``
        Input values.

    f : numpy.ndarray, default ``None``
        Output values of antiderivative function.

    Returns
    -------
    f_prime : numpy.ndarray
        Output values.
    """
    if x is not None:
        f_prime = (x > 0).astype(np.float64)

        return f_prime

    if f is not None:
        f_prime = (f != 0).astype(np.float64)

        return f_prime

    raise ValueError('One of parameters x or f must be passed')


def leaky_relu(
    x: NDArray[np.float64],
    alpha: float = 0.01,
) -> NDArray[np.float64]:
    """
    Leaky ReLU (rectified linear unit) function.

    .. math::
        \\text{leakyrelu}(x) = \\max(\\alpha x, x)

    Parameters
    ----------
    x : numpy.ndarray
        Input values.

    alpha : float
        Negative slope :math:`\\alpha`, where :math:`0 \\le \\alpha \\le 1`.

    Returns
    -------
    f : numpy.ndarray
        Output values.
    """
    if alpha < 0 or alpha > 1:
        raise ValueError('alpha must be a value between 0 and 1')

    f = np.maximum(alpha * x, x)

    return f


def leaky_relu_derivative(
    x: NDArray[np.float64] | None = None,
    f: NDArray[np.float64] | None = None,
    alpha: float = 0.01,
) -> NDArray[np.float64]:
    """
    Leaky ReLU (rectified linear unit) derivative function.

    If ``x`` is provided, the output is computed directly:

    .. math::
        \\text{leakyrelu}'(x) = \\begin{cases}
            \\alpha & \\text{ if } x \\le 0 \\\\
            1 & \\text{ if } x > 0
        \\end{cases}

    If the output of antiderivative ``f`` is provided, the output is computed
    using the antiderivative:

    .. math::
        \\text{leakyrelu}'(x) = \\begin{cases}
            \\alpha & \\text{ if } \\text{leakyrelu}(x) \\le 0 \\\\
            1 & \\text{ if } \\text{leakyrelu}(x) > 0
        \\end{cases}

    Parameters
    ----------
    x : numpy.ndarray, default ``None``
        Input values.

    f : numpy.ndarray, default ``None``
        Output values of antiderivative function.

    Returns
    -------
    f_prime : numpy.ndarray
        Output values.
    """
    if alpha < 0 or alpha > 1:
        raise ValueError('alpha must be a value between 0 and 1')

    if x is not None:
        f_prime = np.where(x < 0, alpha, 1)

        return f_prime

    if f is not None:
        f_prime = np.where(f < 0, alpha, 1)

        return f_prime

    raise ValueError('One of parameters x or f must be passed')
