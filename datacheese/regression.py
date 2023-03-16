import numpy as np
from numpy.typing import NDArray
from .utils import assert_ndarray_shape, assert_fitted, pad_array


class Regression:
    """
    Linear regression implementation model.

    Examples
    --------
    >>> import numpy as np
    >>> from datacheese.regression import Regression

    Generate data:

    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> # y = 1 * x_0 + 2 * x_1 + 3
    >>> y = np.dot(X, np.array([1, 2])) + 3

    Fit model using data:

    >>> model = Regression()
    >>> model.fit(X, y)

    Use model to make predictions:

    >>> model.predict(np.array([[3, 5]]))
    array([16.])

    Setting ``Lamdba`` to non-zero value performs ridge regression:

    >>> X = np.array([[0, 0], [0, 0], [1, 1]])
    >>> y = np.array([0, .1, 1])
    >>> model.fit(X, y, Lambda=0.5)
    >>> model.predict(np.array([[0.8, 0.9]]))
    array([0.71555556])
    """

    def __init__(self):
        self.fitted = False

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        Lambda: np.float64 = 0.0,
    ):
        """
        Fit linear regression model to training data.

        Parameters
        ----------
        X : numpy.ndarray
            2D training features array, of shape ``n x d``, where ``n`` is the
            number of training examples and ``d`` is the number of dimensions.

        y : numpy.ndarray
            1D training target values array, of length ``n``, where ``n`` is
            the number of training examples.

        Lambda : float, default 0.0
            Regularization constant, lambda, to be used as penalty term weight
            in ridge regression. Default is 0.0, which is the special case of
            ordinary least squares regression.
        """
        assert_ndarray_shape(X, shape=(None, None))
        n, self.d = X.shape
        assert_ndarray_shape(y, shape=n)

        # get one-padded training data
        Xp = pad_array(X, 'right', 1)
        # construct identity matrix
        I = np.identity(self.d + 1, dtype=np.float64)
        # solve linear system
        self.w = np.linalg.solve(
            (Xp.T @ Xp) + (Lambda * I),
            Xp.T @ y,
        )
        # set model as fitted
        self.fitted = True

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Use fitted weights to predict target values for test data.

        Parameters
        ----------
        X : ndarray
            2D testing features array, of shape ``m x d``, where ``m`` is the
            number of testing examples and ``d`` is the number of dimensions.

        Returns
        -------
        y_pred : ndarray
            Array of predicted target values.
        """
        assert_fitted(self.fitted, self.__class__.__name__)
        assert_ndarray_shape(X, shape=(None, self.d))

        # get one-padded testing data
        Xp = pad_array(X, 'right', 1)
        # compute predicted target values
        y_pred = Xp @ self.w

        return y_pred

    def score(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> float:
        """
        Use fitted weights to predict target values for test data and compute
        R-squared value using actual target values.

        Parameters
        ----------
        X : ndarray
            2D testing features array, of shape ``m x d``, where ``m`` is the
            number of testing examples and ``d`` is the number of dimensions.

        y : ndarray
            1D testing target values array, of length ``m``.

        Returns
        -------
        r_squared : float
            R-squared score, a value between 0 and 1.
        """
        assert_ndarray_shape(X, shape=(None, self.d))
        m, _ = X.shape
        assert_ndarray_shape(y, shape=m)

        # get predicted target values
        y_pred = self.predict(X)
        # compute square root of squared sum of regression
        sqrt_ssr = np.linalg.norm(y - y_pred)
        # compute square root of total sum of squares
        sqrt_sst = np.linalg.norm(y - np.mean(y))
        # compute r squared
        r_squared = 1 - ((sqrt_ssr / sqrt_sst) ** 2)

        return r_squared
