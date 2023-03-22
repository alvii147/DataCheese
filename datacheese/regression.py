import numpy as np
from numpy.typing import NDArray
from .activations import sigmoid, sigmoid_derivative
from .utils import (
    assert_ndarray_shape,
    assert_fitted,
    assert_str_choice,
    pad_array,
)


class LinearRegression:
    """
    Ordinary least squares linear regression model.

    Examples
    --------
    >>> import numpy as np
    >>> from datacheese.regression import LinearRegression

    Generate input data:

    >>> X = np.array(
    ...     [
    ...         [1, 1, 0],
    ...         [1, 2, -1],
    ...         [2, -2, 0],
    ...         [0, -3, 7],
    ...         [8, -6, 0],
    ...     ],
    ...     dtype=np.float64,
    ... )
    >>> X
    array([[ 1.,  1.,  0.],
           [ 1.,  2., -1.],
           [ 2., -2.,  0.],
           [ 0., -3.,  7.],
           [ 8., -6.,  0.]])

    Generate target values using equations :math:`y_0 = x_0 + 2 x_1 + x_2 + 3`
    and :math:`y_1 = 2 x_0 - x_1 - 3 x_2 - 2`:

    >>> Y = np.matmul(
    ...     X,
    ...     np.array([[1, 2], [2, -1], [1, -3]]),
    ... ) + np.array([3, -2])
    >>> Y
    array([[  6.,  -1.],
           [  7.,   1.],
           [  1.,   4.],
           [  4., -20.],
           [ -1.,  20.]])

    Fit model using data:

    >>> model = LinearRegression()
    >>> model.fit(X, Y)

    Use model to make predictions:

    >>> X_test = np.array([[3, 5, -2], [-2, 4, 3]], dtype=np.float64)
    >>> X_test
    array([[ 3.,  5., -2.],
           [-2.,  4.,  3.]])
    >>> Y_test = np.matmul(
    ...     X_test,
    ...     np.array([[1, 2], [2, -1], [1, -3]]),
    ... ) + np.array([3, -2])
    >>> Y_test
    array([[ 14.,   5.],
           [ 12., -19.]])
    >>> model.predict(X_test)
    array([[ 14.,   5.],
           [ 12., -19.]])

    Compute :math:`R^2` accuracies:

    >>> model.score(X_test, Y_test)
    array([1., 1.])

    Setting ``Lamdba`` to non-zero value performs ridge regression:

    >>> model.fit(X, Y, Lambda=0.5)
    >>> model.predict(X_test)
    array([[ 13.6669421 ,   4.50426919],
           [ 11.51303651, -18.88126742]])
    >>> model.score(X_test, Y_test)
    array([0.99827693, 0.99946753])
    """

    def __init__(self):
        self.fitted = False

    def fit(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        Lambda: float = 0.0,
    ):
        """
        Fit linear regression model to training data.

        Parameters
        ----------
        X : numpy.ndarray
            2D training features array, of shape ``n x d``, where ``n`` is the
            number of training examples and ``d`` is the number of dimensions.

        Y : numpy.ndarray
            2D training target values array, of shape ``n x t``, where ``n`` is
            the number of training examples and ``t`` is the number of targets.

        Lambda : float, default 0.0
            Regularization constant to be used as L2 penalty term weight in
            ridge regression. Default is 0.0, which is the special case of
            ordinary least squares regression.
        """
        assert_ndarray_shape(X, shape=(None, None))
        n, self.d = X.shape
        assert_ndarray_shape(Y, shape=(n, None))

        # get one-padded training data
        Xp = pad_array(X, 'left', 1)
        # construct identity matrix
        I = np.identity(self.d + 1, dtype=np.float64)
        # solve linear system
        self.w = np.linalg.solve(
            (Xp.T @ Xp) + (Lambda * I),
            Xp.T @ Y,
        )
        # set model as fitted
        self.fitted = True

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Use fitted weights to predict target values for test data.

        Parameters
        ----------
        X : numpy.ndarray
            2D testing features array, of shape ``m x d``, where ``m`` is the
            number of testing examples and ``d`` is the number of dimensions.

        Returns
        -------
        Y_pred : numpy.ndarray
            2D array of predicted target values.
        """
        assert_fitted(self.fitted, self.__class__.__name__)
        assert_ndarray_shape(X, shape=(None, self.d))

        # get one-padded testing data
        Xp = pad_array(X, 'left', 1)
        # compute predicted target values
        y_pred = Xp @ self.w

        return y_pred

    def score(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Use fitted weights to predict target values for test data and compute
        :math:`R^2` accuracies using actual target values.

        Parameters
        ----------
        X : numpy.ndarray
            2D testing features array, of shape ``m x d``, where ``m`` is the
            number of testing examples and ``d`` is the number of dimensions.

        Y : numpy.ndarray
            2D testing target values array, of shape ``m x t``, where ``m`` is
            the number of testing examples and ``t`` is the number of targets.

        Returns
        -------
        r_squared : numpy.ndarray
            Array of :math:`R^2` accuracy scores, i.e. values between 0 and 1.
        """
        assert_ndarray_shape(X, shape=(None, self.d))
        m, _ = X.shape
        assert_ndarray_shape(Y, shape=(m, None))

        # get predicted target values
        Y_pred = self.predict(X)
        # compute square root of squared sum of regression
        sqrt_ssr = np.linalg.norm(Y - Y_pred, axis=0)
        # compute square root of total sum of squares
        sqrt_sst = np.linalg.norm(Y - np.mean(Y), axis=0)
        # compute r squared
        r_squared = 1 - ((sqrt_ssr / sqrt_sst) ** 2)

        return r_squared


class LogisticRegression:
    """
    Binary logistic regression model.

    Examples
    --------
    >>> import numpy as np
    >>> from datacheese.regression import LogisticRegression

    Generate input data:

    >>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    >>> X
    array([[0., 0.],
           [0., 1.],
           [1., 0.],
           [1., 1.]])

    Generate target values based on an OR logic gate:

    >>> y = np.any(X, axis=1).astype(int)
    >>> y
    array([0, 1, 1, 1])

    Fit model using data:

    >>> model = LogisticRegression()
    >>> model.fit(X, y)

    Use model to make predictions:

    >>> model.predict(X)
    array([0, 1, 1, 1])

    Use model to obtain prediction probabilities:

    >>> model.predict_prob(X)
    array([0.05523008, 0.99999961, 0.98907499, 0.98196914])

    Compute accuracy:

    >>> model.score(X, y)
    1.0

    Compute negative log probability by changing the ``metric`` parameter:

    >>> model.score(X, y, metric='log_loss')
    0.09550699766960563
    """

    def __init__(self):
        self.fitted = False

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        lr: float = 0.1,
        Lambda: float = 0.0,
        tolerance: float = 0.0001,
        max_iters: int = 1000,
        method: str = 'gradient',
    ):
        """
        Fit logistic regression model to training data.

        Parameters
        ----------
        X : numpy.ndarray
            2D training features array, of shape ``n x d``, where ``n`` is the
            number of training examples and ``d`` is the number of dimensions.

        y : numpy.ndarray
            1D training target values array, of length ``n``, where ``n`` is
            the number of training examples.

        lr : float, default 0.1
            Learning rate for weight update, only used with gradient descent.

        Lambda : float, default 0.0
            Regularization constant, lambda, to be used as penalty term weight.

        tolerance : float, default 0.0001
            Tolerance of maximum element in gradient vector, used to for
            termination criteria.

        max_iters : int, default 1000
            Maximum number of iterations

        method : str, default ``gradient``
            Method to use for computation. Must be either ``gradient``,
            representing gradient descent, or ``newton``, representing Newton's
            method.
        """
        assert_ndarray_shape(X, shape=(None, None))
        n, self.d = X.shape
        assert_ndarray_shape(y, shape=n)
        assert_str_choice(
            method,
            ['gradient', 'newton'],
            str_name='method',
            case_insensitive=True,
        )

        # get one-padded training data
        Xp = pad_array(X, 'left', 1)
        # construct identity matrix
        I = np.identity(self.d + 1)
        # declare random number generator
        rng = np.random.default_rng()
        # initialize weights using Gaussian distribution
        self.w = rng.normal(loc=0, scale=1, size=self.d + 1)

        for i in range(max_iters):
            # compute target probabilities
            y_prob = sigmoid(Xp @ self.w)
            # compute gradient of loss
            del_L = Xp.T @ (y_prob - y) + (Lambda * self.w)
            # terminate if absolute maximum of gradient is below tolerance
            if np.amax(np.abs(del_L)) < tolerance:
                break

            # if gradient descent method chosen
            if method == 'gradient':
                # change in weights
                del_w = lr * del_L
            # if newton's method chosen
            elif method == 'newton':
                # compute Hessian matrix
                R = np.diagflat(sigmoid_derivative(f=y_prob))
                H = (Xp.T @ R @ Xp) + (Lambda * I)
                # change in weights
                del_w = np.linalg.inv(H) @ del_L

            # update weights
            self.w -= del_w

        # set model as fitted
        self.fitted = True

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
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
        Xp = pad_array(X, 'left', 1)
        # apply weights on test data and pass through step function
        y_pred = np.where(Xp @ self.w > 0, 1, 0)

        return y_pred

    def predict_prob(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Use fitted weights to compute target probabilities for test data.

        Parameters
        ----------
        X : ndarray
            2D testing features array, of shape ``m x d``, where ``m`` is the
            number of testing examples and ``d`` is the number of dimensions.

        Returns
        -------
        y_prob : ndarray
            Array of target probabilities.
        """
        assert_fitted(self.fitted, self.__class__.__name__)
        assert_ndarray_shape(X, shape=(None, self.d))

        # get one-padded testing data
        Xp = pad_array(X, 'left', 1)
        # apply weights on test data and pass through sigmoid function
        y_prob = sigmoid(Xp @ self.w)

        return y_prob

    def score(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        metric: str = 'accuracy',
    ) -> float:
        """
        Use fitted weights to predict target values for test data and compute
        prediction score. This can be classification accuracy or log loss,
        depending on the chosen metric.

        Parameters
        ----------
        X : ndarray
            2D testing features array, of shape ``m x d``, where ``m`` is the
            number of testing examples and ``d`` is the number of dimensions.

        y : ndarray
            1D testing target values array, of length ``m``.

        metric : str
            Chosen metric. Must be one of ``accuracy`` or ``log_loss``,
            corresponding to classification accuracy or log loss respectively.

        Returns
        -------
        score : float
            Prediction score.
        """
        assert_ndarray_shape(X, shape=(None, self.d))
        m, _ = X.shape
        assert_ndarray_shape(y, shape=m)
        assert_str_choice(
            metric,
            ['accuracy', 'log_loss'],
            str_name='metric',
            case_insensitive=True,
        )

        # if accuracy metric chosen
        if metric == 'accuracy':
            # get predicted targets
            y_pred = self.predict(X)
            # get mean accuracy
            score = np.mean(y_pred == y)
        # if log loss metric is chosen
        elif metric == 'log_loss':
            # get target probability values
            y_prob = self.predict_prob(X)
            # compute negative loss probabilities
            # ignore errors related to logarithm as we replace nan to 0
            with np.errstate(divide='ignore'):
                ones_loss = np.sum(
                    np.nan_to_num(
                        y * np.log(y_prob),
                        nan=0.0,
                        posinf=np.inf,
                        neginf=-np.inf,
                    )
                )
                zeros_loss = np.sum(
                    np.nan_to_num(
                        (1 - y) * np.log(1 - y_prob),
                        nan=0.0,
                        posinf=np.inf,
                        neginf=-np.inf,
                    )
                )

            score = -(ones_loss + zeros_loss)

        return score
