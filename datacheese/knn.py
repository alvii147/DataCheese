import numpy as np
from numpy.typing import NDArray
from typing import Any
from .utils import (
    assert_ndarray_shape,
    assert_fitted,
    pairwise_distances,
    array_mode_value,
)


class KNN:
    """
    K-nearest neighbours classification model.

    Examples
    --------
    >>> import numpy as np
    >>> from datacheese.knn import KNN

    Generate input data:

    >>> X = np.array([[0], [1], [2], [3]], dtype=np.float64)
    >>> X
    array([[0.],
           [1.],
           [2.],
           [3.]])

    Generate target values:

    >>> y = np.array([0, 0, 1, 1], dtype=int)
    >>> y
    array([0, 0, 1, 1])

    Fit model using data:

    >>> model = KNN()
    >>> model.fit(X, y)

    Use model to make predictions:

    >>> X_test = np.array([[1.1]], dtype=np.float64)
    >>> X_test
    array([[1.1]])
    >>> y_test = np.array([0], dtype=int)
    >>> y_test
    array([0])
    >>> model.predict(X_test, k=3)
    array([0])

    Compute prediction accuracy:

    >>> model.score(X_test, y_test, k=3)
    1.0
    """

    def __init__(self, seed: int | None = None):
        self.seed = seed
        self.fitted = False

    def fit(self, X: NDArray[np.float64], y: NDArray[Any]):
        """
        Fit model by processing and storing training data.

        Parameters
        ----------
        X : numpy.ndarray
            2D training features array, of shape ``n x d``, where ``n`` is the
            number of training examples and ``d`` is the number of dimensions.

        y : numpy.ndarray
            1D training target values array of shape ``n``, where ``n`` is the
            number of training examples.
        """
        assert_ndarray_shape(X, shape=(None, None))
        n, d = X.shape
        assert_ndarray_shape(y, shape=n)

        self.fitted = True
        self.n = n
        self.d = d
        self.X_train = X.copy()
        self.y_train = y.copy()

    def predict(self, X: NDArray[np.float64], k: int) -> NDArray[np.int64]:
        """
        Use stored training data to predict target values for test data.

        Parameters
        ----------
        X : numpy.ndarray
            2D testing features array, of shape ``m x d``, where ``m`` is the
            number of testing examples and ``d`` is the number of dimensions.

        k : int
            Number of neighbours.

        Returns
        -------
        y_pred : numpy.ndarray
            Array of predicted target values.
        """
        assert_fitted(self.fitted, class_name=self.__class__.__name__)
        assert_ndarray_shape(X, shape=(None, self.d))

        # compute pairwise distances between training data and test data
        distances = pairwise_distances(self.X_train, X, p=2)
        # get top-k neighbours
        neighbours = self.y_train[np.argpartition(distances, k, axis=1)[:, :k]]
        # get predicted labels using mode
        y_pred = np.apply_along_axis(
            lambda A: array_mode_value(A, self.seed),
            1,
            neighbours,
        )

        return y_pred

    def score(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        k: int,
    ) -> float:
        """
        Use stored training data to predict target values for test data and
        compute prediction score.

        Parameters
        ----------
        X : numpy.ndarray
            2D testing features array, of shape ``m x d``, where ``m`` is the
            number of testing examples and ``d`` is the number of dimensions.

        y : numpy.ndarray
            1D testing target values array of shape ``m``, where ``m`` is the
            number of testing examples.

        k : int
            Number of neighbours.

        Returns
        -------
        accuracy : float
            Prediction score, a value between 0 and 1.
        """
        assert_ndarray_shape(X, shape=(None, self.d))
        m, _ = X.shape
        assert_ndarray_shape(y, shape=m)

        # get predicted labels
        y_pred = self.predict(X, k=k)
        # compute accuracy
        accuracy = (y_pred == y).sum() / y.shape[0]

        return accuracy
