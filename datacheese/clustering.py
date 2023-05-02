import numpy as np
from numpy.typing import NDArray
from typing import Any
from .utils import (
    assert_ndarray_shape,
    assert_fitted,
    assert_str_choice,
    pairwise_distances,
)


class KMeans:
    """
    K-means clustering model.

    Examples
    --------
    >>> import numpy as np
    >>> from datacheese.clustering import KMeans

    Generate input data:

    >>> X = np.array(
    ...     [
    ...         [1, 2],
    ...         [1, 4],
    ...         [1, 0],
    ...         [10, 2],
    ...         [10, 4],
    ...         [10, 0],
    ...     ],
    ...     dtype=np.float64,
    ... )
    >>> X
    array([[ 1.,  2.],
           [ 1.,  4.],
           [ 1.,  0.],
           [10.,  2.],
           [10.,  4.],
           [10.,  0.]])

    Fit model using data:

    >>> model = KMeans()
    >>> labels, centroids = model.fit(X, k=2)
    >>> labels
    array([1, 1, 1, 0, 0, 0], dtype=int64)
    >>> centroids
    array([[10.,  2.],
           [ 1.,  2.]])

    Use model to make predictions:

    >>> X_test = np.array([[2, 1], [11, 2]], dtype=np.float64)
    >>> X_test
    array([[ 2.,  1.],
           [11.,  2.]])
    >>> model.predict(X_test)
    array([1, 0], dtype=int64)

    Compute within-clusters sum of squares:

    >>> model.score(X_test, metric='wcss')
    3.0000000000000004

    Compute between-clusters sum of squares:

    >>> model.score(X_test, metric='bcss')
    40.5
    """

    def __init__(self, seed: int | None = None):
        self.seed = seed
        self.fitted = False

        # initialize random number generator
        self.rng = np.random.default_rng(seed=seed)

    def fit(
        self,
        X: NDArray[np.float64],
        k: int,
        max_iters: int = 1000,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """
        Fit model by clustering on given data.

        Parameters
        ----------
        X : numpy.ndarray
            2D features array, of shape ``n x d``, where ``n`` is the number of
            data points and ``d`` is the number of dimensions.

        k : int
            Number of clusters.

        max_iters : int, default 1000
            Maximum number of iterations.

        Returns
        -------
        labels : numpy.ndarray
            1D array, of shape ``n``, containing labels for each data point.

        centroids : numpy.ndarray
            2D array, of shape ``k x d``, contraining centroid coordinates.
        """
        assert_ndarray_shape(X, shape=(None, None))
        n, self.d = X.shape

        # initialize and randomize one-hot encoded labels array
        labels_row_indexer = np.arange(n)
        ohe_labels = np.zeros((n, k), dtype=bool)
        ohe_labels[
            labels_row_indexer,
            self.rng.integers(low=0, high=k, size=n),
        ] = 1
        # generate centroids from randomized labels
        self.centroids = np.zeros((k, self.d), dtype=np.float64)

        for _ in range(max_iters):
            # get frequencies of each label
            label_counts = np.sum(ohe_labels, axis=0, dtype=np.float64)
            # set labels that are not used to infinity
            # this is to avoid overflows and zero division errors
            label_counts[label_counts == 0] = np.inf
            # compute new centroids
            new_centroids = (ohe_labels.T @ X) / label_counts[:, None]
            # terminate if centroids have not changed
            if np.array_equal(self.centroids, new_centroids):
                break

            # save new centroids
            self.centroids[:] = new_centroids
            # compute distances from each data point to centroids
            distances = pairwise_distances(X, self.centroids, p=2)
            # set all labels to zeros
            ohe_labels[:, :] = 0
            # recompute labels based on distances
            ohe_labels[
                labels_row_indexer,
                np.argmin(distances, axis=0),
            ] = 1

        # get integer labels from one-hot encoded labels
        labels = np.argmax(ohe_labels, axis=1)
        # set model as fitted
        self.fitted = True

        return labels, self.centroids

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Use stored centroids to predict labels for given data.

        Parameters
        ----------
        X : numpy.ndarray
            2D features array, of shape ``m x d``, where ``m`` is the number of
            data points and ``d`` is the number of dimensions.

        Returns
        -------
        labels : numpy.ndarray
            1D array, of shape ``m``, containing labels for each data point.
        """
        assert_fitted(self.fitted, class_name=self.__class__.__name__)
        assert_ndarray_shape(X, shape=(None, self.d))

        # compute pairwise distances between given data and centroids
        distances = pairwise_distances(X, self.centroids, p=2)
        # get labels based on minimum distances to clusters
        labels = np.argmin(distances, axis=0)

        return labels

    def score(self, X: NDArray[np.float64], metric: str = 'wcss') -> float:
        """
        Use stored centroids to predict labels for given data and compute
        clustering score. This can be the within-clusters sum of squares, or
        the between-clusters sum of squares, depending on the chosen metric.

        Parameters
        ----------
        X : numpy.ndarray
            2D features array, of shape ``m x d``, where ``m`` is the number of
            data points and ``d`` is the number of dimensions.

        metric : str
            Chosen metric. Must be one of ``wcss`` or ``bcss``, corresponding
            to within-clusters sum of squares or the between-clusters sum of
            squares respectively.

        Returns
        -------
        score : float
            Clustering score.
        """
        assert_fitted(self.fitted, class_name=self.__class__.__name__)
        assert_ndarray_shape(X, shape=(None, self.d))
        assert_str_choice(
            metric,
            ['wcss', 'bcss'],
            str_name='metric',
            case_insensitive=True,
        )

        # if WCSS metric is chosen
        if metric.lower() == 'wcss':
            # compute pairwise distances between given data and centroids
            distances = pairwise_distances(X, self.centroids, p=2)
            # compute squared sum of distances between data and centroids
            score = np.sum(np.amin(distances ** 2, axis=0))
        # if BCSS metric is chosen
        elif metric.lower() == 'bcss':
            # compute global centroid by taking average of centroids
            global_centroid = np.mean(self.centroids, axis=0)
            # compute distance between centroids and global centroid
            distances = pairwise_distances(
                self.centroids,
                global_centroid[None, :],
            )[0, :]
            # compute squared sum of distances between centroids and
            # global centroid
            score = np.sum(distances ** 2)

        return score
