import numpy as np
import pytest
from datacheese.clustering import KMeans


rng = np.random.default_rng()
kmeans_parameters = (
    (
        rng.uniform(low=-100, high=100, size=(1, 2)),
        1,
        rng.uniform(low=-100, high=100, size=(1, 2)),
    ),
    (
        rng.uniform(low=-100, high=100, size=(10, 5)),
        3,
        rng.uniform(low=-100, high=100, size=(6, 5)),
    ),
    (
        rng.uniform(low=-100, high=100, size=(50, 5)),
        6,
        rng.uniform(low=-100, high=100, size=(10, 5)),
    ),
    (
        rng.uniform(low=-100, high=100, size=(100, 10)),
        12,
        rng.uniform(low=-100, high=100, size=(25, 10)),
    ),
)


@pytest.mark.parametrize(
    'X, k, X_test',
    kmeans_parameters,
)
def test_kmeans(
    X,
    k,
    X_test,
):
    model = KMeans()
    labels, centroids = model.fit(X, k=k)

    for i, centroid in enumerate(centroids):
        assert np.allclose(centroid, np.mean(X[labels == i], axis=0))

    for i, x in enumerate(X):
        min_distance_label = -1
        min_distance = np.inf
        for j, centroid in enumerate(centroids):
            distance = np.linalg.norm(x - centroid)
            if distance < min_distance:
                min_distance_label = j
                min_distance = distance

        assert min_distance_label == labels[i]

    labels = model.predict(X_test)

    wcss = 0
    for i, x in enumerate(X_test):
        min_distance_label = -1
        min_distance = np.inf
        for j, centroid in enumerate(centroids):
            distance = np.linalg.norm(x - centroid)
            if distance < min_distance:
                min_distance_label = j
                min_distance = distance

        wcss += min_distance ** 2
        assert min_distance_label == labels[i]

    assert np.allclose(model.score(X_test, metric='wcss'), wcss)

    global_centroid = np.zeros(centroids.shape[1], dtype=np.float64)
    for i, centroid in enumerate(centroids):
        global_centroid += centroid

    global_centroid = global_centroid / centroids.shape[0]
    bcss = 0
    for i, centroid in enumerate(centroids):
        bcss += np.linalg.norm(centroid - global_centroid) ** 2

    assert np.allclose(model.score(X_test, metric='bcss'), bcss)
