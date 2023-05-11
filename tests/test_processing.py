import numpy as np
import pytest
from datacheese.processing import KFoldCrossValidation


k_fold_cross_validation_parameters = (
    (
        (15, 4, 2),
        3,
    ),
    (
        (15, 4, 2),
        5,
    ),
    (
        (15, 4, 2),
        7,
    ),
    (
        (15, 4, 2),
        8,
    ),
)


@pytest.mark.parametrize(
    'data_shape, k',
    k_fold_cross_validation_parameters,
)
def test_KFoldCrossValidation(data_shape, k):
    rng = np.random.default_rng()
    data_range = np.prod(data_shape)
    X = rng.choice(data_range, size=data_shape, replace=False)
    test_data_seen = []

    for train_data, test_data in KFoldCrossValidation(X, k, randomize=True):
        allowed_fold_sizes = (X.shape[0] // k, (X.shape[0] // k) + 1)

        assert train_data.shape[0] + test_data.shape[0] == X.shape[0]
        assert test_data.shape[0] in allowed_fold_sizes
        assert train_data.shape[1:] == X.shape[1:]
        assert test_data.shape[1:] == X.shape[1:]

        flattened_train_data = train_data.flatten()
        flattened_test_data = test_data.flatten()
        all_data = np.concatenate((flattened_train_data, flattened_test_data))
        all_unique_data = list(set(list(all_data)))

        assert len(all_unique_data) == data_range
        assert np.amin(all_unique_data) == 0
        assert np.amax(all_unique_data) == data_range - 1

        for i in flattened_test_data:
            test_data_seen.append(i)

    unique_test_data_seen = list(set(list(test_data_seen)))
    assert len(unique_test_data_seen) == data_range
    assert np.amin(unique_test_data_seen) == 0
    assert np.amax(unique_test_data_seen) == data_range - 1
