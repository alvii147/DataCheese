import numpy as np
from numpy.typing import NDArray
from typing import Any


class KFoldCrossValidation:
    """
    K-fold cross validation iterator class.

    Parameters
    ----------
    data : numpy.ndarray
        Array to split into k-folds. Splitting is always done on axis 0.

    k : int
        Number of folds.

    randomize : bool, default True
        Whether or not to shuffle the data before splitting.

    seed : int or None, default None
        Random seed used to shuffle the data.

    Examples
    --------
    >>> import numpy as np
    >>> from datacheese.processing import KFoldCrossValidation

    Generate data:

    >>> X = np.arange(12).reshape(6, 2)
    >>> X
    array([[ 0,  1],
           [ 2,  3],
           [ 4,  5],
           [ 6,  7],
           [ 8,  9],
           [10, 11]])

    Split into 3 folds and iterate over them:

    >>> for i, (train_data, test_data) in enumerate(
    ...     KFoldCrossValidation(X, k=3)
    ... ):
    ...     print(f'Fold {i}')
    ...     print('Train Data:')
    ...     print(train_data)
    ...     print('Test Data:')
    ...     print(test_data)
    Fold 0
    Train Data:
    [[ 4  5]
     [ 2  3]
     [ 8  9]
     [10 11]]
    Test Data:
    [[6 7]
     [0 1]]
    Fold 1
    Train Data:
    [[ 6  7]
     [ 0  1]
     [ 8  9]
     [10 11]]
    Test Data:
    [[4 5]
     [2 3]]
    Fold 2
    Train Data:
    [[6 7]
     [0 1]
     [4 5]
     [2 3]]
    Test Data:
    [[ 8  9]
     [10 11]]
    """

    def __init__(
        self,
        data: NDArray[Any],
        k: int,
        randomize: bool = True,
        seed: int | None = None,
    ):
        # create copy of data
        data_copy = data.copy()
        # if randomization enabled, shuffle data
        if randomize:
            rng = np.random.default_rng(seed=seed)
            p = rng.permutation(data.shape[0])
            data_copy[:] = data[p]

        # split into k-folds
        self.folds = np.array_split(data_copy, k, axis=0)
        # initialize iteration index
        self._i = 0

    def __iter__(self):
        """
        Get iterator.

        Returns
        -------
        self : KFoldCrossValidation
            Iterator object.
        """
        return self

    def __next__(self):
        """
        Get next iteration item.

        Returns
        -------
        train_data : numpy.ndarray
            Training data in current fold.

        test_data : numpy.ndarray
            Testing data in current fold.
        """
        # stop iteration if iteration index is out of bounds
        if self._i >= len(self.folds):
            raise StopIteration

        # construct training data
        train_data = np.concatenate(
            (self.folds[:self._i] + self.folds[self._i + 1:])
        )
        # get testing data
        test_data = self.folds[self._i]
        # increment iteration index
        self._i += 1

        return train_data, test_data
