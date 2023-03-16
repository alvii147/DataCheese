import numpy as np
from numpy.typing import NDArray
from .activations import (
    sigmoid,
    sigmoid_derivative,
    tanh,
    tanh_derivative,
    relu,
    relu_derivative,
    leaky_relu,
    leaky_relu_derivative,
)
from .utils import assert_ndarray_shape, assert_fitted


class BaseLayer:
    """
    Base class for neural network layer.

    Parameters
    ----------
    n_inputs : int
        Number of inputs, usually equal to the number of nodes in the previous
        layer.

    n_nodes : int
        Number of neuron nodes.

    **kwargs : dict
        Layer specific parameters.
    """

    def __init__(self, n_inputs: int, n_nodes: int, **kwargs):
        rng = np.random.default_rng()

        self.n_inputs = n_inputs
        self.n_nodes = n_nodes

        # output of layer
        # this is of side n_nodes + 1, last node is bias node
        self.out = np.zeros(self.n_nodes + 1, dtype=np.float64)
        # set bias node output to 1
        self.out[-1] = 1
        # delta errors
        self.delta = np.zeros(self.n_nodes, dtype=np.float64)
        # weights, randomly chosen over Gaussian distribution
        self.w = rng.normal(
            loc=0,
            scale=1,
            size=(self.n_nodes, self.n_inputs + 1),
        )

    def activation(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Activation function.

        Parameters
        ----------
        x : numpy.ndarray
            Input values.

        Returns
        -------
        f : numpy.ndarray
            Output values.
        """
        raise NotImplementedError

    def activation_derivative(
        self,
        x: NDArray[np.float64] | None = None,
        f: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """
        Activation derivative function.

        If ``x`` is provided, the output is computed directly. If the output
        of antiderivative ``f`` is provided, the output is computed using the
        antiderivative.

        Parameters
        ----------
        x : numpy.ndarray, default ``None``
            Input values.

        f : numpy.ndarray, default ``None``
            Output values of antiderivative function.

        Returns
        -------
        f : numpy.ndarray
            Output values.
        """
        raise NotImplementedError

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict output of given inputs using current weights.

        Parameters
        ----------
        x : numpy.ndarray
            Input values.

        Returns
        -------
        y_pred : numpy.ndarray
            Output values.
        """
        assert_ndarray_shape(x, shape=self.n_inputs)

        # predict output using weights
        x = np.concatenate((x, [1]))
        # predict output
        y_pred = self.activation(self.w @ x)

        return y_pred

    def feed_forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Feed forward given training input example.

        Parameters
        ----------
        x : numpy.ndarray
            Given training input.

        Returns
        -------
        out : numpy.ndarray
            Output values.
        """
        assert_ndarray_shape(x, shape=self.n_inputs)

        # predict output using weights
        self.out[:-1] = self.predict(x)
        # save input for backpropagation
        self.x = x.copy()

        return self.out[:-1]

    def back_propagate(
        self,
        e: NDArray[np.float64],
        lr: float,
    ) -> NDArray[np.float64]:
        """
        Perform backpropagation using forward layer errors.

        Parameters
        ----------
        e : numpy.ndarray
            Errors propagated from forward layer.

        lr : float
            Learning rate for weight updates.

        Returns
        -------
        e_next : numpy.ndarray
            Errors to be propagated to backward layer.
        """
        assert_ndarray_shape(e, shape=self.n_nodes)

        # compute delta errors
        self.delta = self.activation_derivative(f=self.out[:-1]) * e

        # update weights
        xv, dv = np.meshgrid(np.concatenate((self.x, [1])), self.delta)
        self.w += lr * (dv * xv)

        # compute propagated error for backward layer
        e_back = (self.delta @ self.w)[:-1]

        return e_back


class LinearLayer(BaseLayer):
    """
    Layer with linear activation function.

    Parameters
    ----------
    n_inputs : int
        Number of inputs, usually equal to the number of nodes in the previous
        layer.

    n_nodes : int
        Number of neuron nodes.
    """

    def activation(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return x

    def activation_derivative(
        self,
        x: NDArray[np.float64] | None = None,
        f: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        shape = x.shape if x is not None else f.shape

        return np.ones(shape, dtype=np.float64)


class SigmoidLayer(BaseLayer):
    """
    Layer with sigmoid activation function.

    Parameters
    ----------
    n_inputs : int
        Number of inputs, usually equal to the number of nodes in the previous
        layer.

    n_nodes : int
        Number of neuron nodes.
    """

    def activation(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return sigmoid(x)

    def activation_derivative(
        self,
        x: NDArray[np.float64] | None = None,
        f: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        return sigmoid_derivative(x=x, f=f)


class TanhLayer(BaseLayer):
    """
    Layer with hyperbolic tangent activation function.

    Parameters
    ----------
    n_inputs : int
        Number of inputs, usually equal to the number of nodes in the previous
        layer.

    n_nodes : int
        Number of neuron nodes.
    """

    def activation(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return tanh(x)

    def activation_derivative(
        self,
        x: NDArray[np.float64] | None = None,
        f: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        return tanh_derivative(x=x, f=f)


class ReLULayer(BaseLayer):
    """
    Layer with rectified linear unit activation function.

    Parameters
    ----------
    n_inputs : int
        Number of inputs, usually equal to the number of nodes in the previous
        layer.

    n_nodes : int
        Number of neuron nodes.
    """

    def activation(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return relu(x)

    def activation_derivative(
        self,
        x: NDArray[np.float64] | None = None,
        f: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        return relu_derivative(x=x, f=f)


class LeakyReLULayer(BaseLayer):
    """
    Layer with leaky rectified linear unit activation function.

    Parameters
    ----------
    n_inputs : int
        Number of inputs, usually equal to the number of nodes in the previous
        layer.

    n_nodes : int
        Number of neuron nodes.

    alpha : float, default 0.01
        Negative slope :math:`\\alpha`.
    """

    def __init__(self, n_inputs, n_nodes, **kwargs):
        super().__init__(n_inputs, n_nodes, **kwargs)

        self.alpha = kwargs.get('alpha', 0.01)

    def activation(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return leaky_relu(x)

    def activation_derivative(
        self,
        x: NDArray[np.float64] | None = None,
        f: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        return leaky_relu_derivative(x=x, f=f)


class MultiLayerPerceptron:
    """
    Multi-layer percetron feed-forward neural network that implements
    backpropagation.

    Parameters
    ----------
    lr : float
        Learning rate.

    Examples
    --------
    >>> import numpy as np
    >>> from datacheese.neural_networks import (
    ...    MultiLayerPerceptron,
    ...    SigmoidLayer,
    ...    ReLULayer,
    ... )

    Generate training data:

    >>> n_patterns = 5
    >>> n_dimensions = 3
    >>> n_classes = 2
    >>> rng = np.random.default_rng()
    >>> X = rng.random(size=(n_patterns, n_dimensions))
    >>> Y = rng.random(size=(n_patterns, n_classes))

    Initialize model with 2 hidden layers with ReLU and Sigmoid activations
    respectively:

    >>> model = MultiLayerPerceptron(lr=0.5)
    >>> model.add_layer(ReLULayer(n_dimensions, 4))
    >>> model.add_layer(SigmoidLayer(4, n_classes))

    Train model over 20 epochs:

    >>> model.fit(X, Y, epochs=20, verbose=1)
    Epoch: 0, Loss: 0.15181599599950849
    Epoch: 4, Loss: 0.13701115369406147
    Epoch: 8, Loss: 0.11337662383705667
    Epoch: 12, Loss: 0.10121139637335393
    Epoch: 16, Loss: 0.09388681525946835

    Use model to make predictions:

    >>> Y_pred = model.predict(X)
    >>> np.mean((Y_pred - Y) ** 2)
    0.05310463606057757
    """

    def __init__(self, lr: float = 0.5):
        self.fitted = False
        self.lr = lr
        self.layers = []

    def add_layer(self, layer: BaseLayer):
        """
        Add layer to network.

        Parameters
        ----------
        layer : BaseLayer
            Layer object to add to network.
        """
        self.layers.append(layer)

    def feed_forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Feed forward given training input example through each layer.

        Parameters
        ----------
        x : numpy.ndarray
            Given training input.

        Returns
        -------
        out : numpy.ndarray
            Output values.
        """
        assert_ndarray_shape(x, shape=None)

        # feed forward given training input through each layer
        out = x.copy()
        for layer in self.layers:
            out = layer.feed_forward(out)

        return out

    def back_propagate(self, y: NDArray[np.float64], lr: float):
        """
        Perform backpropagation through layers in the network.

        Parameters
        ----------
        y : numpy.ndarray
            Actual output of last forwarded training input.

        lr : float
            Learning rate for weight updates.
        """
        assert_ndarray_shape(y, shape=None)

        # compute error between actual and predicted output of last layer
        e = y - self.layers[-1].out[:-1]
        # perform backpropagation through each layer
        for layer in reversed(self.layers):
            e = layer.back_propagate(e, lr)

    def fit(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        epochs: int,
        verbose: int = 0,
    ):
        """
        Train network weights using training data over given number of epochs.

        Parameters
        ----------
        X : numpy.ndarray
            2D array of input patterns of shape ``n x d``, where ``n`` is the
            number of training examples and ``d`` is the number of dimensions.

        Y : numpy.ndarray
            2D array of output patterns  of shape ``n x c``, where ``n`` is the
            number of training examples and ``c`` is the number of classes.

        epochs : int
            Number of epochs to train over.

        verbose : int, default 0
            Logging verbosity.
        """
        assert_ndarray_shape(X, shape=(None, None))
        n, _ = X.shape
        assert_ndarray_shape(Y, shape=(n, None))

        losses = []
        for i in range(epochs):
            for x, y in zip(X, Y):
                # feed forward input
                y_pred = self.feed_forward(x)
                # perform backpropagation
                self.back_propagate(y, self.lr)
                # compute and store mean squared loss
                loss = np.sum((y_pred - y) ** 2) / y.shape[0]
                losses.append(loss)

            if verbose > 0 and i % int(np.sqrt(epochs)) == 0:
                # log epoch and mean loss over current epoch
                print(f'Epoch: {i}, Loss: {np.mean(losses)}')

        # set model as fitted
        self.fitted = True

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict output of given inputs using current network layers.

        Parameters
        ----------
        X : numpy.ndarray
            2D array of input patterns of shape ``m x d``, where ``m`` is the
            number of testing examples and ``d`` is the number of dimensions.

        Returns
        -------
        Y_pred : numpy.ndarray
            2D array of predicted output values.
        """
        assert_fitted(self.fitted, self.__class__.__name__)
        assert_ndarray_shape(X, shape=(None, None))

        # initialize predicted output array
        Y_pred = np.zeros((X.shape[0], self.layers[-1].n_nodes))
        # iterate over input patterns
        for i, x in enumerate(X):
            # predict output using network layers
            for layer in self.layers:
                x = layer.predict(x)

            Y_pred[i] = x

        return Y_pred
