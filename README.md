[![Free Palestine](https://github.com/alvii147/hinds-banner/blob/main/github/free-palestine-olive.svg)](https://www.pcrf.net/)

<p align="center">
    <img alt="DataCheese logo" src="docs/img/logo_full.png" width=600 />
</p>

<p align="center">
    <strong><i>DataCheese</i></strong> is a Python library with implementations of popular data science and machine learning algorithms.
</p>

<div align="center">

[![](https://img.shields.io/github/actions/workflow/status/alvii147/DataCheese/github-ci.yml?branch=master&label=GitHub%20CI&logo=github)](https://github.com/alvii147/DataCheese/actions) [![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) [![Documentation](https://img.shields.io/badge/Sphinx-Documentation-000000?logo=sphinx)](https://alvii147.github.io/DataCheese/build/html/index.html)

</div>

## Installation

### :one: [Install Python](https://www.python.org/)

Python 3.10 or above required.

### :two: Install package using [pip](https://pypi.org/project/pip/)

Install directly from the repository:

```bash
pip3 install git+https://github.com/alvii147/DataCheese.git
```

## Usage

The `MultiLayerPerceptron` model can be used to train a feed-forward neural network using data:

```python
import numpy as np
from datacheese.neural_networks import (
    MultiLayerPerceptron,
    SigmoidLayer,
    ReLULayer,
)

# number of data patterns
n_patterns = 5
# number of feature dimensions
n_dimensions = 3
# number of target classes
n_classes = 2

# generate random data
rng = np.random.default_rng()
X = rng.random(size=(n_patterns, n_dimensions))
Y = rng.random(size=(n_patterns, n_classes))

# initialize multi-layer perceptron model
model = MultiLayerPerceptron(lr=0.5)
# add relu layer
model.add_layer(ReLULayer(n_dimensions, 4))
# add sigmoid layer
model.add_layer(SigmoidLayer(4, n_classes))
# fit model to data
model.fit(X, Y, epochs=20, verbose=1)
```

When `verbose` is non-zero, progress is logged:

```
Epoch: 0, Loss: 0.15181599599950849
Epoch: 4, Loss: 0.13701115369406147
Epoch: 8, Loss: 0.11337662383705667
Epoch: 12, Loss: 0.10121139637335393
Epoch: 16, Loss: 0.09388681525946835
```

The model can then be used to make predictions:

```python
# predict target values
Y_pred = model.predict(X)
# compute mean squared loss
print(np.mean((Y_pred - Y) ** 2))
```

This outputs the following:

```
0.05310463606057757
```

For more details, visit the [documentation pages](https://alvii147.github.io/DataCheese/build/html/index.html).
