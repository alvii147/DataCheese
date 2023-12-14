<p align="center">
    <img alt="DataCheese logo" src="docs/img/logo_full.png" width=600 />
</p>

<p align="center">
    <strong><i>DataCheese</i></strong> is a Python library with implementations of popular data science and machine learning algorithms.
</p>

<div align="center">

[![](https://img.shields.io/github/actions/workflow/status/alvii147/DataCheese/github-ci.yml?branch=master&label=Tests&logo=github)](https://github.com/alvii147/DataCheese/actions)
[![](https://img.shields.io/badge/Documentation-00CC99?style=flat&logo=data:image/png;base64,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)](https://alvii147.github.io/DataCheese/build/html/index.html)

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
