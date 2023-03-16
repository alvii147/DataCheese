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