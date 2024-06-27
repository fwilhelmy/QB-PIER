QMLBenchmark
==============================

Benchmarking adversarial robustness for classical and quantum classifier

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── results            <- Logs & results dump to keep track of all our experimentations
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

Credits to Samuel Richard for helping write the initial version of the code in this repository.
Especially the QuanvolutionLayer and the VQCLayer.

Credits to ChatGPT for helping structure the code and some bowlerplate functions.

Credits For LeNet  https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/
- For helping work and setup the first original model from which everything started

Credits for initial code of the FGSM attacks:
**Author:** [Nathan Inkawhich](https://github.com/inkawhich)

Computations were made on the supercomputer name-of-the-server, managed by Calcul Québec and the Digital Research Alliance of Canada. The operation of this supercomputer is funded by the Canada Foundation for Innovation (CFI), Ministère de l’Économie et de l’Innovation du Québec (MEI) and le Fonds de recherche du Québec (FRQ).

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
