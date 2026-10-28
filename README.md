# squonk2-predict-local

A Squonk2 Data Manager job that takes a pre-trained scikit-learn model and a
set of molecules, calculates the descriptors the model needs and predicts
values for the molecules.

# Setting up the environment
The development environment requires [Poetry](https://python-poetry.org/docs/main#installing-with-pipx).
From the project root run:


    poetry install --no-root

When done, set up pre-commit:


    pre-commit install -t commit-msg -t pre-commit
