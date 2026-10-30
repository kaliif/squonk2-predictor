# squonk2-predict-local

A Squonk2 Data Manager job that takes a pre-trained scikit-learn model and a
set of molecules, calculates the descriptors the model needs and predicts
values for the molecules.

# Supported models
The model file is loaded with a restricted unpickler that only allows
scikit-learn estimators and the numpy and scikit-learn internals they need, so
a model file can't run code. Supported formats:

- jaqpotpy 1.0.x `MolecularModel` files (`.jmodel`) with a scikit-learn
  estimator and Morgan, MACCS, RDKit or Mordred (2D) descriptors, without
  external features or preprocessing steps

A bare scikit-learn estimator is rejected, because nothing records which
descriptors it was trained on. The job image pins scikit-learn 1.3.2; models
trained with another version are flagged, and rejected if they can't run.

# Verification with probes
Descriptor implementations change between library versions and estimators can
behave differently in another scikit-learn version, so a model can load and
run but give different predictions from when it was created. To catch this,
supply a CSV of probe molecules with the predictions the model gave at
training time:

    smiles,prediction,score
    CCO,0,0.12

`score` is optional: the probability of the second class, or the decision
score, of a binary classifier. A model that doesn't reproduce its probes is
rejected; without probes, predictions are marked `unverified`.

# Output
A CSV with a row per input molecule, in input order: `id`, `smiles`, the
prediction (named after the model's target), depending on the model
`<target>_probability` or `<target>_score` (one column per class for
multiclass models), `<target>_leverage` and `<target>_in_domain` for models
with an applicability domain, `model_status` (`verified` or `unverified`) and
`error` for molecules that couldn't be predicted.

# Running locally

    poetry run python src/predict.py --model models/caco2_wang.jmodel \
        --molecules data/candidates-10.sdf --outfile predictions.csv

SMILES input has the SMILES in the first column and an optional id in the
second (`--delimiter`, `--read-header`).

# Tests

    poetry run pytest

Tests against the legacy models run only when `models/` holds the `.jmodel`
files and their `.probes.csv` files, which aren't in the repository.

# Job tests
The job tests in `data-manager/jobs.yaml` run with
[jote](https://pypi.org/project/im-jote/) against the container image. Build
the image from the working tree and point jote at it, otherwise it tests
whatever image the job definition's tag refers to:

    docker build -t informaticsmatters/squonk2-predict-local:dev .
    jote --image-tag dev

`jote --dry-run` only validates the job definition. The test data is described
in `data/README.md`.

# Probes for jaqpotpy models
`scripts/make_jaqpotpy_probes.py` writes a probe file for a jaqpotpy model
using jaqpotpy itself, so the probes record what the original software
predicted. It runs in a separate environment that reproduces the one the
models were trained in; its docstring says how to set that up.

The job uses the original `mordred` 1.2.0 rather than `mordredcommunity`,
because the latter perceives rings differently and changes ring counts and
descriptors derived from them, which changed predictions of the legacy
Mordred models for some molecules.

# Setting up the environment
The development environment requires [Poetry](https://python-poetry.org/docs/main#installing-with-pipx).
From the project root run:


    poetry install --no-root

When done, set up pre-commit:


    pre-commit install -t commit-msg -t pre-commit
