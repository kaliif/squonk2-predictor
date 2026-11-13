# squonk2-predictor

A Squonk2 Data Manager job that takes a pre-trained scikit-learn model and a
file of precomputed descriptors and predicts values for the molecules. The
descriptors are calculated by a separate descriptor job (see
`docs/descriptor-jobs-design.md`); this job only applies the model, so it isn't
tied to any descriptor library version.

# Supported models
The model file is loaded with a restricted unpickler that only allows
scikit-learn estimators and the numpy and scikit-learn internals they need, so
a model file can't run code. Supported formats:

- jaqpotpy 1.0.x `MolecularModel` files (`.jmodel`) with a scikit-learn
  estimator and Morgan, MACCS, RDKit or Mordred (2D) descriptors, without
  external features or preprocessing steps

A bare scikit-learn estimator is rejected, because nothing records which
descriptors it was trained on. The job runs scikit-learn 1.9.1; models
trained with another version are flagged, and rejected if they can't run.
Trees pickled by scikit-learn < 1.4, which includes the jaqpotpy 1.0.x models,
are converted to the current format on loading.

# Descriptor input
The predictor reads a descriptor file and a sidecar manifest (JSON) that
describes it: the featurizer and its parameters, the library versions, the
fragment method, the fill used for uncomputable descriptors, and the descriptor
column names. The file is delimited text (descriptor values as columns) or an
SDF (values in fields).

The manifest's featurizer name and parameters are checked strictly against the
model — a mismatch (e.g. a Morgan file at radius 3 against a radius-2 model)
changes the feature values silently, so the job refuses it. The library
versions and fragment method are only reported as warnings, because a bare
jaqpotpy model records no expectation to check them against. Fingerprints are
stored as one bit-string column and expanded here to the per-bit columns the
model names. The manifest format is documented in `docs/descriptor-jobs-design.md`.

# Unverified predictions
Descriptor implementations change between library versions and estimators can
behave differently in another scikit-learn version, so a model can load and
run but give different predictions from when it was created. A model file
doesn't record the predictions it made, so the job can't check this and warns
that its predictions are unverified.

# Output
A CSV with a row per input record, in input order: `id`, `smiles`, the
prediction (named after the model's target), depending on the model
`<target>_probability` or `<target>_score` (one column per class for
multiclass models), `<target>_leverage` and `<target>_in_domain` for models
with an applicability domain, and `error` for records that couldn't be
predicted.

# Job versions
One container holds one scikit-learn version, so models saved with an
incompatible version need another version of the job, which users choose in the
Squonk UI. Each such version lives on its own branch and differs from `main`
only in `pyproject.toml`, `poetry.lock`, the `Dockerfile`,
`data-manager/jobs.yaml` and the job tests; the code and tests are shared and
run on every supported version. Fixes go to `main` and are merged into the
branches. When a branch is cut, add its job version to `src/job_versions.py`
on `main` and merge that into every branch, so that a job version refusing a
model can name the version that reads it.

| Branch        | Job version | Image tag | scikit-learn | Python |
| ------------- | ----------- | --------- | ------------ | ------ |
| `main`        | 1.1.0       | `latest`  | 1.9.1        | 3.12   |
| `sklearn-1.2` | 1.0.1       | `1.0.1`   | 1.2.2        | 3.11   |

scikit-learn 1.3 changed the format of tree models, so trees saved before
1.3 load only on `sklearn-1.2`, and trees saved with 1.3 or later only on
`main`. The `data/models/rf-maccs-sklearn-*.jmodel` test models show this;
`scripts/make_version_test_model.py` makes one with the scikit-learn version
that runs it.

To load both versions into Squonk, load the manifest
`data-manager/manifest.yaml` from each branch, and build and push each
branch's image under the tag its `jobs.yaml` names.

# Running locally

    poetry run python src/predict.py --model models/caco2_wang.jmodel \
        --descriptors descriptors.smi --manifest descriptors.manifest.json \
        --outfile predictions.csv

The manifest defaults to a sidecar next to the descriptor file
(`descriptors.smi` → `descriptors.manifest.json`); `--delimiter` sets the
delimiter of a text descriptor file.

# Tests

    poetry run pytest

The tests are self-contained: they build models and descriptor files in
temporary directories and check that the predictor loads and checks models,
reads descriptor files and manifests, and reproduces the estimator's own
predictions. Reproducing the original jaqpotpy predictions end to end is the
descriptor jobs' responsibility, since the descriptors are calculated there.

# Job tests
The job tests in `data-manager/jobs.yaml` run with
[jote](https://pypi.org/project/im-jote/) against the container image. Build
the image from the working tree and point jote at it, otherwise it tests
whatever image the job definition's tag refers to:

    docker build -t kaliif/squonk2-predictor:dev .
    jote --image-tag dev

`jote --dry-run` only validates the job definition. The test data is described
in `data/README.md`.

# Setting up the environment
The development environment requires [Poetry](https://python-poetry.org/docs/main#installing-with-pipx).
From the project root run:


    poetry install --no-root

When done, set up pre-commit:


    pre-commit install -t commit-msg -t pre-commit
