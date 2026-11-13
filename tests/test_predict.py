"""Tests for predict.py: loading and checking models, scikit-learn version
handling, output columns and reporting. The input is always a precomputed
descriptor file; the descriptor-reading logic itself is tested in
test_descriptor_input."""

import os
import pickle

import numpy as np
import pandas as pd
import pytest
import sklearn
import yaml
from conftest import DATA_DIR, ROOT, make_obj, write_descriptor_file
from sklearn import ensemble, linear_model, svm

import job_versions
import predict
from model_spec import UnsupportedModelError
from safe_loader import ModelLoadError, UnsafeModelError, _version_tuple

_RNG = np.random.default_rng(0)
X_TRAIN = _RNG.integers(0, 2, size=(6, 167)).astype(float)
Y = [0, 1, 1, 0, 0, 1]

# a committed MACCS descriptor file (167 bits) for the 10 candidate molecules,
# usable by any MACCS model
MACCS_DESCRIPTORS = DATA_DIR / "descriptors" / "clearance-maccs.smi"
MACCS_MANIFEST = DATA_DIR / "descriptors" / "clearance-maccs.manifest.json"


def _jmodel(jq, estimator, with_domain=True):
    doa = None
    if with_domain:
        doa = make_obj(
            jq.Leverage,
            _data=X_TRAIN,
            _doa_matrix=np.linalg.pinv(X_TRAIN.T @ X_TRAIN),
            _a=0.9,
        )
    return make_obj(
        jq.MolecularModel,
        _model=estimator,
        _descriptors="MACCSKeysFingerprint",
        _doa=doa,
        _X=[f"f{i}" for i in range(167)],
        _Y=["Y"],
        _library=["sklearn"],
        _jaqpotpy_version="1.0.104",
        model_name="activity",
    )


@pytest.fixture
def classifier():
    return linear_model.LogisticRegression().fit(X_TRAIN, Y)


@pytest.fixture
def classifier_file(fake_jaqpotpy, write_pickle, classifier):
    return write_pickle(_jmodel(fake_jaqpotpy, classifier))


def _run(tmp_path, model, descriptors, manifest=None, **kwargs):
    outfile = tmp_path / "out" / "predictions.csv"
    counts = predict.run(model, descriptors, outfile, manifest=manifest, **kwargs)
    # keep_default_na=False so empty cells read back as '' rather than NaN
    return pd.read_csv(outfile, keep_default_na=False, dtype=str), counts


# --- loading ----------------------------------------------------------------


def test_bare_estimator_is_rejected(write_pickle, classifier):
    with pytest.raises(UnsupportedModelError, match="bare LogisticRegression"):
        predict.load_spec(write_pickle(classifier))


def test_unrecognised_contents_are_rejected(write_pickle):
    with pytest.raises(UnsupportedModelError, match="unrecognised"):
        predict.load_spec(write_pickle({"model": 1}))


def test_unsafe_file_is_rejected(tmp_path):
    class Payload:
        def __reduce__(self):
            return os.system, ("true",)

    path = tmp_path / "evil.jmodel"
    path.write_bytes(pickle.dumps(Payload()))
    with pytest.raises(UnsafeModelError):
        predict.run(
            path, MACCS_DESCRIPTORS, tmp_path / "out.csv", manifest=MACCS_MANIFEST
        )
    assert not (tmp_path / "out.csv").exists()


def test_loader_warnings_are_kept(fake_jaqpotpy, write_pickle, classifier, monkeypatch):
    monkeypatch.setattr("sklearn.base.__version__", "0.0.1")
    path = write_pickle(_jmodel(fake_jaqpotpy, classifier))
    monkeypatch.undo()
    spec = predict.load_spec(path)
    assert any("0.0.1" in w for w in spec.warnings)


# --- output -----------------------------------------------------------------


def test_predictions_match_the_estimator(tmp_path, classifier_file, classifier):
    spec = predict.load_spec(classifier_file)
    descfile, manifest = write_descriptor_file(tmp_path, spec, X_TRAIN)
    out, counts = _run(tmp_path, classifier_file, descfile, manifest=manifest)
    assert list(out.columns) == [
        "id",
        "smiles",
        "activity",
        "activity_probability",
        "activity_leverage",
        "activity_in_domain",
        "error",
    ]
    assert counts["read"] == counts["predicted"] == 6
    np.testing.assert_allclose(
        out.activity_probability.astype(float), classifier.predict_proba(X_TRAIN)[:, 1]
    )
    assert out.activity.tolist() == [str(v) for v in classifier.predict(X_TRAIN)]
    assert set(out.error) == {""}


@pytest.mark.parametrize(
    "estimator, columns",
    [
        (
            ensemble.RandomForestClassifier(n_estimators=5, random_state=0),
            [
                "activity_probability_a",
                "activity_probability_b",
                "activity_probability_c",
            ],
        ),
        (svm.SVC(), ["activity_score_a", "activity_score_b", "activity_score_c"]),
        (ensemble.RandomForestRegressor(n_estimators=5, random_state=0), []),
    ],
)
def test_output_columns(tmp_path, fake_jaqpotpy, write_pickle, estimator, columns):
    y = ["a", "b", "c", "a", "b", "c"]
    if not columns:
        y = np.arange(6.0)
    path = write_pickle(_jmodel(fake_jaqpotpy, estimator.fit(X_TRAIN, y), False))
    spec = predict.load_spec(path)
    descfile, manifest = write_descriptor_file(tmp_path, spec, X_TRAIN)
    out, _ = _run(tmp_path, path, descfile, manifest=manifest)
    assert list(out.columns) == ["id", "smiles", "activity", *columns, "error"]


# --- scikit-learn versions -------------------------------------------------

VERSION_MODELS = sorted((DATA_DIR / "models").glob("rf-maccs-sklearn-*.jmodel"))


def _same_tree_format(version):
    change = job_versions.TREE_FORMAT_CHANGE
    return (_version_tuple(version) >= change) == (
        _version_tuple(sklearn.__version__) >= change
    )


@pytest.mark.parametrize("model", VERSION_MODELS, ids=lambda p: p.name)
def test_models_from_other_scikit_learn_versions(tmp_path, model):
    """Models made by scripts/make_version_test_model.py with each version. All
    use MACCS keys, so the committed MACCS descriptor file feeds them."""
    version = model.name.removeprefix("rf-maccs-sklearn-").removesuffix(".jmodel")
    if _same_tree_format(version):
        _, counts = _run(tmp_path, model, MACCS_DESCRIPTORS, manifest=MACCS_MANIFEST)
        assert counts["predicted"] == 10
    else:
        with pytest.raises(ModelLoadError, match="use job version") as info:
            predict.load_spec(model)
        hint = str(info.value)
        assert f"running scikit-learn {sklearn.__version__}" in hint
        # the hint names exactly the job versions that can read the model
        for job, skl in job_versions.JOB_VERSIONS.items():
            can_read = not _same_tree_format(skl)
            assert (f"{job} (scikit-learn {skl})" in hint) == can_read


def test_job_versions_match_job_definition():
    """This branch's job version runs the scikit-learn it is listed with."""
    jobs = yaml.safe_load((ROOT / "data-manager" / "jobs.yaml").read_text())
    version = jobs["jobs"]["predict"]["version"]
    assert job_versions.JOB_VERSIONS[version] == sklearn.__version__


def test_warns_that_predictions_are_unverified(
    tmp_path, classifier_file, classifier, capsys
):
    spec = predict.load_spec(classifier_file)
    descfile, manifest = write_descriptor_file(tmp_path, spec, X_TRAIN)
    predict.run(classifier_file, descfile, tmp_path / "out.csv", manifest=manifest)
    assert "predictions can't be verified" in capsys.readouterr().out
