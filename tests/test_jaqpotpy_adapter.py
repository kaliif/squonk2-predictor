"""Tests for the jaqpotpy adapter: building a ModelSpec from a .jmodel stub.

Descriptors are no longer calculated in the job, so these check the adapter
reads the featurizer identity, its parameters, the feature list, the input
dtype, the applicability domain and the target correctly, and that check_spec
applies its model-level checks. Actual descriptor values are the descriptor
jobs' concern (see test_descriptor_input)."""

import numpy as np
import pytest
from conftest import make_obj
from sklearn import ensemble, linear_model, naive_bayes, neighbors, svm, tree

import jaqpotpy_adapter
from featurizers import (
    MACCSKeys,
    MordredDescriptors,
    MorganFingerprint,
    RDKitDescriptors,
)
from model_spec import UnsupportedModelError, check_spec
from safe_loader import load_model_file

N = 6
Y_CLF = np.array([0, 1, 1, 0, 0, 1])
Y_REG = np.arange(N, dtype=float)
_RNG = np.random.default_rng(0)


def _features(n_features, binary=True):
    """An arbitrary feature matrix to fit estimators on; the values don't
    matter now that the job doesn't compute descriptors."""
    if binary:
        return _RNG.integers(0, 2, size=(N, n_features)).astype(float)
    return _RNG.random((N, n_features))


def _jmodel(
    jq, estimator, descriptors="MACCSKeysFingerprint", n_features=167, **overrides
):
    attrs = dict(
        _model=estimator,
        _descriptors=descriptors,
        _doa=None,
        _X=[f"f{i}" for i in range(n_features)],
        _Y=["Y"],
        _library=["sklearn"],
        _version=["1.3.2"],
        _jaqpotpy_version="1.0.104",
        _external_feats=None,
        model_name="test_model",
    )
    attrs.update(overrides)
    return make_obj(jq.MolecularModel, **attrs)


def _leverage(jq, X):
    return make_obj(
        jq.Leverage,
        _data=X,
        _doa_matrix=np.linalg.pinv(X.T @ X),
        _a=3 * (X.shape[1] + 1) / X.shape[0],
    )


def _adapt(write_pickle, model):
    spec = jaqpotpy_adapter.adapt(load_model_file(write_pickle(model)).obj)
    check_spec(spec)
    return spec


def test_maccs_model(fake_jaqpotpy, write_pickle):
    X = _features(167)
    est = linear_model.LogisticRegression().fit(X, Y_CLF)
    spec = _adapt(
        write_pickle, _jmodel(fake_jaqpotpy, est, _doa=_leverage(fake_jaqpotpy, X))
    )

    assert isinstance(spec.featurizer, MACCSKeys)
    assert spec.featurizer.name == "maccs" and spec.featurizer.binary
    assert spec.featurizer.missing_value == 0.0
    assert spec.name == spec.target_name == "test_model"
    assert spec.output_kind == "probability"
    out = spec.predict_features(X)
    np.testing.assert_array_equal(out["prediction"], est.predict(X))
    np.testing.assert_allclose(out["probability"], est.predict_proba(X))
    assert out["in_domain"].dtype == bool
    np.testing.assert_array_equal(spec.training_features, X)


def test_input_dtype_follows_jaqpotpy(fake_jaqpotpy, write_pickle):
    """jaqpotpy gives MACCS keys to the estimator as int8; with float input
    nearest neighbour models break distance ties differently."""
    X = _features(167)
    est = neighbors.KNeighborsClassifier(n_neighbors=2).fit(X.astype(np.int8), Y_CLF)
    spec = _adapt(write_pickle, _jmodel(fake_jaqpotpy, est))
    assert spec.input_dtype is np.int8
    seen = []
    spec.estimator.predict = lambda X: seen.append(X.dtype) or np.zeros(len(X))
    spec.predict_features(X)
    assert seen and set(seen) == {np.dtype(np.int8)}

    desc = make_obj(
        fake_jaqpotpy.TopologicalFingerprint,
        radius=2,
        size=1024,
        chiral=False,
        bonds=True,
        features=False,
        sparse=False,
    )
    est = naive_bayes.BernoulliNB().fit(_features(1024), Y_CLF)
    assert (
        _adapt(write_pickle, _jmodel(fake_jaqpotpy, est, desc, 1024)).input_dtype
        is np.float64
    )


def test_morgan_parameters(fake_jaqpotpy, write_pickle):
    desc = make_obj(
        fake_jaqpotpy.TopologicalFingerprint,
        radius=3,
        size=1024,
        chiral=True,
        bonds=False,
        features=True,
        sparse=False,
        smiles=False,
    )
    X = _features(1024)
    est = naive_bayes.BernoulliNB().fit(X, Y_CLF)
    spec = _adapt(write_pickle, _jmodel(fake_jaqpotpy, est, desc, 1024))
    assert isinstance(spec.featurizer, MorganFingerprint)
    assert spec.featurizer.params() == {
        "radius": 3,
        "n_bits": 1024,
        "chiral": True,
        "bond_types": False,
        "feature_invariants": True,
    }
    assert spec.featurizer.feature_names == [f"f{i}" for i in range(1024)]
    np.testing.assert_array_equal(
        spec.predict_features(X)["prediction"], est.predict(X)
    )


def test_rdkit_descriptors(fake_jaqpotpy, write_pickle):
    # jaqpotpy stores RDKitDescriptors by name; the model's own feature list is
    # kept, and columns are later selected by name from a descriptor file
    names = ["MolWt", "TPSA", "NumHDonors", "NumHAcceptors"]
    X = _features(len(names), binary=False)
    est = tree.DecisionTreeRegressor(random_state=0).fit(X, Y_REG)
    spec = _adapt(
        write_pickle, _jmodel(fake_jaqpotpy, est, "RDKitDescriptors", _X=names)
    )
    assert isinstance(spec.featurizer, RDKitDescriptors)
    assert spec.featurizer.params() == {"use_fragment": True, "ipc_avg": True}
    assert spec.feature_names == names
    np.testing.assert_array_equal(
        spec.predict_features(X)["prediction"], est.predict(X)
    )


def test_mordred_model(fake_jaqpotpy, write_pickle):
    names = [f"mordred_{i}" for i in range(10)]
    X = _features(len(names), binary=False)
    est = ensemble.RandomForestRegressor(5, random_state=0).fit(X, Y_REG)
    desc = make_obj(fake_jaqpotpy.MordredDescriptors, ignore_3D=True)
    spec = _adapt(write_pickle, _jmodel(fake_jaqpotpy, est, desc, _X=names))
    assert isinstance(spec.featurizer, MordredDescriptors)
    assert spec.featurizer.params() == {"ignore_3D": True}
    assert spec.feature_names == names


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"_jaqpotpy_version": "6.1.0"}, "jaqpotpy version"),
        ({"_library": ["torch_geometric", "torch"]}, "only scikit-learn"),
        ({"_model": "not a model"}, "scikit-learn estimator"),
        ({"_external_feats": ["logP"]}, "external features"),
        ({"_preprocessors": ["scaler"]}, "feature preprocessing"),
        ({"_preprocessors_y": ["scaler"]}, "target preprocessing"),
        ({"_X": None}, "feature names"),
        ({"_descriptors": "SomethingElse"}, "unsupported featurizer"),
    ],
)
def test_rejections(fake_jaqpotpy, write_pickle, overrides, message):
    est = naive_bayes.BernoulliNB().fit(_features(167), Y_CLF)
    path = write_pickle(_jmodel(fake_jaqpotpy, est, **overrides))
    with pytest.raises(UnsupportedModelError, match=message):
        jaqpotpy_adapter.adapt(load_model_file(path).obj)


@pytest.mark.parametrize(
    "make_desc,message",
    [
        (
            lambda jq: make_obj(
                jq.TopologicalFingerprint,
                radius=2,
                size=2048,
                chiral=False,
                bonds=True,
                features=False,
                sparse=True,
            ),
            "sparse",
        ),
        (
            lambda jq: make_obj(
                jq.TopologicalFingerprint,
                radius="2",
                size=2048,
                chiral=False,
                bonds=True,
                features=False,
                sparse=False,
            ),
            "radius",
        ),
        (lambda jq: make_obj(jq.MordredDescriptors, ignore_3D=False), "3D"),
        (lambda jq: make_obj(jq.SmilesToImage), "SmilesToImage"),
    ],
)
def test_featurizer_rejections(fake_jaqpotpy, write_pickle, make_desc, message):
    est = naive_bayes.BernoulliNB().fit(_features(167), Y_CLF)
    path = write_pickle(_jmodel(fake_jaqpotpy, est, make_desc(fake_jaqpotpy)))
    with pytest.raises(UnsupportedModelError, match=message):
        jaqpotpy_adapter.adapt(load_model_file(path).obj)


def test_domain_rejections(fake_jaqpotpy, write_pickle):
    X = _features(167)
    est = naive_bayes.BernoulliNB().fit(X, Y_CLF)
    for doa, message in [
        (make_obj(fake_jaqpotpy.MeanVar), "unsupported applicability domain"),
        (
            make_obj(fake_jaqpotpy.Leverage, _doa_matrix=np.eye(3), _a=0.5, _data=X),
            "malformed",
        ),
    ]:
        path = write_pickle(_jmodel(fake_jaqpotpy, est, _doa=doa))
        with pytest.raises(UnsupportedModelError, match=message):
            jaqpotpy_adapter.adapt(load_model_file(path).obj)


def test_check_rejects_feature_count_mismatch(fake_jaqpotpy, write_pickle):
    est = naive_bayes.BernoulliNB().fit(_features(167)[:, :100], Y_CLF)
    spec = jaqpotpy_adapter.adapt(
        load_model_file(write_pickle(_jmodel(fake_jaqpotpy, est))).obj
    )
    with pytest.raises(UnsupportedModelError, match="fitted on 100"):
        check_spec(spec)


def test_check_rejects_estimator_that_cannot_run(fake_jaqpotpy, write_pickle):
    est = linear_model.LogisticRegression().fit(_features(167), Y_CLF)
    del est.coef_
    spec = jaqpotpy_adapter.adapt(
        load_model_file(write_pickle(_jmodel(fake_jaqpotpy, est))).obj
    )
    with pytest.raises(UnsupportedModelError, match="can't run here"):
        check_spec(spec)


@pytest.mark.parametrize(
    "estimator,y,expected",
    [
        (svm.SVC(), Y_CLF, ["decision score", "no applicability domain"]),
        (
            ensemble.VotingClassifier(
                [("a", svm.SVC()), ("b", tree.DecisionTreeClassifier())]
            ),
            Y_CLF,
            ["labels only", "no applicability domain"],
        ),
        (
            neighbors.KNeighborsClassifier(2),
            Y_CLF,
            ["tie-breaking", "no applicability domain"],
        ),
        (
            ensemble.VotingRegressor(
                [("a", svm.SVR()), ("b", neighbors.KNeighborsRegressor(2))]
            ),
            Y_REG,
            ["tie-breaking", "no applicability domain"],
        ),
        (linear_model.Ridge(), Y_REG, ["no applicability domain"]),
    ],
)
def test_warnings(fake_jaqpotpy, write_pickle, estimator, y, expected):
    est = estimator.fit(_features(167), y)
    spec = _adapt(write_pickle, _jmodel(fake_jaqpotpy, est))
    assert len(spec.warnings) == len(expected)
    for text in expected:
        assert any(text in w for w in spec.warnings), (text, spec.warnings)


def test_weak_domain_warning(fake_jaqpotpy, write_pickle):
    # 6 molecules and 167 features: threshold far above 1
    X = _features(167)
    est = linear_model.Ridge().fit(X, Y_REG)
    spec = _adapt(
        write_pickle, _jmodel(fake_jaqpotpy, est, _doa=_leverage(fake_jaqpotpy, X))
    )
    assert len(spec.warnings) == 1
    assert "domain check is weak" in spec.warnings[0]
