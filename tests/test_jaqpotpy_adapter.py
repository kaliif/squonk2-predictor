import numpy as np
import pandas as pd
import pytest
from conftest import MODELS_DIR, TEST_DATA_DIR, make_obj
from rdkit import Chem
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

SMILES = [
    "CCO",
    "c1ccccc1C(=O)O",
    "CC(C)NCC(O)COc1ccc(NC=O)cc1",
    "CCN(CC)CC",
    "OC1CCCCC1",
    "c1ccncc1",
]
MOLS = [Chem.MolFromSmiles(s) for s in SMILES]
Y_CLF = np.array([0, 1, 1, 0, 0, 1])
Y_REG = np.arange(len(SMILES), dtype=float)


def _maccs_features():
    X, _ = MACCSKeys().compute(MOLS)
    return X


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
    X = _maccs_features()
    est = linear_model.LogisticRegression().fit(X, Y_CLF)
    spec = _adapt(
        write_pickle, _jmodel(fake_jaqpotpy, est, _doa=_leverage(fake_jaqpotpy, X))
    )

    assert isinstance(spec.featurizer, MACCSKeys)
    assert spec.featurizer.missing_value == 0.0
    assert spec.name == spec.target_name == "test_model"
    assert spec.output_kind == "probability"
    features, ok = spec.features(MOLS)
    assert ok.all()
    out = spec.predict_features(features)
    np.testing.assert_array_equal(out["prediction"], est.predict(X))
    np.testing.assert_allclose(out["probability"], est.predict_proba(X))
    assert out["in_domain"].dtype == bool
    np.testing.assert_array_equal(spec.training_features, X)


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
    X, _ = MorganFingerprint(3, 1024, True, False, True).compute(MOLS)
    est = naive_bayes.BernoulliNB().fit(X, Y_CLF)
    spec = _adapt(write_pickle, _jmodel(fake_jaqpotpy, est, desc, 1024))
    assert spec.featurizer.params() == {
        "radius": 3,
        "n_bits": 1024,
        "chiral": True,
        "bond_types": False,
        "feature_invariants": True,
    }
    np.testing.assert_array_equal(
        spec.predict_features(spec.features(MOLS)[0])["prediction"], est.predict(X)
    )


def test_rdkit_descriptors_by_name(fake_jaqpotpy, write_pickle):
    # jaqpotpy stores RDKitDescriptors by name; features are selected by name
    fz = RDKitDescriptors()
    names = [n for n in fz.feature_names if n not in ("SPS", "AvgIpc")][::-1]
    full, _ = fz.compute(MOLS)
    X = full[:, [fz.feature_names.index(n) for n in names]]
    est = tree.DecisionTreeRegressor(random_state=0).fit(X, Y_REG)
    spec = _adapt(
        write_pickle, _jmodel(fake_jaqpotpy, est, "RDKitDescriptors", _X=names)
    )
    np.testing.assert_array_equal(
        spec.predict_features(spec.features(MOLS)[0])["prediction"], est.predict(X)
    )


def test_mordred_model(fake_jaqpotpy, write_pickle):
    fz = MordredDescriptors(missing_value=0.0)
    X, _ = fz.compute(MOLS)
    est = ensemble.RandomForestRegressor(5, random_state=0).fit(X, Y_REG)
    desc = make_obj(fake_jaqpotpy.MordredDescriptors, ignore_3D=True)
    spec = _adapt(write_pickle, _jmodel(fake_jaqpotpy, est, desc, _X=fz.feature_names))
    assert isinstance(spec.featurizer, MordredDescriptors)
    np.testing.assert_array_equal(
        spec.predict_features(spec.features(MOLS)[0])["prediction"], est.predict(X)
    )


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
    est = naive_bayes.BernoulliNB().fit(_maccs_features(), Y_CLF)
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
    est = naive_bayes.BernoulliNB().fit(_maccs_features(), Y_CLF)
    path = write_pickle(_jmodel(fake_jaqpotpy, est, make_desc(fake_jaqpotpy)))
    with pytest.raises(UnsupportedModelError, match=message):
        jaqpotpy_adapter.adapt(load_model_file(path).obj)


def test_domain_rejections(fake_jaqpotpy, write_pickle):
    X = _maccs_features()
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


def test_check_rejects_unknown_feature_names(fake_jaqpotpy, write_pickle):
    est = tree.DecisionTreeRegressor().fit(np.zeros((2, 2)), [0, 1])
    model = _jmodel(
        fake_jaqpotpy, est, "RDKitDescriptors", _X=["MolWt", "NotADescriptor"]
    )
    spec = jaqpotpy_adapter.adapt(load_model_file(write_pickle(model)).obj)
    with pytest.raises(UnsupportedModelError, match="NotADescriptor"):
        check_spec(spec)


def test_check_rejects_feature_count_mismatch(fake_jaqpotpy, write_pickle):
    est = naive_bayes.BernoulliNB().fit(_maccs_features()[:, :100], Y_CLF)
    spec = jaqpotpy_adapter.adapt(
        load_model_file(write_pickle(_jmodel(fake_jaqpotpy, est))).obj
    )
    with pytest.raises(UnsupportedModelError, match="fitted on 100"):
        check_spec(spec)


def test_check_rejects_estimator_that_cannot_run(fake_jaqpotpy, write_pickle):
    est = linear_model.LogisticRegression().fit(_maccs_features(), Y_CLF)
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
    est = estimator.fit(_maccs_features(), y)
    spec = _adapt(write_pickle, _jmodel(fake_jaqpotpy, est))
    assert len(spec.warnings) == len(expected)
    for text in expected:
        assert any(text in w for w in spec.warnings), (text, spec.warnings)


def test_weak_domain_warning(fake_jaqpotpy, write_pickle):
    # 6 molecules and 167 features: threshold far above 1
    X = _maccs_features()
    est = linear_model.Ridge().fit(X, Y_REG)
    spec = _adapt(
        write_pickle, _jmodel(fake_jaqpotpy, est, _doa=_leverage(fake_jaqpotpy, X))
    )
    assert len(spec.warnings) == 1
    assert "domain check is weak" in spec.warnings[0]


# --- the real legacy models, when present ------------------------------------

REFERENCE = pd.read_csv(TEST_DATA_DIR / "jaqpotpy_reference.csv")


@pytest.mark.skipif(not MODELS_DIR.is_dir(), reason="legacy models not available")
@pytest.mark.parametrize("model_name", sorted(REFERENCE.model.unique(), key=str.lower))
def test_legacy_model_matches_reference(model_name):
    """Predictions match those computed independently (separate scratch
    implementation and environment) for TDC ADMET test molecules."""
    path = MODELS_DIR / f"{model_name}.jmodel"
    if not path.exists():
        pytest.skip(f"{path.name} not available")
    spec = jaqpotpy_adapter.adapt(load_model_file(path).obj)
    check_spec(spec)

    ref = REFERENCE[REFERENCE.model == model_name]
    X, ok = spec.features([Chem.MolFromSmiles(s) for s in ref.smiles])
    assert ok.all()
    out = spec.predict_features(X)
    np.testing.assert_allclose(
        out["prediction"].astype(float), ref.prediction, rtol=1e-6
    )
    if spec.output_kind == "probability":
        np.testing.assert_allclose(
            out["probability"][:, 1], ref.score, rtol=1e-6, atol=1e-9
        )
    elif spec.output_kind == "decision_function":
        np.testing.assert_allclose(out["decision_function"], ref.score, rtol=1e-6)
