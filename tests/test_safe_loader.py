import io
import os
import pickle
import sys

import joblib
import numpy as np
import pytest
from conftest import make_obj
from sklearn import (
    compose,
    decomposition,
    ensemble,
    feature_selection,
    gaussian_process,
    impute,
    linear_model,
    naive_bayes,
    neighbors,
    neural_network,
    pipeline,
    preprocessing,
    svm,
    tree,
)
from sklearn.exceptions import InconsistentVersionWarning

from safe_loader import (
    ModelLoadError,
    Stub,
    UnsafeModelError,
    load_model_file,
    stub_name,
)

# --- malicious files --------------------------------------------------------


class _Reduce:
    """Pickles as a call of func(*args) when unpickled."""

    def __init__(self, func, *args):
        self.func, self.args = func, args

    def __reduce__(self):
        return self.func, self.args


def _protocol0(*lines):
    """Hand-written protocol 0 pickle, for references Python won't pickle."""
    return "\n".join(lines).encode() + b"\n."


@pytest.fixture
def marker(tmp_path):
    path = tmp_path / "pwned"
    yield path
    assert not path.exists(), "payload was executed"


@pytest.mark.parametrize("compress", [None, 0, 3, "gzip", "xz"])
def test_rejects_os_system(tmp_path, marker, compress):
    payload = _Reduce(os.system, f"touch {marker}")
    path = tmp_path / "evil.pkl"
    if compress is None:
        path.write_bytes(pickle.dumps(payload))
    else:
        joblib.dump(payload, path, compress=compress)
    with pytest.raises(UnsafeModelError) as exc:
        load_model_file(path)
    assert "posix.system" in str(exc.value) or "os.system" in str(exc.value)


@pytest.mark.parametrize(
    "func,args",
    [
        (eval, ("__import__('os').getcwd()",)),
        (exec, ("x = 1",)),
        (getattr, ("text", "upper")),
        (__import__, ("subprocess",)),
        (open, ("/etc/passwd",)),
    ],
)
def test_rejects_builtins(write_pickle, func, args):
    with pytest.raises(UnsafeModelError):
        load_model_file(write_pickle(_Reduce(func, *args)))


@pytest.mark.parametrize(
    "module,name",
    [
        # resolves any module attribute by name: a known pickle gadget
        ("numpy.core", "_ufunc_reconstruct"),
        ("numpy", "load"),
        ("numpy", "fromfile"),
        ("numpy.core.multiarray", "fromfile"),
        ("numpy.testing", "assert_equal"),
        # sklearn functions (not estimator classes) are not allowed
        ("sklearn.datasets", "load_svmlight_file"),
        ("sklearn.utils", "_print_elapsed_time"),
        # nested names could reach other objects through an allowed one
        ("sklearn.svm", "SVC.__init__.__globals__"),
        ("joblib", "load"),
        ("_codecs", "decode"),
    ],
)
def test_rejects_disallowed_globals(tmp_path, module, name):
    path = tmp_path / "evil.pkl"
    path.write_bytes(_protocol0(f"c{module}", name, "(tR"))
    with pytest.raises(UnsafeModelError) as exc:
        load_model_file(path)
    assert f"{module}.{name}" in exc.value.rejected


def test_ufunc_gadget_not_executed(tmp_path, marker):
    path = tmp_path / "evil.pkl"
    path.write_bytes(
        _protocol0(
            "cnumpy.core",
            "_ufunc_reconstruct",
            "(Vos",
            "Vsystem",
            "tR(Vtouch " + str(marker),
            "tR",
        )
    )
    with pytest.raises(UnsafeModelError):
        load_model_file(path)


def test_reports_all_rejected_references(write_pickle):
    payload = {
        "a": _Reduce(os.getcwd),
        "b": _Reduce(eval, "1"),
        "model": linear_model.Ridge(),
    }
    with pytest.raises(UnsafeModelError) as exc:
        load_model_file(write_pickle(payload))
    assert len(exc.value.rejected) == 2


@pytest.mark.parametrize("compress", [0, 3])
def test_joblib_object_array_is_restricted(tmp_path, marker, compress):
    # joblib itself unpickles object arrays with a plain pickle.load
    payload = np.array([_Reduce(os.system, f"touch {marker}"), 1], dtype=object)
    path = tmp_path / "evil.joblib"
    joblib.dump({"array": payload}, path, compress=compress)
    with pytest.raises(UnsafeModelError):
        load_model_file(path)


def test_codecs_encode_only_latin1(tmp_path):
    path = tmp_path / "codec.pkl"
    path.write_bytes(_protocol0("c_codecs", "encode", "(Vabc", "Vrot13", "tR"))
    with pytest.raises(ModelLoadError):
        load_model_file(path)


def test_rejects_persistent_ids(tmp_path):
    class P(pickle.Pickler):
        def persistent_id(self, obj):
            return "x" if obj == "secret" else None

    buf = io.BytesIO()
    P(buf).dump(["secret"])
    path = tmp_path / "pid.pkl"
    path.write_bytes(buf.getvalue())
    with pytest.raises(ModelLoadError):
        load_model_file(path)


def test_garbage_file(tmp_path):
    path = tmp_path / "junk.pkl"
    path.write_bytes(b"this is not a pickle")
    with pytest.raises(ModelLoadError):
        load_model_file(path)


# --- legitimate models ------------------------------------------------------

rng = np.random.default_rng(0)
X = rng.random((60, 6))
Y_REG = X[:, 0] * 3 + rng.random(60)
Y_CLF = (Y_REG > Y_REG.mean()).astype(int)
Y_STR = np.where(Y_CLF == 1, "active", "inactive")

ESTIMATORS = {
    "rf_random_state_object": (
        ensemble.RandomForestRegressor(5, random_state=np.random.RandomState(0)),
        Y_REG,
    ),
    "extra_trees": (ensemble.ExtraTreesClassifier(5, random_state=0), Y_CLF),
    "gradient_boosting": (ensemble.GradientBoostingClassifier(n_estimators=5), Y_CLF),
    "hist_gradient_boosting": (
        ensemble.HistGradientBoostingRegressor(max_iter=5),
        Y_REG,
    ),
    "adaboost": (ensemble.AdaBoostClassifier(n_estimators=5), Y_CLF),
    "voting_hard": (
        ensemble.VotingClassifier(
            [("a", svm.SVC()), ("b", tree.DecisionTreeClassifier())]
        ),
        Y_CLF,
    ),
    "stacking": (
        ensemble.StackingRegressor([("a", linear_model.Ridge()), ("b", svm.SVR())]),
        Y_REG,
    ),
    "logistic_string_labels": (linear_model.LogisticRegression(), Y_STR),
    "sgd": (linear_model.SGDRegressor(), Y_REG),
    "svc_proba": (svm.SVC(probability=True, random_state=0), Y_CLF),
    "knn_kd_tree": (neighbors.KNeighborsClassifier(algorithm="kd_tree"), Y_CLF),
    "knn_ball_tree": (neighbors.KNeighborsRegressor(algorithm="ball_tree"), Y_REG),
    "naive_bayes": (naive_bayes.BernoulliNB(), Y_CLF),
    "gaussian_process": (
        gaussian_process.GaussianProcessRegressor(
            kernel=gaussian_process.kernels.RBF()
            + gaussian_process.kernels.WhiteKernel()
        ),
        Y_REG,
    ),
    "mlp": (neural_network.MLPRegressor(max_iter=20, random_state=0), Y_REG),
    "pipeline": (
        pipeline.make_pipeline(
            impute.SimpleImputer(),
            preprocessing.StandardScaler(),
            feature_selection.VarianceThreshold(),
            decomposition.PCA(3),
            linear_model.Ridge(),
        ),
        Y_REG,
    ),
    "target_transform_ufunc": (
        compose.TransformedTargetRegressor(
            regressor=linear_model.Ridge(), func=np.log1p, inverse_func=np.expm1
        ),
        Y_REG,
    ),
}


@pytest.fixture(scope="module")
def fitted():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {k: est.fit(X, y) for k, (est, y) in ESTIMATORS.items()}


@pytest.mark.parametrize("name", ESTIMATORS)
@pytest.mark.parametrize(
    "fmt", ["pickle", "joblib", "joblib-zlib", "joblib-gzip", "joblib-xz", "joblib-bz2"]
)
def test_round_trip(tmp_path, fitted, name, fmt):
    est = fitted[name]
    path = tmp_path / "model"
    if fmt == "pickle":
        path.write_bytes(pickle.dumps(est, protocol=pickle.HIGHEST_PROTOCOL))
    else:
        compress = {
            "joblib": 0,
            "joblib-zlib": 3,
            "joblib-gzip": "gzip",
            "joblib-xz": "xz",
            "joblib-bz2": "bz2",
        }[fmt]
        joblib.dump(est, path, compress=compress)
    loaded = load_model_file(path)
    assert type(loaded.obj) is type(est)
    np.testing.assert_array_equal(loaded.obj.predict(X), est.predict(X))


def test_joblib_array_layouts(tmp_path):
    arrays = {
        "fortran": np.asfortranarray(rng.random((4, 3))),
        "big_endian": np.arange(5, dtype=">f8"),
        "empty": np.zeros((0, 3)),
        "scalar": np.array(2.5),
        "structured": np.array([(1, 2.0)], dtype=[("a", "i4"), ("b", "f8")]),
    }
    path = tmp_path / "arrays.joblib"
    joblib.dump(arrays, path)
    loaded = load_model_file(path).obj
    for key, value in arrays.items():
        np.testing.assert_array_equal(loaded[key], value)
        assert loaded[key].dtype == value.dtype.newbyteorder("=")


def test_records_sklearn_versions(write_pickle, fitted, monkeypatch):
    import sklearn.base

    # sklearn records its version when pickling
    monkeypatch.setattr(sklearn.base, "__version__", "0.0.1")
    path = write_pickle(fitted["rf_random_state_object"])
    monkeypatch.undo()
    result = load_model_file(path)
    assert result.sklearn_versions == ["0.0.1"]
    # reported from sklearn_versions rather than a warning per estimator class
    assert not any("0.0.1" in w for w in result.warnings)


def _as_sklearn_1_3(estimators, monkeypatch):
    """Make fitted trees look as scikit-learn 1.3 pickled them: classifier
    nodes hold weighted class counts rather than fractions, and there is no
    monotonic_cst. Returns the pickle."""
    import sklearn.base

    for est in estimators:
        del est.monotonic_cst
        if isinstance(est, tree.DecisionTreeClassifier):
            state = est.tree_.__getstate__()
            weights = est.tree_.weighted_n_node_samples[:, None, None]
            state["values"] = state["values"] * weights
            est.tree_.__setstate__(state)
    monkeypatch.setattr(sklearn.base, "__version__", "1.3.2")
    data = pickle.dumps(estimators)
    monkeypatch.undo()
    return data


def test_old_classifier_trees_are_converted(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    X, y = rng.random((60, 4)), rng.integers(0, 3, 60)
    forest = ensemble.RandomForestClassifier(n_estimators=3, random_state=0)
    forest.fit(X, y, sample_weight=rng.random(60) + 0.5)
    regressor = tree.DecisionTreeRegressor(random_state=0).fit(X, y.astype(float))
    expected = forest.predict_proba(X), regressor.predict(X)

    path = tmp_path / "old.pkl"
    path.write_bytes(
        _as_sklearn_1_3([forest, *forest.estimators_, regressor], monkeypatch)
    )
    # unconverted, the counts give wrong probabilities
    with pytest.warns(InconsistentVersionWarning):
        plain = pickle.loads(path.read_bytes())[0]
    assert not np.allclose(plain.predict_proba(X), expected[0])

    result = load_model_file(path)
    loaded_forest, loaded_regressor = result.obj[0], result.obj[-1]
    np.testing.assert_allclose(loaded_forest.predict_proba(X), expected[0])
    np.testing.assert_array_equal(loaded_regressor.predict(X), expected[1])
    assert loaded_regressor.monotonic_cst is None
    assert result.sklearn_versions == ["1.3.2"]
    # the forest's trees are listed twice in the pickle but stored once
    assert any("converted 3 classifier trees" in w for w in result.warnings)


# --- stubbed packages -------------------------------------------------------


def test_jaqpotpy_objects_are_stubbed(fake_jaqpotpy, write_pickle, monkeypatch, fitted):
    model = make_obj(
        fake_jaqpotpy.MolecularModel,
        _model=fitted["naive_bayes"],
        _descriptors=make_obj(
            fake_jaqpotpy.TopologicalFingerprint, radius=2, size=2048
        ),
    )
    path = write_pickle(model)
    for name in [m for m in sys.modules if m.split(".")[0] == "jaqpotpy"]:
        monkeypatch.delitem(sys.modules, name)

    obj = load_model_file(path).obj

    assert isinstance(obj, Stub)
    assert stub_name(obj) == "jaqpotpy.models.base_classes.MolecularModel"
    assert stub_name(obj._descriptors).endswith("TopologicalFingerprint")
    assert obj._descriptors.radius == 2
    assert isinstance(obj._model, naive_bayes.BernoulliNB)
    assert "jaqpotpy" not in sys.modules
