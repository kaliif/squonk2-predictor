import gzip
import os
import pickle

import numpy as np
import pandas as pd
import pytest
from conftest import DATA_DIR, MODELS_DIR, make_obj
from rdkit import Chem
from sklearn import ensemble, linear_model, svm

import predict
from featurizers import MACCSKeys
from model_spec import UnsupportedModelError
from probes import ProbeError, check_probes, read_probes
from safe_loader import UnsafeModelError

SMILES = [
    "CCO",
    "c1ccccc1C(=O)O",
    "CC(C)NCC(O)COc1ccc(NC=O)cc1",
    "CCN(CC)CC",
    "OC1CCCCC1",
    "c1ccncc1",
]
MOLS = [Chem.MolFromSmiles(s) for s in SMILES]
X_TRAIN = MACCSKeys().compute(MOLS)[0]
SDF = DATA_DIR / "candidates-10.sdf"


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
    return linear_model.LogisticRegression().fit(X_TRAIN, [0, 1, 1, 0, 0, 1])


@pytest.fixture
def classifier_file(fake_jaqpotpy, write_pickle, classifier):
    return write_pickle(_jmodel(fake_jaqpotpy, classifier))


def _maccs(mols):
    return MACCSKeys(missing_value=0.0).compute(mols)[0]


def _run(tmp_path, model, molecules, **kwargs):
    outfile = tmp_path / "out" / "predictions.csv"
    counts = predict.run(model, molecules, outfile, **kwargs)
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
        predict.run(path, SDF, tmp_path / "out.csv")
    assert not (tmp_path / "out.csv").exists()


def test_loader_warnings_are_kept(fake_jaqpotpy, write_pickle, classifier, monkeypatch):
    monkeypatch.setattr("sklearn.base.__version__", "0.0.1")
    path = write_pickle(_jmodel(fake_jaqpotpy, classifier))
    monkeypatch.undo()
    spec = predict.load_spec(path)
    assert any("0.0.1" in w for w in spec.warnings)


# --- output -----------------------------------------------------------------


def test_sdf_predictions(tmp_path, classifier_file, classifier):
    out, counts = _run(tmp_path, classifier_file, SDF)
    assert list(out.columns) == [
        "id",
        "smiles",
        "activity",
        "activity_probability",
        "activity_leverage",
        "activity_in_domain",
        "model_status",
        "error",
    ]
    mols = list(Chem.SDMolSupplier(str(SDF)))
    assert counts["read"] == counts["predicted"] == 10
    assert out.id.tolist() == [m.GetProp("_Name") for m in mols]
    X = _maccs(mols)
    np.testing.assert_allclose(
        out.activity_probability.astype(float), classifier.predict_proba(X)[:, 1]
    )
    assert out.activity.tolist() == [str(v) for v in classifier.predict(X)]
    assert set(out.model_status) == {"unverified"}
    assert set(out.error) == {""}
    assert (out.activity_in_domain == "False").sum() == counts["out_of_domain"]


def test_smiles_file_formats(tmp_path, classifier_file, classifier):
    lines = [
        "smiles\tname",
        "CCO\tethanol",
        "not_a_smiles\tbad",
        "",
        "c1ccccc1O phenol",
        "CCN",
    ]
    path = tmp_path / "in.smi.gz"
    with gzip.open(path, "wt") as fh:
        fh.write("\n".join(lines) + "\n")

    out, counts = _run(tmp_path, classifier_file, path, read_header=True)
    assert out.id.tolist() == ["ethanol", "bad", "phenol", ""]
    assert out.smiles.tolist() == ["CCO", "not_a_smiles", "c1ccccc1O", "CCN"]
    assert out.error.tolist() == ["", "invalid molecule", "", ""]
    assert counts["read"] == 4 and counts["predicted"] == 3
    # integer labels stay integers next to an empty row
    expected = classifier.predict(_maccs([Chem.MolFromSmiles(s) for s in ("CCO",)]))
    assert out.activity[0] == str(expected[0]) and out.activity[1] == ""


def test_comma_separated_smiles(tmp_path, classifier_file):
    path = tmp_path / "in.csv"
    path.write_text("CCO,ethanol,extra\n")
    out, _ = _run(tmp_path, classifier_file, path, delimiter=",")
    assert out.id.tolist() == ["ethanol"] and out.smiles.tolist() == ["CCO"]


def test_batches_give_same_output(tmp_path, classifier_file, monkeypatch):
    whole, _ = _run(tmp_path, classifier_file, SDF)
    monkeypatch.setattr(predict, "BATCH_SIZE", 3)
    batched, _ = _run(tmp_path, classifier_file, SDF)
    pd.testing.assert_frame_equal(whole, batched)


def test_all_molecules_failing(tmp_path, classifier_file):
    path = tmp_path / "in.smi"
    path.write_text("xyz\n")
    out, counts = _run(tmp_path, classifier_file, path)
    assert "activity_probability" in out.columns
    assert counts["predicted"] == 0 and out.error.tolist() == ["invalid molecule"]


def test_empty_input(tmp_path, classifier_file):
    path = tmp_path / "in.smi"
    path.write_text("")
    out, counts = _run(tmp_path, classifier_file, path)
    assert out.empty and "activity" in out.columns and counts["read"] == 0


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
    out, _ = _run(tmp_path, path, SDF)
    assert list(out.columns) == [
        "id",
        "smiles",
        "activity",
        *columns,
        "model_status",
        "error",
    ]


# --- probes -----------------------------------------------------------------


def _probes(tmp_path, classifier):
    X = _maccs(MOLS)
    probes = pd.DataFrame(
        {
            "smiles": SMILES,
            "prediction": classifier.predict(X),
            "score": classifier.predict_proba(X)[:, 1],
        }
    )
    path = tmp_path / "probes.csv"
    probes.to_csv(path, index=False)
    return path


def test_probes_verify_model(tmp_path, classifier_file, classifier):
    out, _ = _run(tmp_path, classifier_file, SDF, probes=_probes(tmp_path, classifier))
    assert set(out.model_status) == {"verified"}


def test_probes_without_score(tmp_path, classifier_file, classifier):
    path = _probes(tmp_path, classifier)
    pd.read_csv(path).drop(columns="score").to_csv(path, index=False)
    predict.run(classifier_file, SDF, tmp_path / "out.csv", probes=path)


@pytest.mark.parametrize(
    "column, message",
    [("prediction", "prediction"), ("score", "probability 0.\\d+, expected 0.123")],
)
def test_probe_mismatch_rejects_model(
    tmp_path, classifier_file, classifier, column, message
):
    probes = _probes(tmp_path, classifier)
    df = pd.read_csv(probes)
    df.loc[2, column] = 1 - df.loc[2, column] if column == "prediction" else 0.123
    df.to_csv(probes, index=False)
    with pytest.raises(ProbeError, match=f"1 probe values.*{message}"):
        predict.run(classifier_file, SDF, tmp_path / "out.csv", probes=probes)
    assert not (tmp_path / "out.csv").exists()


@pytest.mark.parametrize(
    "content, message",
    [
        ("smiles\nCCO\n", "no prediction column"),
        ("smiles,prediction\n", "no molecules"),
        ("smiles,prediction\nCCO,\n", "without a prediction"),
        ("smiles,prediction\nxyz,1\n", "invalid probe SMILES: xyz"),
    ],
)
def test_bad_probe_files(tmp_path, classifier_file, content, message):
    path = tmp_path / "probes.csv"
    path.write_text(content)
    with pytest.raises(ProbeError, match=message):
        predict.run(classifier_file, SDF, tmp_path / "out.csv", probes=path)


def test_probe_score_needs_single_score(tmp_path, fake_jaqpotpy, write_pickle):
    est = ensemble.RandomForestRegressor(n_estimators=5, random_state=0)
    est.fit(X_TRAIN, np.arange(6.0))
    spec = predict.load_spec(write_pickle(_jmodel(fake_jaqpotpy, est)))
    probes = pd.DataFrame(
        {
            "smiles": SMILES,
            "prediction": est.predict(_maccs(MOLS)),
            "score": 0.5,
        }
    )
    with pytest.raises(ProbeError, match="no single score"):
        check_probes(spec, probes)
    check_probes(spec, probes.drop(columns="score"))
    probes.loc[0, "prediction"] += 0.01
    with pytest.raises(ProbeError, match="1 probe values"):
        check_probes(spec, probes.drop(columns="score"))


LEGACY_PROBES = sorted(MODELS_DIR.glob("*.probes.csv"), key=lambda p: p.name.lower())


@pytest.mark.skipif(not LEGACY_PROBES, reason="legacy models not available")
@pytest.mark.parametrize("probes", LEGACY_PROBES, ids=lambda p: p.name)
def test_legacy_model_reproduces_probes(probes):
    """The legacy models reproduce predictions made by jaqpotpy itself (see
    scripts/make_jaqpotpy_probes.py) on TDC ADMET test molecules."""
    model = probes.with_name(probes.name.removesuffix(".probes.csv") + ".jmodel")
    check_probes(predict.load_spec(model), read_probes(probes))
