"""Tests for the descriptor input path: reading a descriptor file and its
manifest, checking they suit the model, building the feature matrix, and
reproducing the estimator's own predictions."""

import json

import numpy as np
import pandas as pd
import pytest
from conftest import make_obj, write_descriptor_file
from rdkit import Chem
from sklearn import ensemble, linear_model

import predict
from descriptor_input import (
    DescriptorFileError,
    Manifest,
    build_matrix,
    check_manifest,
    read_manifest,
    sidecar_manifest_path,
)
from model_spec import UnsupportedModelError

N = 8
_RNG = np.random.default_rng(1)
X_BITS = _RNG.integers(0, 2, size=(N, 167)).astype(float)
X_DESC = _RNG.random((N, 5))
RDKIT_NAMES = ["MolWt", "TPSA", "NumHDonors", "NumHAcceptors", "MolLogP"]
Y = [0, 1, 1, 0, 1, 0, 1, 0]


def _jmodel(jq, estimator, descriptors, feature_names):
    return make_obj(
        jq.MolecularModel,
        _model=estimator,
        _descriptors=descriptors,
        _doa=None,
        _X=feature_names,
        _Y=["Y"],
        _library=["sklearn"],
        _jaqpotpy_version="1.0.104",
        model_name="activity",
    )


@pytest.fixture
def maccs_model(fake_jaqpotpy, write_pickle):
    est = linear_model.LogisticRegression().fit(X_BITS, Y)
    return write_pickle(
        _jmodel(
            fake_jaqpotpy, est, "MACCSKeysFingerprint", [f"f{i}" for i in range(167)]
        )
    )


@pytest.fixture
def rdkit_model(fake_jaqpotpy, write_pickle):
    est = ensemble.RandomForestRegressor(n_estimators=5, random_state=0).fit(
        X_DESC, np.arange(N, dtype=float)
    )
    return write_pickle(_jmodel(fake_jaqpotpy, est, "RDKitDescriptors", RDKIT_NAMES))


def _read(path):
    return pd.read_csv(path, keep_default_na=False, dtype=str)


# --- predictions reproduce the estimator ------------------------------------


def test_fingerprint_predictions_reproduce_estimator(tmp_path, maccs_model):
    spec = predict.load_spec(maccs_model)
    descfile, manifest = write_descriptor_file(tmp_path, spec, X_BITS)
    out, counts = _run(tmp_path, maccs_model, descfile, manifest)
    assert counts["read"] == counts["predicted"] == N
    expected = spec.estimator.predict(X_BITS.astype(spec.input_dtype))
    assert out.activity.tolist() == [str(v) for v in expected]
    np.testing.assert_allclose(
        out.activity_probability.astype(float),
        spec.estimator.predict_proba(X_BITS.astype(spec.input_dtype))[:, 1],
    )


def test_descriptor_predictions_reproduce_estimator(tmp_path, rdkit_model):
    spec = predict.load_spec(rdkit_model)
    descfile, manifest = write_descriptor_file(tmp_path, spec, X_DESC)
    out, counts = _run(tmp_path, rdkit_model, descfile, manifest)
    assert counts["predicted"] == N
    np.testing.assert_allclose(
        out.activity.astype(float), spec.estimator.predict(X_DESC)
    )


def _run(tmp_path, model, descfile, manifest):
    out_path = tmp_path / "out.csv"
    counts = predict.run(model, descfile, out_path, manifest=manifest)
    return _read(out_path), counts


def test_failed_row_is_carried_through(tmp_path, maccs_model):
    spec = predict.load_spec(maccs_model)
    descfile, manifest = write_descriptor_file(tmp_path, spec, X_BITS, failed={2, 5})
    out, counts = _run(tmp_path, maccs_model, descfile, manifest)
    assert counts["read"] == N and counts["predicted"] == N - 2
    assert out.error[2] == "no usable descriptors in the file"
    assert out.activity[2] == "" and out.activity[5] == ""
    assert out.error[0] == ""


def test_empty_descriptor_file(tmp_path, maccs_model):
    spec = predict.load_spec(maccs_model)
    descfile, manifest = write_descriptor_file(tmp_path, spec, X_BITS[:0])
    out, counts = _run(tmp_path, maccs_model, descfile, manifest)
    assert out.empty and "activity" in out.columns and counts["read"] == 0


def test_batches_give_same_output(tmp_path, maccs_model, monkeypatch):
    spec = predict.load_spec(maccs_model)
    descfile, manifest = write_descriptor_file(tmp_path, spec, X_BITS)
    whole, _ = _run(tmp_path, maccs_model, descfile, manifest)
    monkeypatch.setattr(predict, "BATCH_SIZE", 3)
    batched, _ = _run(tmp_path, maccs_model, descfile, manifest)
    # the last digit of a probability can depend on the batch size (BLAS)
    probability = "activity_probability"
    pd.testing.assert_frame_equal(
        whole.drop(columns=probability), batched.drop(columns=probability)
    )
    np.testing.assert_allclose(
        whole[probability].astype(float), batched[probability].astype(float)
    )


def test_default_manifest_is_the_sidecar(tmp_path, maccs_model):
    spec = predict.load_spec(maccs_model)
    descfile, _ = write_descriptor_file(tmp_path, spec, X_BITS, name="d")
    # d.smi -> d.manifest.json, found without passing a manifest
    counts = predict.run(maccs_model, descfile, tmp_path / "out.csv")
    assert counts["predicted"] == N


# --- SDF input --------------------------------------------------------------


def test_sdf_descriptor_input(tmp_path, maccs_model):
    spec = predict.load_spec(maccs_model)
    sdf = tmp_path / "desc.sdf"
    writer = Chem.SDWriter(str(sdf))
    for i in range(N):
        mol = Chem.MolFromSmiles("C" * (i % 3 + 1))
        mol.SetProp("_Name", f"m{i}")
        mol.SetProp("FP", "".join("1" if v else "0" for v in X_BITS[i]))
        writer.write(mol)
    writer.close()
    manifest = tmp_path / "desc.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "featurizer": {"name": "maccs", "params": {}},
                "columns": ["FP"],
                "id_column": "_Name",
                "fragment_method": "none",
                "missing_value": 0,
            }
        )
    )
    out, counts = _run(tmp_path, maccs_model, sdf, manifest)
    assert counts["predicted"] == N
    assert out.id.tolist() == [f"m{i}" for i in range(N)]
    expected = spec.estimator.predict(X_BITS.astype(spec.input_dtype))
    assert out.activity.tolist() == [str(v) for v in expected]


# --- checking ---------------------------------------------------------------


def _manifest(**overrides):
    base = dict(
        featurizer="maccs",
        featurizer_params={},
        columns=["FP"],
        id_column="id",
        smiles_column="smiles",
        fragment_method="none",
        library_versions={"rdkit": "2023.9.6"},
        missing_value=0,
    )
    base.update(overrides)
    return Manifest(**base)


def test_featurizer_name_mismatch_is_refused(maccs_model):
    spec = predict.load_spec(maccs_model)
    with pytest.raises(UnsupportedModelError, match="morgan.*needs maccs"):
        check_manifest(spec, _manifest(featurizer="morgan"))


def test_featurizer_param_mismatch_is_refused(rdkit_model):
    spec = predict.load_spec(rdkit_model)
    bad = _manifest(
        featurizer="rdkit_descriptors",
        featurizer_params={"use_fragment": False, "ipc_avg": True},
        columns=list(RDKIT_NAMES),
    )
    with pytest.raises(UnsupportedModelError, match="use_fragment"):
        check_manifest(spec, bad)


def test_missing_descriptor_columns_are_refused(rdkit_model):
    spec = predict.load_spec(rdkit_model)
    bad = _manifest(
        featurizer="rdkit_descriptors",
        featurizer_params={"use_fragment": True, "ipc_avg": True},
        columns=["MolWt"],  # far short of what the model needs
    )
    with pytest.raises(UnsupportedModelError, match="missing"):
        check_manifest(spec, bad)


def test_fragment_and_library_only_warn_without_model_expectation(maccs_model):
    spec = predict.load_spec(maccs_model)
    warnings = check_manifest(spec, _manifest(fragment_method="hac"))
    assert any("fragment method" in w and "hac" in w for w in warnings)
    assert any("rdkit version" in w for w in warnings)


def test_fragment_mismatch_is_refused_when_model_expects_one(maccs_model):
    spec = predict.load_spec(maccs_model)
    spec.expected_fragment_method = "none"
    with pytest.raises(UnsupportedModelError, match="fragment method"):
        check_manifest(spec, _manifest(fragment_method="hac"))


def test_library_mismatch_is_refused_when_model_expects_one(maccs_model):
    spec = predict.load_spec(maccs_model)
    spec.expected_library_versions = {"rdkit": "2023.9.6"}
    with pytest.raises(UnsupportedModelError, match="rdkit version"):
        check_manifest(spec, _manifest(library_versions={"rdkit": "2026.3.6"}))


# --- manifest parsing -------------------------------------------------------


def test_read_manifest_round_trip(tmp_path):
    path = tmp_path / "m.manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "featurizer": {"name": "morgan", "params": {"radius": 2}},
                "columns": ["FP"],
                "id_column": "id",
                "smiles_column": "smiles",
                "fragment_method": "hac",
                "library_versions": {"rdkit": "2026.3.6"},
                "missing_value": 0,
            }
        )
    )
    man = read_manifest(path)
    assert man.featurizer == "morgan" and man.featurizer_params == {"radius": 2}
    assert man.is_fingerprint and man.columns == ["FP"]


@pytest.mark.parametrize(
    "data, match",
    [
        ({"schema_version": 2}, "schema version"),
        ({"schema_version": 1}, "featurizer name"),
        (
            {"schema_version": 1, "featurizer": {"name": "maccs"}},
            "descriptor columns",
        ),
        (
            {"schema_version": 1, "featurizer": {"name": "maccs"}, "columns": ["FP"]},
            "id_column",
        ),
    ],
)
def test_read_manifest_rejects_bad(tmp_path, data, match):
    path = tmp_path / "m.manifest.json"
    path.write_text(json.dumps(data))
    with pytest.raises(DescriptorFileError, match=match):
        read_manifest(path)


def test_sidecar_path():
    assert sidecar_manifest_path("a/b/foo.smi").name == "foo.manifest.json"
    assert sidecar_manifest_path("foo.smi.gz").name == "foo.manifest.json"


# --- matrix building --------------------------------------------------------


def test_build_matrix_expands_fingerprint_subset(maccs_model):
    spec = predict.load_spec(maccs_model)
    # pretend the model only wants bits 0, 2, 5
    spec.feature_names = ["f0", "f2", "f5"]
    bits = "1" + "0" + "1" + "0" + "0" + "1" + "0" * 161  # 167 chars
    records = [("id0", "CCO", {"FP": bits})]
    X, ok = build_matrix(spec, _manifest(), records)
    assert ok.tolist() == [True]
    np.testing.assert_array_equal(X[0], [1.0, 1.0, 1.0])


def test_build_matrix_wrong_bit_length_fails_row(maccs_model):
    spec = predict.load_spec(maccs_model)
    records = [("id0", "CCO", {"FP": "101"})]  # too short
    _, ok = build_matrix(spec, _manifest(), records)
    assert ok.tolist() == [False]
