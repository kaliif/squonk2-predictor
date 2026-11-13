import csv
import json
import pickle
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
TEST_DATA_DIR = Path(__file__).parent / "data"


def write_descriptor_file(
    tmp_path, spec, X, name="desc", ids=None, smiles=None, failed=(), **manifest
):
    """Write a descriptor file and manifest the way a descriptor job would, from
    a feature matrix X in the model's feature order. For a fingerprint model X
    is the full bit matrix, written as one bit-string column; for a descriptor
    model each feature is its own column. Rows in `failed` are written empty.

    Extra keyword args override manifest fields. Returns (descfile, manifest).
    """
    feat = spec.featurizer
    n = len(X)
    ids = ids if ids is not None else [f"id{i}" for i in range(n)]
    smiles = smiles if smiles is not None else ["C"] * n
    cols = ["FP"] if feat.binary else list(spec.feature_names)

    descfile = tmp_path / f"{name}.smi"
    with open(descfile, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["id", "smiles", *cols])
        for i in range(n):
            if i in failed:
                writer.writerow([ids[i], smiles[i], *[""] * len(cols)])
            elif feat.binary:
                writer.writerow(
                    [ids[i], smiles[i], "".join("1" if v else "0" for v in X[i])]
                )
            else:
                writer.writerow([ids[i], smiles[i], *[repr(float(v)) for v in X[i]]])

    data = {
        "schema_version": 1,
        "featurizer": {"name": feat.name, "params": feat.params()},
        "columns": cols,
        "id_column": "id",
        "smiles_column": "smiles",
        "fragment_method": "none",
        "library_versions": {"rdkit": "2023.9.6"},
        "missing_value": 0,
    }
    data.update(manifest)
    manpath = tmp_path / f"{name}.manifest.json"
    manpath.write_text(json.dumps(data))
    return descfile, manpath


def _fake_class(module, name):
    cls = type(name, (), {"__module__": module})
    cls.__qualname__ = name
    return cls


@pytest.fixture
def fake_jaqpotpy(monkeypatch):
    """Temporarily importable stand-ins for the jaqpotpy classes, so tests can
    write pickles that look like jaqpotpy 1.0.x model files.

    Returns a namespace of classes. The modules are removed again at the end
    of the test, which also proves the loader never needs jaqpotpy.
    """
    classes = {}
    for module, names in {
        "jaqpotpy.models.base_classes": ["MolecularModel"],
        "jaqpotpy.doa.doa": ["Leverage", "MeanVar"],
        "jaqpotpy.descriptors.molecular.topological_fingerprints": [
            "TopologicalFingerprint"
        ],
        "jaqpotpy.descriptors.molecular.maccs_keys_fingerprints": [
            "MACCSKeysFingerprint"
        ],
        "jaqpotpy.descriptors.molecular.mordred": ["MordredDescriptors"],
        "jaqpotpy.descriptors.molecular.rdkit": ["RDKitDescriptors"],
        "jaqpotpy.descriptors.molecular.smiles_to_image": ["SmilesToImage"],
    }.items():
        parts = module.split(".")
        for i in range(1, len(parts) + 1):
            name = ".".join(parts[:i])
            if name not in sys.modules or not getattr(
                sys.modules[name], "_fake", False
            ):
                mod = types.ModuleType(name)
                mod._fake = True
                monkeypatch.setitem(sys.modules, name, mod)
        for cls_name in names:
            cls = _fake_class(module, cls_name)
            setattr(sys.modules[module], cls_name, cls)
            classes[cls_name] = cls
    return types.SimpleNamespace(**classes)


def make_obj(cls, **attrs):
    obj = cls.__new__(cls)
    obj.__dict__.update(attrs)
    return obj


@pytest.fixture
def write_pickle(tmp_path):
    counter = iter(range(10**6))

    def _write(obj, protocol=4):
        path = tmp_path / f"model_{next(counter)}.jmodel"
        path.write_bytes(pickle.dumps(obj, protocol=protocol))
        return path

    return _write
