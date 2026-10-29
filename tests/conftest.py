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
