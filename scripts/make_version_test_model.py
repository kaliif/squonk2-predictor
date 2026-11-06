"""Write a small jaqpotpy-format model with the scikit-learn version running
this script, for testing which job versions can use it.

The model is a random forest classifier on MACCS keys, trained on the
molecules and labels of data/models/CYP2D6_Veith.reference.csv. It is pickled
with stand-ins for the jaqpotpy classes, so jaqpotpy isn't needed. Run it with
the Python of the environment whose scikit-learn version the model should
have, from the repository root:

    python scripts/make_version_test_model.py \\
        --outfile data/models/rf-maccs-sklearn-1.9.1.jmodel
"""

import argparse
import pickle
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# pylint: disable=wrong-import-position
import pandas as pd
import sklearn
from rdkit import Chem, RDLogger
from sklearn.ensemble import RandomForestClassifier

from featurizers import MACCSKeys

TRAINING = ROOT / "data" / "models" / "CYP2D6_Veith.reference.csv"


def _fake_class(module, name):
    """A class that pickles as module.name, standing in for jaqpotpy's."""
    parts = module.split(".")
    for i in range(1, len(parts) + 1):
        sys.modules.setdefault(".".join(parts[:i]), types.ModuleType(parts[i - 1]))
    cls = type(name, (), {"__module__": module})
    cls.__qualname__ = name
    setattr(sys.modules[module], name, cls)
    return cls


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--outfile", required=True)
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")
    training = pd.read_csv(TRAINING)
    mols = [Chem.MolFromSmiles(s) for s in training.smiles]
    X, _ = MACCSKeys().compute(mols)
    estimator = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0)
    estimator.fit(X, training.prediction.astype(int))

    model_class = _fake_class("jaqpotpy.models.base_classes", "MolecularModel")
    model = model_class.__new__(model_class)
    model.__dict__.update(
        _model=estimator,
        _descriptors="MACCSKeysFingerprint",
        _doa=None,
        _X=[f"f{i}" for i in range(X.shape[1])],
        _Y=["CYP2D6_test"],
        _library=["sklearn"],
        _version=[sklearn.__version__],
        _jaqpotpy_version="1.0.104",
        _external_feats=None,
        model_name="CYP2D6_test",
    )
    Path(args.outfile).write_bytes(pickle.dumps(model, protocol=4))
    print(f"{args.outfile}: scikit-learn {sklearn.__version__}", file=sys.stderr)


if __name__ == "__main__":
    main()
