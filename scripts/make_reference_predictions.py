"""Write reference predictions for a jaqpotpy 1.0.x model with jaqpotpy.

The tests compare the job's predictions with these (see tests/reference.py).
They must come from the software that created the model, not from this job's
reimplementation, or they only show that the job agrees with itself. This script therefore does NOT run in the job environment. It needs an
environment that reproduces the one the legacy models were trained in:

    python3.11 -m venv venvjq
    venvjq/bin/pip install numpy==1.26.4 scikit-learn==1.3.2 pandas==2.0.3 \\
        rdkit==2023.9.6 mordred==1.2.0 pydantic requests pyjwt tqdm
    venvjq/bin/pip install --no-deps jaqpotpy==1.0.104

numpy >= 1.24 matters: with it the original mordred fails on some
descriptors and jaqpotpy fills them with 0, which is what the Mordred models
were trained on. jaqpotpy imports torch and other deep learning packages at
import time; they aren't needed for scikit-learn models, so empty stand-ins
are used instead of installing them.

The model files are unpickled with plain pickle, so only run this on trusted
models. Molecules are taken from a CSV with the SMILES in a 'Drug'
column (the TDC ADMET test sets were used for the legacy models) and spread
evenly over the range of predictions, so both classes and the extremes are
covered.

    venvjq/bin/python -I scripts/make_reference_predictions.py \\
        --model models/AMES.jmodel --molecules admet_group/ames/test.csv \\
        --outfile models/AMES.reference.csv
"""

import argparse
import importlib.abc
import importlib.machinery
import sys
import types

N_MOLECULES = 50
FAKE_PACKAGES = (
    "torch",
    "torch_geometric",
    "torch_scatter",
    "torch_sparse",
    "dgl",
    "pymatgen",
    "skl2onnx",
    "onnxruntime",
    "onnx",
    "matplotlib",
    "deepchem",
    "kennard_stone",
)


class _Anything(type):
    """Class that any attribute access, call or subscript also yields, enough
    to get through jaqpotpy's module level uses of torch."""

    def __getattr__(cls, name):
        return _Anything(name, (), {})

    def __call__(cls, *args, **kwargs):
        return _Anything("obj", (), {})

    def __getitem__(cls, item):
        return cls

    def __or__(cls, other):
        return cls

    __ror__ = __or__


class _FakeModule(types.ModuleType):
    __path__: list = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Anything(name, (), {})


class _FakeFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    # pylint: disable-next=unused-argument
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in FAKE_PACKAGES:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return _FakeModule(spec.name)

    def exec_module(self, module):
        pass


def jaqpotpy_features(model, smiles):
    """Feature matrix exactly as jaqpotpy's MolecularModel.infer() computes
    it, which stores some featurizers by name and creates them on use."""
    # pylint: disable=import-outside-toplevel,protected-access
    from jaqpotpy.descriptors.molecular import MACCSKeysFingerprint, RDKitDescriptors

    desc = model._descriptors
    if desc == "RDKitDescriptors":
        desc = RDKitDescriptors()
    elif desc == "MACCSKeysFingerprint":
        desc = MACCSKeysFingerprint()
    return desc.featurize_dataframe(smiles)[model._X].to_numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--molecules", required=True, help="CSV with a Drug column")
    parser.add_argument("--outfile", required=True)
    parser.add_argument("-n", type=int, default=N_MOLECULES)
    args = parser.parse_args()

    sys.meta_path.insert(0, _FakeFinder())
    # pylint: disable=import-outside-toplevel
    import numpy as np
    import pandas as pd
    from jaqpotpy.models import MolecularModel
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    model = MolecularModel.load(args.model)
    smiles = [
        s
        for s in pd.read_csv(args.molecules).Drug.drop_duplicates()
        if Chem.MolFromSmiles(s) is not None
    ]
    model(smiles)
    prediction = np.array(model.prediction)
    estimator = model.model

    if model.probability:
        score = np.array(model.probability)[:, 1]
    elif hasattr(estimator, "decision_function"):
        # jaqpotpy doesn't give decision scores; compute them on the features
        # exactly as jaqpotpy's infer() does
        score = estimator.decision_function(jaqpotpy_features(model, smiles))
    else:
        score = None

    if score is not None:
        order = np.argsort(score, kind="stable")
    else:
        # labels only, or a regressor: spread over the predicted values
        order = np.argsort(prediction, kind="stable")
    picks = np.unique(np.linspace(0, len(order) - 1, args.n).round().astype(int))
    rows = order[picks]

    reference = pd.DataFrame(
        {"smiles": [smiles[i] for i in rows], "prediction": prediction[rows]}
    )
    if score is not None:
        reference["score"] = score[rows]
    reference.to_csv(args.outfile, index=False)
    print(
        f"{args.outfile}: {len(reference)} of {len(smiles)} molecules, "
        f"predictions {sorted(set(reference.prediction))[:3]}...",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
