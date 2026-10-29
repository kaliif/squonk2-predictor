"""Molecular featurizers that reproduce how supported model formats computed
their training features.

Every featurizer exposes the names of the features it computes, so a model's
expected feature list can be checked against it before any molecule is
processed, and features are always selected by name, never by position.
"""

from abc import ABC, abstractmethod

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


def prepare_molecule(mol):
    """Normalise a molecule to what parsing its SMILES would give.

    The supported model formats were trained from SMILES, so molecules read
    from SDF (often with explicit hydrogens and 3D coordinates) are brought to
    the same representation. Returns None if the molecule can't be processed.
    """
    if mol is None:
        return None
    try:
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception:
        return None
    return Chem.MolFromSmiles(smiles)


class Featurizer(ABC):
    """Computes a fixed, named set of features per molecule."""

    name: str
    # features are 0/1 bits, which makes exact distance ties common
    binary: bool = False

    def __init__(self, missing_value=None):
        # value used for descriptors that can't be calculated; None keeps NaN
        self.missing_value = missing_value

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Names of all features this featurizer produces, in order."""

    @abstractmethod
    def _compute(self, mol) -> np.ndarray:
        """Feature values for one prepared molecule, in feature_names order."""

    def params(self) -> dict:
        return {}

    def describe(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.params().items())
        return f"{self.name}({params})"

    def compute(self, mols) -> tuple[np.ndarray, np.ndarray]:
        """Features for each molecule, as (matrix, ok) where ok marks the rows
        that could be calculated. Failed rows are NaN."""
        n = len(self.feature_names)
        X = np.full((len(mols), n), np.nan)
        ok = np.zeros(len(mols), dtype=bool)
        for i, mol in enumerate(mols):
            mol = prepare_molecule(mol)
            if mol is None:
                continue
            try:
                values = np.asarray(self._compute(mol), dtype=float)
            except Exception:
                continue
            if self.missing_value is not None:
                values = np.where(np.isnan(values), self.missing_value, values)
            X[i] = values
            ok[i] = True
        return X, ok


class MorganFingerprint(Featurizer):
    name = "morgan"
    binary = True

    def __init__(
        self,
        radius=2,
        n_bits=2048,
        chiral=False,
        bond_types=True,
        feature_invariants=False,
        missing_value=None,
    ):
        super().__init__(missing_value)
        self.radius = radius
        self.n_bits = n_bits
        self.chiral = chiral
        self.bond_types = bond_types
        self.feature_invariants = feature_invariants

    def params(self):
        return {
            "radius": self.radius,
            "n_bits": self.n_bits,
            "chiral": self.chiral,
            "bond_types": self.bond_types,
            "feature_invariants": self.feature_invariants,
        }

    @property
    def feature_names(self):
        return [f"f{i}" for i in range(self.n_bits)]

    def _compute(self, mol):
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol,
            self.radius,
            nBits=self.n_bits,
            useChirality=self.chiral,
            useBondTypes=self.bond_types,
            useFeatures=self.feature_invariants,
        )
        return np.asarray(fp, dtype=float)


class MACCSKeys(Featurizer):
    name = "maccs"
    binary = True

    @property
    def feature_names(self):
        return [f"f{i}" for i in range(167)]

    def _compute(self, mol):
        return np.asarray(rdMolDescriptors.GetMACCSKeysFingerprint(mol), dtype=float)


class RDKitDescriptors(Featurizer):
    name = "rdkit_descriptors"

    def __init__(self, use_fragment=True, ipc_avg=True, missing_value=None):
        super().__init__(missing_value)
        self.use_fragment = use_fragment
        # Ipc can be huge; jaqpotpy uses the averaged form
        self.ipc_avg = ipc_avg
        self._descriptors = [
            (name, fn)
            for name, fn in Descriptors.descList
            if use_fragment or not name.startswith("fr_")
        ]

    def params(self):
        return {"use_fragment": self.use_fragment, "ipc_avg": self.ipc_avg}

    @property
    def feature_names(self):
        return [name for name, _ in self._descriptors]

    def _compute(self, mol):
        values = []
        for name, fn in self._descriptors:
            if name == "Ipc" and self.ipc_avg:
                values.append(fn(mol, avg=True))
            else:
                values.append(fn(mol))
        return np.asarray(values, dtype=float)


class MordredDescriptors(Featurizer):
    name = "mordred"

    def __init__(self, missing_value=None):
        super().__init__(missing_value)
        # imported here, mordred is slow to import and only some models need it
        from mordred import Calculator, descriptors

        self._calc = Calculator(descriptors, ignore_3D=True)
        self._names = [str(d) for d in self._calc.descriptors]

    def params(self):
        return {"ignore_3D": True}

    @property
    def feature_names(self):
        return self._names

    def _compute(self, mol):
        values = self._calc(mol).values()
        # failed descriptors come back as error objects rather than numbers;
        # boolean descriptors (e.g. Lipinski) count as 0/1
        return np.array(
            [
                (
                    float(v)
                    if isinstance(v, (int, float, np.integer, np.floating, np.bool_))
                    else np.nan
                )
                for v in values
            ],
            dtype=float,
        )
