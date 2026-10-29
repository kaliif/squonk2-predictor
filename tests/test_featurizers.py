import numpy as np
import pytest
from conftest import DATA_DIR
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from featurizers import (
    MACCSKeys,
    MordredDescriptors,
    MorganFingerprint,
    RDKitDescriptors,
    prepare_molecule,
)

SMILES = ["CCO", "c1ccccc1C(=O)O", "CC(C)NCC(O)COc1ccc(NC=O)cc1", "[Na+].[Cl-]"]


@pytest.fixture(scope="module")
def mordred():
    return MordredDescriptors()


@pytest.fixture(scope="module")
def sdf_molecules():
    return list(Chem.SDMolSupplier(str(DATA_DIR / "candidates-10.sdf"), removeHs=False))


def test_morgan_matches_rdkit():
    fz = MorganFingerprint(radius=2, n_bits=1024)
    X, ok = fz.compute([Chem.MolFromSmiles(s) for s in SMILES])
    assert ok.all()
    for row, smi in zip(X, SMILES):
        expected = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(smi), 2, nBits=1024
        )
        np.testing.assert_array_equal(row, np.asarray(expected, dtype=float))
    assert fz.feature_names == [f"f{i}" for i in range(1024)]


def test_maccs_has_167_bits():
    X, ok = MACCSKeys().compute([Chem.MolFromSmiles(s) for s in SMILES])
    assert ok.all() and X.shape == (len(SMILES), 167)


def test_rdkit_descriptors_names_and_ipc():
    fz = RDKitDescriptors()
    assert fz.feature_names == [name for name, _ in Descriptors.descList]
    mol = Chem.MolFromSmiles(SMILES[2])
    X, _ = fz.compute([mol])
    ipc = X[0, fz.feature_names.index("Ipc")]
    assert ipc == pytest.approx(Descriptors.Ipc(mol, avg=True))
    no_fragments = RDKitDescriptors(use_fragment=False)
    assert not any(n.startswith("fr_") for n in no_fragments.feature_names)


def test_mordred_missing_values(mordred):
    # many descriptors are undefined for a single atom
    X, ok = mordred.compute([Chem.MolFromSmiles("C")])
    assert ok.all() and np.isnan(X).any()
    filled = MordredDescriptors(missing_value=0.0)
    X0, _ = filled.compute([Chem.MolFromSmiles("C")])
    assert not np.isnan(X0).any()
    np.testing.assert_array_equal(X0[~np.isnan(X)], X[~np.isnan(X)])


def test_failed_molecules_are_marked():
    X, ok = MACCSKeys().compute([None, Chem.MolFromSmiles("CCO")])
    assert ok.tolist() == [False, True]
    assert np.isnan(X[0]).all()


def test_prepare_removes_explicit_hydrogens(sdf_molecules):
    mol = sdf_molecules[0]
    assert any(a.GetAtomicNum() == 1 for a in mol.GetAtoms())
    prepared = prepare_molecule(mol)
    assert not any(a.GetAtomicNum() == 1 for a in prepared.GetAtoms())
    # without preparation explicit hydrogens change the fingerprint
    raw = np.asarray(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    clean = np.asarray(
        rdMolDescriptors.GetMorganFingerprintAsBitVect(prepared, 2, nBits=2048)
    )
    assert (raw != clean).any()


@pytest.mark.parametrize(
    "make", [MorganFingerprint, MACCSKeys, RDKitDescriptors, "mordred"]
)
def test_sdf_input_matches_smiles_input(sdf_molecules, make, mordred):
    """Features of 3D SDF molecules with explicit hydrogens equal those of the
    same molecules parsed from SMILES, which is how the models were trained."""
    fz = mordred if make == "mordred" else make()
    from_sdf, ok_sdf = fz.compute(sdf_molecules)
    from_smiles, ok_smi = fz.compute(
        [Chem.MolFromSmiles(m.GetProp("enum_smi")) for m in sdf_molecules]
    )
    assert ok_sdf.all() and ok_smi.all()
    np.testing.assert_allclose(from_sdf, from_smiles, rtol=1e-9, equal_nan=True)
