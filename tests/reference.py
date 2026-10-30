"""Comparison of a model's predictions with reference predictions.

A reference file is a CSV with a 'smiles' column, a 'prediction' column with
the value (or class label) the software that created the model predicted, and
optionally a 'score' column with the probability of the second class, or the
decision score, of a binary classifier. scripts/make_reference_predictions.py
writes them for jaqpotpy models.

Recomputing the predictions here tests the whole chain at once: that the
descriptors are still calculated as they were at training time and that the
estimator behaves the same in this scikit-learn version.
"""

import numpy as np
import pandas as pd
from rdkit import Chem

from model_spec import ModelSpec

RTOL = 1e-5
ATOL = 1e-6


class ReferenceMismatch(Exception):
    """The model doesn't reproduce the reference predictions."""


def read_reference(path) -> pd.DataFrame:
    reference = pd.read_csv(path)
    assert {"smiles", "prediction"} <= set(reference.columns), path
    assert not reference.empty and reference.prediction.notna().all(), path
    return reference


def check_reference(spec: ModelSpec, reference: pd.DataFrame) -> None:
    """Raise ReferenceMismatch unless the model reproduces every prediction."""
    mols = [Chem.MolFromSmiles(str(s)) for s in reference.smiles]
    table, ok = spec.predict(mols)
    if not ok.all():
        failed = reference.smiles[~ok]
        raise ReferenceMismatch(
            f"no prediction for {len(failed)} molecules, e.g. {failed.iloc[0]}"
        )

    if spec.is_classifier:
        same = np.array(
            [_same_label(a, b) for a, b in zip(table.prediction, reference.prediction)]
        )
    else:
        same = _close(table.prediction, reference.prediction)
    mismatches = [
        f"{smi}: prediction {got}, expected {exp}"
        for smi, got, exp, s in zip(
            reference.smiles, table.prediction, reference.prediction, same
        )
        if not s
    ]

    if "score" in reference and reference.score.notna().any():
        column = next(c for c in ("probability", "score") if c in table)
        given = reference.score.notna()
        same = _close(table[column][given], reference.score[given])
        mismatches += [
            f"{smi}: {column} {got:.6g}, expected {exp:.6g}"
            for smi, got, exp, s in zip(
                reference.smiles[given],
                table[column][given],
                reference.score[given],
                same,
            )
            if not s
        ]

    if mismatches:
        raise ReferenceMismatch(
            f"model doesn't reproduce {len(mismatches)} reference values: "
            + "; ".join(mismatches[:5])
        )


def _close(got, expected) -> np.ndarray:
    return np.isclose(
        np.asarray(got, dtype=float),
        np.asarray(expected, dtype=float),
        rtol=RTOL,
        atol=ATOL,
    )


def _same_label(got, expected) -> bool:
    # labels read from CSV may be numbers or strings, e.g. 1 vs '1' vs 1.0
    try:
        return float(got) == float(expected)
    except (TypeError, ValueError):
        return str(got) == str(expected)
