"""Verification of a model against probe molecules with known predictions.

A probe file is a CSV with a 'smiles' column, a 'prediction' column with the
value (or class label) the model gave when it was created, and optionally a
'score' column with the probability of the second class, or the decision
score, of a binary classifier.

Recomputing the probes here tests the whole chain at once: that the
descriptors are still calculated as they were at training time and that the
estimator behaves the same in this scikit-learn version.
"""

import numpy as np
import pandas as pd
from rdkit import Chem

from model_spec import ModelSpec

RTOL = 1e-5
ATOL = 1e-6


class ProbeError(Exception):
    """The probes are unusable or the model doesn't reproduce them."""


def read_probes(path) -> pd.DataFrame:
    try:
        probes = pd.read_csv(path)
    except Exception as exc:
        raise ProbeError(f"can't read probe file: {exc}") from exc
    missing = {"smiles", "prediction"} - set(probes.columns)
    if missing:
        raise ProbeError(f"probe file has no {' or '.join(sorted(missing))} column")
    if probes.empty:
        raise ProbeError("probe file has no molecules")
    if probes.prediction.isna().any():
        raise ProbeError("probe file has molecules without a prediction")
    return probes


def check_probes(spec: ModelSpec, probes: pd.DataFrame) -> None:
    """Raise ProbeError unless the model reproduces every probe."""
    mols = [Chem.MolFromSmiles(str(s)) for s in probes.smiles]
    bad = [s for s, m in zip(probes.smiles, mols) if m is None]
    if bad:
        raise ProbeError(f"invalid probe SMILES: {', '.join(map(str, bad[:5]))}")

    table, ok = spec.predict(mols)
    if not ok.all():
        failed = probes.smiles[~ok]
        raise ProbeError(
            f"descriptors can't be calculated for {len(failed)} probe molecules, "
            f"e.g. {failed.iloc[0]}"
        )

    if spec.is_classifier:
        same = np.array(
            [_same_label(a, b) for a, b in zip(table.prediction, probes.prediction)]
        )
    else:
        same = _close(table.prediction, probes.prediction)
    mismatches = [
        f"{smi}: prediction {got}, expected {exp}"
        for smi, got, exp, s in zip(
            probes.smiles, table.prediction, probes.prediction, same
        )
        if not s
    ]

    if "score" in probes and probes.score.notna().any():
        column = next((c for c in ("probability", "score") if c in table), None)
        if column is None:
            raise ProbeError(
                "probes have scores but the model gives no single score per "
                "molecule (it is a regressor, a multiclass or a labels-only "
                "classifier)"
            )
        given = probes.score.notna()
        same = _close(table[column][given], probes.score[given])
        mismatches += [
            f"{smi}: {column} {got:.6g}, expected {exp:.6g}"
            for smi, got, exp, s in zip(
                probes.smiles[given], table[column][given], probes.score[given], same
            )
            if not s
        ]

    if mismatches:
        raise ProbeError(
            f"model doesn't reproduce {len(mismatches)} probe values: "
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
