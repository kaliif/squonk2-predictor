"""Format-independent description of a loaded model.

Adapters turn a model file of a particular format into a ModelSpec. Everything
after that (checks, prediction, reporting) works on the spec only.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, is_classifier

from featurizers import Featurizer


class UnsupportedModelError(Exception):
    """The model can't be used; the message says why."""


@dataclass
class LeverageDomain:
    """Applicability domain from the leverage h = x (X'X)^-1 x' of a molecule
    relative to the training set. A molecule is in domain if h < threshold."""

    matrix: np.ndarray
    threshold: float

    def leverage(self, X: np.ndarray) -> np.ndarray:
        return np.einsum("ij,jk,ik->i", X, self.matrix, X)

    def in_domain(self, X: np.ndarray) -> np.ndarray:
        return self.leverage(X) < self.threshold


@dataclass
class ModelSpec:
    name: str
    source_format: str
    estimator: Any
    featurizer: Featurizer
    # features the estimator expects, in order, as named by the featurizer
    feature_names: list[str]
    target_name: str
    domain: LeverageDomain | None = None
    # feature matrix of the training set, when the format stores it
    training_features: np.ndarray | None = None
    # dtype of the matrix the estimator is given; it can matter even when the
    # values are the same, e.g. nearest neighbour searches break distance
    # ties differently for integer and float input
    input_dtype: type = np.float64
    warnings: list[str] = field(default_factory=list)

    @property
    def is_classifier(self) -> bool:
        return is_classifier(self.estimator) or hasattr(self.estimator, "classes_")

    @property
    def output_kind(self) -> str:
        """'value' for regression; for classifiers 'probability',
        'decision_function' or 'label' depending on what the model can give."""
        if not self.is_classifier:
            return "value"
        # hasattr respects sklearn's available_if, e.g. SVC(probability=False)
        # or VotingClassifier(voting='hard') have no predict_proba
        if hasattr(self.estimator, "predict_proba"):
            return "probability"
        if hasattr(self.estimator, "decision_function"):
            return "decision_function"
        return "label"

    @property
    def classes(self) -> list | None:
        classes = getattr(self.estimator, "classes_", None)
        return None if classes is None else list(classes)

    def features(self, mols) -> tuple[np.ndarray, np.ndarray]:
        """Model input matrix for the molecules, plus a mask of molecules that
        could be featurised."""
        X, ok = self.featurizer.compute(mols)
        index = {name: i for i, name in enumerate(self.featurizer.feature_names)}
        return X[:, [index[n] for n in self.feature_names]], ok

    def predict(self, mols) -> tuple[pd.DataFrame, np.ndarray]:
        """Predictions for a list of molecules: a table with a row per
        molecule, empty for molecules that couldn't be featurised, and the
        mask of molecules that were predicted."""
        X, ok = self.features(mols)
        table = pd.DataFrame(index=range(len(mols)), columns=self.output_columns)
        if ok.any():
            rows = np.flatnonzero(ok)
            outputs = self._output_columns(self.predict_features(X[ok]))
            for col, values in outputs.items():
                # object dtype keeps e.g. integer labels from turning into
                # floats when other rows are empty
                table[col] = pd.Series(values, index=rows, dtype=object)
        return table, ok

    @property
    def output_columns(self) -> list[str]:
        """Names of the columns of the predict() table."""
        zero = np.zeros((1, len(self.feature_names)))
        return list(self._output_columns(self.predict_features(zero)))

    def _output_columns(self, out: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Flatten predict_features output into named columns: 'prediction';
        'probability' or 'score' for binary classifiers (of the second class),
        or 'probability_<class>' / 'score_<class>' for multiclass ones;
        'leverage' and 'in_domain'."""
        cols = {"prediction": out["prediction"]}
        for key, name in (
            ("probability", "probability"),
            ("decision_function", "score"),
        ):
            if key not in out:
                continue
            values = out[key]
            if values.ndim == 1:
                cols[name] = values
            elif values.shape[1] == 2:
                cols[name] = values[:, 1]
            else:
                for cls, column in zip(self.classes, values.T):
                    cols[f"{name}_{cls}"] = column
        for key in ("leverage", "in_domain"):
            if key in out:
                cols[key] = out[key]
        return cols

    def predict_features(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Raw model outputs for a feature matrix with no failed rows."""
        X = np.asarray(X, dtype=self.input_dtype)
        out = {"prediction": self.estimator.predict(X)}
        kind = self.output_kind
        if kind == "probability":
            out["probability"] = self.estimator.predict_proba(X)
        elif kind == "decision_function":
            out["decision_function"] = self.estimator.decision_function(X)
        if self.domain is not None:
            out["leverage"] = self.domain.leverage(X)
            out["in_domain"] = out["leverage"] < self.domain.threshold
        return out


def iter_estimators(estimator):
    """The estimator and every estimator nested inside it (pipeline steps,
    ensemble members, ...)."""
    seen = set()
    stack = [estimator]
    while stack:
        obj = stack.pop()
        if isinstance(obj, BaseEstimator):
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            yield obj
            stack.extend(obj.__dict__.values())
        elif isinstance(obj, (list, tuple)):
            stack.extend(obj)
        elif isinstance(obj, dict):
            stack.extend(obj.values())


def check_spec(spec: ModelSpec) -> None:
    """Checks that apply to any model, whatever its file format.

    Raises UnsupportedModelError for problems that make predictions
    unreliable, and appends warnings to spec.warnings for limitations the user
    should know about.
    """
    available = set(spec.featurizer.feature_names)
    missing = [n for n in spec.feature_names if n not in available]
    if missing:
        raise UnsupportedModelError(
            f"model needs {len(missing)} features that {spec.featurizer.describe()} "
            f"does not produce here, e.g. {', '.join(missing[:5])}"
        )

    n_in = getattr(spec.estimator, "n_features_in_", None)
    if n_in is not None and n_in != len(spec.feature_names):
        raise UnsupportedModelError(
            f"model lists {len(spec.feature_names)} features but the estimator "
            f"was fitted on {n_in}"
        )

    versions = sorted(
        {
            v
            for e in iter_estimators(spec.estimator)
            if (v := getattr(e, "_sklearn_version", None))
        }
    )
    # an estimator unpickled from another sklearn version may fail or silently
    # behave differently; failing is caught by a test prediction
    try:
        spec.predict_features(np.zeros((1, len(spec.feature_names))))
    except Exception as exc:
        trained = ", ".join(versions) or "an unknown version"
        raise UnsupportedModelError(
            f"estimator can't run here (trained with scikit-learn {trained}, "
            f"this job has {sklearn.__version__}): {exc}"
        ) from exc
    other = [v for v in versions if v != sklearn.__version__]
    if other:
        spec.warnings.append(
            f"trained with scikit-learn {', '.join(other)}, running {sklearn.__version__}; "
            "predictions are not verified to be identical"
        )

    kind = spec.output_kind
    if kind == "label":
        spec.warnings.append(
            "classifier gives class labels only, no probability or score"
        )
    elif kind == "decision_function":
        spec.warnings.append("classifier gives a decision score, not a probability")

    if spec.featurizer.binary and any(
        type(e).__name__.startswith(("KNeighbors", "RadiusNeighbors"))
        for e in iter_estimators(spec.estimator)
    ):
        spec.warnings.append(
            "nearest-neighbour model on binary features: equally distant neighbours "
            "are common, so some predictions depend on arbitrary tie-breaking"
        )

    if spec.domain is None:
        spec.warnings.append("no applicability domain information")
    elif spec.domain.threshold >= 1:
        spec.warnings.append(
            f"leverage threshold is {spec.domain.threshold:.2f} (>= 1): the training "
            "set is small for the number of features, so the domain check is weak"
        )
