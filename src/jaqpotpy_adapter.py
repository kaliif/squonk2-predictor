"""Adapter for jaqpotpy 1.0.x MolecularModel files (.jmodel).

These pickles hold a scikit-learn estimator together with a description of
the featurizer it was trained with, the ordered feature names and, optionally,
a leverage applicability domain that includes the training feature matrix.

The safe loader replaces all jaqpotpy objects with Stubs, so the attributes
are read from the pickled state and jaqpotpy is never imported. Featurisation
is reimplemented in featurizers.py to match jaqpotpy 1.0.104, which replaces
descriptors that can't be calculated with 0.
"""

import numpy as np
from sklearn.base import BaseEstimator

from featurizers import (
    MACCSKeys,
    MordredDescriptors,
    MorganFingerprint,
    RDKitDescriptors,
)
from model_spec import LeverageDomain, ModelSpec, UnsupportedModelError
from safe_loader import Stub, stub_name

MOLECULAR_MODEL = "jaqpotpy.models.base_classes.MolecularModel"
LEVERAGE = "jaqpotpy.doa.doa.Leverage"
DESCRIPTORS_PREFIX = "jaqpotpy.descriptors.molecular."

# jaqpotpy fills descriptors it can't calculate with 0
MISSING_VALUE = 0.0


def is_jaqpotpy_model(obj) -> bool:
    return stub_name(obj) == MOLECULAR_MODEL


def adapt(obj, name=None) -> ModelSpec:
    """Build a ModelSpec from a loaded MolecularModel stub, or raise
    UnsupportedModelError saying which part isn't supported."""
    attrs = obj.__dict__

    version = attrs.get("_jaqpotpy_version")
    if not (isinstance(version, str) and version.startswith("1.0.")):
        raise UnsupportedModelError(
            f"jaqpotpy version {version!r} is not supported, only 1.0.x"
        )

    library = attrs.get("_library")
    if library != ["sklearn"]:
        raise UnsupportedModelError(
            f"only scikit-learn models are supported, this one uses {library}"
        )

    estimator = attrs.get("_model")
    if not isinstance(estimator, BaseEstimator):
        raise UnsupportedModelError(
            "model file does not contain a scikit-learn estimator"
        )

    for attr, what in (
        ("_external_feats", "external features"),
        ("_preprocessors", "feature preprocessing steps"),
        ("_preprocessors_y", "target preprocessing steps"),
    ):
        if attrs.get(attr):
            raise UnsupportedModelError(f"models with {what} are not supported")

    feature_names = attrs.get("_X")
    if not (
        isinstance(feature_names, list)
        and all(isinstance(n, str) for n in feature_names)
    ):
        raise UnsupportedModelError("model has no list of feature names")

    target = attrs.get("_Y")
    target_name = target[0] if isinstance(target, list) and len(target) == 1 else None
    model_name = (
        attrs.get("model_name") or attrs.get("_model_title") or name or "prediction"
    )
    # jaqpotpy models generally call their target just 'Y'
    if target_name in (None, "Y"):
        target_name = model_name

    domain, training_features = _leverage_domain(attrs.get("_doa"), len(feature_names))
    featurizer = _featurizer(attrs.get("_descriptors"))

    return ModelSpec(
        name=model_name,
        source_format=f"jaqpotpy {version}",
        estimator=estimator,
        featurizer=featurizer,
        feature_names=feature_names,
        target_name=target_name,
        domain=domain,
        training_features=training_features,
        # jaqpotpy gives MACCS keys to the estimator as int8, everything else
        # as float64
        input_dtype=np.int8 if isinstance(featurizer, MACCSKeys) else np.float64,
    )


def _featurizer(desc):
    # some featurizers are stored by name and created with defaults at predict time
    if desc == "MACCSKeysFingerprint":
        return MACCSKeys(missing_value=MISSING_VALUE)
    if desc == "RDKitDescriptors":
        return RDKitDescriptors(missing_value=MISSING_VALUE)

    qualname = stub_name(desc)
    if not (qualname and qualname.startswith(DESCRIPTORS_PREFIX)):
        raise UnsupportedModelError(f"unsupported featurizer {qualname or desc!r}")
    kind = qualname.rsplit(".", 1)[1]
    params = desc.__dict__

    if kind == "TopologicalFingerprint":
        if params.get("sparse"):
            raise UnsupportedModelError(
                "sparse (count) Morgan fingerprints are not supported"
            )
        return MorganFingerprint(
            radius=_param(params, "radius", int),
            n_bits=_param(params, "size", int),
            chiral=_param(params, "chiral", bool),
            bond_types=_param(params, "bonds", bool),
            feature_invariants=_param(params, "features", bool),
            missing_value=MISSING_VALUE,
        )
    if kind == "MACCSKeysFingerprint":
        return MACCSKeys(missing_value=MISSING_VALUE)
    if kind == "RDKitDescriptors":
        return RDKitDescriptors(
            use_fragment=params.get("use_fragment", True),
            ipc_avg=params.get("ipc_avg", True),
            missing_value=MISSING_VALUE,
        )
    if kind == "MordredDescriptors":
        if params.get("ignore_3D", True) is not True:
            raise UnsupportedModelError(
                "Mordred models with 3D descriptors are not supported"
            )
        return MordredDescriptors(missing_value=MISSING_VALUE)
    raise UnsupportedModelError(f"unsupported featurizer {kind}")


def _param(params, key, typ):
    value = params.get(key)
    if not isinstance(value, typ):
        raise UnsupportedModelError(
            f"featurizer parameter {key}={value!r} is not a valid {typ.__name__}"
        )
    return value


def _leverage_domain(doa, n_features):
    if doa is None:
        return None, None
    if stub_name(doa) != LEVERAGE:
        raise UnsupportedModelError(
            f"unsupported applicability domain {stub_name(doa) if isinstance(doa, Stub) else type(doa).__name__}"
        )
    matrix = doa.__dict__.get("_doa_matrix")
    threshold = doa.__dict__.get("_a")
    data = doa.__dict__.get("_data")
    if not (
        isinstance(matrix, np.ndarray)
        and matrix.shape == (n_features, n_features)
        and isinstance(threshold, (int, float))
    ):
        raise UnsupportedModelError("applicability domain data is malformed")
    training = None
    if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == n_features:
        training = data.astype(float)
    return LeverageDomain(matrix.astype(float), float(threshold)), training
