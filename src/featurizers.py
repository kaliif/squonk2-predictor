"""Descriptor-set descriptions for supported model formats.

The job no longer calculates descriptors itself: a descriptor job does that and
the predictor reads the result (see docs/descriptor-jobs-design.md). What stays
here is a lightweight description of the featurizer a model was trained with —
its name, the parameters that define its output, and, for fingerprints, the
per-bit feature names — so a descriptor file's manifest can be checked against
the model and fingerprint bit strings expanded to the columns the model names.

No rdkit or mordred is needed for this; the names below match the featurizer
names a descriptor job records in its manifest.
"""


class Featurizer:
    """A model's featurizer, as metadata only."""

    name: str
    # features are 0/1 bits (a fingerprint), stored as one bit-string column
    binary: bool = False

    def __init__(self, missing_value=None):
        # value a descriptor file's uncomputable cells were filled with, used
        # when reading descriptor columns back in
        self.missing_value = missing_value

    @property
    def feature_names(self) -> list[str] | None:
        """The featurizer's own feature names, for fingerprints (f0..fN). For
        descriptor sets the model's own feature list is the source of truth, so
        this is None."""
        return None

    def params(self) -> dict:
        return {}

    def describe(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.params().items())
        return f"{self.name}({params})"


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


class MACCSKeys(Featurizer):
    name = "maccs"
    binary = True

    @property
    def feature_names(self):
        return [f"f{i}" for i in range(167)]


class RDKitDescriptors(Featurizer):
    name = "rdkit_descriptors"

    def __init__(self, use_fragment=True, ipc_avg=True, missing_value=None):
        super().__init__(missing_value)
        self.use_fragment = use_fragment
        self.ipc_avg = ipc_avg

    def params(self):
        return {"use_fragment": self.use_fragment, "ipc_avg": self.ipc_avg}


class MordredDescriptors(Featurizer):
    name = "mordred"

    def params(self):
        return {"ignore_3D": True}
