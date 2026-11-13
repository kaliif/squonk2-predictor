"""Read precomputed descriptors for the predictor.

A descriptor job (see docs/descriptor-jobs-design.md) writes a descriptor file
and a sidecar manifest JSON describing it. This module reads that pair, checks
it suits the model, and builds the model's feature matrix from the file's
columns, so the predictor can predict without recomputing descriptors.

The manifest records the featurizer and its parameters, the library versions,
the fragment method, the fill used for uncomputable descriptors and the
descriptor column names. Featurizer name and parameters are checked strictly
against the model (a mismatch silently changes the feature values); library
versions and the fragment method can only be checked when the model records
what it expects, which bare legacy models don't, so they are reported as
warnings instead.

Fingerprints are stored as a single bit-string column and expanded here to the
per-bit columns (f0..fN) the model's feature list names.
"""

import csv
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rdkit import Chem

from model_spec import ModelSpec, UnsupportedModelError

SCHEMA_VERSION = 1


class DescriptorFileError(Exception):
    """The descriptor file or its manifest can't be used; the message says why."""


@dataclass
class Manifest:
    """What a descriptor file records about how it was produced."""

    featurizer: str
    featurizer_params: dict
    columns: list[str]
    id_column: str
    smiles_column: str | None = None
    fragment_method: str | None = None
    library_versions: dict = field(default_factory=dict)
    missing_value: float | None = None
    schema_version: int = SCHEMA_VERSION

    @property
    def is_fingerprint(self) -> bool:
        return self.featurizer in ("morgan", "maccs")

    def describe(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.featurizer_params.items())
        return f"{self.featurizer}({params})"


def read_manifest(path) -> Manifest:
    """Load a manifest JSON, or raise DescriptorFileError if it's malformed."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise DescriptorFileError(f"can't read manifest {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise DescriptorFileError("manifest is not a JSON object")

    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise DescriptorFileError(
            f"manifest schema version {version!r} is not supported, only "
            f"{SCHEMA_VERSION}"
        )

    featurizer = data.get("featurizer")
    if not isinstance(featurizer, dict) or not isinstance(featurizer.get("name"), str):
        raise DescriptorFileError("manifest has no featurizer name")
    columns = data.get("columns")
    if not (isinstance(columns, list) and all(isinstance(c, str) for c in columns)):
        raise DescriptorFileError("manifest has no list of descriptor columns")
    id_column = data.get("id_column")
    if not isinstance(id_column, str):
        raise DescriptorFileError("manifest has no id_column")

    return Manifest(
        featurizer=featurizer["name"],
        featurizer_params=featurizer.get("params") or {},
        columns=columns,
        id_column=id_column,
        smiles_column=data.get("smiles_column"),
        fragment_method=data.get("fragment_method"),
        library_versions=data.get("library_versions") or {},
        missing_value=data.get("missing_value"),
        schema_version=version,
    )


def sidecar_manifest_path(descriptors_path) -> Path:
    """Default manifest path for a descriptor file: foo.smi -> foo.manifest.json."""
    base = str(descriptors_path)
    if base.endswith(".gz"):
        base = base[: -len(".gz")]
    return Path(base).with_suffix(".manifest.json")


def check_manifest(spec: ModelSpec, manifest: Manifest) -> list[str]:
    """Check a descriptor file suits the model. Raises UnsupportedModelError on
    a mismatch that would make predictions wrong; returns warnings for things
    that can't be checked against the model."""
    if manifest.featurizer != spec.featurizer.name:
        raise UnsupportedModelError(
            f"descriptor file was made with the {manifest.featurizer} featurizer, "
            f"but the model needs {spec.featurizer.name}"
        )

    # only the parameters that change the feature values, which is exactly what
    # the featurizer reports; missing_value is handled separately
    expected = spec.featurizer.params()
    differing = [
        f"{key}={manifest.featurizer_params.get(key)!r} (model needs {value!r})"
        for key, value in expected.items()
        if manifest.featurizer_params.get(key) != value
    ]
    if differing:
        raise UnsupportedModelError(
            f"descriptor file's {spec.featurizer.name} parameters don't match the "
            f"model: {', '.join(differing)}"
        )

    if not manifest.is_fingerprint:
        available = set(manifest.columns)
        missing = [n for n in spec.feature_names if n not in available]
        if missing:
            raise UnsupportedModelError(
                f"descriptor file is missing {len(missing)} columns the model "
                f"needs, e.g. {', '.join(missing[:5])}"
            )

    warnings = []
    warnings += _check_or_warn(
        "fragment method",
        manifest.fragment_method,
        spec.expected_fragment_method,
    )
    for library, version in sorted(manifest.library_versions.items()):
        expected_version = (spec.expected_library_versions or {}).get(library)
        warnings += _check_or_warn(f"{library} version", version, expected_version)
    return warnings


def _check_or_warn(what, got, expected) -> list[str]:
    """Raise if the model expects a specific value and it differs; otherwise
    warn that the value couldn't be checked."""
    if expected is not None:
        if got != expected:
            raise UnsupportedModelError(
                f"descriptor file's {what} is {got!r}, but the model needs "
                f"{expected!r}"
            )
        return []
    if got is None:
        return [f"descriptor file records no {what}"]
    return [
        f"descriptor file's {what} is {got!r}; the model doesn't record what it "
        "expects, so this can't be checked"
    ]


def read_descriptors(path, manifest: Manifest, delimiter="\t"):
    """Yield (id, smiles, values) for each record of a descriptor file, where
    values maps each manifest column to its raw string (empty if absent).

    Supports delimited text (columns named by a header) and SDF (descriptors in
    fields). A record whose descriptor columns are all empty is a failed row
    from the descriptor job and is yielded with empty values.
    """
    path = str(path)
    if path.removesuffix(".gz").endswith((".sdf", ".sd")):
        yield from _read_sdf(path, manifest)
    else:
        yield from _read_text(path, manifest, delimiter)


def _opener(path):
    return gzip.open if path.endswith(".gz") else open


def _read_text(path, manifest: Manifest, delimiter):
    with _opener(path)(path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        if reader.fieldnames is None:
            return
        header = set(reader.fieldnames)
        if manifest.id_column not in header:
            raise DescriptorFileError(
                f"descriptor file has no {manifest.id_column!r} column"
            )
        for col in manifest.columns:
            if col not in header:
                raise DescriptorFileError(
                    f"descriptor file has no {col!r} column named by the manifest"
                )
        for row in reader:
            name = (row.get(manifest.id_column) or "").strip()
            smiles = ""
            if manifest.smiles_column:
                smiles = (row.get(manifest.smiles_column) or "").strip()
            values = {col: (row.get(col) or "").strip() for col in manifest.columns}
            yield name, smiles, values


def _read_sdf(path, manifest: Manifest):
    with _opener(path)(path, "rb") as fh:
        for mol in Chem.ForwardSDMolSupplier(fh, sanitize=True):
            if mol is None:
                yield "", "", {col: "" for col in manifest.columns}
                continue
            if manifest.id_column == "_Name":
                name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            else:
                name = (
                    mol.GetProp(manifest.id_column)
                    if mol.HasProp(manifest.id_column)
                    else ""
                )
            smiles = Chem.MolToSmiles(mol)
            values = {
                col: (mol.GetProp(col) if mol.HasProp(col) else "")
                for col in manifest.columns
            }
            yield name.strip(), smiles, values


def build_matrix(spec: ModelSpec, manifest: Manifest, records):
    """Feature matrix for a batch of descriptor records, aligned to the model's
    feature names, with a mask of rows that had usable descriptors.

    records is a sequence of (id, smiles, values) from read_descriptors.
    """
    n = len(records)
    X = np.full((n, len(spec.feature_names)), np.nan)
    ok = np.zeros(n, dtype=bool)
    fill = spec.featurizer.missing_value

    if manifest.is_fingerprint:
        bit_column = manifest.columns[0]
        n_bits = len(spec.featurizer.feature_names)
        # the model's feature names are a subset of f0..f(n_bits-1)
        take = [int(name[1:]) for name in spec.feature_names]
        for i, (_, _, values) in enumerate(records):
            bits = values.get(bit_column, "")
            if len(bits) != n_bits or any(c not in "01" for c in bits):
                continue
            row = np.frombuffer(bits.encode("ascii"), dtype=np.uint8) - ord("0")
            X[i] = row[take].astype(float)
            ok[i] = True
    else:
        for i, (_, _, values) in enumerate(records):
            if all(values[col] == "" for col in manifest.columns):
                continue
            row = np.array(
                [_to_float(values.get(name, "")) for name in spec.feature_names]
            )
            if fill is not None:
                row = np.where(np.isnan(row), fill, row)
            X[i] = row
            ok[i] = True
    return X, ok


def _to_float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return np.nan
