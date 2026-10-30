"""Load user-supplied model files without letting them execute code.

Model files are pickles (plain, or written by joblib.dump, possibly compressed).
Unpickling can call any importable function, so a plain pickle.load on an
uploaded file is arbitrary code execution. This loader resolves every class or
function the file asks for through an allowlist:

- scikit-learn estimators and the internal helpers they need to unpickle
  (trees, losses, distance metrics, ...), numpy arrays, numpy ufuncs and
  random states are resolved to the real objects
- jaqpotpy, mordred and onnx objects are replaced with inert Stub instances
  that only hold the pickled state, so their code is never imported or run
- anything else is recorded, replaced with an inert placeholder, and the load
  fails with UnsafeModelError listing every rejected reference

joblib stores numpy arrays as raw bytes after a NumpyArrayWrapper marker. Those
are read here rather than by joblib, because joblib unpickles object arrays
with an unrestricted pickle.load. joblib's compression is undone with the
standard library, so joblib itself isn't needed.

Estimators pickled by an older scikit-learn are brought up to date where the
pickled state has changed meaning: see _upgrade_estimator_state.
"""

import bz2
import codecs
import contextlib
import gzip
import importlib
import io
import lzma
import pickle
import re
import warnings
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# top-level packages whose objects are replaced by inert stubs, never imported
STUBBED_PACKAGES = ("jaqpotpy", "mordred", "onnx")

_NUMPY_ARRAY_NAMES = {
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    # protocol 5 pickles rebuild arrays from a buffer
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.numeric", "_frombuffer"),
    ("numpy.core.memmap", "memmap"),
    ("numpy._core.memmap", "memmap"),
    ("numpy.random._pickle", "__randomstate_ctor"),
    ("numpy.random._pickle", "__bit_generator_ctor"),
    ("numpy.random._pickle", "__generator_ctor"),
}

_SKLEARN_INTERNAL_NAMES = {
    ("sklearn.tree._tree", "Tree"),
    ("sklearn.utils._bunch", "Bunch"),
    ("sklearn.neighbors._kd_tree", "KDTree"),
    ("sklearn.neighbors._kd_tree", "newObj"),
    ("sklearn.neighbors._ball_tree", "BallTree"),
    ("sklearn.neighbors._ball_tree", "newObj"),
    ("sklearn.metrics._dist_metrics", "newObj"),
    ("sklearn.ensemble._hist_gradient_boosting.predictor", "TreePredictor"),
}

# modules whose classes (and Cython unpickle helpers) are all plain data holders
_SKLEARN_INTERNAL_MODULES = (
    "sklearn._loss.loss",
    "sklearn._loss.link",
    "sklearn._loss._loss",
    "sklearn.ensemble._gb_losses",
    "sklearn.neural_network._stochastic_optimizers",
)

_JOBLIB_WRAPPER = ("joblib.numpy_pickle", "NumpyArrayWrapper")


def _latin1_encode(text, encoding="latin1"):
    """Stand-in for _codecs.encode, which protocol 2 pickles use to store bytes.
    Only the latin-1 use is allowed, not arbitrary codecs."""
    if encoding not in ("latin1", "latin-1"):
        raise pickle.UnpicklingError(f"codec {encoding!r} is not allowed")
    return codecs.encode(text, encoding)


class UnsafeModelError(Exception):
    """The file references code that is not allowed to be loaded."""

    def __init__(self, rejected):
        self.rejected = sorted(rejected)
        super().__init__(
            "Model file references disallowed code: " + ", ".join(self.rejected)
        )


class ModelLoadError(Exception):
    """The file could not be read as a model."""


class Stub:
    """Inert stand-in for an object of a stubbed package.

    Keeps the constructor arguments and pickled state so adapters can read
    them. The qualified name of the original class is in _stub_name.
    """

    _stub_name = "?"

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj.__dict__["_stub_args"] = args
        return obj

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) == 2:
            # (dict state, slot state)
            merged = dict(state[0] or {})
            merged.update(state[1] or {})
            state = merged
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_stub_state"] = state

    # stubbed dict and list subclasses are populated item by item
    def __setitem__(self, key, value):
        self.__dict__.setdefault("_stub_items", {})[key] = value

    def append(self, value):
        self.__dict__.setdefault("_stub_list", []).append(value)

    def extend(self, values):
        self.__dict__.setdefault("_stub_list", []).extend(values)

    def __repr__(self):
        return f"<Stub {self._stub_name}>"


class _Rejected(Stub):
    """Placeholder for a disallowed reference, so loading can carry on and
    report every rejected name at once."""


def stub_name(obj) -> str | None:
    """Qualified name of the class a Stub stands in for, or None."""
    return getattr(obj, "_stub_name") if isinstance(obj, Stub) else None


@dataclass
class LoadResult:
    obj: Any
    warnings: list[str] = field(default_factory=list)
    # scikit-learn versions the estimators in the file were pickled with
    sklearn_versions: list[str] = field(default_factory=list)


def _version_tuple(version: str) -> tuple[int, int]:
    # e.g. '1.3.2' or 'pre-0.18', which old pickles without a version get
    match = re.search(r"(\d+)\.(\d+)", version)
    return (int(match[1]), int(match[2])) if match else (0, 0)


def _upgrade_estimator_state(estimator, state: dict, version: str) -> int:
    """Update the pickled state of an estimator from an older scikit-learn to
    what this version expects, in place. Returns the number of trees whose
    data was converted.

    scikit-learn doesn't support unpickling estimators across versions, and
    for trees from before 1.4 it silently gives wrong results: classifier trees
    then stored weighted class counts in their nodes, later versions store
    class fractions, so predicted probabilities came out wrong. Trees from
    before 1.4 also lack monotonic_cst, which 1.4 to 1.8 can't predict without.
    """
    # pylint: disable=import-outside-toplevel
    from sklearn.base import ClassifierMixin
    from sklearn.tree import BaseDecisionTree
    from sklearn.tree._tree import Tree  # pylint: disable=no-name-in-module

    if _version_tuple(version) >= (1, 4) or not isinstance(estimator, BaseDecisionTree):
        return 0
    state.setdefault("monotonic_cst", None)
    tree = state.get("tree_")
    # the estimator class decides, as the instance has no attributes yet
    if not (isinstance(estimator, ClassifierMixin) and isinstance(tree, Tree)):
        return 0
    tree_state = tree.__getstate__()
    values = tree_state["values"]
    totals = values.sum(axis=2, keepdims=True)
    tree_state["values"] = np.divide(
        values, totals, out=np.zeros_like(values), where=totals > 0
    )
    tree.__setstate__(tree_state)
    return 1


def _is_sklearn_allowed(module: str, name: str, obj) -> bool:
    from sklearn.base import BaseEstimator
    from sklearn.gaussian_process.kernels import Kernel
    from sklearn.metrics import DistanceMetric

    if (module, name) in _SKLEARN_INTERNAL_NAMES:
        return True
    if module in _SKLEARN_INTERNAL_MODULES:
        return isinstance(obj, type) or name.startswith("__pyx_unpickle_")
    if isinstance(obj, type):
        return issubclass(obj, (BaseEstimator, Kernel, DistanceMetric))
    return False


def _resolve(module: str, name: str):
    """Import module.name if it is allowed, otherwise return None."""
    package = module.split(".")[0]
    if package not in ("numpy", "sklearn"):
        return None
    if "." in name:
        # nested attribute lookups could walk out of the allowed object
        return None
    try:
        obj = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError):
        return None

    if package == "numpy":
        if (module, name) in _NUMPY_ARRAY_NAMES:
            return obj
        # ufuncs such as log1p are pickled by name, e.g. for
        # TransformedTargetRegressor(func=np.log1p)
        return obj if isinstance(obj, np.ufunc) else None

    return obj if _is_sklearn_allowed(module, name, obj) else None


class _JoblibArrayWrapper(Stub):
    """Marker joblib writes in place of a numpy array; the raw data follows."""

    _stub_name = "joblib.numpy_pickle.NumpyArrayWrapper"


class RestrictedUnpickler(pickle._Unpickler):  # pylint: disable=protected-access
    """Pure-Python unpickler: unlike the C one, its BUILD handling can be
    overridden, which is needed to read joblib's inline arrays."""

    dispatch = pickle._Unpickler.dispatch.copy()  # pylint: disable=protected-access

    def __init__(self, file, rejected=None, stubs=None):
        super().__init__(file)
        # joblib writes array data outside pickle frames, read from the file directly
        self._fh = file
        self.rejected = rejected if rejected is not None else set()
        self.stubs = stubs if stubs is not None else {}
        self.sklearn_versions: set[str] = set()
        self.converted_trees = 0

    def find_class(self, module, name):
        if module.split(".")[0] in STUBBED_PACKAGES:
            return self._stub_class(module, name, Stub)
        if (module, name) == _JOBLIB_WRAPPER:
            return _JoblibArrayWrapper
        if (module, name) == ("_codecs", "encode"):
            return _latin1_encode
        obj = _resolve(module, name)
        if obj is None:
            self.rejected.add(f"{module}.{name}")
            return self._stub_class(module, name, _Rejected)
        return obj

    def _stub_class(self, module, name, base):
        key = (module, name, base)
        if key not in self.stubs:
            # same class object for every reference, so isinstance checks
            # between stubbed objects behave as they did originally
            self.stubs[key] = type(name, (base,), {"_stub_name": f"{module}.{name}"})
        return self.stubs[key]

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("persistent IDs are not supported")

    def load_build(self):
        from sklearn.base import (
            BaseEstimator,  # pylint: disable=import-outside-toplevel
        )

        state, instance = self.stack[-1], self.stack[-2]
        if isinstance(instance, BaseEstimator) and isinstance(state, dict):
            version = state.get("_sklearn_version")
            if isinstance(version, str):
                self.sklearn_versions.add(version)
                self.converted_trees += _upgrade_estimator_state(
                    instance, state, version
                )
        super().load_build()
        if isinstance(self.stack[-1], _JoblibArrayWrapper):
            wrapper = self.stack.pop()
            self.append(self._read_joblib_array(wrapper))

    dispatch[pickle.BUILD[0]] = load_build

    def _read_joblib_array(self, wrapper):
        dtype = getattr(wrapper, "dtype", None)
        shape = getattr(wrapper, "shape", None)
        order = getattr(wrapper, "order", "C")
        if not isinstance(dtype, np.dtype) or not isinstance(shape, tuple):
            raise ModelLoadError("malformed joblib array marker")
        if dtype.hasobject:
            # nested pickle stream for the array's Python objects
            nested = RestrictedUnpickler(self._fh, self.rejected, self.stubs)
            array = nested.load()
            self.sklearn_versions |= nested.sklearn_versions
            self.converted_trees += nested.converted_trees
            return array

        if getattr(wrapper, "numpy_array_alignment_bytes", None) is not None:
            padding = int.from_bytes(self._fh.read(1), byteorder="little")
            self._fh.read(padding)

        count = int(np.prod(shape, dtype=np.int64))
        nbytes = count * dtype.itemsize
        data = self._fh.read(nbytes)
        if len(data) != nbytes:
            raise ModelLoadError("truncated array data in joblib file")
        array = np.frombuffer(data, dtype=dtype, count=count).copy()
        if order == "F":
            array = array.reshape(shape[::-1]).transpose()
        else:
            array = array.reshape(shape)
        if not dtype.isnative:
            array = array.byteswap().view(dtype.newbyteorder("="))
        return array


# joblib.dump compresses with a standard format, recognised by its first bytes
_DECOMPRESSORS = (
    (b"\x1f\x8b", lambda f: gzip.GzipFile(fileobj=f)),
    (b"BZh", bz2.BZ2File),
    (b"\xfd7zXZ\x00", lambda f: lzma.LZMAFile(f, format=lzma.FORMAT_XZ)),
    (b"]\x00\x00", lambda f: lzma.LZMAFile(f, format=lzma.FORMAT_ALONE)),
    # zlib streams start with 0x78 for the default window size, which no
    # pickle opcode does
    (b"\x78", lambda f: io.BytesIO(_zlib_decompress(f.read()))),
)
_UNSUPPORTED = (
    (b"\x04\x22\x4d\x18", "lz4 compressed files are not supported"),
    # joblib < 0.10, which also reads companion files from disk
    (b"ZF", "joblib files older than version 0.10 are not supported"),
)


def _zlib_decompress(data):
    try:
        return zlib.decompress(data)
    except zlib.error as exc:
        raise ModelLoadError(f"could not read model file: {exc}") from exc


@contextlib.contextmanager
def _open_model_file(path):
    """The file, decompressed if it is compressed."""
    with open(path, "rb") as f:
        head = f.read(6)
        f.seek(0)
        for magic, message in _UNSUPPORTED:
            if head.startswith(magic):
                raise ModelLoadError(message)
        for magic, open_ in _DECOMPRESSORS:
            if head.startswith(magic):
                with open_(f) as decompressed:
                    yield decompressed
                return
        yield f


def load_model_file(path) -> LoadResult:
    """Load a pickle or joblib model file through the allowlist."""
    # pylint: disable=import-outside-toplevel
    from sklearn.exceptions import InconsistentVersionWarning

    rejected: set[str] = set()
    with _open_model_file(path) as fobj:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            unpickler = RestrictedUnpickler(fobj, rejected)
            try:
                obj = unpickler.load()
            except UnsafeModelError:
                raise
            except Exception as exc:
                # placeholders for rejected names can make later steps fail;
                # the rejection is the real reason
                if rejected:
                    raise UnsafeModelError(rejected) from exc
                raise ModelLoadError(f"could not read model file: {exc}") from exc

    if rejected:
        raise UnsafeModelError(rejected)
    # version mismatches are reported from sklearn_versions instead of
    # scikit-learn's warning for every estimator class
    messages = [
        str(w.message)
        for w in caught
        if not issubclass(w.category, InconsistentVersionWarning)
    ]
    if unpickler.converted_trees:
        messages.append(
            f"converted {unpickler.converted_trees} classifier trees from the "
            "format of scikit-learn < 1.4"
        )
    return LoadResult(
        obj,
        list(dict.fromkeys(messages)),
        sorted(unpickler.sklearn_versions, key=_version_tuple),
    )
