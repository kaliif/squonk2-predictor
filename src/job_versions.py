"""The versions of this job and the scikit-learn version each one runs.

Each job version is built from its own branch with its own scikit-learn (see
"Job versions" in README.md), but this table is the same on every branch, so
any version can tell users which other version can read their model. Add a
row when a branch is cut, on main, and merge it into the branches.
"""

import sklearn

# job version: scikit-learn version
JOB_VERSIONS = {
    "1.1.0": "1.9.1",
    "1.0.1": "1.2.2",
}

# scikit-learn 1.3 changed the layout of tree nodes: trees pickled on one side
# of the change can't be read on the other
TREE_FORMAT_CHANGE = (1, 3)
_CHANGE = ".".join(map(str, TREE_FORMAT_CHANGE))


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split(".")[:2])


def _new_tree_format(version: str) -> bool:
    return _version_tuple(version) >= TREE_FORMAT_CHANGE


def tree_format_message() -> str:
    """Why a model whose trees this scikit-learn can't read was refused, and
    which job versions can read it."""
    running = sklearn.__version__
    if _new_tree_format(running):
        saved = f"before {_CHANGE}"
    else:
        saved = f"{_CHANGE} or later"
    message = (
        f"the model has tree estimators saved with scikit-learn {saved}, which "
        f"this job version, running scikit-learn {running}, can't read"
    )
    others = [
        f"{job} (scikit-learn {skl})"
        for job, skl in JOB_VERSIONS.items()
        if _new_tree_format(skl) != _new_tree_format(running)
    ]
    if others:
        message += f"; use job version {' or '.join(others)}"
    return message
