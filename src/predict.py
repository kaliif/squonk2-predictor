"""Predict values for molecules with a pre-trained model.

The model file is loaded without running any code it contains and its format is
recognised. Descriptors are not calculated here: a descriptor job computes them
and the predictor reads that file and its manifest, checks it suits the model,
and applies the estimator. Models that don't say which descriptors they need,
or whose parts aren't supported, are rejected.

Nothing in a model file records the predictions the model made when it was
created, so whether the descriptors and estimator here still reproduce them
can't be checked; the job warns about this.

The output CSV has a row per input record, in input order, with the id, SMILES,
the prediction and, depending on the model, a probability or score and the
applicability domain. Records that can't be predicted have an error instead.

Example:

    python src/predict.py --model models/caco2_wang.jmodel \\
        --descriptors descriptors.smi --outfile predictions.csv
"""

import argparse
import csv
import logging
from itertools import islice
from pathlib import Path

from dm_job_utilities.dm_log import DmLog
from rdkit import RDLogger
from sklearn.base import BaseEstimator

import jaqpotpy_adapter
import job_versions
from descriptor_input import (
    DescriptorFileError,
    build_matrix,
    check_manifest,
    read_descriptors,
    read_manifest,
    sidecar_manifest_path,
)
from model_spec import ModelSpec, UnsupportedModelError, check_spec
from safe_loader import ModelLoadError, UnsafeModelError, load_model_file, stub_name

BATCH_SIZE = 1000
DELIMITERS = {"tab": "\t", "space": " ", "comma": ",", "semicolon": ";", "pipe": "|"}


def _event(msg, level=logging.INFO):
    # DmLog truncates its shared buffer without rewinding it, which pads every
    # message after the first with NUL characters
    DmLog.string_buffer.seek(0)
    DmLog.emit_event(msg, level=level)


def _fatal(msg):
    DmLog.string_buffer.seek(0)
    DmLog.emit_fatal_event(msg)


def load_spec(path) -> ModelSpec:
    """Load and check a model file, or raise saying why it can't be used."""
    try:
        result = load_model_file(path)
    except ModelLoadError as exc:
        # scikit-learn's message lists the record types of tree nodes
        if "node array from the pickle has an incompatible dtype" in str(exc):
            raise ModelLoadError(job_versions.tree_format_message()) from exc
        raise
    obj = result.obj
    if jaqpotpy_adapter.is_jaqpotpy_model(obj):
        spec = jaqpotpy_adapter.adapt(obj, name=Path(path).stem)
    elif isinstance(obj, BaseEstimator):
        raise UnsupportedModelError(
            f"the file holds a bare {type(obj).__name__} with no record of the "
            "descriptors it was trained on"
        )
    else:
        what = stub_name(obj) or type(obj).__name__
        raise UnsupportedModelError(f"unrecognised model file contents ({what})")
    spec.sklearn_versions = result.sklearn_versions
    spec.warnings[:0] = result.warnings
    check_spec(spec)
    return spec


def _report_model(spec: ModelSpec) -> None:
    """Emit the model description and every warning, including that predictions
    can't be verified."""
    _event(
        f"Model {spec.name} ({spec.source_format}): {type(spec.estimator).__name__} "
        f"predicting {spec.target_name} from {len(spec.feature_names)} "
        f"{spec.featurizer.describe()} features"
    )
    for warning in spec.warnings:
        _event(f"Warning: {warning}", level=logging.WARNING)
    _event(
        "Warning: predictions can't be verified: it is not known whether the "
        "descriptors and estimator here reproduce the predictions made when "
        "the model was created",
        level=logging.WARNING,
    )


def _output_columns(spec: ModelSpec) -> list[str]:
    """Prediction column names, each named after the model's target so the
    outputs of several models can be combined."""
    target = spec.target_name
    return [
        target if col == "prediction" else f"{target}_{col}"
        for col in spec.output_columns
    ]


def _write_predictions(spec: ModelSpec, outfile, batches) -> dict:
    """Write the output CSV from an iterator of (metas, table, ok, errors)
    batches, where metas is a list of (id, smiles), and errors gives the error
    text for failed rows. Returns counts of records read and predicted."""
    columns = _output_columns(spec)
    counts = {"read": 0, "predicted": 0, "out_of_domain": 0}
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "smiles", *columns, "error"])
        for metas, table, ok, errors in batches:
            for i, (name, smiles) in enumerate(metas):
                if ok[i]:
                    values = list(table.iloc[i])
                    error = ""
                else:
                    values = [""] * len(columns)
                    error = errors[i]
                writer.writerow([name, smiles, *values, error])
            counts["read"] += len(metas)
            counts["predicted"] += int(ok.sum())
            if "in_domain" in table:
                counts["out_of_domain"] += int(
                    (~table.in_domain[ok].astype(bool)).sum()
                )

    summary = f"Predicted {counts['predicted']} of {counts['read']} molecules"
    if spec.domain is not None:
        summary += f", {counts['out_of_domain']} outside the applicability domain"
    _event(summary)
    DmLog.emit_cost(counts["predicted"])
    return counts


def run(
    model,
    descriptors,
    outfile,
    manifest=None,
    delimiter="\t",
) -> dict:
    """Predict from a precomputed descriptor file and its manifest, and write
    the output CSV. Raises if the model, the file or the manifest can't be used.
    Returns counts of records read and predicted."""
    spec = load_spec(model)
    manifest_path = manifest or sidecar_manifest_path(descriptors)
    manifest = read_manifest(manifest_path)
    _event(
        f"Descriptors from {manifest.describe()}"
        + (
            f", libraries {manifest.library_versions}"
            if manifest.library_versions
            else ""
        )
        + (
            f", fragment method {manifest.fragment_method}"
            if manifest.fragment_method
            else ""
        )
    )
    spec.warnings += check_manifest(spec, manifest)
    _report_model(spec)
    records = read_descriptors(descriptors, manifest, delimiter=delimiter)

    def batches():
        while batch := list(islice(records, BATCH_SIZE)):
            X, ok = build_matrix(spec, manifest, batch)
            table, ok = spec.predict_matrix(X, ok)
            metas = [(name, smiles) for name, smiles, _ in batch]
            errors = ["no usable descriptors in the file"] * len(batch)
            yield metas, table, ok, errors

    return _write_predictions(spec, outfile, batches())


def main():
    parser = argparse.ArgumentParser(
        description="Predict values for molecules with a pre-trained model"
    )
    parser.add_argument("--model", required=True, help="Model file")
    parser.add_argument(
        "--descriptors",
        required=True,
        help="Precomputed descriptor file to predict from (.smi, .csv or .sdf)",
    )
    parser.add_argument(
        "--manifest",
        help="Manifest JSON for the descriptor file (default: next to it)",
    )
    parser.add_argument("--outfile", required=True, help="Output CSV file")
    parser.add_argument(
        "--delimiter",
        default="tab",
        help=f"Delimiter of a text descriptor file: {', '.join(DELIMITERS)} "
        "or a character",
    )
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")
    delimiter = DELIMITERS.get(args.delimiter, args.delimiter)
    try:
        run(
            model=args.model,
            descriptors=args.descriptors,
            outfile=args.outfile,
            manifest=args.manifest,
            delimiter=delimiter,
        )
    except UnsafeModelError as exc:
        _fatal(f"Model rejected as unsafe: {exc}")
    except (ModelLoadError, UnsupportedModelError) as exc:
        _fatal(f"Model can't be used: {exc}")
    except DescriptorFileError as exc:
        _fatal(f"Descriptor file can't be used: {exc}")


if __name__ == "__main__":
    main()
