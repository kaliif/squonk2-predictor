"""Predict values for molecules with a pre-trained model.

The model file is loaded without running any code it contains, its format is
recognised and the descriptors it was trained on are calculated for the
molecules. Models that don't say which descriptors they need, or whose
parts aren't supported, are rejected.

Nothing in a model file records the predictions the model made when it was
created, so whether the descriptors and estimator here still reproduce them
can't be checked; the job warns about this.

The output CSV has a row per input molecule, in input order, with the id,
SMILES, the prediction and, depending on the model, a probability or score
and the applicability domain. Molecules that can't be predicted have an
error instead.

Example:

    python src/predict.py --model models/caco2_wang.jmodel \\
        --molecules data/candidates-10.sdf --outfile predictions.csv
"""

import argparse
import csv
import gzip
import logging
from collections.abc import Iterator
from itertools import islice
from pathlib import Path

from dm_job_utilities.dm_log import DmLog
from rdkit import Chem, RDLogger
from sklearn.base import BaseEstimator

import jaqpotpy_adapter
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
    result = load_model_file(path)
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


def read_molecules(
    path, delimiter="\t", read_header=False
) -> Iterator[tuple[str, str, Chem.Mol | None]]:
    """(id, SMILES, molecule) for each record of an SDF or SMILES file.

    The molecule is None if the record can't be parsed. SMILES files have the
    SMILES in the first column and an optional id in the second; columns are
    separated by the delimiter or, in the first column, by whitespace.
    """
    path = str(path)
    opener = gzip.open if path.endswith(".gz") else open
    if path.removesuffix(".gz").endswith(".sdf"):
        with opener(path, "rb") as fh:
            for mol in Chem.ForwardSDMolSupplier(fh, removeHs=False):
                if mol is None:
                    yield "", "", None
                    continue
                name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
                yield name, Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False)), mol
    else:
        with opener(path, "rt", encoding="utf-8", newline="") as fh:
            rows = csv.reader(fh, delimiter=delimiter, skipinitialspace=True)
            if read_header:
                next(rows, None)
            for row in rows:
                # SMILES can't contain whitespace, so anything after it in the
                # first column is the id of a space separated file
                first = row[0].split(None, 1) if row else []
                if not first:
                    continue
                smiles = first[0]
                if len(row) > 1:
                    name = row[1].strip()
                else:
                    name = first[1].strip() if len(first) > 1 else ""
                yield name, smiles, Chem.MolFromSmiles(smiles)


def run(
    model,
    molecules,
    outfile,
    delimiter="\t",
    read_header=False,
) -> dict:
    """Predict the molecules and write the output CSV. Raises if the model
    can't be used. Returns counts of molecules read and predicted."""
    spec = load_spec(model)
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

    target = spec.target_name
    # prediction columns are named after the target so outputs of several
    # models can be combined
    columns = [
        target if col == "prediction" else f"{target}_{col}"
        for col in spec.output_columns
    ]
    counts = {"read": 0, "predicted": 0, "out_of_domain": 0}
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    records = read_molecules(molecules, delimiter=delimiter, read_header=read_header)
    with open(outfile, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "smiles", *columns, "error"])
        while batch := list(islice(records, BATCH_SIZE)):
            table, ok = spec.predict([mol for _, _, mol in batch])
            for i, (name, smiles, mol) in enumerate(batch):
                if ok[i]:
                    values = list(table.iloc[i])
                    error = ""
                else:
                    values = [""] * len(columns)
                    error = (
                        "invalid molecule"
                        if mol is None
                        else "descriptors can't be calculated"
                    )
                writer.writerow([name, smiles, *values, error])
            counts["read"] += len(batch)
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


def main():
    parser = argparse.ArgumentParser(
        description="Predict values for molecules with a pre-trained model"
    )
    parser.add_argument("--model", required=True, help="Model file")
    parser.add_argument(
        "--molecules", required=True, help="Molecules to predict (.sdf or .smi)"
    )
    parser.add_argument("--outfile", required=True, help="Output CSV file")
    parser.add_argument(
        "--delimiter",
        default="tab",
        help=f"Delimiter of a SMILES file: {', '.join(DELIMITERS)} or a character",
    )
    parser.add_argument(
        "--read-header",
        action="store_true",
        help="The SMILES file has a header line",
    )
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")
    try:
        run(
            model=args.model,
            molecules=args.molecules,
            outfile=args.outfile,
            delimiter=DELIMITERS.get(args.delimiter, args.delimiter),
            read_header=args.read_header,
        )
    except UnsafeModelError as exc:
        _fatal(f"Model rejected as unsafe: {exc}")
    except (ModelLoadError, UnsupportedModelError) as exc:
        _fatal(f"Model can't be used: {exc}")


if __name__ == "__main__":
    main()
