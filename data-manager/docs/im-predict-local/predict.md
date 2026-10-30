# Job: predict

This describes how to run the `predict` job in the `im-predict-local`
collection.

## What the job does

The job predicts a property of molecules with a pre-trained scikit-learn
model that you supply. It reads from the model file which descriptors the
model was trained on, calculates them for your molecules and predicts.

A model can load and run in a different software environment from the one it
was trained in, yet give different predictions: descriptor implementations
change between library versions, and estimators can behave differently in
another scikit-learn version. Nothing fails, the numbers are just wrong. To
guard against this, supply **probe molecules**: a few molecules with the
predictions the model gave when it was created. The model is used only if it
reproduces them all. Without probes the job still runs but marks every
prediction as **unverified**.

## Supported models

Model files are loaded so that they can't run code: only scikit-learn
estimators and the parts they need are accepted, and any other content is
rejected.

Currently supported are jaqpotpy 1.0.x `MolecularModel` files (`.jmodel`)
holding a scikit-learn estimator trained on one of:

* Morgan fingerprints
* MACCS keys
* RDKit descriptors
* Mordred 2D descriptors

Models with external features or preprocessing steps are not supported. A
file holding only a scikit-learn estimator is rejected, because nothing in it
records which descriptors the estimator was trained on.

The job runs scikit-learn 1.9.1, RDKit 2023.09.6 and mordred 1.2.0. A model
trained with another scikit-learn version is reported in the job's events,
and rejected if it can't run. Decision trees saved by scikit-learn before 1.4
are converted to the current format; supply probes to confirm the converted
model reproduces its predictions.

## Implementation details

* Job implementation: [predict.py](/src/predict.py)
* Job definition: `predict` in [jobs.yaml](../../jobs.yaml)

## How to run the job

### Inputs

* **Model**: the model file (`.jmodel`).
* **Molecules to predict**: molecules in SDF or SMILES format. A SMILES file
  has the SMILES in the first column and optionally an id in the second.
* **Probe molecules** (optional): a CSV file with the columns:
    * `smiles`
    * `prediction`: the value, or class label, the model predicted when it
      was created
    * `score` (optional): for a binary classifier, the probability of the
      second class, or the decision score for models without probabilities

### Options

* **Output file name**: name of the CSV file to create. Default:
  `predictions.csv`.
* **Delimiter of a SMILES input file**: tab, space, comma, semicolon or
  pipe. Default: tab. Ignored for SDF.
* **SMILES input file has a header line**: Default: false. Ignored for SDF.

### Output

A CSV file with a row for every input molecule, in input order:

* `id`: the molecule's name (SDF) or second column (SMILES)
* `smiles`
* the prediction, in a column named after the model's target, e.g. `dili`
* depending on the model:
    * `<target>_probability`: probability of the second class for binary
      classifiers, or `<target>_probability_<class>` for each class of a
      multiclass classifier
    * `<target>_score`: decision score for classifiers without probabilities
    * `<target>_leverage` and `<target>_in_domain`: the applicability domain,
      for models that have one. Predictions for molecules outside the domain
      are less reliable.
* `model_status`: `verified` or `unverified`
* `error`: why a molecule couldn't be predicted, otherwise empty

The job's events report the model, any limitations found (e.g. no
applicability domain, a different scikit-learn version), whether the model was
verified, and how many molecules were predicted.

The job fails without creating an output file if the model is unsafe or
unsupported, or doesn't reproduce its probes. The reason is reported in the
job's events.

## Limitations

1. Only jaqpotpy 1.0.x models are supported so far.
2. Probes are a sample: a change in descriptor calculation that affects only
   rare molecules can go unnoticed if no probe molecule is affected.
