# squonk2-predictor: open design questions

Kalev Takkis, 30 October 2026. Status: for discussion.

The `predict` job works: it loads jaqpotpy 1.0.x models safely, reproduces
jaqpotpy's own predictions for all 22 legacy models, and has run on the
Squonk test platform. Before we extend it to models users train themselves,
a few design questions need answers. Two of them shape the model manifest
and should be settled before it is built. The rest can follow without
breaking anything.

## Where we are

- **Safe loading.** Model files are unpickled with an allowlist, so a model
  file can't run code. Anything outside scikit-learn estimators and their
  numpy internals is rejected.
- **Supported models.** Only jaqpotpy 1.0.x `.jmodel` files with Morgan,
  MACCS, RDKit or Mordred 2D descriptors. A bare scikit-learn estimator is
  rejected because nothing records which descriptors it was trained on.
- **No verification.** The job can't check that a model still gives the
  predictions it gave when it was created, and warns about this on every
  run. We tried user-supplied probe molecules and dropped them: users can't
  produce them.
- **Old pinned environment.** The image is held back to keep the legacy
  models exact (table below). scikit-learn is current (1.9.1); trees saved
  before 1.4 are converted on loading.
- **Next step planned.** A sidecar manifest (JSON next to the model) naming
  the featurizer and its parameters, so bare scikit-learn estimators can be
  used.

| Library change     | Effect on the legacy models                                    |
| ------------------ | -------------------------------------------------------------- |
| RDKit 2024.03      | `fr_unbrch_alkane` changes                                     |
| RDKit 2024.03.6    | segfault in MACCS keys for one molecule                        |
| RDKit 2025.09      | H-bond acceptor counts change                                  |
| RDKit 2026.03      | Mordred information-content indices change                     |
| RDKit 2026.09      | ring perception changes; two molecules refused                 |
| mordredcommunity   | rings perceived differently from mordred 1.2.0                 |
| scikit-learn ≥ 1.4 | old classifier trees give wrong probabilities unless converted |

Each of these changed predictions without raising an error. That is the
background to every question below.

## Decisions needed

| #   | Question                                   | When            | Recommendation                                                              |
| --- | ------------------------------------------ | --------------- | --------------------------------------------------------------------------- |
| 1   | Who writes the manifest?                   | Before manifest | A packaging step we own; hand-written manifests accepted but unverified     |
| 2   | What must a manifest describe?             | Before manifest | Featurizer, molecule preparation, target meaning, feature list, versions    |
| 3   | Pipelines and custom code                  | Can follow      | Allow standard scikit-learn steps; never user classes                       |
| 4   | Model file format                          | Can follow      | Evaluate ONNX for the estimator part                                        |
| 5   | Upgrades and environments                  | Can follow      | Reference predictions as a release gate; split environments when one fails |
| 6   | Applicability domain without training data | Can follow      | Optional in the manifest; say so in the output when it's missing            |

## 1. Who writes the manifest, and can we trust it?

*Decide before the manifest is built.*

**Problem.** A hand-written manifest is a claim the job can't check. If a
user names the wrong featurizer and the feature count happens to match, the
job predicts nonsense without an error. This is the same gap the probe
molecules were meant to close.

**Options.**

a. **Hand-written manifests only.** Simple, and covers any model. Every
   prediction stays unverified.
b. **A packaging step we own:** a small helper library, or a Squonk "train"
   or "package model" job. It writes the manifest at training time,
   including the featurizer, library versions, the scikit-learn version and
   a handful of reference predictions on training molecules. The predict job
   recomputes those references and refuses the model if they don't match.
   This is the probe idea, made automatic, so users don't have to do
   anything.
c. **Both**, with the job reporting which kind it got.

**Recommendation.** Option c. Treat packaged models as the supported route,
and accept hand-written manifests as best effort with the existing
unverified warning. The question for the team is whether arbitrary
scikit-learn files are a goal in themselves, or whether "models packaged our
way" is acceptable to users.

## 2. What must a manifest describe?

*Decide before the manifest is built.*

**Problem.** The featurizer alone isn't enough. Each item below gives
silently wrong results when it's missing, and whatever goes into version 1 of
the format has to be supported from then on.

- **Molecule preparation.** Salt stripping, neutralisation, tautomer
  canonicalisation, stereo handling. The job currently only removes
  hydrogens, which matches jaqpotpy. A user's model may have been trained on
  standardised molecules.
- **Target meaning.** Units and any transform (log10, pIC50) so values aren't
  misread. For classifiers, what each label means.
- **Feature list and order.** Many models use a subset of descriptors after
  variance or correlation filtering. The manifest should name the features
  in order, not only the featurizer.
- **Versions.** RDKit, mordred and scikit-learn versions at training time,
  so the job can warn or refuse when its own differ.
- **Format version.** A schema version field, so the format can change
  later.

**Recommendation.** Agree a version 1 schema with all five items before the
first user writes a manifest. Fields can be optional, but the slots should
exist.

## 3. Pipelines and custom code

*Can follow.*

**Problem.** Real models are often scikit-learn `Pipeline`s with scalers,
imputers, feature selectors or a `ColumnTransformer`. Standard steps can be
added to the loader's allowlist. A custom transformer, meaning the user's
own class, can't be loaded without running their code.

**Recommendation.** Allow standard scikit-learn pipeline steps as they are
needed, each with a test. Keep custom classes a hard refusal, and make the
error message say which class it was and why it's refused. Where to draw the
line is a product decision for the team.

## 4. Model file format

*Can follow.*

**Problem.** Pickles are tied to the scikit-learn version, in both
directions. Old models need conversion code, as the trees did. Models
trained with a scikit-learn newer than the image's may not load at all, and
that direction gets worse over time because users upgrade faster than the
image does.

**Options.**

- **Pickle**, with conversion code that grows with each scikit-learn
  release.
- **skops**: safe to load, but still tied to the scikit-learn version.
- **ONNX**: removes scikit-learn drift for the estimator, though not for the
  descriptors. Needs onnxruntime in the image, and not every estimator
  converts.

**Recommendation.** Evaluate ONNX as the estimator format for packaged
models (question 1). Separately, some legacy `.jmodel` files carry an ONNX
copy of the estimator (the dili model does; the others haven't been
checked). That copy could give an independent check of the estimator part.

## 5. Upgrades and environments

*Can follow.*

**Problem.** Any image upgrade can quietly change predictions, as the table
above shows. As long as old models need old RDKit and numpy, one image can't
be fully current. The lasting fix is separate environments for each
generation of models, with the job working out which one a model needs. That
needs a rule for when a new environment is created.

**Questions.**

- Must every supported model, or every model type, come with reference
  predictions before it's supported? The test suite already checks all 22
  legacy models and the three test models this way.
- When an upgrade breaks one model, does that block the upgrade, or does
  that model move to an older image?
- How does the job pick an environment: from the manifest's recorded
  versions, from the model format, or by the user choosing a job version?

**Recommendation.** Treat the reference-prediction tests as a release gate
for the image. An upgrade that fails the gate for some models creates a new
environment for the new models rather than breaking the old ones.

## 6. Applicability domain without training data

*Can follow.*

**Problem.** jaqpotpy stored the training data needed for the leverage
check. A bare estimator has none. The leverage matrix grows with the square
of the number of descriptors, so for Mordred models it is large.

**Recommendation.** Make the domain optional in the manifest, written by the
packaging step when there is one. Models without it run, and the output and
events say no domain check was made, as the job already does for legacy
models without one.

## Smaller items

- **Mordred speed.** Mordred descriptors are slow on large inputs. This
  affects cost and job resources, not the design.
- **Provenance.** A model is just a file in a project, so nothing records
  which version of a model produced which predictions. A model hash in the
  output would be a cheap start.

Questions 1 and 2 block the manifest. Questions 3 to 6 can be decided later
without changing what is already built.
