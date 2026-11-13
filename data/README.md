# Data
Test data for the job tests in `data-manager/jobs.yaml`.

- `10.smi`, `10-error.smi`, `candidates-10.sdf`, `candidates-10-error.sdf`:
  molecules, the same as in the squonk2-jaqpot job tests (the `-error` files
  have one invalid molecule). The job no longer takes molecules directly; these
  are kept as the source for generating the descriptor fixtures below.
- `models/CYP2D6_Veith.jmodel`, `models/clearance_hepatocyte_az.jmodel`:
  jaqpotpy 1.0.104 models from squonk2-jaqpot (Morgan and MACCS). Used by the
  job tests. `CYP2D6_Veith.reference.csv` is kept as the molecule (SMILES)
  source for `scripts/make_version_test_model.py`.
- `models/dili-3trees.jmodel`: the squonk2-jaqpot dili model (Mordred
  descriptors) cut down to 3 of its 250 trees and without its applicability
  domain and ONNX copy, to keep it small. No longer referenced by a test.
- `models/rf-maccs-sklearn-<version>.jmodel`: the same small random forest
  on MACCS keys, saved by scikit-learn 1.2.2, 1.3.2 and 1.9.1 with
  `scripts/make_version_test_model.py`. scikit-learn 1.3 changed the tree
  format, so each job version reads only some of them.
- `models/bare-estimator.pkl`: a scikit-learn estimator without any record of
  its descriptors, which the job must reject.
- `descriptors/cyp2d6-morgan.{smi,manifest.json}`,
  `descriptors/clearance-maccs.{smi,manifest.json}`,
  `descriptors/clearance-maccs.sdf` with
  `descriptors/clearance-maccs-sdf.manifest.json`: precomputed descriptors for
  `candidates-10.sdf` and matching manifests, for the `predict` job tests. Made
  from each model's own featurizer (so the values match), as a descriptor job
  would write them: a tab-separated file (or SDF) with `id`, `smiles` and a
  single `FP` bit-string column (an `FP` field in the SDF). The maccs file
  feeds the morgan model in the mismatch-rejection test.
