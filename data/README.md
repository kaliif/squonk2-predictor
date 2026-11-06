# Data
Test data for the job tests in `data-manager/jobs.yaml`.

- `10.smi`, `10-error.smi`, `candidates-10.sdf`, `candidates-10-error.sdf`:
  molecules to predict, the same as in the squonk2-jaqpot job tests. The
  `-error` files have one invalid molecule.
- `models/CYP2D6_Veith.jmodel`, `models/clearance_hepatocyte_az.jmodel`:
  jaqpotpy 1.0.104 models from squonk2-jaqpot, with reference predictions
  (`.reference.csv`, used by the pytest tests) made by
  `scripts/make_reference_predictions.py` from the TDC ADMET test sets.
- `models/dili-3trees.jmodel`: the squonk2-jaqpot dili model (Mordred
  descriptors) cut down to 3 of its 250 trees and without its applicability
  domain and ONNX copy, to keep it small. Its reference predictions were made
  with jaqpotpy for the cut-down model.
- `models/rf-maccs-sklearn-<version>.jmodel`: the same small random forest
  on MACCS keys, saved by scikit-learn 1.2.2, 1.3.2 and 1.9.1 with
  `scripts/make_version_test_model.py`. scikit-learn 1.3 changed the tree
  format, so each job version reads only some of them.
- `models/bare-estimator.pkl`: a scikit-learn estimator without any record of
  its descriptors, which the job must reject.
