# Descriptor jobs and the predictor: design note

Kalev Takkis, 13 November 2026. Status: decisions taken; for build.

This note proposes splitting descriptor calculation out of the predictor into
separate per-library jobs, defines the file contract between them, and lists
what has to be built. It follows the request to continue the predictor's
branch-per-version pattern and to make molecule preparation the user's choice.
The decisions taken on the first round are folded in below and summarised at
the end.

## Update (13 November, later): predictor is now descriptors-only

The predictor no longer calculates descriptors at all. There is **one** `predict`
job and its only molecular input is a descriptor file (+ manifest); the
`--molecules` path, `featurizers.py`'s calculators and `mordred` were removed,
and `rdkit` is kept only to read SDF descriptor files (its version no longer
matters). The predictor is now a thin "apply the model to precomputed features"
job, independent of descriptor-library versions.

Consequence: the predictor can't reproduce a legacy model on its own — a
descriptor job must run first — and the reference-prediction tests that checked
the 22 legacy models against jaqpotpy were removed from this repo; that
correctness check moves to the descriptor jobs' legacy branches (not yet built).
The sections below describe the file contract and checks, which are built and
unchanged by this pivot; the "bundled path stays" framing in them is superseded.

## Goal and principle

One descriptor job should contain **a single version of a single descriptor
library**, nothing else. That avoids the combinatorial explosion of baking
mutually incompatible RDKit / mordred / numpy versions into one container,
which is the problem the predictor already runs into for the legacy models.

A user picks a descriptor job whose library version matches their model,
runs it to produce a descriptor file, then runs a matching predictor job that
reads that file. When Squonk's workflow feature lands, the two steps become
one workflow and the hand-off is wired automatically; until then the user
chains them by hand and the descriptor file carries a manifest so the
predictor can check the pieces fit.

## Where we are

Three repositories, each a Squonk Data Manager job. All are InformaticsMatters'
(the predictor is an early dev repo on the kaliif remote, to be moved into the
org later); work on and push them as needed.

| Repo | Image | Jobs | Library | Python |
| --- | --- | --- | --- | --- |
| `squonk2-predictor` (this) | `kaliif/squonk2-predictor` | predict | sklearn 1.9.1 (+1.2.2 branch) | 3.12 / 3.11 |
| `squonk2-desc-rdkit` | `informaticsmatters/rdkit-descriptors` | rdkit-descriptor-generator, rdkfp-generator, morgan-fingerprint-generator | rdkit ≥ 2026.3.6 | 3.13 |
| `squonk2-desc-mordred` | `informaticsmatters/mordred-descriptors` | mordred-descriptor-generator | mordredcommunity ≥ 2.0.6, rdkit ≥ 2026.3.6 | 3.13 |

The predictor currently computes its own descriptors (Morgan, MACCS, RDKit,
Mordred) from pinned libraries (rdkit 2023.9.6, mordred 1.2.0, numpy 1.26),
and reproduces jaqpotpy's reference predictions for all 22 legacy models.
That bundled path stays; the descriptor-file path is an addition, not a
replacement.

## The reproduction problem

The descriptor jobs as they stand **cannot** reproduce the legacy models.
They run libraries newer than the ones the models were trained on, and those
newer libraries change descriptor values without raising any error:

| Change | What it alters |
| --- | --- |
| rdkit 2024.03 → | `fr_unbrch_alkane` |
| rdkit 2025.09 → | H-bond acceptor counts |
| rdkit 2026.03 → | Mordred information-content indices |
| rdkit 2026.09 → | ring perception (some molecules refused) |
| mordredcommunity vs mordred 1.2.0 | ring perception, counts derived from it |
| numpy 2 vs 1.26 | mordred 1.2.0 won't import at all under numpy 2 |

So feeding a 2026-library descriptor file to a legacy model gives silently
wrong predictions. To reproduce the legacy models through the descriptor-file
path there must be a **legacy-environment branch** of each descriptor job,
pinned to rdkit 2023.9.6, mordred 1.2.0, numpy 1.26 — the same pins the
predictor carries, reusable here. **Decision:** build these legacy branches.
We would prefer models developed in Squonk, but users bring their own, so we
support the legacy versions — within reason; don't go below Python 3.11 (the
predictor's current floor) for an environment.

## Version matching: branches and images

Reuse the predictor's scheme exactly:

- Each library-version generation of a descriptor job lives on its own
  branch, builds its own image under its own tag, and is a selectable job
  version in the Squonk UI.
- A table like the predictor's `src/job_versions.py` records, per branch, the
  library version it runs, so a mismatch can name the version that fits.
- `main` carries the current libraries; a `legacy` branch (name TBD) carries
  the pinned legacy environment.

The predictor and descriptor versions are matched by the user (and later by
the workflow). A model trained on rdkit 2023.9.6 descriptors needs the legacy
descriptor branch **and** a predictor job whose sklearn reads its estimator.

## Molecule preparation (decided)

jaqpotpy does **not** strip salts: it featurises the whole multi-fragment
SMILES, counterion included. This was confirmed empirically — the predictor's
whole-molecule output reproduces jaqpotpy's own reference values exactly,
while largest-fragment diverges (e.g. solubility salt: whole −5.697 = ref,
largest −3.777). 7 of the 22 reference sets contain salts.

We judge jaqpotpy's whole-molecule behaviour to be a **mistake**, not a
convention to honour. So:

- Every descriptor job exposes `--fragment-method` (`hac` / `mw` / `none`) as
  a `jobs.yaml` option (the existing jobs already do).
- **Stripping to the largest fragment (`hac`) is the default.** Keeping the
  whole molecule (`none`) is the deliberate opt-in a user must choose.
- Nothing in a model file records how its molecules were prepared, so the job
  can't infer it — the user has to know their model.

**Consequence to keep in mind.** With `hac` as the default, a legacy model run
through the descriptor-file path will **not** match the stored reference
predictions for the 7 salt-containing sets, because those references carry
jaqpotpy's whole-molecule quirk. Exact reproduction of the legacy models needs
`--fragment-method none`. Two follow-ons:

- The bundled predictor keeps reproducing the legacy models exactly (whole
  molecule), so the two paths disagree on salts by design — the file path is
  the "more correct" one.
- The legacy descriptor branch's reference-value tests must run with `none` to
  compare against the jaqpotpy references; a separate check can record how far
  `hac` moves each salt.

## The file contract

### Descriptor output

Tab-separated text today: SMILES column, ID column, passed-through input
fields, then descriptor columns. Read and written with `im-rdkit-utilities`
(`create_reader` / `create_writer`, id/mol columns and delimiter), supporting
`.smi` and `.sdf`. Two decisions:

1. **Fingerprint bits stay a single bit-string column.** The Morgan job writes
   one `MorganFingerprint` column, and a MACCS job (167 bits) would do the
   same. The descriptor file stays compact; the predictor expands the string
   to per-bit columns (`f0..fN`) only when the model needs them. (The
   predictor is the job that expands — see below.)
2. **Failed molecules get an empty / flagged row**, not a dropped row, so the
   output lines up with the input and it's visible on inspection which
   molecule failed and why.

### Manifest (sidecar JSON next to the descriptor file)

The descriptor job writes a small JSON describing what it produced, so the
predictor can check the file suits the model instead of trusting column names
to line up. Proposed v1 fields:

- `schema_version`
- `featurizer`: name and parameters (e.g. Morgan radius, bits, useBondTypes;
  or the RDKit/Mordred descriptor-set identity)
- `library_versions`: rdkit, mordred, numpy
- `fragment_method`: `hac` / `mw` / `none`
- `missing_value`: the fill used for uncomputable descriptors
- `columns`: the descriptor column names, in order
- `id_column`, `smiles_column`

This is the same information design-question 2 says a model manifest needs;
here it describes the descriptor file rather than the model. The predictor
accepts both job-written manifests and **hand-written ones** (custom models
come with custom manifests), reporting which kind it got. Once workflows wire
the jobs together with matched versions, the manifest's checking role falls
away (the workflow guarantees the match), but the field set is still the right
description of the output.

## Predictor's new feature: load precomputed descriptors

A `--descriptors <file>` mode that takes the descriptor file as the predictor's
**primary input** — it carries the molecule id and the descriptor values per
row (`.smi` / `.csv`), or an `.sdf` with the values in fields, so no separate
structure file is needed. It reads input the same way the descriptor jobs do
(`.smi` / `.sdf` / `.csv` via the shared utilities). The mode:

1. Reads the descriptor file and its manifest.
2. Selects the columns the model's feature list names, by name, never by
   position (the predictor already selects features this way internally).
3. Expands a fingerprint bit string to per-bit columns when the featurizer is
   a fingerprint.
4. Fills missing values as the model was trained (0 for the legacy models).
5. Checks the manifest's featurizer, parameters, library versions and fragment
   method against what the model expects, and **warns or refuses** on a
   mismatch — this is the check the split buys us, and the reason the manifest
   exists.
6. Joins descriptor rows to inputs by id, emitting the usual `error` column for
   molecules with no descriptors (the flagged failure rows from the descriptor
   job).

The model still can't state what it expects in machine-readable form (a bare
`.jmodel` records its feature names but not its library versions or fragment
method), so for now the check is best-effort against what the predictor knows
about each legacy model, plus whatever the future model manifest adds.

## What has to be built

| Work | Where | Size |
| --- | --- | --- |
| Legacy-environment branch (rdkit 2023.9.6, mordred 1.2.0, numpy 1.26, Python ≥ 3.11) | both desc repos | medium; reuse predictor pins |
| MACCS keys job (missing; 9 legacy models use it) | desc-rdkit | small |
| Morgan defaults to match jaqpotpy (radius 2, bits, useBondTypes) via options | desc-rdkit | small |
| Manifest output (sidecar JSON) | both desc repos | small–medium |
| Empty / flagged row for failed molecules instead of dropping | both desc repos | small |
| `--descriptors` primary-input mode + manifest check + bit expansion | predictor (this repo) | medium |
| Reference-value tests (not just line counts), run with `none` on the legacy branch | both desc repos | medium |
| Templating/doc cleanups (bugs noted below) | both desc repos | small |

Known small bugs in the descriptor repos: boolean options templated with
`{% if x is defined %}` fire whenever the option has a value (even `false`),
so true-by-default flags can't be switched off; `missingVal` is used but not
declared as an option; duplicated `molColumn`/`idColumn` titles; "Generate
Calculate" descriptions; a bare `except:` and the DmLog NUL-padding bug in the
mordred job; stray `~` backup files and a commented-out dead job block.

## Decisions taken

1. **Reproduction scope** — build legacy-version descriptor branches so the
   legacy models work through the file path too; keep environments at Python
   3.11 or above.
2. **Bit expansion** — descriptor jobs emit the compact bit string; the
   predictor expands it to per-bit columns.
3. **Failure representation** — empty / flagged row.
4. **Manifest authority** — accept both job-written and hand-written manifests.
5. **MACCS home** — new job in `squonk2-desc-rdkit`.
6. **Molecule prep** — `hac` (strip salts) is the default; `none` is the
   opt-in; jaqpotpy's whole-molecule behaviour is treated as a quirk.
7. **Repos** — all InformaticsMatters'; push as needed; predictor moved to the
   org later.

## Mismatch handling (decided)

Step 5 **refuses** (does not predict) on a mismatch that silently changes the
feature values or their meaning:

- **Library version** — rdkit / mordred / numpy differs from what the model
  needs.
- **Fragment method** — `hac` / `mw` / `none` differs from the model's.
- **Featurizer parameters** — the knobs that change the numbers with molecule
  and library fixed:
  - Morgan: `radius`, `n_bits`, `useBondTypes`, `useChirality`, `useFeatures`.
  - RDKit descriptors: `use_fragment` (includes/excludes the `fr_*` columns),
    `ipc_avg` (changes the `Ipc` value by orders of magnitude).
  - Mordred: `ignore_3D` (includes/excludes 3D descriptors).
  - MACCS: none — a fixed 167-key definition, nothing to mismatch.

It **warns** (and proceeds) on a difference that doesn't change a computable
value, e.g. `missing_value` — the fill used for uncomputable cells, not a
featurizer parameter.

## Manifest v1 schema (as built in the predictor)

The predictor's `--descriptors` mode is built and reads this shape; the
descriptor jobs must emit it.

```json
{
  "schema_version": 1,
  "featurizer": {"name": "morgan", "params": {"radius": 2, "n_bits": 2048,
                 "chiral": false, "bond_types": true, "feature_invariants": false}},
  "columns": ["FP"],
  "id_column": "id",
  "smiles_column": "smiles",
  "fragment_method": "none",
  "library_versions": {"rdkit": "2023.9.6", "numpy": "1.26.4"},
  "missing_value": 0
}
```

- `featurizer.name` is one of `morgan`, `maccs`, `rdkit_descriptors`, `mordred`
  (the predictor's featurizer names); `params` are the value-affecting knobs,
  checked strictly against the model. For `rdkit_descriptors` the params are
  `use_fragment`, `ipc_avg`; for `mordred`, `ignore_3D`; `maccs` has none.
- `columns`: for a fingerprint, the single bit-string column (any name, e.g.
  `FP` or `MorganFingerprint`); for descriptors, every descriptor column name,
  which must cover the model's feature list.
- `id_column` (required) and `smiles_column` (optional) name those columns;
  for an SDF, `id_column` may be `_Name` and the descriptors are read from
  fields.
- `fragment_method`, `library_versions` and `missing_value` are reported and,
  where the model records an expectation, checked — a bare legacy model records
  none, so they only warn.

### How the predictor reads it

- Fingerprints: the bit-string column is expanded to the per-bit columns the
  model names; a wrong-length or non-`01` string fails that row.
- Descriptors: columns are selected by the model's feature names; empty cells
  are filled with the model's training fill (`0` for the legacy models); a row
  whose descriptor columns are all empty is a failed row from the descriptor
  job and comes out with an `error`.
- The manifest defaults to a sidecar next to the descriptor file
  (`foo.smi` → `foo.manifest.json`); `--manifest` overrides it.

Still open: whether the descriptor text files should carry an explicit `error`
column for failed rows (currently inferred from empty descriptor cells), and
the delimiter — the predictor takes it from `--delimiter`, not the manifest.
