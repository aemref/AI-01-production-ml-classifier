# Model Artifact and Rollback Policy

The inference service loads a checked-in, versioned JSON artifact instead of
training a model when the process starts. The current artifact is
`artifacts/logistic-regression-98b12889accb.json`; its version suffix is the
first 12 characters of the complete source-dataset SHA-256 digest.

## Build and verify

Build a new candidate from the validated dataset:

```bash
python -m src.build_artifact
```

The builder fits the validation-selected, class-balanced logistic regression on
the training partition only. It records the feature and class contracts,
training split settings, scaler parameters, classifier parameters, dataset
identity, and row counts. Output is deterministic, reviewable JSON accompanied
by a SHA-256 sidecar.

Verify the published artifact before testing or packaging it:

```bash
python -m src.verify_artifact
```

Verification fails if the artifact or sidecar is missing, bytes have changed,
the schema/model/features/classes are incompatible, or the checked-in source
dataset no longer matches the recorded checksum and row count. CI performs this
check before building the container, and the API repeats artifact integrity and
contract validation at startup.

The sidecar detects accidental or uncoordinated modification; it is not a
digital signature because an attacker with repository write access could alter
both files. A production release would require signed provenance and a trusted
verification key outside the image.

## Release procedure

1. Run the three-model comparison and confirm logistic regression remains the
   validation-selected model without consulting test results for selection.
2. Build a new artifact. Keep its dataset-derived filename; do not overwrite a
   previous version.
3. Update `DEFAULT_ARTIFACT_PATH` in `src/predictor.py` and review the JSON diff.
4. Run the full test, verification, benchmark, and container smoke suite.
5. Record the artifact version and evidence in release notes before tagging.

## Rollback procedure

Keep prior artifact JSON and checksum pairs in version control. To roll back,
change `DEFAULT_ARTIFACT_PATH` to the last verified version in a normal commit,
run the same full verification suite, and rebuild the image. Do not overwrite
the old model file, rewrite Git history, or claim rollback success until both
`/health` and `/predict` report the restored model version in the rebuilt
container.
