# Repository publication

The current default-branch snapshot contains the current complete experiment. Update source,configs,recorded results,figures and documentation together. Preserve the source provider attribution and data license terms.

Run scripts/verify_results.py and the tests, then regenerate MANIFEST.sha256 with scripts/build_manifest.py before committing. A new scientific run should use a fresh run_outputs directory and must not overwrite the archived results.

The manuscript itself is maintained separately; this repository contains experiment materials and does not include the unpublished manuscript document.
