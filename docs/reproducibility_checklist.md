# Current reproducibility checklist

1. Retrieve Git LFS source products if rebuilding input.
2. Validate the exact station table and its SHA-256.
3. Install the recorded GP environment.
4. Run scripts/verify_results.py and the tests.
5. For a fresh experiment, run scripts/run_experiment.py all into an empty output directory.
6. Verify nine k candidates select55 and20 GST candidates select(.75,.5), using pointwise original-scale means before pooled OOF metrics.
7. Compare205-station ensembles, individual seeds, expressions, generation histories and branch counts.
8. Reproduce external baselines in their separate recorded environment if desired.
9. Interpret runtime as descriptive completed-batch timing, including parent/archive cost.
