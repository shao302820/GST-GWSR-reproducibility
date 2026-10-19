# Release verification status

The current snapshot contains only the current experiment: nine k candidates, four normalized GST thresholds (20 alpha/tau settings), and the locked k55/alpha.75/tau.5 outer205 comparison, plus the eight retained external baselines used in that comparison.

Recorded artifacts were copied byte-for-byte where possible. results/source_provenance.json records source SHA-256 fingerprints and copied artifact digests. Curated selection/audit metadata use repository-relative locations. MANIFEST.sha256 covers the released repository files.

scripts/verify_results.py independently checks prediction alignment, three-seed means, original-scale metrics, selected parameters, histories,5/20 budgets and the independent k55 reference. tests/ also checks archive invariants and portable preparation/planning against recorded arrays. This packaging process does not rerun the full scientific searches. Fresh-run commands are provided separately in README.md.

The eight external baseline implementations are retained with their original tuning grids. Stochastic baselines use seeds42,123,456; OLS,SVR,GWR,MGWR use deterministic single fits. Their recorded environment differs from the GP environment; see results/external_baselines/protocol.json.
