# Current data preparation

The current deterministic builder is `scripts/prepare_dataset.py`. It uses the bundled clipped heat-flow source table and harmonized eight-predictor grids to construct the1,024-row model input. See `docs/data_processing_workflow.md` for the command and source provenance.

The authoritative input remains `data/processed/heat_flow_stations.csv`, SHA-256 `b28b789a818500f718f81cbccc2b6a68a72c1a164c9d536016ff099a8583fe05`. The experiment entry points use that table directly. Exploratory feature-screening and alternative-model programs are excluded from this current release.
