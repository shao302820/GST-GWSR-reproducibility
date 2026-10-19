# Raw and source-derived data

This repository contains the source files needed to inspect the data path:

- `source_products/`: IHFC metadata, the clipped heat-flow source table, GEBCO, WGM2012, GlobSed, CAM2016Litho, LITHO1.0, Curie-depth, GVP, EMAG2, coastline, and source metadata snapshots.
- `predictor_grids/`: the eight harmonized 0.25-degree predictor tables used by the final model.
- `heat_flow_source/`: the original station table used as the local experiment input.

Large raster and grid files are tracked with Git LFS. Run `git lfs pull` after cloning to retrieve them.
