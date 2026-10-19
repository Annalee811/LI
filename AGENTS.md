# Current dataset identity

Read `dataset/current/identity_policy.json` and `dataset/current/README.md` before new dataset indexing, model training, splitting, batch prediction, plots, or reports.

- Canonical ID: `Q1-1-Z2`; `0353-1-Z2` is a legacy alias of the same four images. Its tube group is `Q1-1`.
- Exclude `Q1-3-Z3` from current analyses, paper figures, new model fitting/evaluation, and prediction batches. Preserve old files as history.
- Use `analysis-scripts/dataset_identity.py` and the active manifest. Keep every date at the same position in the same data partition; do not treat tiles/dates as independent tubes.
- Do not alter old model weights, labels, or quantitative results for this identity correction. Renaming does not resolve the coarse/unverified registration link at `Q1-1-Z2`.
