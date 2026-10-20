# Preserved exploratory stage1 for the V18 root-identity refinement

This source snapshot and `results/B3_temporal_root_identity_v18_stage1` retain the actual32-epoch first pass. The final published V18uses a further16-epoch refinement with lane-aware junction costs and a more conservative final reference. This stage1has no lateral junction-position term; its exact revision counts are recorded in revision_summary.json and revision_log.json. Original human JSON is unchanged. It is retained for reproducibility, not presented as the final result.

For a new stage1directory, use this folder's prepare_teacher.py,prepare.py,setup_routes.py,run_all.py with the delivered V17teacher/root inputs and V14retained DeepLab/SegFormer starting heads. This folder's defaults run32epochs; the final folder defaults to16. Actual histories and hashes are in the stage1artifact archive.
