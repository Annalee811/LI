# V68: GAT optimization for new roots and tip extension

Temporal targets are uploaded18 only. The user explicitly authorized only the142
user_defined root IDs of upload17 as connection targets; assistant proposals and
unassigned handles are ignored. Identity names, hierarchy, colors and user cuts
are never neural inputs or candidate graph construction rules. No previous GAT
weights or optimizer state is loaded.

Binary contours, widths and lengths remain exactly fixed. The completed V66 heads
are the current frozen baseline. Each new GAT/MLP correction starts with zero
output residual, exactly reproducing that baseline at initialization. This is
new GAT refinement training, not a new from-scratch binary CNN experiment.

Each predicted skeleton is cut at crossings/branches into12px computational
chunks. Candidate graph edges use predicted geometry only. One edge head learns
same-root links from confirmed17 IDs; another learns class boundaries from18.
Same root does not imply same class: true persistent-to-extension transitions
are preserved as separate supervised targets. Pixel-wise residuals can change
classes within chunks. No forced same-color connection, hard class rule, root
dilation or pixel synthesis is used.

Six new modules: three GAT refiners and three same-input MLP controls,100 real
epochs each. Minority sampling, weighted focal CE and new/extension-focused Dice
are shared by GAT and control. Validation-only selection emphasizes new and tip
classes; all gains/losses and persistent-class changes will be reported.

All four dates remain a single outer development position. Fixed spatial masks
are inner train/val/holdout masks, not independent positions/tubes. Connection
supervision uses pixels only within the associated supervision mask. Graph
features are transductive and the frozen CNN has same-field exposure; the saved
scores are internal reference agreement, not independent generalization.

Resume after an interruption only with this run's source/run_all.py, after
ensuring no orchestrator is running. Atomic checkpoints retain model/optimizer/
scheduler/Torch RNG and every real epoch. STOPPED_BY_USER.json blocks recovery.
