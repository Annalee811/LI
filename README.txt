Joint ConvLSTM + GAT ablation and compact ABC figure
Date: 2026-11-23

The original study and original Full checkpoints remain unchanged.
Three actual joint-ablation runs were added in experiments/no_temporal_relation.
The prior single-component LSTM/GAT runs are preserved in the earlier release but are not shown as separate components in the revised compact plot.

Figure files: figure/V90_joint_module_ABC.{jpg,png,tif,pdf,svg}
Resolution: 190 x 100 mm, 600 dpi; Times New Roman, black text, white background.
Panel A: two-class Dice; B: joint-module and decoder effects; C: same-field T1-T4 RGB predictions.
Independent accuracy remains unavailable without human temporal ground truth.

Replot from the supplied CSVs and RGB/mask cache:
python figure/make_ABC_joint_module.py --fonts /path/to/fonts
Requires numpy, matplotlib, Times New Roman.ttf and Times New Roman Bold.ttf.
Font files are not redistributed.

code/study_model.py and code/train_study.py are the exact code used for the new training runs.
Training inputs are reused from /workspace/work/V90_validation_20261123/data; data is linked locally and is not duplicated in this figure/experiment supplement.
Original data, complete Full comparators and the broader study are available in the prior release referenced in Joint_module_protocol.json and the source release below. The training script retains original project asset paths, so this supplement alone is not a standalone raw-RGB deployment bundle.

Source figure release: https://github.com/Annalee811/LI/tree/c91f8c354a129338b69efb6d691d5a72fc726324
Source two-class study release: https://github.com/Annalee811/LI/tree/3839df9feaa49ecd30ffeec5f1d6593d205686d5
Source complete training study: https://github.com/Annalee811/LI/tree/4a4e61a9e14ece88f6c8ac5bb3bc044a9170a01e

Each new run includes weights, optimizer resume state, history, configuration and predictions. Reports verify source checkpoint preservation and confusion matrices against saved predictions. See English caption and Chinese notes for evaluation boundaries and limitations.
