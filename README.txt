V90 joint learned structure model — training release
2026-11-24

This release trains the temporal backbone and differentiable structural refinement as one network. It is a true two-channel existing/new classifier, not a relabeling of an old three-channel head.

Architecture:
Frozen root-region/temporal-prior features -> frame CNN + ConvLSTM + gated GAT -> learned identity-weighted Axis refinement -> learned soft-matching Junction refinement -> two-class logits.
Identity, axis, junction and root-state supervision is included during training. Classification gradients reach the structural modules. No fixed legacy Axis/Junction class-repainting rules are used.

Scope and interpretation:
- Upstream U-Net segmentation and V66 priors are frozen and reused. This is not raw-RGB training of the entire upstream stack.
- The dataset is B3 internal, foreground-conditioned, and the frozen prior has historical B3 exposure. Scores do not establish independent cross-tube generalization.
- Existing means persistent + elongating portions of an existing root axis; new means a newly appearing root axis. The output head has two actual channels.
- All 12 runs use 100 full-graph optimizer updates each, at most 8192 sampled training root pixels/update. These are not 100 full-pixel dataset epochs.
- Three random seeds describe optimization variability, not biological replication.
- The neural baseline retains common auxiliary identity/pair supervision but disables structural refinement and the corresponding extra consistency terms. Ablations remove the specified refinement plus its extra loss, rather than all identity or all junction supervision.
- Internal prediction PNGs cover eligible typed pixels only. They are evaluation masks, not complete full-field segmentation products. Codes: 0 outside eligible support, 1 existing, 2 new.
- The 18-position / 66-image preparation and prediction task is paused at the user's request. Existing partial files are preserved locally and excluded from this release. Do not automatically resume batch scripts. No independent accuracy is reported.

Archive layout (extract archives into the same NEW directory):
1. Code_and_selected_model.zip: code/, selected_model/, requirements, protocol and core checks.
2. B3_training_inputs.zip: data/ and original/new input audits. These are the prepared B3 training features/labels, not the whole original image collection.
3. Runs_joint_full.zip and the three comparison archives: experiments/<variant>/ and matching logs, including checkpoints, optimizer resume states, histories, result JSON and internal masks.
4. Reports_and_figures.zip: CSV/XLSX metrics, editable Word report, checks, vector PDF/SVG and 600-dpi graphics.

Train/reproduce (from the extracted common root):
python -m pip install -r requirements-training.txt
python code/check_implementation.py
python code/train.py --variant joint_full --seed 20261124 --updates 100
python code/run_study.py

The study driver selects the full-model checkpoint after training. For separately run seeds, use python code/select_model.py after all three full-model runs complete.

Training automatically resumes an unfinished run with the identical config; completed study runs are skipped by run_study.py. To start a fresh independent reproduction, extract the code and input archives into another empty directory and omit the runs archives. Only validation macro Dice chooses checkpoints. No test-based model or seed selection is permitted.

Reports:
python -m pip install -r requirements-report.txt
python code/summarize.py
python code/identity_diagnostics.py
python code/figures.py --fonts /path/to/your/Times-New-Roman-fonts
python code/write_report.py

Font files are not redistributed. PDF fonts are embedded; SVG text remains editable. Replotting uses Times New Roman; training does not require fonts. See reports/Environment.json for the exact CPU/PyTorch environment and conditional CPU bfloat16 behavior.

selected_model/best.pt is the full architecture checkpoint selected among the three predeclared full-model seeds by validation macro Dice only. It does not assert that the full architecture outperforms all controls. See Mean_SD_metrics.csv and Paired_effects.csv for the actual outcome.

Raw-RGB deployment and multi-position prediction remain outside this paused delivery stage. The optional predict.py consumes a label-free prepared graph folder; it does not by itself recreate the frozen raw-image preprocessing stack. New image accuracy cannot be guaranteed without appropriate external human labels.
