# 0353-2-B3 native-pixel refinement v6

Three-model root-length supplement: `three_model_regression.py` reads the unchanged v6 masks and length pairs, verifies the skeleton lengths and existing agreement statistics, and exports U-Net/DeepLabV3+/SegFormer-B0 regression and Bland–Altman figures. The archive carries a minimal `run_snapshot` for exact reproduction. Agreement R² and OLS regression R² are explicitly distinct. No model fitting, image selection or predictions are changed by this figure supplement.

RGB-only root/background segmentation; temporal labels and death are not trained. Original RGB and human references remain unchanged. This is a single-position exploratory experiment; C4 is a repeatedly inspected spatial benchmark, not a fresh independent test. The actual delivered validation/repeated-C4 Dice values are 0.7539/0.6227, below the established 0.95 gate.

The four downloadable archives contain publication-format figures, native RGB/root products and isolated-column probabilities, model weights/scripts, and the lossless 384-pixel training export. Source annotations are preserved in full. Exported patches number120/40/40 (train/validation/repeated benchmark), from30/10/10 spatial windows×4 dates; overlapping patches are not biological replicates.

## Environment

Python3.12.14. Install `requirements.txt` in a virtual environment. PyTorch must be installed with the CPU/CUDA wheel appropriate to the machine; scripts automatically detect CUDA. The recorded run used CPU,3 threads. This stage does not require TensorFlow: the previously trained original TF1.12 DeepLabV3+ is retained as frozen validation/test score arrays. Its full original weights, source and runtime are in the immutable [v5 model archive](https://raw.githubusercontent.com/Annalee811/LI/e068b2d6d503135da6271a52303fca3826260887/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/root-background-stage-v5/0353-2-B3_root_background_v5_models.zip).

The new U-Net ImageNet initialization is the official [ResNet18 checkpoint](https://download.pytorch.org/models/resnet18-f37072fd.pth), SHA256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`. This initialization is only needed to repeat training, not for inference with saved weights. Keep TLS verification enabled. No credentials or private annotation uploads are required by these scripts.

## Repeat analysis from delivered weights

Extract the data and model ZIPs. Set DATA to the extracted directory containing `dataset/upper` and `original_human_progress.json`; set RUN to the model directory containing `unet_pretrained` and `legacy`. Run the commands below from the extracted model directory. The sibling source helper layout must be kept intact.

```bash
DATA=/absolute/path/to/0353-2-B3_native384_training_data_v6
RUN=/absolute/path/to/0353-2-B3_refined_models_v6
SRC="$RUN/source/paper_v6"
python "$SRC/verify.py" --data "$DATA" --output "$RUN"
python "$SRC/prepare_baselines.py" --data "$DATA" --legacy "$RUN/legacy" --output "$RUN"
python "$SRC/select_refinement.py" --data "$DATA" --run "$RUN"
python "$SRC/choose_delivery.py" --data "$DATA" --run "$RUN"
python "$SRC/evaluate_refinement.py" --data "$DATA" --legacy "$RUN/legacy" --run "$RUN" --selection delivery_selection.json --threads 3
python "$SRC/figures.py" --data "$DATA" --run "$RUN" --selection delivery_selection.json
```

These commands regenerate all baseline comparisons as well as the delivered model; they may take minutes on CPU. Use a copy of the extracted model directory if preserving the exact published JSON bytes: paths/provenance in regenerated metadata may differ. GPU floating-point results can differ from this CPU run. Frozen TF baseline scores, rather than new TF training, are explicitly used.

## Repeat new training

```bash
python "$SRC/train.py" --data "$DATA" --output /absolute/path/to/new_run --initial /absolute/path/to/resnet18-f37072fd.pth --epochs 75 --threads 3
```

The published run stopped at60 epochs and selected epoch35. This command trains the new pretrained U-Net only; it does not claim to retrain the three prior architectures. Selection used C3: threshold calibration, native384 context, four flips, fixed equal-weight ensembles, and a thin-root-preserving compact-component grid. The final delivery filter was added after initial ranking, from the existing C3 grid, and deliberately trades a small Recall/Dice loss for Precision. Both decisions, every tested configuration and the exact noise audit are retained. C4 was not used to choose this filter, but had already been inspected; it cannot support a new independent-performance claim.

`finalize.py` is the repository delivery tool; it requires the original v5 run, cached v6 outputs and publication directory. It verifies saved PNG confusion counts, RGB identity, RGBA foreground colors and archive CRCs, then writes reports and four ZIPs. `prepare_data.py` regenerates the native export from the v5 upper RGB/masks with no resizing or label edits.

## Interpretation

Foreground Precision/Recall/strict pixel Dice/IoU are primary. Boundary F1 at1px and clDice are separate diagnostics. Length agreement compares skeletons derived from the brush reference with predicted skeletons in pixels; no independent manual lengths or physical scale exist. The32 block–date pairs are correlated. Unpainted upper pixels are background under the user's completeness declaration; original full progress and temporal attributes remain intact. Whole-upper-field images include training pixels and are qualitative; reported metrics use isolated448-pixel C3/C4 columns. Removed small objects are candidates, not independently adjudicated noise.600dpi is the figure export setting, not a claim of additional source-image information or publication-level predictive validity.
