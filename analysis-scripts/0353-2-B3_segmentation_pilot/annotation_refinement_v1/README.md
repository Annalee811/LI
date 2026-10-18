# 0353-2-B3: revised root references, 200 training views, three-model fine-tuning

Only the annotated upper four-period RGB field is used (2541×1108px). Original RGB pixels, human stroke JSON and original validation/test masks are retained exactly. Original/revised references are separate versions. No temporal or death classifier is trained in this experiment.

The previous36base256px tiles have a fixed random25/5/6train/validation/test allocation. All four dates at a position share the same split. Each training position receives a second native256px view translated12px in x/y, inward at image boundaries. There are200training views,20validation and24internal-test images. All extra windows explicitly exclude every held-out pixel. The two training views overlap;200views are not200independent acquisitions. No easy or dense-root tile filtering is applied.

The previous U-Net is used together with bright tubular RGB support to supplement narrow/missing supplied root strokes. Small gaps require both model/image support. High-confidence model structures far from original strokes additionally require elongated shape. Original root strokes are retained; uncertain differences are recorded rather than assigned invented root anatomy. See `revision/provenance.json` and the pixel-addition log. These references are **model-assisted revisions, not independent human truth**. Width changes and probable fine-root additions are not guaranteed to be correct pixel by pixel.

The full-field revised masks are qualitative illustrations. Training labels are generated independently INSIDE each training window using its own RGB/model context; local normalization and border context can therefore differ from simply cropping a full-field revision. The exact supervised masks are the delivered `data/patches/train/mask` files and their hashes in the manifest. The full-field display is not silently substituted as a different training target.

Only TRAINING masks are revised. ORIGINAL validation masks select checkpoints/thresholds; ORIGINAL test masks provide the primary metrics. Separate revised-reference scores quantify agreement with the teacher-influenced reference and must not be promoted as independent accuracy. All matching previous networks are fine-tuned for20epochs with64augmented draws per epoch, batch2, initial learning rate0.00008, hard-negativeBCE+Dice+0.15softclDice. Epoch0can be retained if validation does not improve. All3models see the same actual training-view sequence. Training is not repeated until a desired test Dice is obtained.

## Use delivered packages

Extract the ZIPs into one parent folder. They share `0353-2-B3_final`. Enter that folder. The training-data ZIP contains the exact revised200training masks and immutable20/24held-out references. Model ZIPs include actual selected weights, training histories, thresholds and scripts. Evaluation ZIPs provide native probabilities and full-upper/native test outputs. The revised-annotation ZIP contains original RGBs, original masks, revised masks, RGB overlays, added-pixel masks and unresolved-disagreement masks. No Word documents are generated.

Use a separate copied run directory for new experiments: the commands below overwrite that named experiment's own model outputs. Never overwrite a previous version or a published checkpoint. The uploaded original manual JSON retains temporal attributes; revised binary masks do not invent new/persistent/extension classes.

## Reproduce training

Host Python3.12 dependencies are pinned in `requirements.txt`. CUDA detection is automatic; the delivered run used CPU. Use the active environment interpreter (in the cloud checkout, `.venv/bin/python`). First restore the immutable STARTING checkpoints, not the newly trained result weights:

```sh
python analysis-scripts/0353-2-B3_segmentation_pilot/annotation_refinement_v1/fetch_starting_assets.py --root .
python analysis-scripts/0353-2-B3_segmentation_pilot/annotation_refinement_v1/train_torch.py --run results/B3_annotation_refinement_v1 --model unet --epochs 20 --threads 1
python analysis-scripts/0353-2-B3_segmentation_pilot/annotation_refinement_v1/train_torch.py --run results/B3_annotation_refinement_v1 --model segformer --epochs 20 --threads 1
```

The original TensorFlow1.12ResNet-v2-50DeepLabV3+model is used without architectural substitution. The vendor and LICENSE are included in its model ZIP. The SHA-pinned runtime builder can reconstruct the CPU Docker image:

```sh
python runtime/prepare_deeplab_runtime.py --context tf112_cpu_build --build
docker run --rm --tmpfs /tmp:rw,size=128m --tmpfs /ram-python:rw,exec,size=1300m --mount type=bind,src="$PWD",dst=/work --workdir /work -e OMP_NUM_THREADS=2 -e PYTHONDONTWRITEBYTECODE=1 b3-tf112-mkl:local sh analysis-scripts/0353-2-B3_segmentation_pilot/annotation_refinement_v1/container_entry.sh --run results/B3_annotation_refinement_v1 --vendor results/B3_uploaded_deeplab_v1/vendor --epochs 20 --threads 2
```

The disposable container root is cleaned of unused compiler/header files and its Python libraries relocated into an executable memory filesystem; `/work` is never removed. It needs1.3GBfreeRAM plus model memory and enough disk for Docker's initial root copy. Preserve session proxy/CA trust in managed-cloud networked builds. Do not run large figure exports concurrently with container startup on a nearly full disk.

## Final revised-reference figures and metrics

With all3evaluation packages, training data and model packages extracted, prepare the final-model-only report and recreate its outputs:

```sh
python analysis-scripts/0353-2-B3_segmentation_pilot/annotation_refinement_v1/make_revised_report.py --run results/B3_annotation_refinement_v1
python analysis-scripts/0353-2-B3_segmentation_pilot/annotation_refinement_v1/analyze.py --run results/B3_annotation_refinement_v1/revised_report
python analysis-scripts/0353-2-B3_segmentation_pilot/annotation_refinement_v1/comparison.py --run results/B3_annotation_refinement_v1/revised_report
python analysis-scripts/0353-2-B3_segmentation_pilot/annotation_refinement_v1/figures.py --run results/B3_annotation_refinement_v1/revised_report
python analysis-scripts/0353-2-B3_segmentation_pilot/annotation_refinement_v1/report.py --run results/B3_annotation_refinement_v1/revised_report
```

All final comparison figures show the REVISED reference and only this round's final trained models. S2/S3/S4/S5 and the headline final-report tables quantify model-assisted revised-reference agreement, NOT independent human accuracy. Actual checkpoint/threshold selection still used original validation labels. S1 retains those real original-validation monitoring values instead of inventing revised-label historical training curves. The original retained-reference metrics are preserved separately in `original_holdout_metrics.csv` for audit.

The helper creates a separate `revised_report`; it never overwrites original training outputs. Full-field revised masks are qualitative, while exact supervised train masks are the independently prepared native tile files. Revised test references have not been used to select weights or thresholds. Whole-upper views include training pixels. Green in binary comparison figures means all roots, not newborn roots.

For optional complete reference regeneration, `fetch_starting_assets.py --with-original-data` retrieves SHA-verified original archives, then use `refine.py --source results/B3_random_tiles_v1 --run <new_run>` and `expand200.py --run <new_run>`. The delivered labels are already prepared; this is optional and refuses to overwrite an existing reference version.

Independent checkpoint reload checks: `verify_torch.py --run ...`; for TensorFlow use the same Docker invocation with `verify --run ... --vendor ...` after `container_entry.sh`. The final `verify.py` checks source hashes, fixed original held-out masks, added native RGB windows, zero cross-split pixel overlap, training draw equality and actual checkpoint outputs. `package.py` writes ordinary GitHub ZIPs below99MiB and tests ZIP CRCs. Public download hashes are verified separately.

600dpiTIFF/PNG and PDF are provided; quantitative charts include editableSVG. Main final-report figures: Fig2 revised-reference metrics, Fig3 four-date RGB/revised-reference/three-model comparison, Fig4 three-model length agreement. S1 training curves, S2 per-date Dice, S3 length deviations, S4 test-tile heterogeneity and S5 reference-relative errors. S5greenTP,pinkFP,orangeFN are defined against revised reference masks. Native full-upper/local background-removed RGB and black-root masks are also included. Original human JSON and native RGBs remain in the training-data archive. Plant Phenomics notes are Markdown, never Word. Independent external test images:0; do not claim independent manual or cross-tube accuracy from teacher-influenced reference agreement.

For later RGB prediction use `predict_torch_RGB.py --model unet --model-dir <selected_model_dir> --input <RGB_path> --output <new_output>` (or `--model segformer`). DeepLab uses `predict-rgb --model-dir ... --vendor ... --input ... --output ...` after the Docker entry script. Native dimensions and foreground RGB pixels are retained; confidence thresholds are validation-frozen. No new-image accuracy is inferred without external labels.
