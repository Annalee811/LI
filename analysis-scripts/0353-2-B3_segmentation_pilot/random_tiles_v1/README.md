# Random native tiles, four-period grouping, three fresh models

Only0353-2-B3's human-annotated upper RGB field(2541×1108px) is used. No labels or colors are changed. No death or temporal-class model is fitted here.

Direct256×256 native tiling produces36 tile positions with small gaps. A single `numpy.random.default_rng(20261018).permutation(36)` assigns first25 to training,next5 to validation,last6 to test. Four dates at a tile position always remain together:100/20/24 RGB images. No larger C-regions, density selection, easy-tile filtering, overlapping train/test windows or seed search. This is internal same-position testing, not independent tube validation.

All networks are newly randomly initialized. Do not initialize a new random split with the old B3-trained weights. U-Net uses ResNet18/native RGB skip; SegFormer-B0 has a native RGB residual refiner; DeepLabV3+ uses the supplied original ResNet-v2-50/ASPP/decoder in TensorFlow1.12. Pretraining is absent in all three. Framework optimizers and precision differ and are disclosed.

Extract the training-data and model ZIPs into the same parent folder. They share the top-level folder `0353-2-B3_random_tiles_v1`. Enter that folder; the original relative command paths below are preserved. Existing packages already contain the exact fixed allocation, unchanged inputs and actual trained weights; regeneration is optional.

## Host Python3.12

Install the pinned `requirements.txt` into a virtual environment. Set `PYTHON=path/to/venv/bin/python` in your shell if needed; below `python` means that interpreter.

```sh
python analysis-scripts/0353-2-B3_segmentation_pilot/random_tiles_v1/train_torch.py --run results/B3_random_tiles_v1 --model unet --epochs 50 --threads 3
python analysis-scripts/0353-2-B3_segmentation_pilot/random_tiles_v1/train_torch.py --run results/B3_random_tiles_v1 --model segformer --epochs 50 --threads 1
```

These overwrite the named experiment's model output, so use a copied run directory for new experiments. For whole-upper prediction with existing weights, add `--predict-only`; the saved validation-frozen threshold is used. CUDA detection is automatic. Host training used CPU BF16 autocast; scores and saved probabilities are FP32. All50 epochs use64 augmented draws(batch2), iterating a shuffled queue over all100 training images.

## Original uploaded TensorFlow1.12 model

The DeepLab model package includes the original vendor source and LICENSE, a SHA-pinned Intel TensorFlow1.12 runtime builder in `runtime/prepare_deeplab_runtime.py`, and the adapted source. The only vendor adaptation guards optional pretrained loading; network architecture is unchanged.

```sh
python runtime/prepare_deeplab_runtime.py --context tf112_cpu_build --build
docker run --rm --tmpfs /tmp:rw,size=128m --tmpfs /ram-python:rw,exec,size=1300m --mount type=bind,src="$PWD",dst=/work --workdir /work -e OMP_NUM_THREADS=3 -e PYTHONDONTWRITEBYTECODE=1 b3-tf112-mkl:local sh analysis-scripts/0353-2-B3_segmentation_pilot/random_tiles_v1/container_entry.sh --run results/B3_random_tiles_v1 --vendor results/B3_uploaded_deeplab_v1/vendor --epochs 50 --threads 3
```

`container_entry.sh` modifies only the disposable container root, removes unused compiler/header files, and relocates Python libraries to an executable memory filesystem. It never removes `/work` files. This avoids disk exhaustion with Docker's vfs storage driver and requires at least1.3GB available RAM for the library mount plus model memory. The executable mount is required for loading NumPy/TensorFlow shared libraries. Networked builds/downloads must preserve the environment's proxy, CA trust and checksum verification. The actual cloud run used the already verified `b3-tf112-mkl:local` image; no package/TLS checks were bypassed.

To predict with saved DeepLab weights, append `--predict-only` to the same command. Delivered FP16 NPZ variables are restored to the FP32 original graph before threshold calibration, scoring and inference. Original weight names and SHA256s are saved.

## Recompute evaluation and publication figures

Extract all three evaluation ZIPs too, into the same parent folder. The true training-run probability and selection paths are preserved automatically under `results/B3_random_tiles_v1/<model>`, so no manual file copying is needed. These ZIPs also offer concise native output folders for viewing. Run:

All evaluation, comparison and publication export steps can be executed together with `python analysis-scripts/0353-2-B3_segmentation_pilot/random_tiles_v1/run_analysis.py --run results/B3_random_tiles_v1`. Individual commands follow:

```sh
python analysis-scripts/0353-2-B3_segmentation_pilot/random_tiles_v1/analyze.py --run results/B3_random_tiles_v1
python analysis-scripts/0353-2-B3_segmentation_pilot/random_tiles_v1/comparison.py --run results/B3_random_tiles_v1
python analysis-scripts/0353-2-B3_segmentation_pilot/random_tiles_v1/large_local_comparison.py --run results/B3_random_tiles_v1
python analysis-scripts/0353-2-B3_segmentation_pilot/random_tiles_v1/figures.py --run results/B3_random_tiles_v1
python analysis-scripts/0353-2-B3_segmentation_pilot/random_tiles_v1/report.py --run results/B3_random_tiles_v1
python analysis-scripts/0353-2-B3_segmentation_pilot/random_tiles_v1/verify_torch.py --run results/B3_random_tiles_v1
```

Root/background masks are thresholded directly; no test-selected dilation, bridging, deletion or tolerance-inflated Dice. Headline scores sum native valid-pixel confusion counts over held-out tile-date images. Model selection uses validation only. Validation and test are not averaged. Undefined empty-tile ratios are null. Root-length agreement compares skeleton-derived lengths from unchanged masks in pixel units; it is not independent manual measurement. Every test tile is delivered, including difficult tiles. Whole-upper output includes training pixels and is qualitative only.

RGB comparison sheets keep real soil/droplet backgrounds and overlay actual human/model root masks in the same green. Additional full-upper and local comparisons remove the background to white, offering original foreground RGB and black-root mask versions. No display dilation or smoothing is used. All four dates are included at matched coordinates. Figure explanations are supplied in chat, not Word documents.

An additional768×768 native central field is supplied for easier visual inspection. Its fixed geometric center is not chosen for model performance. It includes fitted pixels and is visibly marked QUALITATIVE; it is not included in the held-out test metrics.

## Optional dataset regeneration

`prepare.py --source results/B3_random_tiles_v1/data --run <new_run>` uses the exact delivered original human JSON and four upper RGB/mask files, verifying the pinned annotation SHA256. The delivered training ZIP already contains those unchanged inputs and all lossless tiles; their file hashes, coordinates, roles and counts are in `data/manifest.csv` and `data/summary.json`. Augmentation is performed after splitting. Extra tiles are not extra independent biological samples. No independent external test images are available.

## Later inference on additional positions

For U-Net or SegFormer, `predict_torch_RGB.py --model unet --model-dir results/B3_random_tiles_v1/unet --input <RGB_file_or_folder> --output <new_folder>` uses the saved, validated threshold without adaptation to the new images. Native image dimensions and RGB values are retained; separate root masks, probabilities, RGB root-only images and RGB overlays are written. Images smaller than256px are edge-padded internally and returned to the original canvas, without resizing. This is prediction only: external accuracy cannot be claimed without corresponding held-out human labels.

For the actual original TensorFlow DeepLabV3+, use the Docker command above but pass `predict-rgb --model-dir results/B3_random_tiles_v1/deeplab --vendor results/B3_uploaded_deeplab_v1/vendor --input <RGB_file_or_folder> --output <new_folder>` after `container_entry.sh` instead of the training arguments. This executes `predict_TF_RGB.py` with all saved original variables, using the same native-size RGB inference/export code as the delivered protocol. The TensorFlow model is not replaced or renamed as a Torch model.
