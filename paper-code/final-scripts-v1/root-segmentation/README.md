# Root/background segmentation

Models: U-Net, DeepLabV3+, SegFormer-B0. The archive contains training, checkpoint selection, threshold calibration and evaluation code, with fixed numerical settings and input/checkpoint hashes. Human-reviewed binary masks use 0=background, 255=root and 128=ignore. Input data are supplied separately.

Python3.12 is used for U-Net/SegFormer. Install `requirements.txt`. DeepLabV3+ uses Python3.5/TensorFlow1.12; `runtime/prepare_runtime.py` downloads the pinned official wheel and writes the pinned Docker build context. The upstream DeepLab source and license are included. The included Docker launcher runs the TensorFlow interpreter. Keep `experiment/` and `initialization/` inside the extracted package directory; execute the commands from that directory. Docker is required for DeepLabV3+.

```bash
python -m pip install -r requirements.txt
export CUDA_VISIBLE_DEVICES=""
python fetch_initialization.py --output initialization
python prepare_data.py --data-root /path/to/data --run experiment --initial initialization
python runtime/prepare_runtime.py --context runtime/build --build
python train_all.py --run experiment --tf-python "sh ./tf_python.sh"
```

Individual commands:

```bash
python train_torch.py --run experiment --model unet --epochs 20 --batch 2 --threads 3
sh tf_python.sh train_tf.py --run experiment --vendor vendor/deeplab --epochs 20 --batch 2 --threads 2
python train_torch.py --run experiment --model segformer --epochs 20 --batch 2 --threads 3
```

The supplied data directory must contain the `patches/` files named in `training_manifest.csv`. Their exact hashes and the data partition are checked before training. Supply the optional `upper/` RGB inputs listed in `inference_inputs.json` to reproduce full-field probability maps and binary masks after training. Samples from the same spatial window retain one partition across dates. Epoch-zero checkpoints are eligible; threshold selection uses validation only. Training saves weights, measured learning curves, draw records, threshold grid, held-out probabilities and Precision/Recall/Dice/IoU. Neural arithmetic can differ across hardware/library builds; the pinned initialization and configuration are required for numerical reproduction.
