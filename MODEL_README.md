# V90 frozen-weight transfer candidate — experimental

This package performs real inference with the original V90 ConvLSTM/GAT, learned relation gates, junction-continuation head and root-state head. It builds a fresh graph for each input position; it does not reuse the fixed B3 graph.

**This is not the complete original V90 workflow and is not a validated production model.** The original dense local-correction preparation was not released. The reconstructed adapter changes the node partition/neighbors, approximates one geometric cue, and starts from V66 priors. These are material method changes. No external accuracy is claimed.

On developmental B3 reference B, macro Dice: V66 71.32%; this transfer candidate 76.45%; V90 automatic output 79.08%. T2–T4 metrics are pooled over valid typed reference-root pixels; background and untyped/ignored pixels are excluded. These are within-position, development-exposed agreements, not independent generalization estimates. Do not reuse B3 scores for other tubes.

## Run

Use Python 3.11 and install requirements in a new virtual environment. CPU inference works; do not install over an existing project environment.

Put registered, matched-field RGB observations in one directory as T1.png, T2.png, T3.png, T4.png. Missing dates must remain absent. At least two observations are required; each side must be at least 256 pixels. The script verifies a common pixel grid but cannot verify anatomical alignment by image dimensions alone. It does not register images.

```bash
python predict_candidate.py --rgb-dir /path/to/registered/position --position sample-1 --output /path/to/new/output
```

For multiple positions, each in its own directory:

```bash
python batch_predict.py --rgb-root /path/to/registered_positions --output /path/to/new_batch
```

The complete workflow uses the available observation sequence offline, including later observations in structural decoding. It is not a live single-frame predictor. Empty-foreground frames are currently rejected with a saved log; the adapter does not invent roots.

The package needs no B3 cached tensors, human labels, root-ID maps or network access for prediction. All three frozen model checkpoints and inference normalization are included. Outputs are under predictions/sample-1/.

## Files

* RGB_temporal_candidate.png: lossless native-grid class overlay on input RGB.
* RGB_uncertainty_review.png: uncertain root foreground shown grey.
* classes.png: 0 background, 1 persistent, 2 newly observed, 3 extending, 4 first-observation baseline.
* classes_with_uncertainty.png: same class codes, with 255 for uncertain foreground.
* uncertainty.png: nonzero marks uncertain foreground, including junctions affected by the degree proxy and unresolved axis decisions.
* ownership.npz: exclusive per-pixel axis ID, inferred cross-date track ID, uncertainty mask. Inferred exclusivity is a consistency property, not proof of biological identity.
* adapter_provenance.json: input hashes and method departures.

The uncertainty map is a heuristic review flag, not calibrated probability or a guarantee that every remaining pixel is correct. Every new position needs independent manual validation. First observations are baseline, not newly emerged roots. With missing intermediate dates, changes refer to the previous observed date, not an invented adjacent observation. The existing Q1-1-Z2 T2/T3 alignment remains coarse/unverified; Q1-3-Z3 remains excluded.

The original V90 weights/source and training assets remain unchanged in https://github.com/Annalee811/Temporal-segmentation-model/tree/4615bcbd37442a3b11e5ad4533242df95f237fb6 . That repository's original tensor-bound reproduction is a separate command, and manual annotation is used for evaluation only.
