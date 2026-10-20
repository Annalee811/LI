# 0353-2-B3 human temporal labels and whole-root-axis tracing — v15

**Correction to v14:** the U-Net temporal teacher no longer replaces the user's temporal classes. All original typed root pixels keep their original classes. U-Net supplies only a narrow morphology adjustment, supported by raw RGB. Four upper RGBs2541×1108are used; lower originals and the full human JSON are retained. T1baseline; subsequent dates relative to the immediately previous date. Death excluded.

## Label meaning and morphology

A pre-existing unchanged root is persistent. On a growing root, the old shaft is persistent and the added terminal part is tip growth. A newly observed root or new lateral is new. Physical touching/crossing alone never proves one root's identity. Prior absence under soil or water does not prove biological birth; these observations still require review.

The full original manual JSON SHA256 is27943e5f9d087bf718668bba0a39058dc9a4871de6c75c5ccc27febad4c06a09. Every original root pixel is retained; every original T2–T4 temporal class is retained. T1original untyped roots are explicitly baseline. Original root thickness/tip tracing can expand at most6nativepixels, only where the latest frozen U-Net root mask/probability≥0.45and local rawRGB contrast support it. Added pixels inherit the nearest original temporal trace; ambiguous added class boundaries stay untyped. Distant model-only roots do not become human temporal truth. This is morphology-assisted reference refinement, not a new independent human reference. No class is changed to inflate a score. Original and width-adjusted references remain separate.

UI temporal codes1new/2growth/3persistent/4baseline differ from model codes1persistent/2new/3growth/4baseline;0background,255untyped in reference targets. New lower-field labels are not invented. The annotation editor embeds all four native upper RGBs and the morphology-adjusted HUMAN reference, not the model's final temporal predictions. It supports60×zoom,tools beside the image,Ctrl+Z,keeps zoom on date switches,and saves/reimports progress. Existing original JSON is separately retained. Real Chromium mouse/undo/save/reimport tests passed.

## Frameworks

Frozen root/background networks are ResNet18U-Net(PyTorch),the originally supplied TensorFlow1.12ResNet-v2-50DeepLabV3+,andSegFormer-B0(PyTorch). Each supplies its own predicted roots and probabilities. Three separate paired temporal heads use PyTorch,as did the preceding temporal experiments. TensorFlow is a framework,not a biological tracking mechanism. No claim is made that switching frameworks solves crossings. Binary root weights are unchanged this round.

## Whole-image root axes and RGB evidence

Before cropping for fitting,the full predicted root skeleton is turned into a graph. Dense junction nodes are clustered; opposite tangent continuity,root width and local originalRGBappearance score the possible routes. At a T junction,the smooth main route can continue while a lateral arm stays separate. Prior-frame predicted-axis support helps resolve intersections; ambiguous alternatives are flagged rather than silently fused. Short existing-tip gaps may be linked only with compatible directions and positive originalRGBline evidence; this does not fill or warp any RGB or root-mask pixels. Candidate axes and their local IDs are geometry-based traces,not independently confirmed biological root instances.

Cross-date correspondence uses nearby predicted skeletons,tangent alignment and dominant prior-axis support. The temporal head receives31channels:the inherited25prediction-only rootRGB/probability/topology channels,four whole-axis match/position maps,and current/previous rawRGBlocalcontrast. Human masks are not inference features. Both full-frame RGB and the background-removed root image contribute:RGBappearance is used in graph routing and foreground/network/context features; background itself is not classified as a root just because it connects two traces.

Full-frame network probabilities are blended FIRST. Then the active structured decoder chooses a whole-axis interpretation:allNEW,or terminalGROWTH–PERSISTENT–terminalGROWTH,with growth anchored to a real previous predicted tip and compatible direction. An all-growth continuation fragment is also allowed when an unresolved crossing/gap separates it from the old shaft,but only with outward previous-tip attachment,little previous overlap and learned growth evidence. A lateral emerging from an old shaft cannot use that exception. NEW andGROWTHnever coexist inside one accepted traced axis. Width ownership is unique at shared skeleton nodes. It remains possible that two separate candidate axes actually belong to the same biological root; unresolved crossings are explicitly recorded. Zero forbidden mixtures is a rule check,**not100%tracking accuracy**.

The raw independent-pixel prediction is retained as a counterfactual. Whole-axis rules can reduce Dice when a route is wrong,the previous tip is missed,or the human label/geometry disagrees. The difference is reported,not hidden. Geometry/tip constraints remain heuristic and need root-instance review.

## Actual training and internal evaluation

The same fixed random25/5/6base256pxspatial windows(seed20261018)are used.25additional shifted12pxtraining views exclude every validation/test pixel. All four dates at a location stay together.200trainingRGBviews,20validation,24internal-testRGBviews correspond to150/15/18adjacent-date pairs. Shifted training views overlap their own bases; they are not additional independent acquisitions. No easy-tile selection is used. Whole-image axis context crosses crop boundaries; quantitative samples are not independent roots or tubes.

All three31-channel temporal heads are initialized from their v13HUMAN-trained evaluation heads,**not v14temporal-distillation weights**. Extra input-channel weights start at0;old class-prior multipliers are reset.24epochs,64draws/epoch,batch4,384optimizerupdates per model. All3share the same actual coordinate/date/augmentation draw order. AdamWlr1e-4,weight_decay1e-4,cosine decay to1e-5;seed20261021is an arbitrary seed,not an experiment date. The inherited loss combines weightedCE,0.5classDice,0.2boundaryBCE,0.05same-label local consistency,0.3same-class skeleton-run loss. References are human temporal labels with morphology expansion,supervised only on the student's own detected root pixels. Remaining background/root detection is frozen.

Epoch0and checks every4epochs compete on width-adjusted human VALIDATION raw-pixel macroDice. Selected epochs12/16/8for U-Net/DeepLab/SegFormer. After fitting,whole-axis geometry strength is selected from0,.5,1,2on the same15validation pairs only. Test values do not select checkpoints or strengths. The delivered full-frame axis-decoded output,not independent tile predictions,is cropped to compute the18internal-test pair metrics. T1is excluded from temporal macroDice. Precision,Recall,Dice,IoU and confusion counts are reported per class;undefined classes are not fabricated. MacroPrecision/Recall summaries use0for an undefined class. Per-pair macroDice excludes a class only when it has noTP,FP,FN.

Original narrow-mask scores and width-adjusted human-reference scores are both given. U-Net revised-reference macroDice0.5639,persistent0.8663,new0.7091,growth0.1163;DeepLab macro0.4718;SegFormer macro0.4014. Growth remains weak. The unconstrained-pixel macroDice values are0.6391/0.4490/0.4390,respectively:anatomical grammar is not a uniform accuracy improvement. These scores cannot be directly compared with v14's U-Net-generated temporal reference agreement. The teacher-reference target itself changed back to original HUMAN temporal semantics.

Historical same-field warm starts and binary fitting have prior pixel exposure,and the workflow has been iteratively developed on this position. The same-field spatial holdout is **not a new blind test**. Independent external test images0. No claim of cross-root/cross-tube generalization,biological root-instance validation,or manuscript-ready temporal accuracy is made. Skeleton pixel-count regressions are mask-derived length diagnostics,not independently measured physical manual root lengths. IdentityR²andOLSassociationR²are separate;no scale or biological replication is invented.

## Packages and files

Extract desired ZIPs into the same parent;all share`0353-2-B3_temporal_v15/`and keep repository paths. Annotation package embeds the human-first morphology reference. Each model result includes four nativeRGBs,temporal overlays,root-only original-colorRGBA,color-rootRGBA,classmaps,assigned-class confidence/probabilities,putative axisIDs,axis/matchJSON,reviewmasks,exact evaluation outputs,rootprobabilitycaches,and axis caches. Huge float feature caches are regenerated,not distributed. Large optional duplicate reviewRGB overlays and intermediate development outputs are omitted;review masks and originalRGBs reconstruct them. Raw probabilities outside roots do not affect the decoder and are omitted from downloadable model ZIPs;native masked class probabilities retain every root score.

Weights packages include frozen binary networks,new best/last temporal heads,and v13manual warm-start heads. Complete scripts include v15,required v13modules,binary architecture/nativeRGBpredictors,identitypolicy,andoriginalTFvendor/runtime sources. Figures are600dpiLZWTIFFwithPNGpreviews;numeric charts also havePDF/SVG. Large duplicate rasterPDFs are omitted from downloads. DirectFig6TIFFis the RGB/root-axis/temporal diagnostic;axisIDs are local candidates and double arrows show orientation,not proven growth polarity. Orange circles mark unresolved crossings. Background full/local comparison figures and S1learningcurves,S2transitionDice,S3skeletonlength,S4test-pairdifferences,S5errors,S6confusions are supplied. No Word files.

## Reproduce without overwriting published results

Use Python3.12and`requirements.txt`;CUDAis detected automatically. Actual temporal fitting usedCPU BF16autocast withFP32objective/weights;full-frame scores are saved withdeclaredFP16quantization. Device/arithmetic differences can cause small reproduction variation. Extract data,all3model results,and weights/scripts. Enter`0353-2-B3_temporal_v15`:

```sh
mkdir -p results/B3_temporal_axis_v15_reproduce
cp -a results/B3_temporal_axis_v15/data results/B3_temporal_axis_v15_reproduce/data
python analysis-scripts/0353-2-B3_segmentation_pilot/temporal_axis_v15/cache.py --run results/B3_temporal_axis_v15_reproduce --roots results/B3_annotation_refinement_v1 --previous results/nonexistent_optional_v14_cache
python analysis-scripts/0353-2-B3_segmentation_pilot/temporal_axis_v15/run_all.py --run results/B3_temporal_axis_v15_reproduce --roots results/B3_annotation_refinement_v1 --previous results/B3_temporal_v13
python analysis-scripts/0353-2-B3_segmentation_pilot/temporal_axis_v15/predict.py --run results/B3_temporal_axis_v15_reproduce --roots results/B3_annotation_refinement_v1 --model unet
```

Repeat the last command for`deeplab`and`segformer`. Cache.py recomputes inherited25-channel features when an optional v14cache does not exist;the large cache is unnecessary. Use a new output directory for a new scientific run;never silently overwrite a published reference/checkpoint. Prepare.py documents original reference regeneration;the delivered exact data already exists. Review editor regeneration uses the included original customHTMLtemplate.

`report.py`,`figures.py`,`axis_figures.py`recreate tables and panels. `test_axes.py`tests crossings,laterals,old-tipgrowth,parallelseparation,and extension-fragment versus old-shaft lateral behavior. `verify.py`checks original hashes,unchanged human temporal labels,boundedmorphology,split integrity,actual draws,frozen weights,allmetrics,reloadoutputs,and forbidden mixtures. `verification.json`records the actual checks. A preliminary geometry-only development pass and a conservative tip-gating pass are preserved locally as development history;they are not extra independent experiments or hidden independent test replications. Decoder changes addressed real anatomical failure modes;all reported results are development results.

## Truly label-free inference

For a four-period registeredRGBposition with this model's own frozen binary masks/probabilities:

```sh
python analysis-scripts/0353-2-B3_segmentation_pilot/temporal_axis_v15/predict_RGB.py --head results/B3_temporal_axis_v15/unet/best_temporal_head.pt --decoder results/B3_temporal_axis_v15/unet/decoder_selection.json --rgb-dir results/B3_temporal_axis_v15/source_RGB/0353-2-B3 --root-dir results/B3_annotation_refinement_v1/unet/upper --output results/new_label_free_prediction
```

This entrypoint reads **no human label**,does not retune on new images,and fixes the decoder from prior validation. It automatically extracts full-frameRGB-supported axes and compares adjacent dates. Root outputs may use`T1_probability.npy`or`T1_RGB_probability.npy`naming. All4observed adjacent dates are required;missing dates are not fabricated. For other positions first run the respective frozen nativeRGBbinary predictor. No other position was predicted here. Active policy canonicalizes0353-1-Z2toQ1-1-Z2and excludesQ1-3-Z3. The label-free entrypoint was run on all4B3frames for EACH of the3actual selected heads and reproduced all delivered temporal maps pixel for pixel.
