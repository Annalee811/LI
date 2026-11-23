"""Create immutable experiment archives; all source and original results stay in place."""
from pathlib import Path
import csv,json,hashlib,shutil,zipfile,sys
W=Path(__file__).resolve().parents[1];OLD=Path('/workspace/work/V90_all_microtubes_20261118')
D=W/'delivery';S=W/'package_staging';D.mkdir(exist_ok=True);S.mkdir(exist_ok=True)
def copy(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
def tree(src,dst,exclude=()):
 for p in sorted(src.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts and p.name not in exclude:copy(p,dst/p.relative_to(src))
def archive(src,name,only_position=None):
 dest=D/name
 with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6)as z:
  for p in sorted(src.rglob('*')):
   if p.is_file() and (only_position is None or p.relative_to(src).parts[0] in [only_position,'README.txt']):z.write(p,src.name+'/'+str(p.relative_to(src)))
 with zipfile.ZipFile(dest)as z:assert z.testzip() is None
 assert dest.stat().st_size<95*2**20, (name,'Must split below GitHub single-blob limit')
 print('ARCHIVE',name,round(dest.stat().st_size/2**20,2),'MiB',flush=True)
 return dest

def main():
 root=S/'V90_validation_20261123';root.mkdir(exist_ok=True)
 tree(W/'code',root/'code',exclude=('branch_review.py',))
 tree(W/'data',root/'data');tree(W/'reports',root/'reports');tree(W/'logs',root/'logs')
 tree(Path('/workspace/LI/dataset/current'),root/'dataset/current')
 copy(Path('/workspace/LI/analysis-scripts/dataset_identity.py'),root/'analysis-scripts/dataset_identity.py')
 tree(W/'experiments',root/'experiments',exclude=('inference.npz',))
 for p in W.glob('*COMPLETE.json'):copy(p,root/p.name)
 copy(W/'ALL_EXPERIMENTS_READY.json',root/'ALL_EXPERIMENTS_READY.json')
 for p in W.glob('*.log'):copy(p,root/'execution_logs'/p.name)
 for seed in [20261123,20261124,20261125]:
  src=W/'decoder_experiments'/str(seed);dst=root/'decoder_experiments'/str(seed)
  copy(src/'completed.json',dst/'completed.json')
  for sub in ['branch/full','axis','junction']:
   for p in(src/sub).rglob('*'):
    if p.is_file()and p.suffix.lower() in ['.png','.npz','.json','.csv']:copy(p,dst/p.relative_to(src))
 text='''V90 supplementary actual experiments — 2026-11-23

This archive adds internal temporal-input controls, retrained module ablations and a
paired structure-decoder study. It does not replace the original V90 weights.
15 runs = 5 variants x 3 fixed optimization seeds x 100 full-graph updates;
8192 sampled training root pixels per update. Checkpoint selection: validation only.
Upstream root segmentation and V66 priors are frozen, historically B3-exposed.
These are conditional internal B3 results, not independent cross-tube accuracy.

Main internal macro Dice: full 63.71 +/- 0.35%; frozen V66 62.56%.
No ConvLSTM 63.42%; no GAT 62.95%; no learned identity gates 63.69%.
The current-frame classifier collapsed to persistent-only under this fixed budget.
It is a negative control, not evidence about a fully tuned strong single-frame model.
Seed SD is optimization variability, not biological replication; no P-values invented.
Scores use eligible typed root pixels T2–T4, fixed foreground; background excluded.
Historical V90 references have a different training history and are not matched ablations.

Paired decoder study (separate whole-field inference context, same outputs per seed):
raw 63.70 +/- 0.34%; axis 73.51 +/- 0.50%; axis+junction 76.71 +/- 1.99%.
Instance pair precision ~95.91%, recall ~49.77%; overfragmentation remains.
Do not mix these whole-field stages into the partition-isolated main protocol.

Independent cohort: 3 tubes, 7 positions, 26 frames. Training tube 0353-2 excluded.
No external human temporal GT exists yet. Accuracy is NOT_EVALUATED_NO_HUMAN_GT.
390 new neural masks plus 52 frozen-reference masks are in the separate prediction ZIP.
Raw class codes: 0 background, 1 persistent, 2 newly observed, 3 extending,
4 first-observation root. Encoded masks look dark in a normal viewer; use the preview PDF.
PNG/native dimensions are preserved. Missing dates are not synthesized.
The experimental transfer adapter has documented geometric approximations.
External RGB had prior unlabelled prediction exposure; this is not a never-seen-image trial.

REPRODUCING CONDITIONAL TRAINING (Python 3.12, PyTorch 2.8 tested on CPU)
  python -m pip install -r requirements-experiments.txt
  python code/reproduce_study.py --output /a/new/empty/study
Uses bundled frozen features, current class GT subset and training-only normalization.
15 best and resume checkpoints, full histories and configs are under experiments/.
Original executed study_model.py and train_study.py retain their recorded SHA256.
The reproduction helper stages one historical baseline-image path to bundled data.
Large derived inference arrays are retained locally and can be regenerated from weights.

REPLAYING INDEPENDENT PREDICTIONS
Unzip Independent_inputs_*.zip beside this study; each supplies positions under
independent_inputs/. They can be extracted into the same directory.
  python code/reproduce_independent.py --inputs-root /path/independent_inputs \
    --model-root . --output /a/new/empty/predictions
This replays the raw heads from prepared features, not a standalone raw-RGB deployment.
Original raw-RGB upstream entry points and weights remain in the existing project.

REPLAYING THE PAIRED STRUCTURE DECODER
Unzip Decoder_inputs.zip, then:
  python code/reproduce_decoder.py --inputs /path/decoder_inputs \
    --model-root . --output /a/new/empty/decoder
All three stages share identical neural outputs within a seed. Human GT is not read
at prediction. Decoder inputs include existing B3 registered RGB, graph and topology.

FUTURE INDEPENDENT SCORING (not run as accuracy while human GT is unavailable)
Use the separate per-tube human-label kit; its all-255 masks are EMPTY TEMPLATES.
Fill from RGB independently, set annotation_state=human_complete and supply annotator
and annotation_source in a copy of reports/annotation_checklist.csv. Keep the locked
26 observations and hashes; no prediction-based cherry-picking or relabeling as GT.
  python code/evaluate_independent.py --manifest completed_checklist.csv \
    --annotation-root /path/annotation-kit --prediction-dir /path/to/one/model/seed \
    --output /a/new/evaluation
T1 is binary-root only. T2–T4 count human-valid background too. Results are grouped
by whole tube. Pending templates are rejected, not scored as all-background or zero.

FIGURES AND TABLES
reports/ contains a complete Chinese TXT report, XLSX and CSV metrics, two 600-dpi
Times New Roman figures/PDFs, and a seven-page external RGB-overlay preview.
Plot sources require a locally licensed Times New Roman installation; configure
the font directory F in plot_and_tables.py / plot_decoder.py when reproducing figures.
Font files are not redistributed. Test labels are not needed to regenerate predictions.

Original source: https://github.com/Annalee811/Temporal-segmentation-model/tree/4615bcbd37442a3b11e5ad4533242df95f237fb6
Canonical dataset policy: Q1-1-Z2 is the old 0353-1-Z2 alias; Q1-3-Z3 excluded.
No original files, weights, labels or earlier results were deleted or overwritten.
'''
 (root/'README.txt').write_text(text)
 (root/'requirements-experiments.txt').write_text('torch==2.8.0\nnumpy==2.2.6\nscipy==1.16.2\nscikit-image==0.25.2\nPillow==11.3.0\n')
 (root/'requirements-figures.txt').write_text('matplotlib>=3.9\nopenpyxl>=3.1\nPyMuPDF>=1.24\n')
 hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(root.rglob('*'))if p.is_file() and p.name!='FILE_SHA256.json'}
 (root/'FILE_SHA256.json').write_text(json.dumps(hashes,indent=2))
 archive(root,'V90_actual_experiments_code_weights_results.zip')
 pred=S/'independent_predictions';tree(W/'independent_predictions',pred)
 copy(root/'README.txt',pred/'README.txt')
 copy(W/'reports/independent_protocol.json',pred/'independent_protocol.json')
 archive(pred,'Independent_predictions_442_masks.zip')
 rows=list(csv.DictReader((W/'reports/annotation_checklist.csv').open(encoding='utf-8-sig')))
 for tube in ['0353-3','A017-3','Q1-2']:
  kit=S/('annotation_kit_'+tube);kit.mkdir(exist_ok=True)
  positions=sorted(set(r['position_id']for r in rows if r['tube_id']==tube))
  for pos in positions:tree(W/'independent_annotation_kit'/pos,kit/pos)
  copy(W/'independent_annotation_kit/README.txt',kit/'README.txt')
  copy(W/'reports/annotation_checklist.csv',kit/'all_locked_26_frame_checklist.csv')
  archive(kit,'Annotation_RGB_templates_'+tube+'.zip')
  inputs=S/('prepared_'+tube)/'independent_inputs';inputs.mkdir(parents=True,exist_ok=True)
  for pos in positions:
   src=OLD/'transfer_candidates'/pos;dst=inputs/pos
   for name in ['graph.npz','extra.npz','frames.npz']:copy(src/'local/data/unet'/name,dst/name)
   copy(src/'adapter_provenance.json',dst/'adapter_provenance.json')
   for p in(W/'external_inputs'/pos).glob('*.npz'):copy(p,dst/p.name)
  (inputs/'README.txt').write_text('Prepared features for conditional neural-head prediction replay. Known transfer adapter approximations apply. No external human temporal GT is included. Merge independent_inputs directories across the three tube ZIPs.\n')
  if tube=='Q1-2':
   for pos in positions:archive(inputs,'Independent_inputs_'+pos+'.zip',only_position=pos)
  else:archive(inputs,'Independent_inputs_'+tube+'.zip')
 # A shared copy replaces three redundant per-seed original topology/RGB inputs.
 src=OLD/'model_source/assets';inputs=S/'decoder_inputs';inputs.mkdir(exist_ok=True)
 tree(src/'base',inputs/'base')
 for rel in ['raw_graph.npz','joint/data/topology.pkl','joint/data/pairs.npz','local/data/spatial.npy','local/data/unet/graph.npz','local/data/unet/ownership.npz','local/data/unet/extra.npz','local/data/unet/frames.npz']+[f'local/data/unet/T{t}_pixels.npz'for t in [1,2,3,4]]:copy(src/rel,inputs/rel)
 tree(Path('/workspace/LI/dataset/current'),inputs/'dataset/current')
 copy(Path('/workspace/LI/analysis-scripts/dataset_identity.py'),inputs/'analysis-scripts/dataset_identity.py')
 archive(inputs,'Decoder_inputs.zip')
 for name in ['Baseline_ablation_SCI.jpg','Baseline_ablation_SCI.pdf','Structured_decoder_SCI.jpg','Structured_decoder_SCI.pdf','Baseline_ablation_and_test_index.xlsx','Experiment_report_CN.txt','Independent_prediction_preview.pdf']:
  copy(W/'reports'/name,D/name)
 manifest={p.name:dict(bytes=p.stat().st_size,SHA256=hashlib.sha256(p.read_bytes()).hexdigest())for p in sorted(D.iterdir())if p.is_file()and p.name!='ARTIFACT_SHA256.json'}
 (D/'ARTIFACT_SHA256.json').write_text(json.dumps(manifest,indent=2))
 print('DELIVERY_READY',len(manifest),'files',sum(d['bytes']for d in manifest.values())/2**20,'MiB')
if __name__=='__main__':main()
