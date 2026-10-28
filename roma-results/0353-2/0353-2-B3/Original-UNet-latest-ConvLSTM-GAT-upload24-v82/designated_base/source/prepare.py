"""Upload18 restart: immutable binary support, new prediction-only inputs, no GAT."""
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path('/workspace/LI')
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'results/B3_temporal_upload18_noGAT_v66_20261028'
UPLOAD = Path('/workspace/attachments/fea66c21-080a-42c8-9b7a-5be64f6558ee/root_annotation_progress (18).json')
OLD_SOURCE = HERE.parent / 'temporal_uploaded10_v51'
BINARY = ROOT / 'results/B3_annotation_refinement_v1'
PERIODS = ['T1', 'T2', 'T3', 'T4']
MODELS = ['unet', 'deeplab', 'segformer']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p, v): Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def unpack(v, shape):
    v = np.asarray(v, np.int64)
    assert len(v)%2 == 0 and (v[1::2]>0).all() and v[1::2].sum()==np.prod(shape)
    assert ((v[::2]>=0)&(v[::2]<=255)).all()
    return np.repeat(v[::2].astype(np.uint8), v[1::2]).reshape(shape)

def main():
    sys.path.insert(0, str(ROOT/'analysis-scripts'))
    from dataset_identity import require_active_position
    assert require_active_position('0353-2-B3') == '0353-2-B3'
    assert not RUN.exists(), 'Never overwrite an experiment'
    doc = json.loads(UPLOAD.read_text())
    assert {e['name'] for e in doc['images']} == {t+'_multiregion_RGB.png' for t in PERIODS}
    RUN.mkdir(); data=RUN/'data'; data.mkdir()
    for d in ['current','full_annotation','upper']: (data/d).mkdir()
    shutil.copy2(UPLOAD, data/'active_progress.json')
    # Copy software only. No old checkpoint, routing model, temporal probability or feature cache.
    shutil.copytree(OLD_SOURCE/'legacy', HERE/'legacy', ignore=shutil.ignore_patterns('__pycache__'))
    immutable=[]
    def remember(p): immutable.append(dict(path=str(p), sha256=sha(p)))
    remember(data/'active_progress.json')
    audits=[]
    for e in sorted(doc['images'], key=lambda e:e['name']):
        t=e['name'][:2]; shape=e['height'], e['width']
        rgb=ROOT/'results/B3_full_canvas_v1/originals'/e['name']
        assert shape==(1867,2541) and Image.open(rgb).size==shape[::-1]
        assert sha(rgb)==e['image_sha256'], 'Uploaded image hash mismatch'
        layers={k:unpack(v,shape) for k,v in e['layers'].items()}
        for k,a in layers.items(): Image.fromarray(a).save(data/'full_annotation'/f'{t}_{k}.png')
        raw=np.array(Image.open(rgb).convert('RGB'))[:1108]
        Image.fromarray(raw).save(data/'upper'/f'{t}_RGB.png')
        mask=layers['mask'][:1108]; temporal=layers['temporal'][:1108]
        assert set(np.unique(mask)).issubset({0,64,128,255})
        assert set(np.unique(temporal)).issubset({0,1,2,3,4})
        target=np.full(mask.shape,255,np.uint8)
        target[mask==0]=0 # Only explicit background; 64/128 remain unreviewed/ignored.
        for encoded,trained in [(3,1),(1,2),(2,3),(4,4)]: target[(mask==255)&(temporal==encoded)]=trained
        if t=='T1':
            assert not np.isin(temporal[mask==255],[1,2,3]).any()
            target[mask==255]=4 # Baseline display semantics only; never supervised as a change class.
        else: assert not (target==4).any()
        Image.fromarray(target).save(data/'current'/f'{t}_classes.png')
        remember(rgb); remember(data/'upper'/f'{t}_RGB.png'); remember(data/'current'/f'{t}_classes.png')
        audits.append(dict(period=t, full_image_sha256=sha(rgb), upper_crop=[0,0,2541,1108],
            root_temporal_codes={str(k):int(v) for k,v in zip(*np.unique(temporal[mask==255],return_counts=True))},
            target_counts={str(k):int(v) for k,v in zip(*np.unique(target,return_counts=True))},
            lower_canvas_preserved_not_trained=True))
    # Retain the predeclared spatial masks as INNER development masks. Dates never become separate outer subjects.
    for name in ['paired_manifest.csv','spatial_manifest.csv']:
        shutil.copy2(ROOT/'results/B3_temporal_uploaded10_v51_20261025/data'/name,data/name); remember(data/name)
    with (data/'paired_manifest.csv').open(encoding='utf-8-sig') as f: pairs=list(csv.DictReader(f))
    masks={s:np.zeros((1108,2541),bool) for s in ['train','val','test']}
    for r in pairs:
        x,y,w,h=[int(r[k]) for k in ['x','y','width','height']]
        masks[r['split']][y:y+h,x:x+w]=True
    assert all(not (masks[a]&masks[b]).any() for a,b in [('train','val'),('train','test'),('val','test')])
    binary_audit=[]
    for m in MODELS:
        (RUN/m).mkdir()
        for t in PERIODS:
            mask=BINARY/m/'upper'/f'{t}_root_mask.png'
            p=BINARY/m/'upper'/f'{t}_probability.npy'
            assert Image.open(mask).size==(2541,1108)
            gate=np.array(Image.open(mask))>0
            previous_gate=np.array(Image.open(ROOT/'results/B3_temporal_gat_v65_20261027'/f'{m}_gat/full'/f'{t}_classes.png'))>0
            assert np.array_equal(gate,previous_gate), 'Must inherit the last experiment binary support exactly'
            remember(mask); remember(p)
            binary_audit.append(dict(model=m,period=t,path=str(mask),sha256=sha(mask),root_pixels=int(gate.sum()),same_as_last_experiment=True))
        remember(BINARY/m/('best_weights_fp16.npz' if m=='deeplab' else 'best.pt'))
    prep=dict(run=str(RUN), position='0353-2-B3',roots=str(BINARY),seed=2026102801,epochs=100,
        learning_rate=1e-4,minimum_learning_rate=3e-6,weight_decay=1e-4,
        batch_size=4,focused_draws_per_epoch=50,training_draws_per_epoch=200,
        sampling='150 unique adjacent-date train windows +50 class-focused repeats; persistent/new/extension/extension pattern',
        annotation_semantics={'annotation_code_1':'newly observed','annotation_code_2':'tip extension',
            'annotation_code_3':'persistent','annotation_code_4':'baseline',
            'T1_untyped_root':'baseline display only, not a training label','code_64_128':'unreviewed/ignored, not true background'},
        uploaded_reference_SHA256=sha(UPLOAD),immutable_inputs=immutable,
        GAT=False,old_temporal_checkpoint_loaded=False,old_temporal_probabilities_used=False,
        learned_routing_used=False,features='25 channels generated fresh from paired RGB and frozen binary predictions only',
        restart_scope='Random temporal heads only; binary weights/probabilities/masks are frozen',
        outer_partition='All four B3 dates in one development position',
        evaluation_scope='Inner spatial validation/holdout on one field, not an independent test or cross-tube accuracy',
        selection='Max validation macro Dice, never use inner test for selection',
        source_upload_has_root_number_annotations=False)
    js(RUN/'preparation.json',prep)
    js(RUN/'reference_verification.json',dict(passed=True,source_SHA256=sha(UPLOAD),copy_SHA256=sha(data/'active_progress.json'),periods=audits,binary=binary_audit,spatial_pixel_overlap=0))
    (RUN/'README.md').write_text('''# Upload18 temporal restart, no GAT (V66)

Latest uploaded `(18).json` is preserved byte for byte. No prior temporal checkpoint,
temporal probabilities, learned router, GAT, root-number labels or prior optimizer
state is used. Three randomly initialized 25-channel convolutional temporal heads
learn persistent, newly observed, and tip extension for T2–T4. T1 is baseline.
The last experiment's three binary supports and probabilities are reused exactly;
binary weights are not loaded into an optimizer or retrained. Root widths/lengths
are consequently unchanged. New prediction-only paired features are recomputed.

The four uploaded full canvases are preserved. Learning covers the same registered
2541×1108 upper main field as the frozen binary results, without resizing. Lower
multi-region canvas annotations remain archived and are not silently used as a
registered temporal view. Mask64/128 and untyped roots are ignored, not background.

100 real epochs, 150 unique training pairs +50 class-focused repeats per epoch,
15 inner validation and18 inner holdout pairs. All dates remain one outer B3
development subject. This is internal reference agreement, not external accuracy;
the historical binary branches have same-field exposure. Validation only selects
the best temporal head; no promise of a target score or monotone convergence.

Atomic epoch checkpoints save new model/optimizer/scheduler and all RNG states.
Interrupted unsaved epochs replay; completed epochs are never filled or simulated.
Resume only this run via `source/run_all.py --run <this-directory>` after ensuring
no orchestrator is active. A STOPPED_BY_USER.json marker blocks recovery.
''')
    shutil.copytree(HERE,RUN/'source',ignore=shutil.ignore_patterns('__pycache__'))
    js(RUN/'pipeline_status.json',dict(stage='preparing_prediction_only_features',GAT=False,epochs=100))
    print(json.dumps(dict(run=str(RUN),source_SHA256=sha(UPLOAD),periods=audits),ensure_ascii=False))

if __name__=='__main__': main()
