"""Apply one explicitly user-confirmed reference correction; never fit or edit predictions."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

ROOT = Path('/workspace/LI')
OUT = ROOT / 'results/B3_reference_local_correction_v84_20261028'
SOURCE = ROOT / 'results/B3_original_UNet_latestCG_upload24_v82_20261028/data'
V83 = ROOT / 'results/B3_root_axis_junction_v83_20261028/final'
ORIGINAL = ROOT / 'results/B3_temporal_upload18_noGAT_v66_20261028/unet/full'
PERIODS = ['T1', 'T2', 'T3', 'T4']
ROI = [1368, 50, 1599, 202]
sys.path.insert(0, str(ROOT / 'analysis-scripts'))
from dataset_identity import require_active_position
sys.path.insert(0, str(ROOT / 'analysis-scripts/0353-2-B3_segmentation_pilot/root_axis_junction_v83'))
from evaluate import metric
from rendering import FONT, isolated


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(p, data):
    Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def unpack(v, shape):
    return np.repeat(np.array(v[::2], np.uint8), np.array(v[1::2], np.int64)).reshape(shape)


def pack(a):
    a = a.ravel()
    starts = np.r_[0, np.flatnonzero(a[1:] != a[:-1]) + 1]
    return np.column_stack((a[starts], np.diff(np.r_[starts, len(a)]))).ravel().tolist()


def main():
    assert require_active_position('0353-2-B3') == '0353-2-B3'
    OUT.mkdir(exist_ok=True)
    source = SOURCE / 'source_progress_24.json'
    immutable = [source, V83.parent / 'joint_best.pt']
    immutable += [folder / (t + '_classes.png') for folder in [ORIGINAL, V83 / 'full', SOURCE / 'reference'] for t in PERIODS]
    hashes = {str(p): sha(p) for p in immutable}
    assert hashes[str(source)] == 'cb5a9326b73003a81b44961300de88806378a771514e886faff168a715bf2c8e'
    old = {t: np.array(Image.open(SOURCE / 'reference' / (t + '_classes.png'))) for t in PERIODS}
    new = {t: a.copy() for t, a in old.items()}
    # Fixed location, identified by surrounding purple hook, green root and upper arc.
    components, _ = ndi.label(old['T2'] == 3, np.ones((3, 3), np.uint8))
    k = int(components[120, 1460])
    assert k > 0
    edit = components == k
    yy, xx = np.nonzero(edit)
    assert [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1] == [1423, 105, 1544, 147]
    assert int(edit.sum()) == 457
    new['T2'][edit] = 2  # Prediction/reference PNG convention: 2=new, 3=extension.
    doc = json.loads(source.read_text())
    entry = next(v for v in doc['images'] if v['name'] == 'T2_RGB.png')
    temporal = unpack(entry['layers']['temporal'], edit.shape)
    assert np.all(temporal[edit] == 2)  # Editable JSON convention: 1=new, 2=extension.
    revised = temporal.copy()
    revised[edit] = 1
    entry['layers']['temporal'] = pack(revised)
    audit = dict(
        source_sha256=hashes[str(source)], period='T2', pixels=457,
        bbox_xyxy=[1423, 105, 1544, 147], screenshot_context_ROI=ROI,
        old_annotation_class='tip_extension', corrected_annotation_class='newly_observed',
        authority='User explicitly confirmed this pictured extension interval is a new root.',
        mask_width_length_and_root_IDs_unchanged=True,
        training=False, prediction_edit=False,
        annotation_code_before=2, annotation_code_after=1,
        derived_PNG_code_before=3, derived_PNG_code_after=2,
    )
    doc['manual_reference_corrections_v84'] = [audit]
    corrected_json = OUT / 'root_annotation_progress_24_user_corrected_T2_new_root.json'
    corrected_json.write_text(json.dumps(doc, ensure_ascii=False, separators=(',', ':')))
    readback = json.loads(corrected_json.read_text())
    original_doc = json.loads(source.read_text())
    for e, e0 in zip(readback['images'], original_doc['images']):
        for field in e0:
            if field != 'layers':
                assert e[field] == e0[field]
        for layer in e0['layers']:
            if e['name'] == 'T2_RGB.png' and layer == 'temporal':
                a = unpack(e['layers'][layer], edit.shape)
                b = unpack(e0['layers'][layer], edit.shape)
                assert np.array_equal(a != b, edit)
            else:
                assert e['layers'][layer] == e0['layers'][layer]
    assert readback['root_structure_annotation'] == original_doc['root_structure_annotation']
    refdir = OUT / 'corrected_reference'
    refdir.mkdir(exist_ok=True)
    for t in PERIODS:
        Image.fromarray(new[t]).save(refdir / (t + '_classes.png'))
        assert np.array_equal(old[t] > 0, new[t] > 0)
        assert np.array_equal(old[t] == 255, new[t] == 255)
    Image.fromarray(edit.astype(np.uint8) * 255).save(OUT / 'T2_reference_changed_pixels.png')
    spatial = np.load(SOURCE / 'spatial.npy')
    audit['changed_pixels_by_split'] = {name: int((edit & (spatial == k)).sum()) for k, name in [(-1, 'unassigned'), (0, 'train'), (1, 'validation'), (2, 'internal_test')]}
    assert audit['changed_pixels_by_split']['internal_test'] == 0
    assert audit['changed_pixels_by_split']['validation'] == 0
    preds = {name: {t: np.array(Image.open(p / (t + '_classes.png'))) for t in PERIODS} for name, p in [('original_U_Net_V66', ORIGINAL), ('current_optimized_U_Net_V83', V83 / 'full')]}
    results, rows = {}, []
    for scope in ['internal_fixed_root_spatial_holdout', 'all_annotated_fixed_root_pixels', 'user_confirmed_local_ROI']:
        results[scope] = {}
        for version, ref in [('source24_before_correction', old), ('source24_user_corrected', new)]:
            gt, pp = [], {name: [] for name in preds}
            for t in PERIODS[1:]:
                valid = np.isin(ref[t], [1, 2, 3]) & (preds['original_U_Net_V66'][t] > 0)
                if scope == 'internal_fixed_root_spatial_holdout':
                    valid &= spatial == 2
                if scope == 'user_confirmed_local_ROI':
                    local = np.zeros(valid.shape, bool)
                    x, y, x1, y1 = ROI
                    if t == 'T2':
                        local[y:y1, x:x1] = True
                    valid &= local
                gt.append(ref[t][valid].astype(np.int64) - 1)
                for name in preds:
                    pp[name].append(preds[name][t][valid].astype(np.int64) - 1)
            results[scope][version] = {}
            for name, p in pp.items():
                m = metric(np.concatenate(gt), np.concatenate(p))
                m['evaluated_pixels'] = int(sum(len(x) for x in gt))
                results[scope][version][name] = m
                for r in m['classes']:
                    rows.append(dict(scope=scope, reference=version, model=name, **r))
    previous = json.loads((V83 / 'evaluation_results.json').read_text())['pixel_metrics']['24']
    for name, prevname in [('original_U_Net_V66', 'original_U_Net'), ('current_optimized_U_Net_V83', 'whole_segment_temporal_graph')]:
        a = results['internal_fixed_root_spatial_holdout']
        assert a['source24_before_correction'][name] == a['source24_user_corrected'][name]
        assert a['source24_user_corrected'][name]['classes'] == previous[prevname]['classes']
    with (OUT / 'U_Net_Dice_before_after_reference_correction.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write_json(OUT / 'evaluation_results.json', results)
    audit['corrected_JSON_sha256'] = sha(corrected_json)
    write_json(OUT / 'reference_correction_audit.json', audit)
    for p, h in hashes.items():
        assert sha(p) == h
    write_json(OUT / 'verification.json', dict(passed=True, no_training=True, no_prediction_edit=True, source24_preserved=True, root_geometries_and_IDs_unchanged=True, actual_changed_reference_pixels=457, original_test_Dice_unchanged=True, immutable_input_sha256=hashes, corrected_JSON_sha256=sha(corrected_json)))
    # Annotated comparison, categorical nearest-neighbor only.
    im = Image.new('RGB', (2220, 650), 'white'); d = ImageDraw.Draw(im)
    font = ImageFont.truetype(str(FONT), 30); small = ImageFont.truetype(str(FONT), 24)
    titles = ['Reference before correction', 'User-corrected reference', 'Current optimized U-Net (unchanged)']
    x, y, x1, y1 = ROI
    for i, (title, a) in enumerate(zip(titles, [old['T2'], new['T2'], preds['current_optimized_U_Net_V83']['T2']])):
        d.text((i * 740 + 20, 20), title, font=font, fill='black')
        im.paste(Image.fromarray(isolated(a[y:y1, x:x1])).resize((700, 461), Image.Resampling.NEAREST), (i * 740 + 20, 90))
    d.text((20, 575), 'T2: 457 reference pixels changed from extension to new. Model weights and predictions are unchanged.', font=small, fill='black')
    d.text((20, 610), 'This location is outside the original internal test mask; its correction does not change the test Dice.', font=small, fill='black')
    im.save(OUT / 'T2_User_corrected_reference_and_unchanged_UNet.tif', compression='tiff_adobe_deflate', dpi=(600, 600))
    im.resize((1554, 455), Image.Resampling.LANCZOS).save(OUT / 'T2_User_corrected_reference_and_unchanged_UNet_preview.png')
    text = '# User-confirmed T2 reference correction, no retraining\n\n'
    text += 'Only the pictured T2 extension interval (457 pixels; bbox [1423,105,1544,147]) is changed to newly observed. The user confirmed the annotation error. All old files, root contours, root IDs, weights and predictions are retained. No extension class outside this interval is changed.\n\n'
    text += 'The editable JSON uses temporal 1=new and 2=extension; derived class PNGs use 2=new and 3=extension. These code systems are deliberately kept distinct. The annotation revision is appended as metadata. No new root identifiers or structural labels are invented.\n\n'
    text += 'The corrected interval has no original validation/test pixels, so the previously reported internal test Dice is unchanged. Full annotated-area metrics are in-sample descriptive reference agreement, not independent accuracy or a training gain. All metrics are conditional on the frozen U-Net root foreground; T1 baseline is excluded from the three-class scores.\n\n'
    text += '| Scope | Reference | Model | New | Persistent | Extension | Macro |\n|---|---|---|---:|---:|---:|---:|\n'
    for scope, versions in results.items():
        for version, models in versions.items():
            for name, m in models.items():
                by = {r['name']: r['Dice'] for r in m['classes']}
                text += f"| {scope} | {version} | {name} | {by['newly_observed']:.6f} | {by['persistent']:.6f} | {by['tip_extension']:.6f} | {m['macro_Dice']:.6f} |\n"
    (OUT / 'RESULTS.md').write_text(text)
    print(json.dumps(dict(correction=audit, metrics=results), indent=2), flush=True)


if __name__ == '__main__':
    main()
