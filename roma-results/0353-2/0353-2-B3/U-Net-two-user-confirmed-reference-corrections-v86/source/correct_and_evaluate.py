"""Two explicit user-confirmed annotation edits; all U-Net predictions stay immutable."""
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path('/workspace/LI')
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/B3_user_confirmed_reference_corrections_v86_20261028'
DATA = ROOT / 'results/B3_original_UNet_latestCG_upload24_v82_20261028/data'
V84 = ROOT / 'results/B3_reference_local_correction_v84_20261028'
V85 = ROOT / 'results/B3_branch_continuity_review_v85_20261028'
V83 = ROOT / 'results/B3_root_axis_junction_v83_20261028/final'
BASE = ROOT / 'results/B3_temporal_upload18_noGAT_v66_20261028'
PERIODS = ['T1', 'T2', 'T3', 'T4']
sys.path.insert(0, str(ROOT / 'analysis-scripts/0353-2-B3_segmentation_pilot/root_axis_junction_v83'))
from prepare import js, sha, csvout
from evaluate import metric
from rendering import Canvas, isolated, overlay


def unpack(a, shape, dtype=np.uint8):
    return np.repeat(np.array(a[::2], dtype), np.array(a[1::2], np.int64)).reshape(shape)


def pack(a):
    a = a.ravel(); starts = np.r_[0, np.flatnonzero(a[1:] != a[:-1]) + 1]
    return np.column_stack((a[starts], np.diff(np.r_[starts, len(a)]))).ravel().tolist()


def main():
    sys.path.insert(0, str(ROOT / 'analysis-scripts'))
    from dataset_identity import require_active_position
    assert require_active_position('0353-2-B3') == '0353-2-B3'
    OUT.mkdir(exist_ok=True); (OUT / 'corrected_reference').mkdir(exist_ok=True)
    source = DATA / 'source_progress_24.json'
    modelpaths = dict(original_U_Net_V66=BASE / 'unet/full', optimized_U_Net_V83=V83 / 'full', reviewed_U_Net_V85=V85 / 'full')
    immutable = [source, V83.parent / 'joint_best.pt', V84 / 'root_annotation_progress_24_user_corrected_T2_new_root.json']
    immutable += [p / (t + '_classes.png') for p in list(modelpaths.values()) + [DATA / 'reference', V84 / 'corrected_reference'] for t in PERIODS]
    checksums = {str(p): sha(p) for p in immutable}
    assert checksums[str(source)] == 'cb5a9326b73003a81b44961300de88806378a771514e886faff168a715bf2c8e'
    original = {t: np.array(Image.open(DATA / 'reference' / (t + '_classes.png'))) for t in PERIODS}
    one = {t: np.array(Image.open(V84 / 'corrected_reference' / (t + '_classes.png'))) for t in PERIODS}
    revised = {t: a.copy() for t, a in one.items()}
    doc = json.loads((V84 / 'root_annotation_progress_24_user_corrected_T2_new_root.json').read_text())
    frame = next(v for v in doc['root_structure_annotation']['periods'] if v['period'] == 'T2')
    own = unpack(frame['owner_map_rle'], (1108, 2541), np.int32)
    # The pictured diagonal is the already user-defined new lateral 0017-N6.
    # This identifier is used ONLY to edit the user-confirmed annotation interval.
    # It is never used to change any prediction or to fit any model.
    ids = [s['index'] for s in frame['segments'] if s['root_id'] == 'R20569']
    known = next(h for h in doc['root_structure_annotation']['root_track_hierarchy'] if h['root_id'] == 'R20569')
    assert known['label'] == '0017-N6' and known['annotation_status_v60'] == 'user_defined'
    edit = np.isin(own, ids) & (one['T2'] == 3)
    yy, xx = np.nonzero(edit)
    bbox = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
    assert edit.sum() == 916 and bbox == [1227, 458, 1392, 653]
    lab, _ = ndi.label(one['T2'] == 3, np.ones((3, 3), np.uint8))
    assert sorted(int((lab[edit] == k).sum()) for k in np.unique(lab[edit])) == [89, 827]
    revised['T2'][edit] = 2
    e = next(v for v in doc['images'] if v['name'] == 'T2_RGB.png')
    temporal = unpack(e['layers']['temporal'], edit.shape)
    assert (temporal[edit] == 2).all()
    temporal[edit] = 1; e['layers']['temporal'] = pack(temporal)
    spatial = np.load(DATA / 'spatial.npy')
    audit = dict(period='T2', root_label='0017-N6', root_id='R20569', bbox=bbox, added_changed_pixels=916,
                 reason='User explicitly confirmed this pictured root is new and the U-Net prediction is correct; extension annotation is erroneous.',
                 changed_pixels_by_split={name: int((edit & (spatial == k)).sum()) for k, name in [(-1, 'unassigned'), (0, 'train'), (1, 'validation'), (2, 'internal_test')]},
                 annotation_code_before=2, annotation_code_after=1, derived_PNG_before=3, derived_PNG_after=2,
                 weights_and_predictions_unchanged=True, no_training=True, root_geometry_and_IDs_unchanged=True)
    doc['manual_reference_corrections_v86'] = [audit]
    corrected_json = OUT / 'root_annotation_progress_24_two_user_confirmed_new_roots.json'
    corrected_json.write_text(json.dumps(doc, ensure_ascii=False, separators=(',', ':')))
    original_doc = json.loads(source.read_text()); readback = json.loads(corrected_json.read_text())
    first_edit = one['T2'] != original['T2']; total = first_edit | edit
    assert first_edit.sum() == 457 and not (first_edit & edit).any() and total.sum() == 1373
    for a, b in zip(readback['images'], original_doc['images']):
        for key in b:
            if key != 'layers': assert a[key] == b[key]
        for layer in b['layers']:
            if a['name'] == 'T2_RGB.png' and layer == 'temporal':
                old = unpack(b['layers'][layer], total.shape); new = unpack(a['layers'][layer], total.shape)
                assert np.array_equal(old != new, total)
            else: assert a['layers'][layer] == b['layers'][layer]
    assert readback['root_structure_annotation'] == original_doc['root_structure_annotation']
    for t in PERIODS:
        assert np.array_equal(revised[t] > 0, original[t] > 0)
        assert np.array_equal(revised[t] == 255, original[t] == 255)
        Image.fromarray(revised[t]).save(OUT / 'corrected_reference' / (t + '_classes.png'))
    Image.fromarray(edit.astype(np.uint8) * 255).save(OUT / 'T2_second_correction_pixels.png')
    Image.fromarray(total.astype(np.uint8) * 255).save(OUT / 'T2_all_correction_pixels.png')
    maps = {name: {t: np.array(Image.open(p / (t + '_classes.png'))) for t in PERIODS} for name, p in modelpaths.items()}
    rows = []; result = {}
    refs = dict(source24_original=original, first_user_correction_V84=one, both_user_corrections_V86=revised)
    for scope in ['internal_fixed_root_spatial_holdout', 'all_annotated_fixed_root_pixels']:
        result[scope] = {}
        for version, reference in refs.items():
            yy = []; pp = {name: [] for name in maps}
            for t in PERIODS[1:]:
                take = np.isin(reference[t], [1, 2, 3]) & (maps['original_U_Net_V66'][t] > 0)
                if scope == 'internal_fixed_root_spatial_holdout': take &= spatial == 2
                yy.append(reference[t][take].astype(np.int64) - 1)
                for name in maps: pp[name].append(maps[name][t][take].astype(np.int64) - 1)
            result[scope][version] = {}
            for name, p in pp.items():
                m = metric(np.concatenate(yy), np.concatenate(p)); result[scope][version][name] = m
                for r in m['classes']: rows.append(dict(scope=scope, reference=version, prediction=name, **r))
    csvout(OUT / 'U_Net_Dice_same_predictions_reference_revision.csv', rows)
    js(OUT / 'evaluation_results.json', result)
    audit['total_corrected_pixels_from_original24'] = 1373
    audit['first_correction_pixels'] = 457
    audit['prediction_types_on_newly_corrected_pixels'] = {name: {str(k): int((a['T2'][edit] == k).sum()) for k in [1, 2, 3]} for name, a in maps.items()}
    js(OUT / 'reference_correction_audit.json', audit)
    for p, h in checksums.items(): assert sha(p) == h
    js(OUT / 'verification.json', dict(passed=True, no_training=True, no_prediction_edit=True, all_weights_and_prior_files_unchanged=True, geometry_and_IDs_unchanged=True, old_source24_preserved=True, added_reference_edits=916, cumulative_reference_edits=1373, immutable_inputs_sha256=checksums, corrected_JSON_sha256=sha(corrected_json)))
    figdir = OUT / 'figures'; figdir.mkdir(exist_ok=True)
    manifest = []
    cv = Canvas(4100, 3820, figdir); cv.text(2050, 25, 'U-Net results with two user-confirmed annotation corrections', 56)
    cv.text(2050, 115, 'Predictions and model weights unchanged | Only the reference is revised in this export', 32)
    rgb = {t: np.array(Image.open(BASE / 'data/upper' / (t + '_RGB.png'))) for t in PERIODS}
    xx = [130, 1130, 2130, 3130]; w = 850; h = 371
    for x, s in zip(xx, ['RGB', 'Latest corrected reference', 'Optimized U-Net V83', 'Reviewed U-Net V85']): cv.text(x + w // 2, 210, s, 34)
    for i, t in enumerate(PERIODS):
        y = 280 + i * 825; cv.text(25, y + 320, t, 44, 'lt'); cv.panel(rgb[t], xx[0], y + 180, w, h)
        for x, a in zip(xx[1:], [revised[t], maps['optimized_U_Net_V83'][t], maps['reviewed_U_Net_V85'][t]]):
            cv.panel(overlay(rgb[t], a), x, y, w, h); cv.panel(isolated(a), x, y + h + 10, w, h, True)
    cv.legend(3700); manifest.append(cv.save('Fig1_Four_period_latest_reference_same_UNet_predictions'))
    for i, roi in enumerate([[1368, 50, 1599, 202], [1182, 429, 1447, 708]], 1):
        x, y, x1, y1 = roi; cv = Canvas(3200, 1200, figdir)
        cv.text(1600, 25, 'User-confirmed new root: annotation correction ' + str(i), 54)
        cv.text(1600, 110, 'T2 | Purple: persistent | Green: new | Yellow: extension | Predictions unchanged', 30)
        for xc, s, a in zip([100, 880, 1660, 2440], ['Original annotation', 'Corrected annotation', 'U-Net V83', 'U-Net V85'], [original['T2'], revised['T2'], maps['optimized_U_Net_V83']['T2'], maps['reviewed_U_Net_V85']['T2']]):
            cv.text(xc + 325, 210, s, 34); cv.panel(isolated(a[y:y1, x:x1]), xc, 280, 650, 650, True)
        cv.text(1600, 1030, 'Reference-only change; no root pixel, root identifier, model weight or prediction is edited.', 30)
        manifest.append(cv.save('Fig2_Reference_correction_' + str(i)))
    js(OUT / 'figure_manifest.json', dict(files=manifest, dpi=600, black_bold_Times_New_Roman=True, geometry_unchanged=True))
    text = '# Two user-confirmed reference corrections; fixed U-Net predictions\n\n'
    text += 'No fitting, retraining, model-weight update, postprocessing update or prediction edit is performed in V86. The original V66, optimized V83 and reviewed V85 prediction maps are each scored unchanged under all reference versions. Their previous quantitative records remain preserved.\n\n'
    text += 'Reference revision1: T2 bbox [1423,105,1544,147],457 extension pixels changed to new, as explicitly confirmed by the user. Revision2: the pictured diagonal new lateral0017-N6 (existing manual root R20569), bbox [1227,458,1392,653],916 extension pixels changed to new. This includes827 pixels on the diagonal and89 on the same-root short attachment-side interval. No mother-root or other root class is changed. Root IDs and all four root masks are unchanged. The cumulative reference change is1373 T2 pixels; all other dates/layers and the original24 JSON are preserved. JSON temporal codes1=new,2=extension differ from derived PNG codes2=new,3=extension.\n\n'
    text += 'These are manual reference corrections prompted by user review, not improvements from training. Scores against the corrected reference are internal reference agreement and cannot be interpreted as newly demonstrated independent accuracy. All scopes are conditional on the immutable root foreground; segmentation omissions are outside this temporal-only task. T1 baseline is excluded from three-class metrics. Both full-area and the unchanged internal-test mask are exported separately.\n\n'
    text += '| Scope | Reference | Prediction | New | Persistent | Extension | Macro |\n|---|---|---|---:|---:|---:|---:|\n'
    for scope, references in result.items():
        for ref, models in references.items():
            for name, m in models.items():
                by = {r['name']: r['Dice'] for r in m['classes']}
                text += f"| {scope} | {ref} | {name} | {by['newly_observed']:.6f} | {by['persistent']:.6f} | {by['tip_extension']:.6f} | {m['macro_Dice']:.6f} |\n"
    text += '\nSecond correction split counts: ' + json.dumps(audit['changed_pixels_by_split']) + '\n'
    (OUT / 'RESULTS.md').write_text(text)
    (OUT / 'source').mkdir(exist_ok=True)
    for p in HERE.glob('*.py'): shutil.copy2(p, OUT / 'source' / p.name)
    js(OUT / 'completion.json', dict(state='complete', no_training=True, no_prediction_edit=True, cumulative_user_confirmed_reference_changes=1373))
    print(json.dumps(dict(audit=audit, test=result['internal_fixed_root_spatial_holdout'], full=result['all_annotated_fixed_root_pixels']), indent=2), flush=True)


if __name__ == '__main__':
    main()
