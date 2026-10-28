"""User-directed T4 reference extension->new reassignment; immutable U-Net outputs."""
import json
import shutil
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path('/workspace/LI')
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/B3_T4_all_extension_reference_to_new_v87_20261028'
PREVIOUS = ROOT / 'results/B3_user_confirmed_reference_corrections_v86_20261028'
DATA = ROOT / 'results/B3_original_UNet_latestCG_upload24_v82_20261028/data'
BASE = ROOT / 'results/B3_temporal_upload18_noGAT_v66_20261028'
V83 = ROOT / 'results/B3_root_axis_junction_v83_20261028/final'
V85 = ROOT / 'results/B3_branch_continuity_review_v85_20261028'
PERIODS = ['T1', 'T2', 'T3', 'T4']
sys.path.insert(0, str(ROOT / 'analysis-scripts/0353-2-B3_segmentation_pilot/root_axis_junction_v83'))
from prepare import js, sha, csvout
from evaluate import metric
from rendering import Canvas, isolated, overlay


def unpack(v, shape):
    return np.repeat(np.array(v[::2], np.uint8), np.array(v[1::2], np.int64)).reshape(shape)


def pack(a):
    a = a.ravel(); s = np.r_[0, np.flatnonzero(a[1:] != a[:-1]) + 1]
    return np.column_stack((a[s], np.diff(np.r_[s, len(a)]))).ravel().tolist()


def score(y, p):
    m = metric(y, p)
    cm = np.array(m['confusion'])
    for k, r in enumerate(m['classes']):
        r['reference_present'] = bool(cm[k].sum())
        r['predicted_pixels'] = int(cm[:, k].sum())
        r['false_positive_pixels'] = int(cm[:, k].sum() - cm[k, k])
        if cm[k].sum() + cm[:, k].sum() == 0:
            r['Dice'] = None
            r['IoU'] = None
        r['class_Dice_status'] = 'reference_class_present' if r['reference_present'] else ('no_reference_pixels_but_false_positives' if r['predicted_pixels'] else 'absent_in_both_reference_and_prediction')
    m['macro_Dice_defined_classes'] = float(np.mean([r['Dice'] for r in m['classes'] if r['Dice'] is not None]))
    m['macro_Dice_reference_present_classes'] = float(np.mean([r['Dice'] for r in m['classes'] if r['reference_present']]))
    return m


def main():
    sys.path.insert(0, str(ROOT / 'analysis-scripts'))
    from dataset_identity import require_active_position
    assert require_active_position('0353-2-B3') == '0353-2-B3'
    OUT.mkdir(exist_ok=True); (OUT / 'corrected_reference').mkdir(exist_ok=True)
    source = PREVIOUS / 'root_annotation_progress_24_two_user_confirmed_new_roots.json'
    paths = dict(original_U_Net_V66=BASE / 'unet/full', optimized_U_Net_V83=V83 / 'full', reviewed_U_Net_V85=V85 / 'full')
    immutable = [source, DATA / 'source_progress_24.json', V83.parent / 'joint_best.pt']
    immutable += [p / (t + '_classes.png') for p in list(paths.values()) + [PREVIOUS / 'corrected_reference', DATA / 'reference'] for t in PERIODS]
    hashes = {str(p): sha(p) for p in immutable}
    parent = json.loads(source.read_text()); doc = json.loads(source.read_text())
    e = next(v for v in doc['images'] if v['name'] == 'T4_RGB.png')
    temporal = unpack(e['layers']['temporal'], (1108, 2541))
    edit = temporal == 2
    assert edit.sum() == 784
    temporal[edit] = 1
    e['layers']['temporal'] = pack(temporal)
    spatial = np.load(DATA / 'spatial.npy')
    audit = dict(period='T4', changed_pixels=784, directive='User explicitly requested every T4 manual extension label to become new; U-Net predictions must remain unchanged.', annotation_code_before=2, annotation_code_after=1, derived_PNG_before=3, derived_PNG_after=2, no_training=True, no_prediction_edit=True, previous_two_T2_corrections_preserved=True, cumulative_changed_pixels_from_original24=2157, changed_pixels_by_split={name: int((edit & (spatial == k)).sum()) for k, name in [(-1, 'unassigned'), (0, 'train'), (1, 'validation'), (2, 'internal_test')]}, parent_JSON_sha256=hashes[str(source)])
    doc['manual_reference_corrections_v87'] = [audit]
    corrected_json = OUT / 'root_annotation_progress_24_user_corrected_T2_and_T4.json'
    corrected_json.write_text(json.dumps(doc, ensure_ascii=False, separators=(',', ':')))
    readback = json.loads(corrected_json.read_text())
    for a, b in zip(readback['images'], parent['images']):
        for k in b:
            if k != 'layers': assert a[k] == b[k]
        for layer in b['layers']:
            if a['name'] == 'T4_RGB.png' and layer == 'temporal':
                new = unpack(a['layers'][layer], edit.shape); old = unpack(b['layers'][layer], edit.shape)
                assert np.array_equal(new != old, edit) and not (new == 2).any()
            else: assert a['layers'][layer] == b['layers'][layer]
    assert readback['root_structure_annotation'] == parent['root_structure_annotation']
    before = {t: np.array(Image.open(PREVIOUS / 'corrected_reference' / (t + '_classes.png'))) for t in PERIODS}
    revised = {t: a.copy() for t, a in before.items()}
    assert np.array_equal(before['T4'] == 3, edit)
    revised['T4'][edit] = 2
    for t in PERIODS:
        assert np.array_equal(before[t] > 0, revised[t] > 0)
        assert np.array_equal(before[t] == 255, revised[t] == 255)
        Image.fromarray(revised[t]).save(OUT / 'corrected_reference' / (t + '_classes.png'))
    assert not (revised['T4'] == 3).any()
    Image.fromarray(edit.astype(np.uint8) * 255).save(OUT / 'T4_reference_changed_pixels.png')
    original = {t: np.array(Image.open(DATA / 'reference' / (t + '_classes.png'))) for t in PERIODS}
    assert sum(int((revised[t] != original[t]).sum()) for t in PERIODS) == 2157
    preds = {name: {t: np.array(Image.open(p / (t + '_classes.png'))) for t in PERIODS} for name, p in paths.items()}
    result = {}; rows = []
    for scope in ['internal_fixed_root_spatial_holdout', 'all_annotated_fixed_root_pixels']:
        result[scope] = {}
        for refname, reference in [('source24_original', original), ('two_T2_corrections_V86', before), ('two_T2_and_all_T4_corrections_V87', revised)]:
            result[scope][refname] = {}
            for period_scope in ['T2_T3_T4_combined', 'T4_only']:
                yy = []; pp = {name: [] for name in preds}
                periods = PERIODS[1:] if period_scope == 'T2_T3_T4_combined' else ['T4']
                for t in periods:
                    take = np.isin(reference[t], [1, 2, 3]) & (preds['original_U_Net_V66'][t] > 0)
                    if scope == 'internal_fixed_root_spatial_holdout': take &= spatial == 2
                    yy.append(reference[t][take].astype(np.int64) - 1)
                    for name in preds: pp[name].append(preds[name][t][take].astype(np.int64) - 1)
                result[scope][refname][period_scope] = {}
                for name, p in pp.items():
                    m = score(np.concatenate(yy), np.concatenate(p))
                    result[scope][refname][period_scope][name] = m
                    for r in m['classes']: rows.append(dict(scope=scope, reference=refname, periods=period_scope, prediction=name, **r))
    csvout(OUT / 'U_Net_Dice_reference_revision_T4_and_combined.csv', rows)
    js(OUT / 'evaluation_results.json', result)
    js(OUT / 'reference_correction_audit.json', audit)
    for p, digest in hashes.items(): assert sha(p) == digest
    js(OUT / 'verification.json', dict(passed=True, no_training=True, no_prediction_edit=True, all_weights_and_prior_files_unchanged=True, root_geometry_and_IDs_unchanged=True, previous_T2_corrections_preserved=True, T4_reference_has_no_extension_labels=True, added_reference_edits=784, cumulative_reference_edits=2157, immutable_inputs_sha256=hashes, corrected_JSON_sha256=sha(corrected_json)))
    figdir = OUT / 'figures'; figdir.mkdir(exist_ok=True); manifest = []
    cv = Canvas(4100, 3820, figdir)
    cv.text(2050, 25, 'U-Net: latest user-corrected temporal reference', 62)
    cv.text(2050, 115, 'T4 reference: extension reassigned to new | U-Net predictions and weights unchanged', 32)
    rgb = {t: np.array(Image.open(BASE / 'data/upper' / (t + '_RGB.png'))) for t in PERIODS}
    xx = [130, 1130, 2130, 3130]; w = 850; h = 371
    for x, label in zip(xx, ['RGB', 'Latest corrected reference', 'U-Net V83 (unchanged)', 'U-Net V85 (unchanged)']): cv.text(x + w // 2, 210, label, 34)
    for i, t in enumerate(PERIODS):
        y = 280 + i * 825; cv.text(25, y + 320, t, 44, 'lt'); cv.panel(rgb[t], xx[0], y + 180, w, h)
        for x, a in zip(xx[1:], [revised[t], preds['optimized_U_Net_V83'][t], preds['reviewed_U_Net_V85'][t]]):
            cv.panel(overlay(rgb[t], a), x, y, w, h); cv.panel(isolated(a), x, y + h + 10, w, h, True)
    cv.legend(3700); manifest.append(cv.save('Fig1_Four_period_corrected_T4_reference_same_UNet'))
    cv = Canvas(3300, 4470, figdir); cv.text(1650, 25, 'T4 reference revision: all extension labels become new', 55)
    cv.text(1650, 110, '784 reference pixels changed | Previous T2 corrections retained | No retraining', 32)
    for i, (title, a) in enumerate([('Previous T4 reference', before['T4']), ('Corrected T4 reference', revised['T4']), ('U-Net V85 prediction (unchanged)', preds['reviewed_U_Net_V85']['T4'])]):
        y = 230 + i * 1330; cv.text(1650, y, title, 36); cv.panel(isolated(a), 250, y + 55, 2800, 1221, True)
    cv.legend(4350); manifest.append(cv.save('Fig2_T4_reference_before_after_and_fixed_UNet'))
    js(OUT / 'figure_manifest.json', dict(files=manifest, dpi=600, categorical_nearest_neighbor=True, black_bold_Times_New_Roman=True))
    text = '# T4 reference-only change: extension to new\n\n'
    text += 'The user explicitly instructed that ALL T4 manual extension labels become new. Exactly784 T4 temporal pixels are reassigned; no extension label remains in the T4 manual temporal layer. The earlier1373 explicitly confirmed T2 corrections remain, for2157 cumulative changed pixels relative to original24. All root masks, widths, lengths, identifiers, other annotation layers and prior files are preserved.\n\n'
    text += 'No model is fitted or retrained and no prediction is edited. Original V66, optimized V83 and reviewed V85 predictions are all rescored unchanged. Score changes between V86 and V87 come only from the reference revision; they are not new training gains. The original24 reference and prior quantitative results remain available. Conditional temporal Dice on the fixed root foreground is reported; binary segmentation omissions are outside this task and T1 baseline is excluded. Full annotated-area scores are descriptive in-sample agreement. Internal spatial masks belong to the same development position and are not an independent tube test.\n\n'
    text += 'T4 has zero reference extension support after this revision. Its extension-class entry is0 when a model still predicts extension (false positives), or undefined when absent in both reference and prediction. For T4-only summaries, the macro over reference-present classes uses persistent and new only; the combined T2/T3/T4 score retains all three classes because extension remains in earlier dates. CSV exports both reference support and predicted extension counts.\n\n'
    text += '| Scope | Reference | Periods | Prediction | Persistent | New | Extension | Three-class macro |\n|---|---|---|---|---:|---:|---:|---:|\n'
    for scope, refs in result.items():
        for ref, groups in refs.items():
            for periods, models in groups.items():
                for name, m in models.items():
                    d = [r['Dice'] for r in m['classes']]
                    text += '| ' + ' | '.join([scope, ref, periods, name] + ['undefined' if v is None else f'{v:.6f}' for v in d] + [f"{m['macro_Dice_defined_classes']:.6f}"]) + ' |\n'
    text += '\nT4 changed pixels by spatial mask: ' + json.dumps(audit['changed_pixels_by_split']) + '\n'
    (OUT / 'RESULTS.md').write_text(text)
    (OUT / 'source').mkdir(exist_ok=True)
    for p in HERE.glob('*.py'): shutil.copy2(p, OUT / 'source' / p.name)
    js(OUT / 'completion.json', dict(state='complete', no_training=True, no_prediction_edit=True, T4_reference_extension_to_new=784, cumulative_user_reference_corrections=2157))
    print(json.dumps(dict(audit=audit, metrics=result['internal_fixed_root_spatial_holdout']), indent=2), flush=True)


if __name__ == '__main__': main()
