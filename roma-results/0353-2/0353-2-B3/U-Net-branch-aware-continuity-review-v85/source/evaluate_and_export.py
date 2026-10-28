"""Freeze inference-only review, then score references and export transparent audits."""
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path('/workspace/LI')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'analysis-scripts/0353-2-B3_segmentation_pilot/root_axis_junction_v83'))
from evaluate import metric
from prepare import csvout, js, sha
from rendering import Canvas, isolated, overlay

OUT = ROOT / 'results/B3_branch_continuity_review_v85_20261028'
PREV = ROOT / 'results/B3_root_axis_junction_v83_20261028/final'
DATA = ROOT / 'results/B3_original_UNet_latestCG_upload24_v82_20261028/data'
BASE = ROOT / 'results/B3_temporal_upload18_noGAT_v66_20261028'
REV = ROOT / 'results/B3_reference_local_correction_v84_20261028'
PERIODS = ['T1', 'T2', 'T3', 'T4']


def main():
    verify = json.loads((OUT / 'candidate_verification.json').read_text())
    assert verify['passed']
    val = json.loads((OUT / 'validation_review.json').read_text())
    assert val['candidate']['macro_Dice'] >= val['before']['macro_Dice']
    # Freeze every prediction before reading any internal-test results.
    (OUT / 'full').mkdir(exist_ok=True)
    for t in PERIODS:
        shutil.copy2(OUT / 'candidate_full' / (t + '_classes.png'), OUT / 'full' / (t + '_classes.png'))
    frozen = {t: sha(OUT / 'full' / (t + '_classes.png')) for t in PERIODS}
    js(OUT / 'prediction_freeze.json', dict(no_training=True, selection='Fixed inference rules; validation macro nondecreasing. Only two validation pixels changed; this is weak support, not proof of generalization.', prediction_sha256=frozen, test_metrics_opened_after_this_freeze=True))
    spatial = np.load(DATA / 'spatial.npy')
    maps = {name: {t: np.array(Image.open(p / (t + '_classes.png'))) for t in PERIODS} for name, p in [('original_U_Net_V66', BASE / 'unet/full'), ('previous_optimized_V83', PREV / 'full'), ('branch_aware_review_V85', OUT / 'full')]}
    refs = {name: {t: np.array(Image.open(p / (t + '_classes.png'))) for t in PERIODS} for name, p in [('source24_original', DATA / 'reference'), ('source24_user_corrected', REV / 'corrected_reference')]}
    results = {}; rows = []; editrows = []
    (OUT / 'changes').mkdir(exist_ok=True)
    for t in PERIODS:
        before = maps['previous_optimized_V83'][t]; after = maps['branch_aware_review_V85'][t]
        assert np.array_equal(before > 0, after > 0)
        if t == 'T1':
            assert np.array_equal(before, after)
        edit = before != after
        Image.fromarray(edit.astype(np.uint8) * 255).save(OUT / 'changes' / (t + '_edited.png'))
        editrows.append(dict(period=t, changed_pixels=int(edit.sum()), foreground_pixels_changed=0,
                             new_to_persistent=int((edit & (before == 2) & (after == 1)).sum()),
                             new_to_extension=int((edit & (before == 2) & (after == 3)).sum()),
                             extension_to_new=int((edit & (before == 3) & (after == 2)).sum()),
                             other_changes=int((edit & ~(((before == 2) & np.isin(after, [1, 3])) | ((before == 3) & (after == 2)))).sum())))
    csvout(OUT / 'Pixel_change_audit.csv', editrows)
    benefits = []
    for scope in ['internal_fixed_root_spatial_holdout', 'all_annotated_fixed_root_pixels']:
        results[scope] = {}
        for refname, gt in refs.items():
            yy = []; pp = {name: [] for name in maps}
            for t in PERIODS[1:]:
                g = gt[t]
                take = np.isin(g, [1, 2, 3]) & (maps['original_U_Net_V66'][t] > 0)
                if scope == 'internal_fixed_root_spatial_holdout':
                    take &= spatial == 2
                b = maps['previous_optimized_V83'][t]; a = maps['branch_aware_review_V85'][t]
                changed = take & (b != a)
                benefits.append(dict(scope=scope, reference=refname, period=t, changed=int(changed.sum()), fixed=int((changed & (b != g) & (a == g)).sum()), harmed=int((changed & (b == g) & (a != g)).sum()), changed_still_wrong=int((changed & (b != g) & (a != g)).sum())))
                yy.append(g[take].astype(np.int64) - 1)
                for name in maps:
                    pp[name].append(maps[name][t][take].astype(np.int64) - 1)
            results[scope][refname] = {}
            for name, p in pp.items():
                m = metric(np.concatenate(yy), np.concatenate(p)); results[scope][refname][name] = m
                for row in m['classes']:
                    rows.append(dict(scope=scope, reference=refname, variant=name, **row))
    csvout(OUT / 'Dice_metrics_two_reference_versions.csv', rows)
    csvout(OUT / 'Edit_benefits_and_harms.csv', benefits)
    js(OUT / 'evaluation_results.json', results)
    # Root-identity targets are audit-only and enter only after predictions have been frozen.
    rootrows = []
    for t in PERIODS[1:]:
        px = dict(np.load(DATA / 'unet' / (t + '_pixels.npz')))
        use = px['split'] == 2
        for rid in np.unique(px['rid'][use]):
            if rid < 0:
                continue
            take = use & (px['rid'] == rid)
            if take.sum() < 10:
                continue
            r = dict(period=t, root_target=int(rid), pixels=int(take.sum()))
            for name in maps:
                v = maps[name][t].ravel()[px['pixel'][take]]
                r[name + '_new_extension_conflict5'] = bool(min((v == 2).sum(), (v == 3).sum()) >= 5)
            rootrows.append(r)
    csvout(OUT / 'Known_root_new_extension_conflicts.csv', rootrows)
    root_summary = {name: sum(r[name + '_new_extension_conflict5'] for r in rootrows) for name in maps}
    # Reuse the SAME 80 scoreable biological junctions; do not change the denominator.
    import pickle
    from topology import fuse_crossing_bridges
    with (PREV.parent / 'data/topology.pkl').open('rb') as f:
        top = pickle.load(f)
    old_junctions = list(csv.DictReader((PREV / 'Confirmed_two_root_junction_third_type_audit.csv').open(encoding='utf-8-sig')))
    for t in PERIODS:
        fuse_crossing_bridges(top['topos'][t])
    junctionrows = []
    for r in old_junctions:
        t = r['period']; hub = top['topos'][t]['hubs'][int(r['hub'])]; q = hub['points']
        types = [int(v) for v in r['allowed_reference_types'].split(',')]
        x0 = max(0, int(q[:, 0].min()) - 2); x1 = min(2541, int(q[:, 0].max()) + 3)
        y0 = max(0, int(q[:, 1].min()) - 2); y1 = min(1108, int(q[:, 1].max()) + 3)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        near = np.min((xx[:, :, None] - q[:, 0]) ** 2 + (yy[:, :, None] - q[:, 1]) ** 2, axis=2) <= 4
        near &= (maps['original_U_Net_V66'][t][y0:y1, x0:x1] > 0) & (spatial[y0:y1, x0:x1] == 2)
        v = dict(period=t, hub=int(r['hub']), allowed_reference_types=r['allowed_reference_types'], core_pixels=int(near.sum()))
        for name in maps:
            v[name + '_third_type_pixels'] = int((near & ~np.isin(maps[name][t][y0:y1, x0:x1], types)).sum())
        junctionrows.append(v)
    csvout(OUT / 'Confirmed_two_root_junction_audit.csv', junctionrows)
    junction_summary = {name: dict(junctions=sum(r[name + '_third_type_pixels'] > 0 for r in junctionrows), pixels=sum(r[name + '_third_type_pixels'] for r in junctionrows)) for name in maps}
    js(OUT / 'structural_audit.json', dict(root_records=len(rootrows), new_extension_conflicts=root_summary, confirmed_two_root_junctions=len(junctionrows), third_type_junctions=junction_summary))
    for p, h in verify['previous_prediction_and_checkpoint_sha256'].items():
        assert sha(p) == h
    assert all(sha(OUT / 'full' / (t + '_classes.png')) == frozen[t] for t in PERIODS)
    js(OUT / 'verification.json', dict(passed=True, no_training=True, unchanged_model_checkpoint=True, U_Net_segmentation_geometry_unchanged=True, all_prior_outputs_unchanged=True, source24_preserved=True, T1_unchanged=True, four_full_fields_processed=True, predictions_frozen_before_test_scoring=True, manual_reference_or_IDs_in_prediction_inputs=False, changes=editrows, prediction_sha256=frozen, reference_correction='Only 457 explicitly user-confirmed T2 pixels are corrected in separate V84; no other annotation is edited.'))

    figdir = OUT / 'figures'; figdir.mkdir(exist_ok=True)
    rgb = {t: np.array(Image.open(BASE / 'data/upper' / (t + '_RGB.png'))) for t in PERIODS}
    figures = []
    cv = Canvas(4100, 3820, figdir)
    cv.text(2050, 30, 'U-Net: branch-aware temporal continuity review', 64)
    cv.text(2050, 120, 'Frozen ConvLSTM + GAT | No retraining | Native root geometry unchanged', 34)
    xx = [130, 1130, 2130, 3130]; w = 850; h = 371
    for x, label in zip(xx, ['RGB', 'User-corrected reference', 'Previous optimized V83', 'Branch-aware review V85']):
        cv.text(x + w // 2, 220, label, 34)
    for i, t in enumerate(PERIODS):
        y = 290 + i * 825; cv.text(30, y + 310, t, 46, 'lt'); cv.panel(rgb[t], xx[0], y + 180, w, h)
        for x, a in zip(xx[1:], [refs['source24_user_corrected'][t], maps['previous_optimized_V83'][t], maps['branch_aware_review_V85'][t]]):
            cv.panel(overlay(rgb[t], a), x, y, w, h); cv.panel(isolated(a), x, y + h + 10, w, h, True)
    cv.legend(3700); figures.append(cv.save('Fig1_Four_period_branch_aware_review'))
    change = sum((maps['previous_optimized_V83'][t] != maps['branch_aware_review_V85'][t]).astype(int) for t in PERIODS[1:])
    windows = [('Fig2_Mother_extension_and_new_lateral', [750, 840, 1050, 1108]), ('Fig3_User_corrected_new_root', [1368, 50, 1599, 202])]
    scored = []
    for y in range(0, 1108 - 256 + 1, 64):
        for x in range(0, 2541 - 256 + 1, 64):
            scored.append((int(change[y:y+256, x:x+256].sum()), [x, y, x+256, y+256]))
    for score, roi in sorted(scored, reverse=True):
        if any(max(0, min(roi[2], q[2]) - max(roi[0], q[0])) * max(0, min(roi[3], q[3]) - max(roi[1], q[1])) > 256**2 * .2 for _, q in windows):
            continue
        windows.append(('Fig' + str(len(windows) + 2) + '_All_field_detail', roi))
        if len(windows) == 4:
            break
    for stem, (x, y, x1, y1) in windows:
        cv = Canvas(3250, 3850, figdir); cv.text(1625, 25, 'Root-axis and branch detail across four periods', 54)
        cv.text(1625, 110, f'Native ROI: [{x}, {y}, {x1}, {y1}] | Same images and unchanged segmentation', 32)
        xxx = [120, 910, 1700, 2490]; width = 650
        for j, t in enumerate(PERIODS):
            topy = 210 + j * 850; cv.text(25, topy + 340, t, 42, 'lt')
            for xc, label in zip(xxx, ['Reference', 'Previous V83', 'Branch-aware V85', 'Changed pixels']):
                cv.text(xc + width // 2, topy, label, 32)
            a = maps['previous_optimized_V83'][t][y:y1, x:x1]; b = maps['branch_aware_review_V85'][t][y:y1, x:x1]
            diff = np.full((*a.shape, 3), 255, np.uint8); diff[a > 0] = [210, 210, 210]; diff[a != b] = [213, 94, 0]
            for xc, img in zip(xxx, [isolated(refs['source24_user_corrected'][t][y:y1, x:x1]), isolated(a), isolated(b), diff]):
                cv.panel(img, xc, topy + 60, width, 650, True)
        cv.legend(3690); figures.append(cv.save(stem))
    js(OUT / 'figure_manifest.json', dict(files=figures, dpi=600, black_bold_Times_New_Roman=True, categorical_nearest_neighbor=True, ROI_selection='Two known user-feedback regions and two non-overlapping highest-change windows; not selected by favorable scores.'))
    text = '# All-field U-Net branch/shaft continuity review, no retraining\n\n'
    text += 'This continues the delivered V83 output, not a replacement U-Net baseline. All model weights stay fixed. The ConvLSTM/GAT continuation and root-identity features are reused at inference. No manual root IDs, source24 temporal classes or revised-reference classes enter this refinement. Source24 root IDs enter only the post-freeze audit.\n\n'
    text += 'Rules distinguish biological axes from mother/daughter families. A short confirmed old shaft may produce a much longer terminal extension; old/root length fraction alone cannot indicate new birth. A mother\'s established distal axis may keep or recover extension when earlier tip evidence or a same-axis stable extension anchor supports it. A separate new lateral remains new, including its attachment to either a persistent OR extending mother. Conversely, newly colored intervals with coherent earlier identity evidence become persistent. New-axis extension islands and short persistent islands are corrected conservatively. Shared crossing pixels stay in the incident-root types. All proposed intervals, junctions and unresolved cases are exported.\n\n'
    text += 'This is inference plus explicit topology/history decoding, not additional neural training or a claim of raw-model-only corrections. Ambiguous old/new axes and possible extension without reliable parent/tip evidence remain unchanged. Every path is processed computationally; this does not mean every biological root was independently verified by a human. No root pixel is added, removed, widened or shortened. T1 is unchanged.\n\n'
    text += 'The separate V84 reference revision changes exactly 457 T2 extension pixels to new at the user\'s explicit direction. That revision has no internal test/validation pixels. No other manual annotation is changed. Both reference versions are scored separately. Full-area reference agreement is in-sample descriptive agreement, not independent accuracy. The entire B3 is a development position. Conditional root-foreground temporal scores exclude binary segmentation misses and T1 baseline.\n\n'
    text += 'Validation evidence is weak: only two validation pixels changed, both corrected; extension validation Dice remains zero. Therefore these checks do not establish reliable extension generalization. Predictions are frozen before test scoring; no rule is fitted to test-score gains. Known user examples are development checks.\n\n'
    text += '| Scope | Reference | Variant | New | Persistent | Extension | Macro |\n|---|---|---|---:|---:|---:|---:|\n'
    for scope, refm in results.items():
        for ref, models in refm.items():
            for name, m in models.items():
                by = {r['name']: r['Dice'] for r in m['classes']}
                text += f"| {scope} | {ref} | {name} | {by['newly_observed']:.6f} | {by['persistent']:.6f} | {by['tip_extension']:.6f} | {m['macro_Dice']:.6f} |\n"
    text += '\nStructural checks (different denominators from pixel Dice):\n\n```json\n' + json.dumps(dict(root_records=len(rootrows), root_conflicts=root_summary, junctions=len(junctionrows), third_type=junction_summary), indent=2) + '\n```\n'
    (OUT / 'RESULTS.md').write_text(text)
    (OUT / 'source').mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):
        shutil.copy2(p, OUT / 'source' / p.name)
    js(OUT / 'completion.json', dict(state='complete', no_training=True, class_detail_review=True, prior_checkpoint_preserved=True, all_four_fields=True, unresolved_cases_retained=True))
    print(json.dumps(dict(changes=editrows, structural=root_summary, junctions=junction_summary, test=results['internal_fixed_root_spatial_holdout']['source24_user_corrected']), indent=2), flush=True)


if __name__ == '__main__':
    main()
