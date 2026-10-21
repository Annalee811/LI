"""Audit saved artifacts and write delivery tables; never train or select models."""
import argparse, csv, hashlib, json, shutil
from pathlib import Path

import numpy as np
from PIL import Image


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def csvout(path, records):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def rank_auc(labels, scores):
    """Independent Mann–Whitney rank calculation with average ranks for ties."""
    order = np.argsort(scores, kind='stable')
    ss = scores[order]
    yy = labels[order].astype(np.int64)
    starts = np.r_[0, np.flatnonzero(ss[1:] != ss[:-1])+1]
    ends = np.r_[starts[1:], len(ss)]
    positives = np.add.reduceat(yy, starts)
    average_ranks = (starts.astype(np.float64)+1+ends)/2
    npos = int(yy.sum())
    nneg = len(yy)-npos
    return float((np.dot(positives, average_ranks)-npos*(npos+1)/2)/(npos*nneg))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    run, source = args.run, args.source
    assert read(run/'completion.json')['state'] == 'complete'
    report = run/'report'
    assert read(report/'verification.json')['passed']
    models = ['unet', 'deeplab', 'segformer']
    names = ['U-Net', 'DeepLabV3+', 'SegFormer-B0']
    manifest = rows(run/'data/manifest.csv')
    allocations = {}
    for row in manifest:
        allocations.setdefault(row['tile_id'], set()).add(row['split'])
    assert all(len(value) == 1 for value in allocations.values())
    assert {s: sum(r['split'] == s for r in manifest) for s in ['train','val','test']} == {'train':200,'val':20,'test':24}
    final_manifest = rows(source/'revised_report/data/manifest.csv')
    final_test = [r for r in final_manifest if r['split'] == 'test']
    test_metrics = {r['model']: r for r in rows(report/'final_test_metrics.csv')}
    summaries, differences, histories = [], [], []
    audit = {}
    for model, name in zip(models, names):
        folder = run/model
        training = read(folder/'training_run.json')
        config = read(folder/'config.json')
        config['continuation'] = read(run/'preparation.json')
        write_json(folder/'config.json', config)
        history = rows(folder/'history.csv')
        original = rows(source/model/'history.csv')
        replay = read(folder/'historical_replay_complete.json')
        assert len(history) == training['epochs_completed']
        assert len(history) >= 30
        assert [int(r['epoch']) for r in history] == list(range(1,len(history)+1))
        for row in history:
            assert all(np.isfinite(float(row[k])) for k in ['train_loss','validation_loss','learning_rate'])
            phase = 'historical_replay_measurement' if int(row['epoch']) <= 20 else 'additional_training_from_original_final_weights'
            assert row['phase'] == phase
            histories.append(dict(model=model, **row))
        old_draws = rows(source/model/'training_draws.csv')
        draws = rows(folder/'training_draws.csv')
        assert len(draws) == 64*len(history)
        for epoch in range(1,21):
            assert [r['image_id'] for r in draws if int(r['epoch']) == epoch] == [r['image_id'] for r in old_draws if int(r['epoch']) == epoch]
            previous = float(original[epoch-1]['train_loss'])
            current = float(history[epoch-1]['train_loss'])
            differences.append(dict(model=model,epoch=epoch,original_training_loss=previous,replayed_training_loss=current,signed_loss_change=current-previous,absolute_loss_change=abs(current-previous),relative_loss_change=abs(current-previous)/abs(previous),replayed_validation_loss=float(history[epoch-1]['validation_loss']),original_validation_loss_was_recorded=False))
        checkpoint = 'best_weights_fp16.npz' if model == 'deeplab' else 'best.pt'
        assert sha(folder/checkpoint) == training['checkpoint_SHA256']
        if not training['baseline_retained']:
            # Check exported weights on validation only, including TF FP16
            # serialization. Test agreement never determines model retention.
            exported = read(folder/'metrics.json')['val']
            baseline = read(folder/'baseline_before_extension.json')
            fixed = baseline['validation_at_original_threshold']
            assert exported['Dice'] > fixed['Dice']+1e-4, 'Exported candidate lost validation-Dice eligibility'
            assert exported['Precision'] >= fixed['Precision']-.005
            assert exported['Recall'] >= fixed['Recall']-.005
            selected_record = history[training['best_epoch']-1]
            assert float(selected_record['validation_loss']) <= baseline['validation_loss']+.01
        preserved_files = 0
        if training['baseline_retained']:
            assert sha(folder/checkpoint) == sha(source/model/checkpoint)
            for subfolder in ['probabilities','upper']:
                for old in sorted((source/model/subfolder).rglob('*')):
                    if old.is_file():
                        new = folder/old.relative_to(source/model)
                        assert sha(new) == sha(old), str(new)
                        preserved_files += 1
        yy, pp = [], []
        for row in final_test:
            gt = np.array(Image.open(source/'revised_report'/row['mask_path']))
            pred = np.load(folder/'probabilities/test'/(row['image_id']+'_probability.npy'))
            valid = gt != 128
            yy.append(gt[valid] == 255)
            pp.append(pred[valid])
        auc = rank_auc(np.concatenate(yy), np.concatenate(pp))
        assert abs(auc-float(test_metrics[model]['ROC_AUC'])) < 1e-10
        rel = [r['training_loss_relative_difference'] for r in replay['epochs']]
        dice_errors = [r['validation_Dice_absolute_difference'] for r in replay['epochs'] if r['validation_Dice_absolute_difference'] is not None]
        summaries.append(dict(model=name,measured_replay_epochs=20,additional_training_epochs=len(history)-20,total_monitoring_epochs=len(history),training_draws_per_epoch=64,total_training_draws=len(draws),total_optimizer_updates=len(draws)//2,max_replay_training_loss_relative_difference=max(rel),mean_replay_training_loss_relative_difference=float(np.mean(rel)),max_replay_logged_Dice_absolute_difference=max(dice_errors),near_original_training_loss=replay['near_original_training_loss'],near_original_logged_Dice=replay['near_original_logged_Dice'],baseline_retained=training['baseline_retained'],selected_epoch=training['best_epoch'],final_Dice=float(test_metrics[model]['Dice']),final_Dice_change=float(test_metrics[model]['Dice_change'])))
        audit[model] = dict(history_and_original_draw_order_verified=True,checkpoint_SHA256=sha(folder/checkpoint),baseline_retained=training['baseline_retained'],old_prediction_files_verified_byte_identical=preserved_files,ROC_AUC_checked_by_independent_rank_method=auc)
    csvout(report/'replay_and_continuation_summary.csv', summaries)
    csvout(report/'original_vs_replayed_training_losses.csv', differences)
    csvout(report/'all_epoch_loss_and_metrics.csv', histories)
    protocol = read(report/'figure_protocol.json')
    unmatched = [r['model'] for r in summaries if not (r['near_original_training_loss'] and r['near_original_logged_Dice'])]
    expected_count = 10+len(unmatched)
    assert len(protocol['outputs']) == expected_count
    primary_rows = rows(report/'primary_loss_plot_data.csv')
    name_to_model = dict(zip(names, models))
    for row in primary_rows:
        summary = next(s for s in summaries if name_to_model[s['model']] == row['model'])
        if int(row['epoch']) <= 20 and summary['model'] in unmatched:
            assert not row['validation_loss'], 'Unmatched replay may not fill original validation loss'
    for record in protocol['outputs']:
        path = report/'figures'/record['file']
        assert sha(path) == record['SHA256']
        with Image.open(path) as image:
            image.load()
            assert image.mode == 'RGB' and image.n_frames == 1
            assert tuple(image.info['dpi']) == (600.,600.)
    text = [
        '# Saved continuation and monitoring results', '',
        'All curves contain actual measurements. Epochs1–20 were replayed from archived starting weights; missing original validation losses were not invented. Epochs21 onward reload the old selected final weights, with a new optimizer. The boundary is marked on the figures.', '',
        '| Model | Replay + additional epochs | Max relative training-loss difference | Max logged Dice difference | Original final retained | Selected epoch |',
        '|---|---:|---:|---:|---|---:|',
    ]
    for row in summaries:
        text.append(f"| {row['model']} | 20 + {row['additional_training_epochs']} | {row['max_replay_training_loss_relative_difference']:.3%} | {row['max_replay_logged_Dice_absolute_difference']:.6f} | {row['baseline_retained']} | {row['selected_epoch']} |")
    text += ['',
        'The proximity flags retain the advance thresholds: relative training-loss difference≤1% and logged validation-Dice difference≤0.005. Every model and epoch is retained in the comparison, including deviations; no unfavorable epochs are omitted. If either check fails, the primary loss figure keeps original epochs1–20training loss and leaves original validation loss missing. A separate supplementary TIFF displays the entire unmatched replay without treating it as reconstructed history.', '',
        'One monitoring epoch comprises32batches×2augmented draws, following the old run; it is not a full sweep of the200training views. These200views are correlated crops of four original dates, not200independent acquisitions. All dates at a spatial coordinate share one allocation.', '',
        'Training-selection monitoring uses the unchanged original validation labels. Final confusion matrices, ROC/PR and test scores use the reporting reference in0353-2-B3_final_paper_figures.zip. These are different reference definitions and their scores must not be combined or substituted. Same-position/teacher-influenced reference agreement is not an independent biological accuracy estimate. External independent annotated images:0.', '',
        f'Loss plots and LR plots are separate for each model, cover all completed epochs, and use no smoothing. All{expected_count}TIFFs are RGB,600dpi,LZW-compressed. Original files and selected baselines remain untouched.', '',
    ]
    (run/'RESULTS.md').write_text('\n'.join(text))
    src = Path(__file__).resolve().parent
    shutil.copytree(src,run/'source',dirs_exist_ok=True)
    write_json(run/'source_manifest.json',[dict(path=str(p.relative_to(src)),bytes=p.stat().st_size,SHA256=sha(p)) for p in sorted(src.glob('*')) if p.is_file()])
    write_json(run/'verification.json',dict(passed=True,models=audit,all_TIFFs_RGB_600dpi=True,figure_count=expected_count,unmatched_replays_not_used_to_impute_original_validation_loss=unmatched,replay_epoch_sequences_complete=True,original_draw_order_checked=True,replay_measurements_not_original_historical_logs=True,annotations_and_split_unchanged=True,test_not_used_for_selection=True,external_independent_test_images=0))
    completion=read(run/'completion.json');completion['figures']='report/figures';completion['audit']='verification.json';write_json(run/'completion.json',completion)
    print('FINAL_ARTIFACT_AUDIT_PASSED',len(summaries),'models /',expected_count,'TIFFs',flush=True)


if __name__ == '__main__':
    main()
