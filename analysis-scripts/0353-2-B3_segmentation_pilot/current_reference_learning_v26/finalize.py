"""Wait for saved monitors, audit the current reference, then build composites."""
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
import common as c


def main():
    parser=argparse.ArgumentParser()
    for key in ['run','source','trajectory']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();here=Path(__file__).resolve().parent
    if (args.run/'completion.json').exists():
        assert c.read(args.run/'verification.json')['passed'];print('REUSE_COMPLETED_FIGURES',flush=True);return
    while True:
        completed=[]
        for model in c.MODELS:
            try:
                if c.read(args.run/model/'completion.json')['state']=='complete':completed.append(model)
            except (FileNotFoundError,json.JSONDecodeError):pass
        c.js(args.run/'finalization_status.json',dict(completed=completed,pending=[m for m in c.MODELS if m not in completed]))
        if len(completed)==3:break
        time.sleep(15)
    data=c.Data(args.run);validation=[r for r in data.rows if r['split']=='val'];npos=0;valid=0
    for row in validation:
        _,mask=data.load(row);npos+=int((mask==255).sum());valid+=int((mask!=128).sum())
    audit={};snapshots=[];metrics=[]
    for model in c.MODELS:
        folder=args.run/model;rows=c.csvread(folder/'history.csv');original=c.csvread(args.trajectory/model/'history.csv')
        assert len(rows)==30
        expected=c.csvread(args.trajectory/model/'training_draws.csv');draws=c.csvread(folder/'training_draws.csv')
        assert [(r['epoch'],r['image_id']) for r in draws]==[(r['epoch'],r['image_id']) for r in expected]
        for row,previous in zip(rows,original):
            assert row['epoch']==previous['epoch']
            assert float(row['learning_rate'])==float(previous['learning_rate'])
            tp,fp,fn,tn=[int(row['validation_'+key]) for key in ['TP','FP','FN','TN']]
            assert tp+fn==npos and tp+fp+fn+tn==valid
            assert abs(float(row['validation_Dice'])-2*tp/(2*tp+fp+fn))<1e-12
            assert abs(float(row['validation_IoU'])-tp/(tp+fp+fn))<1e-12
        suffix='.npz' if model=='deeplab' else '.pt'
        paths=[folder/'epoch_checkpoints'/('epoch_%03d'%epoch+suffix) for epoch in range(1,31)]
        for epoch,path in enumerate(paths,1):
            assert path.is_file() and path.stat().st_size>10_000_000
            snapshots.append(dict(model=model,epoch=epoch,path=str(path),bytes=path.stat().st_size,SHA256=c.sha(path)))
        if model=='deeplab':
            with np.load(paths[-1]) as weights:
                assert len(weights.files)==len(c.read(folder/'weight_names.json'))
                assert weights['v0000'].dtype==np.float32
        else:
            import torch
            saved=torch.load(paths[-1],map_location='cpu',weights_only=True)
            assert saved['epoch']==30 and saved['current_reference_manifest_SHA256']==c.sha(args.run/'data/manifest.csv')
            del saved
        reported=c.read(folder/'completion.json')['reported_final_checkpoint_SHA256']
        old=args.source/model/('best_weights_fp16.npz' if model=='deeplab' else 'best.pt')
        assert reported==c.sha(old)
        audit[model]=dict(epochs=30,per_epoch_checkpoints=30,training_draws=1920,optimizer_updates=960,current_validation_positive_pixels=npos,current_validation_valid_pixels=valid,draw_order_and_LR_match_recorded_trajectory=True,reported_final_checkpoint_unchanged=True,reported_final_checkpoint_SHA256=reported,maximum_training_loss_difference_from_trajectory=max(abs(float(r['train_loss_difference_from_V25'])) for r in rows))
        metrics.append(dict(model=model,epochs=30,last_current_validation_Dice=float(rows[-1]['validation_Dice']),maximum_current_validation_Dice=max(float(r['validation_Dice']) for r in rows),final_reported_model_unchanged=True))
    c.js(args.run/'epoch_checkpoint_manifest.json',snapshots);c.csvout(args.run/'monitor_summary.csv',metrics)
    with (args.run/'figure_build.log').open('w') as log:
        subprocess.run([sys.executable,str(here/'figures.py'),'--run',str(args.run),'--trajectory',str(args.trajectory)],stdout=log,stderr=subprocess.STDOUT,check=True)
    protocol=c.read(args.run/'report/figure_protocol.json');assert protocol['current_reference_only_in_figures'] and not protocol['old_reference_metrics_displayed']
    assert len(protocol['outputs'])==8
    for record in protocol['outputs']:
        path=args.run/'report/figures'/record['file'];assert c.sha(path)==record['SHA256']
        with Image.open(path) as image:
            image.load();assert image.mode=='RGB' and image.n_frames==1 and tuple(image.info['dpi'])==(600.,600.)
    shutil.copytree(here,args.run/'source',dirs_exist_ok=True)
    c.js(args.run/'source_manifest.json',[dict(path=str(p.relative_to(here)),bytes=p.stat().st_size,SHA256=c.sha(p)) for p in sorted(here.glob('*')) if p.is_file()])
    c.js(args.run/'verification.json',dict(passed=True,reference_manifest_SHA256=c.sha(args.run/'data/manifest.csv'),models=audit,every_displayed_validation_score_uses_current_reference=True,old_reference_curves_displayed=False,all90_epoch_checkpoints_saved=True,all_8_TIFFs_RGB_600dpi=True,reported_final_test_Dice_unchanged=True,external_independent_test_images=0))
    c.js(args.run/'completion.json',dict(state='complete',figures='report/figures',models=c.MODELS,reference='current user-confirmed annotation',epochs=30,actual_replay_measurements=True,reported_final_models_unchanged=True))
    print('CURRENT_REFERENCE_COMPOSITES_AUDITED_COMPLETE',flush=True)


if __name__=='__main__':main()
