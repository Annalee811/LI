"""Actual training draws exported before augmentation, without changing labels."""
import argparse,csv
from pathlib import Path
import numpy as np
from PIL import Image
import common as c

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run;out=run/'focused_training';out.mkdir(exist_ok=True);rows=[]
    rgb={t:np.array(Image.open(run/'data/upper'/f'{t}_RGB.png')) for t in c.PERIODS}
    for n in c.MODELS:
        with (run/n/'evaluation_training_draws.csv').open(encoding='utf-8-sig') as f:draws=list(csv.DictReader(f))[:96]
        for i,r in enumerate(draws):
            t=r['current'];prev=c.PERIODS[c.PERIODS.index(t)-1];x,y,side=[int(r[k]) for k in ['x','y','side']];folder=out/n;folder.mkdir(exist_ok=True);row=dict(model=n,**r)
            for label,im in [('previous_RGB',rgb[prev]),('current_RGB',rgb[t]),('human_target',np.array(Image.open(run/'data/targets'/f'{t}_temporal_target.png')))]:
                crop=im[y:y+side,x:x+side];name=f'D{i:03d}_{label}.png';Image.fromarray(crop).save(folder/name);assert np.array_equal(np.array(Image.open(folder/name)),crop);row[label]=str((folder/name).relative_to(out))
            col=int(r['column']);dx=x-c.COLS[col]
            for period,label in [(prev,'previous_root_RGB'),(t,'current_root_RGB')]:
                gate=np.array(Image.open(run/'data/root_gates'/n/f'C{col}'/f'{period}.png'))[y:y+side,dx:dx+side]>0
                im=rgb[period][y:y+side,x:x+side]*gate[:,:,None];name=f'D{i:03d}_{label}.png';Image.fromarray(im).save(folder/name);row[label]=str((folder/name).relative_to(out))
            rows.append(row)
    c.csvout(out/'native_draw_manifest.csv',rows);c.js(out/'verification.json',dict(passed=True,actual_training_pairs=len(rows),selection='First96 training draws per model; no validation sampling; before logged augmentation',original_annotation_preserved=True,temporal_labels_revised=True,exact_native_crop_checks=len(rows)*3,model_specific_junction_pools=True))
    summary=dict(original_RGB_files=4,locations=1,external_independent_test_RGB_files=0,train_date_region_views=12,validation_date_region_views=4,repeated_benchmark_date_region_views=4,reference_grid_pair_counts={'train':216,'validation':72,'repeated_benchmark':72},exported_actual_training_pairs=len(rows),interpretation='Spatial views,overlapping crops and augmentations are correlated derivatives of the same4 images,not new independent images',models={})
    for n in c.MODELS:
        summary['models'][n]={}
        for phase in ['evaluation','final']:
            with (run/n/f'{phase}_training_draws.csv').open(encoding='utf-8-sig') as f:summary['models'][n][phase+'_training_draws']=sum(1 for _ in csv.DictReader(f))
    c.js(run/'dataset_summary.json',summary)
    print('FOCUS_TRAINING_EXPORT',len(rows),flush=True)
if __name__=='__main__':main()
