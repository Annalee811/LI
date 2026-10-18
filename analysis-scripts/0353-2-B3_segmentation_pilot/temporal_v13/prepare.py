"""Export exact adjacent RGB patches and unchanged human temporal targets."""
import argparse,shutil
from pathlib import Path
import numpy as np
from PIL import Image
import common as c

def main():
    a=argparse.ArgumentParser();a.add_argument('--source',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    src=args.source;provenance=__import__('json').loads((src/'dataset/provenance.json').read_text())
    for t in c.PERIODS:
        for layer in ['RGB','mask','temporal']:
            rel=f'dataset/upper/{t}_{layer}.png';assert c.sha(src/rel)==provenance['files'][rel]
    assert c.sha(src/'original_human_progress.json')==provenance['source_progress_sha256']
    shutil.copytree(src/'dataset/upper',out/'upper',dirs_exist_ok=True);shutil.copyfile(src/'original_human_progress.json',out/'original_human_progress.json');(out/'targets').mkdir(exist_ok=True)
    counts=[];rgb={t:np.array(Image.open(out/'upper'/f'{t}_RGB.png')) for t in c.PERIODS};targets={}
    for t in c.PERIODS[1:]:
        mask=np.array(Image.open(out/'upper'/f'{t}_mask.png'));raw=np.array(Image.open(out/'upper'/f'{t}_temporal.png'));y=c.target(mask,raw);targets[t]=y
        Image.fromarray(y).save(out/'targets'/f'{t}_temporal_target.png')
        counts.append(dict(period=t,**{c.NAMES[k]:int((y==k).sum()) for k in [1,2,3]},untyped_root_pixels=int(((y==255)&(mask==255)).sum())))
    rows=[];checks=0
    for col,left in enumerate(c.COLS):
        split='train' if col<3 else 'validation' if col==3 else 'repeated_benchmark'
        for prev,t in zip(c.PERIODS,c.PERIODS[1:]):
            for j,(x,y) in enumerate(c.coords(c.HEIGHT,448)):
                stem=f'{prev}_{t}_C{col}_W{j:02d}';folder=out/'patches'/split;folder.mkdir(parents=True,exist_ok=True)
                row=dict(id=stem,spatial_group=f'C{col}',split=split,previous=prev,current=t,x=left+x,y=y,width=c.SIDE,height=c.SIDE)
                for name,im in [('previous_RGB',rgb[prev]),('current_RGB',rgb[t]),('human_target',targets[t])]:
                    f=folder/f'{stem}_{name}.png';crop=im[y:y+c.SIDE,left+x:left+x+c.SIDE];Image.fromarray(crop).save(f)
                    assert np.array_equal(np.array(Image.open(f)),crop);checks+=1;row[name]=str(f.relative_to(out));row[name+'_sha256']=c.sha(f)
                rows.append(row)
    c.csvout(out/'paired_patch_manifest.csv',rows);c.csvout(out/'human_class_counts.csv',counts)
    c.js(out/'protocol.json',dict(source_annotation_sha256=c.sha(out/'original_human_progress.json'),labels_changed=False,source_images=4,adjacent_transitions=3,upper_size=[2541,1108],training_pairs=sum(r['split']=='train' for r in rows),validation_pairs=sum(r['split']=='validation' for r in rows),repeated_benchmark_pairs=sum(r['split']=='repeated_benchmark' for r in rows),external_independent_test_images=0,source_value_mapping={'1':'newly observed -> output2','2':'tip growth -> output3','3':'persistent -> output1'},T1='baseline only; no temporal class loss',death=False,untyped_root='ignored in temporal loss and temporal metrics',background='Soil/water/debris combined; unpainted pixels treated as background per original B3 training protocol',patches='Native256; overlap128. Patches are correlated and not independent biological samples.',groups='All four dates stay in the same spatial column;64px guards. TrainC0-C2,validationC3,previously inspected benchmarkC4.',final_fit='After freezing head epoch/model selection, fit heads using all B3 upper labels; no unbiased scores claimed for final all-label fit.',lossless_crop_checks=checks))
    c.js(out/'source_integrity.json',dict(passed=True,files={str(f.relative_to(out)):c.sha(f) for f in sorted((out/'upper').glob('*.png'))},annotation_sha256=c.sha(out/'original_human_progress.json')))
    print('TEMPORAL_DATA_READY',len(rows),counts,flush=True)
if __name__=='__main__':main()
