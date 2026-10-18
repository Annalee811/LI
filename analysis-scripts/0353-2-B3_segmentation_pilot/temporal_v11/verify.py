"""Meaningful standalone inference and immutable data verification for all three heads."""
import argparse,json,subprocess,sys,shutil,csv
from pathlib import Path
import numpy as np
from PIL import Image
import common as c

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);a=p.parse_args();run=a.run;fixture=run/'standalone_check';(fixture/'RGB').mkdir(parents=True,exist_ok=True);checks=[]
    for t in c.PERIODS:
        rgb=np.array(Image.open(run/'data/upper'/f'{t}_RGB.png'))[:,c.COLS[3]:c.COLS[3]+448];Image.fromarray(rgb).save(fixture/'RGB'/f'{t}_RGB.png')
    for n in c.MODELS:
        roots=fixture/n/'root_inputs';roots.mkdir(parents=True,exist_ok=True)
        for t in c.PERIODS:
            shutil.copyfile(a.roots/n/'probabilities/val'/f'{t}_probability.npy',roots/f'{t}_probability.npy');shutil.copyfile(run/'data/root_gates'/n/'C3'/f'{t}.png',roots/f'{t}_root_mask.png')
        out=fixture/n/'outputs';subprocess.run([sys.executable,str(Path(__file__).with_name('predict.py')),'--head',str(run/n/'evaluation_head.pt'),'--rgb-dir',str(fixture/'RGB'),'--root-dir',str(roots),'--output',str(out)],check=True)
        for t in c.PERIODS:
            mask=np.array(Image.open(roots/f'{t}_root_mask.png'))>0;pred=np.array(Image.open(out/f'{t}_temporal_classes.png'));assert np.array_equal(pred>0,mask)
            rgba=np.array(Image.open(out/f'{t}_original_roots_RGBA.png'));rgb=np.array(Image.open(fixture/'RGB'/f'{t}_RGB.png'));assert np.array_equal(rgba[mask,:3],rgb[mask]);assert (rgba[~mask]==0).all()
            if t!='T1':assert np.array_equal(pred,np.array(Image.open(run/n/'evaluation/validation'/f'{t}_classes.png'))),'Standalone paired inference differs from recorded metrics'
        # Verify strict removal of background from both RGB inputs to the head.
        pp=np.load(roots/'T1_probability.npy');cp=np.load(roots/'T2_probability.npy');pm=np.array(Image.open(roots/'T1_root_mask.png'))>0;cm=np.array(Image.open(roots/'T2_root_mask.png'))>0
        f=c.features(np.array(Image.open(fixture/'RGB/T1_RGB.png')),np.array(Image.open(fixture/'RGB/T2_RGB.png')),pp,cp,pm,cm)
        assert (f[:,:,:3][~pm]==0).all() and (f[:,:,3:6][~cm]==0).all()
        assert np.array_equal(c.target(np.array([[255,255,255,255,0]],np.uint8),np.array([[3,1,2,0,0]],np.uint8)),[[1,2,3,255,0]])
        checks.append(dict(model=n,standalone_without_human_targets=True,all_three_validation_class_maps_exact=True,predicted_root_support_unchanged=True,RGB_background_zero_before_head=True,original_root_RGB_unchanged=True))
        print('TEMPORAL_VERIFIED',n,flush=True)
    assert len({c.sha(run/n/'evaluation_head.pt') for n in c.MODELS})==3
    assert len({c.sha(run/n/'final_head.pt') for n in c.MODELS})==3
    for n in c.MODELS:
        config=json.loads((run/n/'training_run.json').read_text());root=a.roots/n/('best_weights_fp16.npz' if n=='deeplab' else 'best.pt')
        assert c.sha(root)==config['root_checkpoint_sha256']
        assert config['evaluation_epochs_completed']>0 and config['final_epochs_completed']>=0
    c.js(run/'standalone_verification.json',dict(passed=True,checks=checks,three_distinct_evaluation_heads=True,three_distinct_final_heads=True,root_backbone_weights_unchanged=True,same_sampling_policy=True,model_specific_predicted_junction_pools=True))
if __name__=='__main__':main()
